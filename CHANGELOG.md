# 修改日志（Change Log）

本项目已纳入 git 仓库（`https://github.com/lqy7333-del/graph-uav-cpp.git`），本文件仍保留作为代码变更的可读说明。条目按时间倒序（最新在上）。
每次修改请追加一节，写明：**文件 / 变更（旧→新）/ 原因 / 回滚方法**。

---

## 2026-10-15

### 23. PPO 训练诊断指标：定位「critic 收敛但 reward 不涨」
- **文件**：修改 `coverage_env/rl/ppo/ppo.py`、`coverage_env/rl/trainer/distributed.py`。
- **变更（旧 → 新）**：`PPOAgent.update()` 的返回字典新增
  `policy_loss`（advantage 加权 surrogate，与 entropy bonus 分离）、
  `advantage_std`（**归一化前**的原始 advantage 尺度）、`return_mean` / `return_std`、
  `explained_variance`（critic 对 return 方差的解释度 `1 - Var(target-pred)/Var(target)`）、
  `ratio_std`、`clip_fraction`、`approx_kl`、`actor_grad_norm`
  （`clip_grad_norm_` 在裁剪**前**返回的总范数）；`DistributedTrainer` 把这些写入 CSV，
  并在控制台第二行打印其中最关键几项。空 buffer 时返回全零，保证日志 schema 稳定。
  **只增加统计与记录，不改变任何训练数值**（advantage 归一化、loss、梯度裁剪均原样）。
- **原因**：88 轮正式训练显示 `mean_ep_reward` 从 −42.65 漂到 −43.90（块均值标准误 ≈0.26，
  即约 6 个标准误的**反向**漂移），与最初「`critic_loss` 收敛但 reward 不涨」同一症状；
  而 `entropy` 虽已算出却未记录，导致只能靠推理判断是「熵主导」还是「critic 太弱」。
- **首测（3 轮）已给出关键证据**：

  | update | entropy | policy_loss | explained_var | advantage_std | return_std | grad_norm |
  | --- | --- | --- | --- | --- | --- | --- |
  | 2 | 4.348 | −0.00088 | **+0.000** | 8.540 | 8.542 | 0.1343 |
  | 3 | 4.574 | −0.01005 | **+0.002** | 8.099 | 8.106 | 0.1692 |

  两点：① `explained_variance ≈ 0` 且 `advantage_std ≈ return_std` —— **critic 没有减掉任何
  基线**，advantage 几乎等于原始 return，信号被 return 的方差（sd ≈ 8.5）淹没；
  ② `policy_loss ≈ 0` 而 entropy bonus 约 `0.01 × 4.4 = 0.044`，且 `entropy` 在**上升**
  （4.35 → 4.57）—— 熵正则可能正在把策略推回均匀分布。
- **回滚方法**：`git revert <本提交 SHA>`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。

### 22. 修复跨 episode 泛化的地块摆放崩溃：`sample_components` 增加有限重试
- **文件**：修改 `coverage_env/env/instance_generator.py`、
  `coverage_env/tests/test_domain_randomization.py`。
- **变更（旧 → 新）**：`sample_components` 拆为「重试壳 + `_sample_components_once`」；
  前者最多尝试 `attempts`（默认 8）次，捕获摆放失败（`_sample_center` 抛出的
  `RuntimeError`）后整轮重抽，全部失败才抛出带尝试次数的 `RuntimeError`。
- **原因**：**这是条目 15 引入的回归。** 地块摆放是拒绝采样（`_sample_center` 只试 5000 个
  随机位置），密集 workspace 下即使「通常放得下」的地块数量也可能偶尔失败。旧代码整个训练
  只在启动时生成一次实例，所以几乎碰不到；改为**每 episode 重采样**后，一次正式训练要重采样
  数千次，稀有失败就变成了必然。实测一次 200 轮 4 卡训练在**第 88 轮**崩溃：
  ```
  RuntimeError: Unable to place non-overlapping fields in configured workspace
    environment.reset() → _resample_instance() → sample_components() → _sample_center()
  ```
  单独测量 5~20 地块各 120 次（共 1920 次）摆放失败率为 **0**，即 p 很小（约 1e-4 量级），
  与「3000 次重采样中出现 1 次」一致；按 p=1e-3 估计，重试 8 次后失败概率约 1e-24。
- **不影响采样分布**：重试只是多消耗 RNG，成功路径与原来逐位一致（未触发重试时，
  重抽序列完全相同——所以重启训练会复现前 88 轮的数据）。
- **验证**：新增 3 项测试 —— 前两次摆放失败仍能成功返回、尝试耗尽后抛出且信息含尝试次数、
  以及走完整 `reset()` 路径能恢复。全量 106 项，仍只有 2 个改动前即存在的失败。
- **回滚方法**：`git revert <本提交 SHA>`（把重试壳去掉，`_sample_components_once` 改回
  `sample_components`）。

### 21. DDP 训练日志增加「按 episode」指标，使跨地块数量泛化的趋势可评估
- **文件**：修改 `coverage_env/rl/trainer/distributed.py`。
- **变更（旧 → 新）**：`DistributedTrainer.train()` 的 rollout 循环新增逐 episode 累计
  `episode_reward`，在 `env_done` 时（此刻 `environment` 仍描述刚结束的那一集）记录该集
  reward 与地块数；rank 0 的 CSV 新增 `episodes` / `mean_ep_reward` / `mean_ep_fields`
  三列，控制台行追加 `eps=… mean_ep_reward=… mean_ep_fields=…`。
  **只增加记录，不改动任何训练逻辑**（梯度、rollout 长度、DDP 同步均不变）。
- **原因**：地块数量泛化后，128 步 rollout 会跨越 7~10 集、每集地块数不同，`reward` 只是
  rollout 求和，被实例方差主导（实测 −304 ~ −486）；而 `coverage_ratio` / `total_distance`
  是 rollout 结束瞬间的快照，经常为 0（如第 8 轮）。两者都无法支撑验收标准里的
  「reward 上升趋势」与「不同地块数量之间量级可比」两条。
- **效果（前 7 轮实测）**：`mean_ep_reward` 稳定在 **−41.0 ~ −44.3**，而同期
  `mean_ep_fields` 在 9.2 ~ 14.9 之间变化、`reward` 在 −304 ~ −452 之间摆动。
  即归一化后的「单位面积路程成本」与地块数量无关，验收标准第 1 条的量级可比性获得直接证据。
- **回滚方法**：`git revert <本提交 SHA>`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。

### 20. 可见性图连接器优化第二步：按 region 缓存多边形内部可见性子图
- **文件**：修改 `coverage_env/planner/coverage_path.py`、
  `coverage_env/planner/boundary_aware_sweep.py`、
  `coverage_env/tests/test_visibility_connector.py`。
- **变更（旧 → 新）**：
  1) `_shortest_path_inside_polygon` 新增 keyword-only 参数 `pair_cache`（默认 `None`，
     行为与原来完全一致）；对 `first >= 2` 的顶点对查表/写表，
     `start`/`end` 参与的顶点对永不缓存。
  2) `BoundaryAwareSweepPlanner.plan` 每次调用建一个 `visibility_pairs` 字典，透传给
     `_materialize_strokes` → `_connector`；字典生命周期与 `region_geometry` 一致，
     因此天然有界，不存在全局缓存增长或跨几何体下标串味的问题。
  3) `_connector` 由 `@staticmethod` 改为实例方法以携带该字典（仓库内无其他调用点）。
- **原因**：一次 `plan()` 只构造一个 `region_geometry`，却要为每个候选方向 / `reverse_lanes`
  反复调用 `_connector`；实测同一几何体平均被查 **12.7~19.4 次**。而
  `_shortest_path_inside_polygon` 的 O(V²) 谓词扫描里约 **88%** 是「多边形自身顶点两两之间」
  ——这部分只取决于几何体，与 `start`/`end` 无关，是纯重复计算。
- **不改变算法**：顶点集合、候选对的遍历顺序、边插入顺序（决定 Dijkstra 的并列择优）
  全部保持原样，只是跳过重复的 `covers` 谓词。
- **实测（固定权重与动作序列的 8 步 rollout，三次测量）**：
  改前 3.92s → 改后 **1.34~1.36s（2.90×）**；输出完全一致：
  `distance=10921.693648 coverage=0.397438`，
  trajectory `sha256=18599effde4501b3`（含/不含该步骤三次运行全部相同）。
- **回滚方法**：`git revert a689c83`；或删除 `pair_cache` 参数、`plan()` 里的
  `visibility_pairs`、`_materialize_strokes`/`_connector` 的透传参数，并把 `_connector`
  还原为 `@staticmethod`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。
- **测试**：`test_visibility_connector` 增至 6 项，新增「真实输入上 cached vs uncached
  逐点一致」与「只缓存多边形顶点对、不缓存 start/end」；planner/validation 相关 33 项通过。

### 19. 可见性图连接器优化第一步：prepared geometry 加速重复 covers 谓词
- **文件**：修改 `coverage_env/planner/coverage_path.py`；新增
  `coverage_env/tests/test_visibility_connector.py`。
- **变更（旧 → 新）**：`_shortest_path_inside_polygon` 里把 `polygon.buffer(1e-7)` 提出来
  只算一次，并用 `shapely.prepared.prep(...).covers()` 替换裸 `.covers()`：
  ```python
  visible = prep(polygon.buffer(1e-7))
  if visible.covers(direct): ...
  ```
- **原因**：该函数是 rollout 的 CPU 热点（约占 72%），成本来自约 **23 万次** shapely
  `covers` 谓词与约 **23 万次** `LineString` 构造。prepared geometry 会缓存内部空间索引，
  重复谓词快约 2 倍。（顺带发现：`polygon.buffer(1e-7)` 本身只要 0.025 ms，不是收益来源，
  原先算了两次也只是顺手合并。）
- **不改变算法**：顶点集合、O(V²) 候选对、遍历顺序、边插入顺序、Dijkstra 全部不变，
  只替换谓词实现。
- **实测（20 地块、`seed=42`，同一批 296 次 connector 调用逐输入 A/B）**：
  改前 8.38 ms/call → 改后 4.26 ms/call，**1.97×**，输出逐点完全一致；
  端到端 8 步 rollout：7.58s → 3.88s（约 **1.95×**，重复测稳定）。
- **回滚方法**：`git revert cef3fc2`（把 `prep(...)` 还原为 `polygon.buffer(1e-7).covers(...)`），
  删除 `test_visibility_connector.py`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。
- **测试**：新增 `test_visibility_connector.py` 4 项 —— 凹多边形黄金路径逐点断言（4 用例）、
  端点与包含性、prepared 谓词在合成几何与真实区域几何上与标量 `covers` 逐对一致（>500 对）。
  另跑 planner 相关 30 项测试全通过。

### 18. 修复 PPO replay 用错全局状态：`_global_sources` 改为优先 transition 的 state
- **文件**：修改 `coverage_env/rl/decoder/action_decoder.py`（`_global_sources`）、
  `docs/training_performance_notes.md`；新增
  `coverage_env/tests/test_ppo_replay_state.py`。
- **变更（旧 → 新）**：`CoveragePolicy._global_sources(environment, state)` 原先**先看
  environment**——只要 `environment.uavs` 非空就直接返回 `environment.uavs[0]` /
  `environment.depot`，`state` 只在没有 environment 时才用。现改为**先看 `state`**：
  从 `state["uavs"]` / `state["depot"]` 取 UAV 与 depot 特征，缺哪一项才回落到
  environment（`state=None` 或没有 `uavs` 时行为不变，例如 rollout 期的
  `value_for_graph`）。
- **原因**：`PPOAgent.update()` 调 actor 时不传 `environment`，`forward` 回落到
  trainer 绑定的活环境；而 replay 发生在整段 rollout **之后**，环境早已推进，于是重算出的
  `z_G`（进而 log-prob 与 value）与采集时记录的不一致。这直接破坏 on-policy 假设：
  首个 epoch 的 `ratio ≠ 1`，clipped surrogate 从一开始就在优化错误目标，critic 的
  回归目标也自相矛盾。属于条目 17 记录时顺带发现、当时未修的既有问题。
- **实测（10 条 transition，dropout=0，把活环境改到远离 rollout 终点后重算 log-prob）**：

  | | max abs Δlog-prob | ratio 范围 |
  | --- | --- | --- |
  | 修复前（env 优先） | 4.73e-03 | [0.9953, 1.0024] |
  | 修复后（state 优先） | **0.00e+00** | **[1.0000, 1.0000]** |

  即修复后 replay 与采集**逐位一致**，`ratio ≡ 1`。
- **为什么不影响 rollout**：rollout 时传入的 `state` 就是当前状态，`build_uav_feature` /
  `build_depot_feature` 对「UAV 对象」与「state dict」两条分支产生**完全相同**的向量
  （`get_state()` 已包含 `position/max_energy/max_payload/speed/coverage_width/
  remaining_energy/remaining_payload`）。新增测试
  `test_uav_and_depot_features_agree_between_object_and_state` 与
  `test_rollout_encoding_is_unchanged` 守住这一点。
- **影响**：会改变训练行为（这是修复本身的目的——首个 epoch 的 ratio 从 ≠1 变成 =1）。
  若需复现旧行为（不推荐），把 `_global_sources` 的优先级改回 environment 优先即可。
- **回滚方法**：`git revert <本提交 SHA>`（或恢复 `_global_sources` 的 environment 优先
  逻辑），删除 `test_ppo_replay_state.py`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。
- **测试**：新增 `test_ppo_replay_state.py` 6 项（对象/state 特征一致、`_global_sources`
  优先级与兜底、stale 环境不影响 replay、逐条与批量 replay 均复现 stored log-prob、
  rollout 编码不变）。

### 17. 图 batch 化接入 update()：action embedding 切片 + 批量策略前向（默认关闭）
- **文件**：修改 `coverage_env/rl/encoder/batched_encoder.py`、
  `coverage_env/rl/decoder/action_embedding.py`、
  `coverage_env/rl/decoder/global_state_encoder.py`、
  `coverage_env/rl/decoder/action_decoder.py`、`coverage_env/rl/ppo/ppo.py`、
  `docs/training_performance_notes.md`；新增
  `coverage_env/tests/test_batched_policy_update.py`。
- **变更（旧 → 新）**：
  1) `batched_encoder.py` 把批量前向抽成自由函数 `encode_graph_batch(encoder, batch)`，
     任意 `CoverageGraphEncoder` 实例可直接批量化（不再依赖子类或复制权重）。
  2) `ActionEmbedding.forward` 新增 `field_offset`/`boundary_offset`（默认 0）：把本图
     局部索引重定位到批 embedding 的行区间。
  3) `GlobalStateEncoder.forward_batched` 新增：原 `forward` 的 `reshape(-1)` 只服务单图，
     批量版保持 `[G, ...]` 前导维，输出 `[G, hidden]`。
  4) `CoveragePolicy.forward_batched(contexts)` 新增：**1 次 GNN** over 批图 + 逐
     transition 解码，返回与 `forward` 同构的输出（含 `action_distribution`/`mask`/
     `global_embedding`），两条路径可互换。
  5) `PPOAgent.update(buffer, *, batched=False)` 新增开关：批量路径用
     `_batched_minibatch_rows`（1 次编码 + critic 按 minibatch 一次求值），默认路径抽成
     `_per_transition_minibatch_rows`；**默认 `batched=False`，训练行为完全不变**。
  6) 新增 7 项测试：批量前向 vs 逐条前向（绑定环境 / 仅 state 两种）、分布合法性、
     空输入、update 两条路径损失一致、批量路径确实更新参数、空 buffer。
- **原因**：承接条目 16 的原型，用户要求「接着做 action embedding 切片，把只 batch encoder
  这条打通到可以真正接进 update()」。
- **实测结果**：
  - 策略前向等价（12 transition / 93~123 候选）：mask 完全相同，
    distribution 1.5e-08、log_prob 2.4e-07、entropy 9.5e-07、`global_embedding` 8.9e-07。
  - `update` 端到端（64 transition / epochs=4 / batch=64）：CPU 6.45s → 1.78s（3.62×），
    **CUDA 2.25s → 0.35s（6.39×）**；`actor_loss`/`critic_loss`/`entropy` 相对误差 < 1e-4。
  - 全量测试 90 项，仅剩 2 个改动前即存在的失败（见条目 15 的「已知失败」）。
- **注意事项**：
  1) 默认关闭；`DistributedTrainer`/`Trainer` 本次未接，避免静默改变训练语义。
  2) `dropout > 0` 时两条路径的 dropout mask 排布不同（统计等价、逐位不等价），故等价性
     测试用 `dropout=0`；默认 `dropout=0.1` 下属两种都合法的正则化实现。
  3) `places=5` 之类的**绝对**容差在这里不适用（`critic_loss` ~484，逐次前向差 ~1e-7 经多个
     optimizer step 放大到 ~1e-5），测试改用相对容差 1e-4。
- **顺带发现的既有问题**：`update()` 调 actor 时不传 `environment`，
  `CoveragePolicy.forward` 回落到 `self.environment`，于是 `_global_sources` 优先取
  **当前**环境的 UAV/depot 而非 transition 存下的 `state`——replay 首个 epoch 的 ratio
  因此 ≠ 1。**已由条目 18 修复**（本条提交时批量路径忠实复现了同一行为，故两者仍一致）。
- **回滚方法**：`git revert <本提交 SHA>`；或删除 `test_batched_policy_update.py`、
  `encode_graph_batch`、`ActionEmbedding` 的两个 offset 参数、`GlobalStateEncoder.forward_batched`、
  `CoveragePolicy.forward_batched`、`PPOAgent.update` 的 `batched` 参数与两个 helper
  （恢复内联的逐 transition 循环）。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。

### 16. 图 batch 化原型：disjoint-union 编码器 + 等价性验证（不接入训练）
- **文件**：新增 `coverage_env/rl/encoder/batched_encoder.py`、
  `coverage_env/tests/test_batched_encoder_equivalence.py`；修改
  `docs/training_performance_notes.md`（P1 补充实测结论）。**未改任何训练路径代码。**
- **变更（旧 → 新）**：
  1) `batched_encoder.py` 新增 `collate_graphs()`（把 N 张单图拼成一张不连通大图：
     field 全部在前、boundary 全部在后，边端点重映射到批的统一索引空间，并记录
     `field/boundary/edge_batch` 与 per-graph offsets）、`segment_mean()`（按 graph 分段
     均值）、`batched_graph_context()`（替代全局 mean 池化）、`BatchedGraph` 容器与
     `BatchedCoverageGraphEncoder`（继承 `CoverageGraphEncoder`，共享权重，新增
     `forward_batched()`；原 `forward()` 保持不变）。
  2) 新增 8 项等价性与不变量测试：真实重采样实例逐位一致、跨字段边全空的退化场景、
     单图批、`global_feature` 按图堆叠、边不跨分量、boundary 局部索引不越界（回归守卫）、
     空批拒绝、CUDA 舍入级一致。
  3) `docs/training_performance_notes.md` 的 P1 增补实测小节。
- **原因**：用户要求先做一个「不接入训练、只验证大图拼接前后 encoder 输出逐位一致」的
  原型，用来决定 P1（图 batch 化）是否值得落地。
- **关键结论**：
  1) 由于 batch 采用类型分离布局，单图里的 `edge_index - n_field` **原样成立**（换成总
     field 数即可），`FieldBoundaryEncoder` 无需改动；真正要改的只有「全局 mean 池化 →
     按图分段 mean」一处，比条目 15 文档里估计的工作量小很多。
  2) CPU 上 `field_embedding`/`boundary_embedding`/三类 attention **逐位一致（diff=0.0）**；
     仅 pooled `graph_context` 差 ~2e-7~5e-7（`index_add_` 与 `mean` 累加顺序不同）。
     CUDA 上全部差异 ~1e-6（GEMM tiling）。
  3) 实测加速（64 图 / 4013 节点 / 纯 encoder forward）：CPU 1.22×，**CUDA 14.9×**
     （per-graph 98.1 ms → batched 6.6 ms），印证 GPU kernel launch 开销是主因。
- **踩坑记录**：初版把每条边的 boundary 索引减成「本图内 `[0, n_boundary_k)`」，导致不同
  图的 boundary 特征行**重叠**，输出误差约 1.0 却不抛异常；改为批级统一索引
  （`edge_index - n_field_total`）后归零。该不变量已由
  `test_boundary_indices_stay_inside_their_owning_graph` 守住。
- **未完成（接入训练前仍需做）**：action embedding 的 per-transition 切片、rollout 图快照
  的按 minibatch collate、`attention_debug` 多图语义、critic 侧 `z_G` 的 per-graph 消费。
- **回滚方法**：删除新增的 2 个文件；把 `docs/training_performance_notes.md` 的
  「P1 补充」小节删去即可。不影响任何既有训练/推理路径。
- **验证**：`python -m unittest coverage_env.tests.test_batched_encoder_equivalence` 8/8 通过；
  连同 `test_encoder`/`test_policy_forward`/`test_graph_builder` 共 16 项通过；
  直接运行 `python coverage_env/tests/test_batched_encoder_equivalence.py` 打印完整对比表。

### 15. 跨 episode 泛化 + 总路程 reward 重设计 + 训练效率调研
- **文件**：修改 `coverage_env/env/instance_generator.py`、`coverage_env/env/environment.py`、
  `coverage_env/rl/reward.py`、`coverage_env/config/default.yaml`、
  `coverage_env/rl/trainer/distributed.py`、`coverage_env/rl/trainer/trainer.py`、
  `coverage_env/rl/evaluation/evaluator.py`、
  `coverage_env/tests/test_boundary_guided_planner.py`、`coverage_env/tests/test_rl_interface.py`；
  新增 `coverage_env/tests/test_domain_randomization.py`、`docs/training_performance_notes.md`。
- **变更（旧 → 新）**：
  1) **跨 episode 泛化（任务 1）**：`InstanceGenerator` 拆出 `sample_num_fields()` /
     `sample_components()`（生成 `fields/depot/uavs/resolution`，不建环境、不离散化），
     `generate()` 增加 `randomize_each_reset`（显式 `num_fields` 时默认 `False`，
     省略时默认 `True`）。`CoverageEnvironment.__init__` 新增 `config`/`rng`/
     `randomize_each_reset`；`reset()` 在首次之后每轮调用 `_resample_instance()`，
     用环境持有的 `random.Random` 重新抽地块数量（`environment.num_fields` 的 `[min,max]`）、
     地块几何、depot 和 UAV 参数；新增 `_compute_total_area()` 并在每轮 reset 缓存
     `self.total_area`（离散化可覆盖面积之和）。`default.yaml` 新增 `uav.ranges`
     （`max_energy/max_payload/speed/coverage_width/alpha/beta` 的 min/max），
     `distributed.py` 不再固定 `num_fields=20`（`--num-fields` 默认改为 `None`，给了才固定），
     `Trainer`/`Evaluator` 的 `max_steps` 每轮按当前地块数重算。
  2) **reward 重设计（任务 2）**：`calculate_reward` 由
     `-λ1·d/ds - λ2·e/es + λ3·a/cs - λ4·frag/fs` 改为只保留
     `-distance / total_area * reward_scale`（step-level shaping，episode 累加即
     `-总路程/总面积`）；`environment.step()` 删除 fragmentation 前后计算与
     `_reward_scales()`/`_characteristic_distance()`/`_fragmentation_summary()`，
     改用当前 episode 的 `self.total_area`；`default.yaml` 删除 `rl.reward_weights`、
     `rl.reward_normalization`、`rl.fragmentation_small_component_max_points`，新增
     `rl.reward_scale: 100.0`。
  3) **训练效率调研（任务 3）**：新增 `docs/training_performance_notes.md`，只总结
     逐 transition 前向、DDP all-reduce、rollout planner 长尾的根因、优化方向、
     预期收益/工作量/风险，**未改任何性能代码**。
- **原因**：① 训练期间地块数量与 UAV 参数完全固定，策略过拟合单一场景；② 旧 reward
  由距离/能耗/覆盖/碎片化四项加权、信号混杂，且未对齐「最小化总路程」目标，跨地块数量
  不可比；③ update 的逐 transition 前向与高频 all-reduce 缺少书面调研。
- **取舍说明**：归一化用 `total_area`（离散化可覆盖面积）而非 `num_fields`，可同时消除
  面积差异；实测 5/10/20 地块的单次作业 `总路程/总面积 = 0.55/0.44/0.43`，跨地块数量可比
  （20 地块不再系统性更负）。`reward_scale` 取 **100**（而非示例的 1000）：实测比值约
  0.43~0.55，1000 会把 episode return 抬到 −430~−550、critic MSE ~2.5e5；100 使 episode
  return 落在 O(10~100)（−38~−55），critic MSE ~4e2，更易拟合。scale 与地块数量无关，
  不破坏归一化。task 3 未对 planner 的 CPU 长尾做实现。
- **回滚方法**：`git revert <本提交 SHA>`；或手动恢复 `InstanceGenerator.generate()`/
  `CoverageEnvironment`（删除 `config`/`rng`/`randomize_each_reset`/`_resample_instance`/
  `_replace_instance`/`_compute_total_area`/`total_area`）、`calculate_reward` 旧四项目标、
  `default.yaml` 的 `reward_weights`/`reward_normalization`/`fragmentation_small_component_max_points`
  与 `uav.ranges`、`distributed.py` 的 `num_fields` 固定逻辑、`Trainer`/`Evaluator` 的
  `max_steps`；删除 `test_domain_randomization.py` 与 `docs/`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。
- **验证**：单卡 `python -m unittest coverage_env.tests.test_training` 通过；新增
  `test_domain_randomization`（重采样/可复现/total_area/step reward）4 项通过；4 卡 DDP
  `--num-updates 2 --rollout-steps 8 --ppo-epochs 1 --batch-size 4` 跑通，4 rank
  actor 参数 checksum 一致（928.721299），无死锁、无 unused-parameter 报错；单卡泛化
  训练 6 episode 跑通（reward −38~−49，critic_loss ~4e2~7e2）。
- **已知失败（与本改动无关，改动前已存在）**：`test_uav.UAVSimulationTests.
  test_depot_and_manual_path_execution` 仍硬编码 `energy==40.0`（alpha 0.08 下应为 16.0，
  见条目 14）；`test_nearest_neighbor_episode` 5 地块基线因药量耗尽无法 10/10 完成
  （见条目 13 的「已知影响」）。两者在 `git stash` 回退到改动前代码后同样失败，故未在
  本次修复。

---

## 2026-10-14

### 14. 缩小 workspace + 调低 alpha，使 payload 成为主约束、电量宽裕
- **文件**：修改 `coverage_env/config/default.yaml`。
- **变更（旧 → 新）**：
  1) `environment.workspace`：`min_lat 31.20→31.218`、`max_lat 31.25→31.232`、
     `min_lon 121.42→121.462`、`max_lon 121.52→121.478`（workspace 由约 5.5×9.5 km 缩到
     约 1.5×1.5 km）。
  2) `field_generator.min_field_spacing`：`40.0 → 15.0`。
  3) `uav.energy.alpha`：`0.20 → 0.08`。
- **原因**：workspace 过大导致 20 个小地块散布（实测地块最近邻间距均值约 900 m、最大 3.8 km，
  depot 到地块均值约 3.3 km、最大 7.6 km），`alpha·distance` 的转移能耗把能量吃光，能量先于药量
  耗尽。缩小 workspace 后地块紧凑（间距均值约 280 m、depot 约 700 m），并调低 alpha 降低移动
  能耗，使 payload 成为主约束、电量宽裕。实测（20 地块、不补给 greedy、8 个 seed）：一次作业
  平均覆盖 4.2 个地块，药量先耗尽（剩余约 0.4%），能量还剩约 45.7%。
- **回滚方法**：恢复 `default.yaml` 的 workspace 四个值、`min_field_spacing: 40.0`、
  `alpha: 0.20`。之后运行 `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。

### 13. 地块几何缩小 + UAV 参数按真实农业无人机标定
- **文件**：修改 `coverage_env/config/default.yaml`。
- **变更（旧 → 新）**：
  1) `field_generator.field_size`：`min_area 4000.0 → 1000.0`、`max_area 8000.0 → 6000.0`（地块面积 1000~6000 m²，实测均值约 3813 m²）。
  2) `field_generator.shape`：`min_vertices 6 → 3`、`max_vertices 10 → 5`（顶点数 3~5）。
  3) `uav`：`max_energy 30000.0 → 1500.0`、`max_payload 500.0 → 50.0`、
     `payload_per_area 0.001 → 0.003`。
- **原因**：
  1) 地块顶点数/面积过大导致一次 `step_configuration` 几何规划耗时 5~11 秒（热点在
     sweep planner 的可见性图路径搜索，region_geometry 顶点随格点数膨胀）；缩小后实测降到
     约 0.72 秒，缓解测试/训练慢的问题。
  2) 原 `max_energy=30000`/`max_payload=500` 为「保证 5~20 地块不成为硬约束」的占位值；现按
     真实农业无人机（大疆 T60/T100、极飞 P150）标定：电池约 1500 Wh、药箱约 50 L、亩用量约
     2 L/亩（`payload_per_area=0.003`，即 0.003 药量/m²）。
- **回滚方法**：恢复 `default.yaml` 上述 6 个值（面积 4000/8000、顶点 6/10、
  max_energy 30000、max_payload 500、payload_per_area 0.001）。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。
- **已知影响**：缩小参数后，默认 20 地块场景下单架次仅能覆盖约 3 个地块（能量先耗尽，见
  `alpha` 系数偏高的说明），依赖「能量/药量不成为约束」的旧集成测试（如
  `test_nearest_neighbor_episode.py` 期望 10/10 episode 覆盖完所有地块）将不再成立，需后续
  更新基线或启用自动补给。

### 12. 自动补给（auto resupply）：资源耗尽时自动返航补满
- **文件**：修改 `coverage_env/env/uav.py`、`coverage_env/env/environment.py`、
  `coverage_env/config/default.yaml`；新增 `coverage_env/tests/test_auto_resupply.py`。
- **变更（旧 → 新）**：
  1) `UAV` 新增 `replenish()`（能量/药量补满到上限，返回补入量），并新增累计字段
     `total_energy_recharged`/`total_payload_resupplied`（`__init__`/`reset` 维护）。
  2) `environment.py`：`step()` 中原 `done = self.is_done()` 改为——先判 `_all_fields_complete()`
     （完成则 `mission_complete` 并 `finalize_episode`）；否则看 `get_action_mask` 是否还有可行动作，
     无可行动作时调用 `_attempt_auto_resupply()`：若返航不可行或补满后仍无可行动作则正常终止（记录
     `termination_reason`），否则 UAV 返航 + 补满并继续（`done=False`）。新增
     `_has_feasible_action_at_full_resources()`（在 depot 满资源下探测是否有可行动作）与
     `_attempt_auto_resupply()`（返航 + 补满，info 带 `auto_resupply`/`resupply_count`/返航成本/补入量）。
     `get_summary()` 增加 `resupply_count`/`total_energy_recharged`/`total_payload_resupplied`。
  3) `default.yaml`：新增 `rl.allow_auto_resupply: true`。
- **原因**：此前 UAV 能量/药量耗尽导致所有动作不可行时 episode 直接终止，即使任务还没完成；现在
  在确认「满资源下仍存在可行动作」的前提下自动返航补满，让单次任务可覆盖更大范围（测试用 payload
  受限场景验证跨地块补给）。
- **回滚方法**：删除 `test_auto_resupply.py`；`uav.py` 移除 `replenish()` 与两个累计字段；`environment.py`
  将 `step()` 的 done 判定恢复为 `done = self.is_done()`，删除 `_has_feasible_action_at_full_resources`/
  `_attempt_auto_resupply`/`_resupply_count`/`_termination_reason` 及 `get_summary()` 三个新字段；
  `default.yaml` 删除 `allow_auto_resupply`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。

### 11. PPO 快速可行性筛选与失败 transition
- **文件**：修改 `coverage_env/env/environment.py`、`coverage_env/config/default.yaml`、
  `coverage_env/tests/validation/test_constraint_feasibility.py`、
  `coverage_env/tests/test_boundary_guided_planner.py`；新增
  `coverage_env/tests/test_failed_action_transition.py`。
- **变更（旧 → 新）**：动作 mask 的 workload 估算不再对每个候选 action 调用完整覆盖路径预检；
  改用 entry 距离、估计扫描距离、返航距离、覆盖面积能耗和可用药量的快速保守估算，并缓存
  boundary growth selection 与当前未覆盖状态下的可达面积。新增 `rl.fast_estimate_safety_factor`
  （默认 1.25）用于抬高快速估算成本。选中动作在执行层仍使用实际 planner 和资源量精确校验。
  `step()` 对预检或执行阶段的预期规划/资源错误返回负奖励失败 transition，不抛异常；若执行过程中
  已产生部分 UAV/覆盖更新，会先回滚位置、电量、药量、覆盖状态、轨迹和执行记录。失败 transition
  的状态不变、距离/能耗/药耗/覆盖增益为零，并写入 episode log。连续失败达到
  `rl.failure_termination_threshold`（默认 5）时结束当前 episode，让 Trainer 正常开始后续 episode。
- **原因**：避免训练时对数百个候选动作重复运行几何 planner，并保证单个边缘资源失败不会中断 PPO
  rollout 或留下部分执行状态。
- **回滚方法**：恢复 `environment.py` 的 `_preview_action_feasible()` 实现/调用及原有抛错流程；删除新增失败
  transition 测试，并恢复两个已修改测试中的异常断言；移除 `default.yaml` 中三个 `rl` 新配置项。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。

## 2026-10-13

### 10. Environment 真实药量消耗 + 转移/作业路径分离
- **文件**：新增 `coverage_env/tests/test_payload_and_transfer_separation.py`；修改
  `coverage_env/env/uav.py`、`coverage_env/planner/configuration_executor.py`、
  `coverage_env/env/environment.py`、`coverage_env/config/default.yaml`。
- **变更（旧 → 新）**：
  1) `UAV.execute_path` 返回新增 `payload_consumption`/`remaining_payload`；新增
     `total_payload_consumed` 累计药耗（`__init__`/`reset`/`execute_path` 维护）。
  2) `ConfigurationExecutor.execute` 由单个 `execute_path(total_path, area)` 拆为三次显式调用：
     `entry_transit(area=0)` → `coverage(area=covered_area)` → `exit_transit(area=0)`；
     `CoverageExecutionResult` 新增 `payload`/`transfer_distance`/`coverage_distance`。
  3) `environment.py`：execution record/feedback 记录 `payload`/`transfer_distance`/`coverage_distance`；
     `get_summary` 增加 `total_payload`/`remaining_payload`；`estimate_workload` 缓存 key 加
     `remaining_payload`；`_estimate_workload_uncached` 用 `remaining_payload / payload_per_area`
     封顶可达面积；`_refine_workload_for_execution` 额外校验药量。
  4) `default.yaml`：`payload_per_area: 0.0 → 0.001`，`max_payload: 100.0 → 500.0`（占位，保证当前
     5~20 地块场景药量不成为硬约束；约束逻辑已就绪，后续按真实药量/系数再调）。
- **原因**：此前药量未真实消耗；转移与作业路径在能耗/药量上未显式分离；workload 估算未考虑药量。
  现在转移只耗电 `alpha*distance`（不耗药），作业耗电 `alpha*distance + beta*area` 且耗药
  `gamma*area`（`gamma = payload_per_area`）；药量作为硬约束参与可行性（不足则 action 不可行）。
- **回滚方法**：删除新增测试文件；`uav.py` 移除 `payload_consumption`/`remaining_payload` 返回字段与
  `total_payload_consumed`；`configuration_executor.py` 恢复单次 `execute_path(total_path, area)` 并删除
  result 的三个新字段；`environment.py` 移除 payload/transfer/coverage 记录、`get_summary` 药量字段、
  `estimate_workload` 药量封顶与缓存 key、`_refine` 药量校验；`default.yaml` 恢复
  `payload_per_area: 0.0`、`max_payload: 100.0`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。

### 9. Phase 11 设计修正：Global State + Entry/Exit Role Embedding
- **文件**：新增 `coverage_env/rl/decoder/global_state_encoder.py`、
  `coverage_env/tests/test_global_context_and_role_embedding.py`；修改
  `coverage_env/rl/decoder/action_embedding.py`、`action_decoder.py`、`decoder/__init__.py`、
  `coverage_env/rl/ppo/ppo.py`、`coverage_env/rl/trainer/trainer.py`、
  `coverage_env/env/uav.py`、`coverage_env/env/instance_generator.py`、`coverage_env/config/default.yaml`。
- **变更（旧 → 新）**：
  1) Action embedding 由 `[z_F, z_B(e_in), z_B(e_out)]` + 5 维 `GlobalEncoder(global_feature)` 改为
     `[z_F, z_B(e_in)+r_entry, z_B(e_out)+r_exit, z_G]`（仍为 4×128=512 维）。新增可学习角色向量
     `r_entry`/`r_exit`，显式区分 boundary 入口/出口语义；entry/exit 仍共享同一张 boundary embedding 表。
  2) 新增 `GlobalStateEncoder`：`z_G = MLP(concat(g_graph, x_UAV, x_Depot))`，其中
     `g_graph = concat(MeanPool(Z_F), MeanPool(Z_B))`；`x_UAV` 为 10 维
     `[E_max,Q_max,speed,coverage_width,x,y,E_remaining,Q_remaining,energy_ratio,payload_ratio]`；
     `x_Depot` 为 2 维 depot 坐标。删除旧 `GlobalEncoder`。
  3) Critic（`ppo.py` 的 `ValueNetwork`）由「pool field/boundary + 5 维 global」改为直接消费 `z_G`；
     `value_for_graph` 增加 `environment` 参数，trainer 相应传参。
  4) UAV 新增 `max_payload`/`remaining_payload`/`payload_per_area`（默认 0 不消耗），在 reset/get_state/
     execute_path 维护；instance_generator 与 config 接线。
- **原因**：旧 action embedding 缺少全局状态（无 UAV 能力/药量、无图池化），且 entry/exit 无角色语义，
  策略无法适应不同能力 UAV 或区分入口/出口。
- **回滚方法**：删除新增的 2 个文件；`action_embedding.py` 恢复无 role 向量的 3 段拼接并恢复 `GlobalEncoder`；
  `action_decoder.py` 恢复 `GlobalEncoder(embeddings["global_feature"])` 构建 global_embedding；
  `ppo.py` 的 `ValueNetwork` 恢复 3 输入（field/boundary/global pooling）；`trainer.py` 恢复
  `value_for_graph(next_graph)` 单参调用；`uav.py` 删除 payload 三字段及相关逻辑；`instance_generator.py`/
  `default.yaml` 删除 payload 接线。之后运行 `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。

### 8. 项目初始化为 Git 仓库并推送到 GitHub
- **文件**：`.gitignore`（新增 `.claude/`、`coverage_env/results/` 两行）；项目根目录初始化 `.git/`。
- **变更（旧 → 新）**：项目由「非 git 仓库、靠 CHANGELOG 记录回滚」改为「git 仓库，本地 `main`
  分支已 commit（`initial commit`）并推送到 `https://github.com/lqy7333-del/graph-uav-cpp.git`」。
  `.gitignore` 额外排除本地工具状态（`.claude/`）与可再生成产物（`coverage_env/results/` 约 190MB
  日志/图片）；`__pycache__`、`.pytest_cache` 等此前已排除。
- **原因**：用户要求用 GitHub 做代码版本回退（不额外编写说明文档）。
- **回滚方法**：删除 `.git/` 目录即回到纯文件 + CHANGELOG 记录状态；`.gitignore` 新增的两行直接删去即可。

### 7. Boundary-aware Coverage Executor 重构
- **文件**：新增 `coverage_env/planner/growth_edge.py`、`boundary_region_generator.py`、
  `boundary_aware_sweep.py`、`coverage_env/tests/test_boundary_guided_planner.py`；修改
  `coverage_env/planner/configuration_executor.py`、`coverage_env/planner/__init__.py`、
  `coverage_env/env/environment.py`、`coverage_env/baseline/nearest_neighbor.py`、
  `coverage_env/rl/reward.py`、`coverage_env/config/default.yaml`、
  `coverage_env/visualization/plot.py`、`coverage_env/visualization/episode_visualizer.py`、
  `coverage_env/main.py`、`coverage_env/tests/test_configuration_executor.py`、
  `coverage_env/tests/test_entry_anchor_difference.py`。
- **变更（旧 → 新）**：旧 executor 直接依据 Field/剩余点生成 coverage region 和 sweep path；新流程根据
  entry/exit boundary 选择增长边，从边界种子逐点连通扩张覆盖区，再用有限方向候选和两状态 DP
  联合选择扫描线方向、顺序及入口/出口边上的端点。workload 区域超出增长边可达面积时拒绝执行。
  修正种子点整体排序会造成 workload 前缀断连的问题；窄区域的扫描线放在区域中部，并为超出半个
  swath 的覆盖点补充扫描线。执行结果及可视化增加增长边、剩余覆盖点等诊断信息。baseline 的
  workload 上限改为采用 Environment 的增长边可达估计；reward 增加碎片化惩罚 `lambda4`。
  Coverage action、GNN、Policy 和 PPO 接口保持原样。
- **原因**：避免覆盖区按 workload 截断后不连通、扫描线无法命中窄区域，以及路径候选评估为每个
  方向都重复构建昂贵的可视线连接。DP 先用直线代价下界筛选候选，仅对有希望的候选生成实际连接。
- **回滚方法**：删除本节新增的 4 个文件；将本节列出的已修改文件恢复为 2026-10-12 版本。
  其中 reward/config 的回滚还需移除 `lambda4`、fragmentation normalization 和小碎片配置；环境与
  baseline 恢复旧的 workload/fragmentation 逻辑；executor 恢复旧 `RegionGenerator`/`CoveragePath`
  调用。之后运行 `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证回滚结果。
  该重构随项目文件一并进入 `a166a95` 初始提交，因此 Git 中没有此重构之前的 executor 版本，
  不能单独回退到重构前代码；如需恢复该旧版本，需要从外部备份或旧版文件取回。后续修改应单独提交，
  可使用 `git revert <commit SHA>` 生成可审查、可推送的回滚提交。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证回滚结果。

---

## 2026-10-12

### 6. 基线（NearestNeighbor）加入真实返航
- **文件**：`coverage_env/env/environment.py`、`coverage_env/tests/test_nearest_neighbor_episode.py`
- **变更**：新增 `CoverageEnvironment.finalize_episode()`（任务完成后调用 `return_to_depot()`）；
  `step()` 改为复用 `finalize_episode()`；基线 episode 循环末尾调用 `environment.finalize_episode()`，
  使 baseline 的 summary 也计入返航距离/能耗。
- **原因**：用户要求基线同样走真实返航，而非只预留能量。
- **回滚**：删除 `test_nearest_neighbor_episode.py` 中的 `environment.finalize_episode()` 一行即可
  （`step()` 的返航逻辑保留）。

### 5. 测试夹具 / 演示统一从 config 读取能量系数
- **文件**：`coverage_env/config/__init__.py`（新增 `energy_parameters()`）、`coverage_env/main.py`、
  `coverage_env/tests/test_uav.py`、`test_configuration_executor.py`、`test_entry_anchor_difference.py`、
  `test_mission.py`、`test_rl_interface.py`、`tests/validation/support.py`
- **变更**：把各处硬编码 `energy_consumption_parameters={"alpha": 0.2, "beta": 0.01}` 改为
  `energy_consumption_parameters=energy_parameters()`（从 `default.yaml` 读取）；`test_uav.py` 中
  硬编码的 `assertAlmostEqual(energy, 12.0)` 改为 `uav.alpha * 10 + uav.beta * 1000`（自洽）。
- **原因**：beta 后续还会改，单点来源避免再次漂移。
- **回滚**：把各文件 `energy_parameters()` 改回字面量 `{"alpha": 0.2, "beta": 0.01}`，
  删除 `config/__init__.py` 的 `energy_parameters` 函数及其 `__all__` 导出。

### 4. reward 归一化
- **文件**：`coverage_env/rl/reward.py`、`coverage_env/env/environment.py`、`config/default.yaml`
- **变更**：`calculate_reward` 新增 `distance_scale / energy_scale / coverage_scale` 参数，三项各除以特征尺度；
  λ 默认 0.001/0.01/0.1 → 1.0/1.0/1.0；新增 `rl.reward_normalization` 配置段；`_reward_scales()` 从环境
  派生尺度（总可覆盖面积 / max_energy / 特征距离），可被 `rl.reward_normalization` 覆盖。
- **原因**：旧实现 coverage 项主导（单步 reward ~522，critic MSE 数十万量级）。
- **回滚**：`reward.py` 恢复 `-λ1*d - λ2*e + λ3*a`（λ 默认 0.001/0.01/0.1）；删除 `environment.py`
  的 `_reward_scales` 调用与 `default.yaml` 的 `reward_normalization` 段。

### 3. 真实返航（RL step 接口）
- **文件**：`coverage_env/env/environment.py`、`config/default.yaml`
- **变更**：新增 `CoverageEnvironment.return_to_depot()`（真正飞回并计费、更新轨迹）；`step()` 在
  done 且全部地块完成时自动返航，返航代价计入 terminal reward 与 `info["return_to_depot"]`；
  `get_summary()` 增加 `returned_to_depot` 字段；新增 `rl.return_to_depot: true` 开关。
- **原因**：此前只在 feasibility 估算里"预留"返航能量，不真正飞回 Depot。
- **回滚**：删除 `return_to_depot()`/`finalize_episode()` 方法与 `step()` 中的返航分支，
  删除 `default.yaml` 的 `return_to_depot` 键。

### 2. max_energy 提升
- **文件**：`config/default.yaml`
- **变更**：`uav.max_energy: 20000.0 → 30000.0`。
- **原因**：beta 改为 0.02 后面积能耗翻倍，20 地块基线在 20000 Wh 下出现单种子能量不足；
  提高电池容量使环境自洽（占位值，后续按正式情况再调）。
- **回滚**：`max_energy` 改回 `20000.0`。

### 1. beta 默认值统一为 0.02
- **文件**：`config/default.yaml`、`coverage_env/main.py`、`coverage_env/rl/action_mask.py`
- **变更**：`beta: 0.01 → 0.02`（YAML / main.py / action_mask 回退默认三处对齐）。
- **原因**：spec/README/UAV 类默认均为 0.02 Wh/m²，唯独 YAML 是 0.01，不一致。
- **回滚**：上述三处 `0.02` 改回 `0.01`。

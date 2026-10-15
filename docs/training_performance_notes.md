# 训练效率问题调研笔记

> 本文档只做**调研与总结**，不包含任何性能优化实现。
> 对应任务：update 效率问题分析（PPO 逐 transition 前向 + DDP all-reduce 频繁）。

## 0. 结论摘要

一轮训练 = `rollout_steps=128` 步 rollout + 1 次 PPO `update()`。按代码结构逐项拆解后，
主要开销来自三处：

| 编号 | 问题 | 影响 | 主要改动面 |
| --- | --- | --- | --- |
| P1 | `PPOAgent.update()` 对 minibatch 内**逐个 transition 单独前向** | 单轮 512 次 encoder 前向（4 epoch × 2 minibatch × 64），小图 GPU 严重欠载 | `coverage_env/rl/ppo/ppo.py`、encoder/decoder、graph 构建 |
| P2 | DDP **每次 backward 都 all-reduce**，且 actor/critic 分属两个 DDP 模块 | 单轮 16 次 all-reduce（2 模块 × 8 minibatch），batch 小 → 延迟无法被计算掩盖 | `coverage_env/rl/ppo/ppo.py`、`trainer/distributed.py` |
| P3 | rollout 每步的**几何 planner** 是 CPU 长尾（任务书未列，但实测占比很大） | 每步候选可行性估计 + 真实 planner，单卡实测约 0.11 s 中位、长尾 >1 s | `coverage_env/env/environment.py`、`planner/` |

单卡实测（RTX 5090，20 地块，`seed=42`，共 128 条 transition）：

```
candidate gen + feasibility mask :    ~1.0 ms/步
actor forward (policy)           :   ~14.9 ms/步
env.step (planner + execute)     :  ~109.0 ms/步（中位；长尾显著）
next graph build + encode        :   ~14.7 ms/步
---------------------------------------------------
128 步 rollout                   :   ~56 s
PPO update (epochs=4, batch=64)  :   ~6.1 s（512 次 encoder 前向）
PPO update (epochs=4, batch=16)  :   ~5.1 s（512 次 encoder 前向）
```

因此：**在 planner 便宜、或 GPU 成为瓶颈、或 4 卡 all-reduce 延迟被放大的场景下，
update 会成为主导**，这正是任务书观察到的现象；而本机当前配置下 rollout 的 planner
仍占大头。两者都需要优化，方向不同。

---

## P1. update 逐个 transition 前向

### 根因

`PPOAgent.update()` 的最内层循环：

```python
for index in indices:
    transition = transitions[index]
    policy_state = transition.state
    output = self.actor(                       # ← 每个 transition 一次完整前向
        policy_state["observation"],
        graph=policy_state["graph"],
        candidate_actions=policy_state["actions"],
        feasible_mask=policy_state["mask"],
    )
```

一次 `self.actor(...)` 的代价 = `CoverageGraphEncoder`（`FeatureEmbedding` + 3 个
`RelationGATLayer`）**完整跑一遍** + `GlobalStateEncoder` + `ActionEmbedding` +
`ActionDecoder`。而单张图只有约 `20 地块 + 60~100 边界 ≈ 100` 个节点，属于
"小图、多次调用"：GPU 上每个 `nn.Linear`/`index_add_`/`LayerNorm` 都是独立 kernel，
**kernel launch 与 Python/autograd 调度开销远大于实际算力**。

调用次数（`N=128`、`epochs=4`、`batch_size=64`）：

```
minibatch 数 = epochs × ceil(N / batch_size) = 4 × 2 = 8
encoder 前向 = 8 × 64 = 512 次 / 轮 update
```

此外 rollout 里每一步还要额外 2 次 encoder 前向（`actor(state)` 选动作 +
`encode_graph(next_graph)` 算 next value），且无法与 update 复用。

### 优化方向

| 方向 | 做法 | 预期收益 | 工作量 | 风险 |
| --- | --- | --- | --- | --- |
| **图 batch 化**（首选） | 用 PyG `Batch.from_data_list` / disjoint union 把 minibatch 内多张异构图拼成一张不连通大图，encoder 每 minibatch 只跑 1 次；节点级算子天然支持不连通图，靠 `edge_type` 分池化 | encoder 调用 512 → 8/轮，理论 5~30×；GPU 利用率大幅提升 | 中-高（3~5 天） | 中-高：需重写节点统一索引、`field_id_to_index`→per-graph 映射、action embedding 的 per-graph 取节点、`attention_debug` 与 `encode_graph` 语义；验证掩码/归一化不错位 |
| **只 batch encoder**（折中） | 只把最贵的 GNN 前向批量化，decoder/采样仍逐 transition 用批后的 embedding | 拿到 P1 大部分收益（GNN 占前向 >80%），风险显著更低 | 中（2~3 天） | 中：action embedding 需按 transition 从批 embedding 中切片 |
| **加大 batch_size** | `64 → 128`，minibatch 8 → 4，前向次数 512 → 256 | 1.3~2× | 低（改配置） | 低：需同步调 lr / epochs；PPO 比率估计方差变化 |
| **`torch.compile` / CUDA Graph** | 编译 encoder/decoder 融合 kernel，或对固定形状做 CUDA Graph 捕获 | 1.3~2×（小图收益明显） | 低-中 | 中：地块数每 episode 变化 → 动态形状会反复重编译，需 `dynamic=True` 或 padding 到固定上限 |
| **AMP / bfloat16** | 前向反向用 bf16 | ~2× GPU 吞吐 | 低-中 | 中：数值稳定性（GAT 的 softmax/scatter 一般安全，需回归测试） |

### P1 补充：batch 化原型已实测验证（未接入训练）

原型：`coverage_env/rl/encoder/batched_encoder.py` + 等价性测试
`coverage_env/tests/test_batched_encoder_equivalence.py`（8 项通过）。

结论（重要，降低了工作量估计）：

1. **批量化比预想简单得多**——因为 batch 采用「所有 field 在前、所有 boundary 在后」的
   类型分离布局，单图里的 `edge_index - n_field` 表达式**原样成立**（只需把 `n_field`
   换成总 field 数），`FieldBoundaryEncoder` 也**不用改**。真正需要改的只有一处：
   全局 mean pooling → 按 graph 分段 mean。原先估的「重写节点统一索引 / 掩码」
   大半不必要。
2. **数值等价**（在 CPU 上）：
   - `field_embedding` / `boundary_embedding` / inter·intra·FB **attention**：
     **逐位一致（max abs diff = 0.0）**，6 张不同地块数的真实实例全部通过。
   - 只有 pooled `graph_context` 差 ~2e-7~5e-7（float32 量级）：`index_add_` 的累加
     顺序与 `mean(dim=0)` 不同，属正常舍入。对下游 MLP 无实质影响。
   - CUDA 上全部差异 ~1e-6（GEMM tiling 不同），仍属舍入。
3. **实测加速（64 张图，4013 节点，纯 encoder forward）**：

   | device | per-graph | batched | speedup |
   | --- | --- | --- | --- |
   | cpu | 338.4 ms | 278.3 ms | 1.22× |
   | cuda | 98.1 ms | 6.6 ms | **14.9×** |

   即：GPU 上 kernel launch 开销确实是主因（CPU 本身吞吐就够，收益很小）。
   折算到 update：512 次前向 → 8 次，update 的 encoder 部分预计降到 ~1/15。

4. **踩过的坑（已做成回归测试）**：一开始把每条边的 boundary 索引减成「本图内的
   `[0, n_boundary_k)`」，导致 graph 1 的 boundary 落到 graph 0 的特征行上（索引重叠），
   误差高达 1.0 但**不报错**——正好印证了上表「索引错位是主要坑」。
   `test_boundary_indices_stay_inside_their_owning_graph` 现在守着这个不变量。

5. **尚未验证的部分**：`attention_debug` 的多图语义（诊断用途，接入后可后置）。

### P1 补充 2：已打通到「可接进 update()」（`batched=True` 已可用）

改动（全部向后兼容，**默认行为不变**）：

| 位置 | 改动 |
| --- | --- |
| `encoder/batched_encoder.py` | 抽出自由函数 `encode_graph_batch(encoder, batch)`：任意 `CoverageGraphEncoder` 实例都能批量化，无需子类或复制权重 |
| `decoder/action_embedding.py` | `forward(..., field_offset=0, boundary_offset=0)`：把「本图局部索引」重定位到批 embedding 的行区间（默认 0 即原行为） |
| `decoder/global_state_encoder.py` | 新增 `forward_batched()`：输入 `[G, graph_dim]`，输出 `[G, hidden]`（原 `forward` 的 `reshape(-1)` 只服务单图） |
| `decoder/action_decoder.py` | 新增 `CoveragePolicy.forward_batched(contexts)`：**1 次 GNN** + 逐 transition 解码，返回与 `forward` 同构的输出 |
| `ppo/ppo.py` | `PPOAgent.update(buffer, *, batched=False)`：一个开关切到批量路径；critic 也按 minibatch 批量求值 |

等价性（12 条 transition，候选动作 93~123 个）：

| 量 | max abs diff |
| --- | --- |
| action mask | **完全相同** |
| action distribution | 1.5e-08 |
| log-prob（实际动作） | 2.4e-07 |
| entropy（可行集） | 9.5e-07 |
| `global_embedding`（critic 输入） | 8.9e-07 |

`update` 端到端（64 transition / epochs=4 / batch=64）：

| device | per-transition | batched | speedup |
| --- | --- | --- | --- |
| cpu | 6.45 s | 1.78 s | 3.62× |
| cuda | 2.25 s | 0.35 s | **6.39×** |

损失一致性：`actor_loss` / `critic_loss` / `entropy` 相对误差 < 1e-4。逐次前向只差 ~1e-7，
但一个 update 串联多个 optimizer step，会被 float32 舍入放大到 ~1e-5 绝对值；因此测试用
**相对**容差（真实索引 bug 会大几个数量级，仍能抓住）。

注意事项：

- `batched=True` **默认关闭**，打开只需在调用处传参；`DistributedTrainer` / `Trainer`
  本次未改，避免静默改变训练语义。
- `dropout > 0` 时两条路径的 dropout mask 排布不同：统计等价、逐位不等价。默认
  `rl.encoder.dropout = 0.1`，所以这是「两种都合法的正则化实现」，不是 bug；等价性测试
  用 `dropout=0` 做逐位比对。
- **顺带发现并已修复的既有问题**：`update()` 调用 actor 时不传 `environment`，于是
  `CoveragePolicy.forward` 回落到 `self.environment`（trainer 绑定的那个活环境），而
  `_global_sources` 原先**优先**使用当前环境的 UAV/depot，而不是 transition 里存的
  `state`。replay 时环境已推进过整段 rollout，导致首个 epoch 的 `ratio ≠ 1`，PPO 的
  on-policy 修正被污染。现已改为**优先用 transition 的 `state`**，环境仅作无 state 时的
  兜底。实测：修复前 `max|Δlog-prob| = 4.7e-3`、`ratio ∈ [0.9953, 1.0024]`；修复后
  **`Δ = 0.0`、`ratio ≡ 1.0`**。回归测试见 `coverage_env/tests/test_ppo_replay_state.py`。

复现：`python coverage_env/tests/test_batched_policy_update.py`（打印逐量 diff 与双设备
update 计时）；测试 `python -m unittest coverage_env.tests.test_batched_policy_update`（7 项）。

---

## P2. DDP all-reduce 频繁

### 根因

- `actor` 与 `critic` 是两个独立的 `DistributedDataParallel` 模块，因此每个 minibatch
  触发 **2 次** all-reduce：`actor_loss.backward()` 一次、`critic_loss.backward()` 一次。
- 单轮 all-reduce 次数 = `2 × minibatch 数 = 2 × 8 = 16`。
- batch 很小（64 条、单条图约 100 节点），**通信延迟无法被计算掩盖**：NCCL 在等
  kernel 完成，all-reduce 本身很小但固定延迟高，4 卡上被放大。
- actor/critic 分属两个 optimizer、两次 backward，通信无法合并。
- 跨 episode 泛化后各 rank 图结构不同（地块数不同），进一步加大通信与 kernel 抖动。

### 优化方向

| 方向 | 做法 | 预期收益 | 工作量 | 风险 |
| --- | --- | --- | --- | --- |
| **梯度累积 + `no_sync()`** | 连续 k 个 minibatch 用 `with model.no_sync():` 累积梯度，每 k 次才 all-reduce + `step()` | all-reduce 次数降为 1/k；通信与计算更好重叠 | 低-中 | 低-中：必须保证所有 rank 的 optimizer step 次数严格一致；累积时 loss 需按 k 缩放；与 PPO 逐 minibatch 的 ratio 语义兼容 |
| **合并 actor/critic 为单一 DDP 模块** | 一个 `nn.Module` 包住 actor+critic，一次 backward | all-reduce 次数减半，且可合并通信 | 中（1~2 天） | 中：梯度裁剪需分参数组（或统一）、两个 optimizer 逻辑、checkpoint key 兼容 |
| **加大 batch** | 同 P1，减少 minibatch 数 | 直接线性减少 all-reduce | 低 | 低 |
| **`find_unused_parameters`** | 当前为 `False`（性能考虑） | 无收益，防错 | 低 | 跨 episode 泛化后若某 rank 的 `EDGE_BB_INTER` 为空，`RelationGATLayer` 走空边分支会**跳过多组参数**，DDP 在 `find_unused_parameters=False` 下可能报错/挂起；建议在空边分支里仍让参数参与（或用 `no_sync` 方案），必要时按需开启 |

> 说明：当前 `RelationGATLayer.forward` 在 `edge_index.shape[1] == 0` 时提前返回，
> 不使用 `source_projection`/`target_projection`/`relation_embedding`/attention 参数。
> 20 地块固定场景下 inter edges 基本非空，但泛化到 5~20 地块后需关注这一点。

---

## P3. rollout 的 CPU 几何 planner 长尾（补充）

### 根因

- 每步要枚举候选动作（每地块 `n²` 个 entry/exit，3~5 边界 → 9~25 个/地块，20 地块
  最多约 500 个），并对**每个候选**做可行性估计；`_refine_workload_for_execution`
  还会对入选动作跑真实 planner 预览。
- 选中动作执行时再跑一次 `ConfigurationExecutor.execute`（可见性图路径搜索是热点）。
- 实测 `env.step` 中位 ~109 ms 但长尾可 >1 s；4 卡并行时 4 个进程抢 CPU，planner 更慢。

### 优化方向

| 方向 | 做法 | 预期收益 | 工作量 | 风险 |
| --- | --- | --- | --- | --- |
| **异步 collector / rollout 与 update overlap** | 独立进程或线程持续跑 planner，GPU 只消费数据 | 隐藏大部分 rollout 时间 | 高（3~5 天） | 中-高：on-policy 一致性、跨 rank 同步、内存/队列控制 |
| **增强可行性估计缓存** | 复用已有 `_action_estimate_cache`/`_growth_*_cache`，减少 `_refine` 二分次数，或按状态指纹命中 | 1.2~2× rollout | 低-中 | 低（保持保守估计即可） |
| **候选动作剪枝** | 用启发式（距离下界、可达面积）先淘汰明显不可行/劣质组合 | 减少 30%~70% 候选 | 中 | 中：改变动作空间与策略表达，需重训验证 |
| **几何计算下沉 C++** | 项目名 `graph-uav-cpp` 的长期方向：planner/可见性图用 C++ 实现 | 数量级 | 高 | 中-高：绑定/构建/可维护性 |

---

## 优先级建议

1. **低风险先做**：加大 `batch_size`（P1/P2）、梯度累积 + `no_sync()`（P2）、
   增强可行性缓存（P3）。这三项改动小、可直接从训练日志验证。
2. **中风险高收益**：只 batch encoder（P1 折中）+ `torch.compile`（P1）。
3. **高投入**：异步 collector（P3）、完整图 batch 化（P1）、actor/critic 合并（P2）。

## 实测方法与复现说明

本文档数据来自两个一次性 profiling 脚本（**未纳入仓库**，按需重建）：

1. **rollout 分阶段**：固定 20 地块实例，在**同一进程**内对每一步分别计时
   `get_action_candidates() + get_action_mask()`、`actor(state)`、
   `environment.step(action)`、`build_graph + encode_graph`；用
   `torch.cuda.synchronize()` 夹住 GPU 段，取 8 步的中位数。
2. **update 耗时**：先由同一实例跑出 128 条 transition 填入 `RolloutBuffer`，再用
   `PPOAgent.update()` 计时（`epochs=4`，`batch_size` 分别取 64/16），对比 minibatch
   数与 encoder 前向次数。

重建脚本时可直接复用 `coverage_env.rl.trainer.trainer._clone_graph` 保存 rollout 中
的图快照。

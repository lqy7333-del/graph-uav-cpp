# 修改日志（Change Log）

本项目已纳入 git 仓库（`https://github.com/lqy7333-del/graph-uav-cpp.git`），本文件仍保留作为代码变更的可读说明。条目按时间倒序（最新在上）。
每次修改请追加一节，写明：**文件 / 变更（旧→新）/ 原因 / 回滚方法**。

---

## 2026-10-14

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

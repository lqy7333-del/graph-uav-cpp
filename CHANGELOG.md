# 修改日志（Change Log）

本项目已纳入 git 仓库（`https://github.com/lqy7333-del/graph-uav-cpp.git`），本文件仍保留作为代码变更的可读说明。条目按时间倒序（最新在上）。
每次修改请追加一节，写明：**文件 / 变更（旧→新）/ 原因 / 回滚方法**。

---

## 2026-10-13

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
  本项目当前没有 Git 历史，实际恢复已修改文件需要使用外部备份或对应旧版文件；本条记录列出的文件
  和逆向操作作为回滚清单。

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

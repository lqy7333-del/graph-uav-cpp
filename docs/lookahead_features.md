# 方案 B：给策略加「前瞻输入」

## 1. 为什么需要

条目 86 的结构差异分析给出了明确机制：

> **顺序决定了每次覆盖多少** —— 载荷耗尽时「下一个去哪个地块」决定了各地块的残留量
> ⇒ workload 切分是顺序的**副产品**
> ⇒ 覆盖距离（占总成本 58%）不只由门决定，也被顺序**间接控制**

束搜索（条目 85）据此拿到 5.6–12.8% 的改进，而所有**局部**方法（2-opt/交换/relocate/换门）
都只能做到 0–3%。原因：局部方法不改变「出行结构」，而好解恰恰要求改变它。

现有状态里**已经有** `remaining_payload` / `remaining_energy` / `max_payload` 等量
（`build_uav_feature`，13 维），所以问题**不是信息缺失** ✗，而是：
**这些量没有被组织成「这一步的后果」** ✗。

## 2. 要加的特征

### 2.1 全局前瞻（加到 UAV 节点特征，13 → 17 维）

| 特征 | 定义 | 作用 |
| --- | --- | --- |
| `loads_remaining` | `ceil(uncovered_area × density / remaining_payload) / 8` | **还剩几趟**；这是出行结构的核心量 |
| `payload_ratio` | `remaining_payload / max_payload` | 显式比例（现在是两个绝对值，网络要自己相除）|
| `uncovered_ratio` | `uncovered_area / total_area` | 还剩多少活 |
| `depot_pressure` | `dist(uav, depot) / (remaining_energy / alpha)` | 回程占剩余续航的比例：越大越「该回了」 |

### 2.2 每个候选的前瞻（加到 `action_context`，+4 维）

| 特征 | 定义 | 作用 |
| --- | --- | --- |
| `will_finish` | `remaining_payload ≥ area_i × density` | 这一趟能否做完该地块 |
| `payload_after` | `clamp(remaining_payload − area_i × density, 0) / max_payload` | 做完后还剩多少 |
| `can_continue` | `payload_after ≥ min_other_need` | 做完后是否还能跑下一块（否则**强制回补给**）|
| `exit_to_depot` | `dist(exit_point, depot) / position_scale_m` | **在哪儿结束这一趟**——束搜索正是靠这个省下 462–698 m |

`can_continue` 与 `exit_to_depot` 是**关键**：它们把
「这一步会不会导致我从一个很远的地方被迫飞回补给站」变成**可见的输入** ✓
而这正是当前策略完全看不到的量 ✗。

## 3. 改动位置

| 文件 | 改动 |
| --- | --- |
| `coverage_env/rl/graph/feature_builder.py` | `UavNodeFeatureBuilder` 13 → 17 维 |
| `coverage_env/rl/graph/feature_embedding.py` | UAV 的特征维度常量 |
| `coverage_env/rl/decoder/global_state_encoder.py` | `build_uav_feature` 13 → 17 维；`DEFAULT_SCALES` 不变 |
| `coverage_env/rl/decoder/action_decoder.py` | `_assemble_output` 里 `action_context` 追加 4 列；打分器输入维度 +4 |
| `coverage_env/rl/decoder/hierarchical_decoder.py` | 三级打分的输入维度 +4（或只在第一级加）|
| `coverage_env/rl/decoder/action_embedding.py` | 需要把候选的 entry/exit 点暴露出来（`exit_to_depot` 用）|
| 测试 | 维度断言、`test_every_actor_parameter_receives_gradient` 需同步 |

## 4. 验证协议（与本次审计相同，不重新发明）

1. **短训练**（200 轮，约 4 h）→ 检查维度改动没破坏训练（EV、覆盖率）
2. **配对评估**（20 实例/档，vs 两行规则）→ 看是否从 ~0 变为正值
3. **束搜索 vs 新策略**（`beam_oracle`）→ 看差距是否从 5.6–12.8% 收窄
4. **结构差异**（`structure_diff`）→ 看访问次数是否变得更接近束搜索的解

判据：**配对评估 > +0.5 且束搜索差距 < 3%** 才算 B 成功。

## 5. 风险

| 风险 | 应对 |
| --- | --- |
| 特征只是「更多同样的信息」，网络学不会用 | 先做 200 轮短训验证；若无变化，改加到**解码器的每个候选打分**上而不是全局 |
| `will_finish` 依赖 `density`（`payload_per_area` 已在状态里）| 用状态里的值，不写死 |
| 维度改动使旧 checkpoint 失效 | 接受；新跑一轮 |
| 束搜索的解可能不在策略的可表达范围内 | 若 `structure_diff` 显示访问次数差异无法缩小，说明需要**更长的动作**（宏动作/出行序列），而不是更好的特征 |

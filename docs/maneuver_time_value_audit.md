# 机动时间 A/B 价值目标与回报一致性审计

**2026-10-31 · 审计对象：`869bdd4` 的 6 个正式 run；修复提交：`c390951`**
**不训练、不改算法、不选新 checkpoint；只用固定验证集与已保存的 checkpoint 做审计。**

## 0. 审计结论

**发现一个实现错误，不是“目标本身错误”，也不是“critic 训不动”的独立优化问题：**

- `next_value`（GAE 的 bootstrap 值）在训练时只传了 `environment`，没有传 `next_state`。
- `CoveragePolicy._global_sources` 于是回退到环境的 UAV 对象；该对象**没有 heading 字段**，
  `build_uav_feature` 把方向三通道置零，而下一步 actor 前向用的是带 heading 的 state。
- 因此 `V(s_{t+1})` 在 GAE 里被算成了**另一组输入特征**的值。实测同一状态的
  `V(state)` 与 `V(environment)` 在 heading 有效后差 3.65（5 地块样本），
  在 B43 完整 rollout 中相邻 transition 的 `next_value − 下一步 value` 最大达
  **149.6（update 381）** 和 **174.2（update 387 replay）**，p95 约 **103 / 73**。
- 这直接污染了 GAE 的 `delta` 和 critic 的 return target；B43 的大 return、
  critic loss 44,326 与 validation 退化都出现在这个错误之后。**不能把这些归因于
  时间目标或纯优化不稳定性。**

**同时，审计确认了两件事是对的：**

- **逐 episode reward 闭合正确**：48 个审计 episode（42 完成、6 个 rollout 尾片段、0 失败）
  的 `Σreward` 与 `−(T − A/(80))/100` 最大误差 **7.1e-15**。
- **GAE 公式本身正确**：独立 Python 复算与 `compute_advantage` 的最大差
  **5.2e-5**（float32 舍入级），`done`/`truncated` 边界处理一致。

**因此：这是一个真实的目标计算错误；6 个正式 A/B run 全部受影响，不能作为
“时间目标 vs 距离目标”的方法结论。修复已提交，但尚未重训。**

## 1. 修复与验证

修复提交：**`c390951`**（`fix: use next observation state for GAE bootstrap values`）

- `PPOAgent.value_for_graph(..., state=next_state)`：显式接收并传给
  `CoveragePolicy.encode_graph`。
- `DistributedTrainer` 在 `encode_graph(next_graph, ...)` 时传 `state=next_state`。
- 单进程 `Trainer` 同样传 `state=next_state`。
- 新增回归测试：
  `PpoReplayStateTests.test_bootstrap_value_uses_the_next_observation_heading`。
- 全量测试 269 项，仍只有 2 个 `5490b97` 基线失败；新测试通过。

修复后同一 update 381 的 `next_value` 一致性（相邻 transition 的
`next_value − 下一步 value`）从 **max 149.56** 降到 **max 9.16e-5**。

## 2. 表一：逐 episode reward 闭合

数据：`coverage_env/results/maneuver_time/diagnostics/episode_reward_closure.csv`
（B43，update 381 replay 与原始 381 日志接近；update 387 replay；4 个 rank）。

公式（B 组，time 模式，`gamma=1`，`service_time=0`）：

```
expected = -(sum(time_total) - sum(coverage_gain)/(10*8)) / 100 + (-1)*failed_steps
```

| update | episodes | 完成 | rollout 尾片段 | 失败步 | max abs error | mean error |
|---:|---:|---:|---:|---:|---:|---:|
| 381 | 23 | 20 | 3 | 0 | 7.105e-15 | 5.41e-16 |
| 387 | 25 | 22 | 3 | 0 | 7.105e-15 | −2.49e-16 |

- 失败与截断没有混入完成 episode 的平均；这里 0 失败步，尾片段单列。
- **回报账本没有错误**；大 target 不是 reward 公式或面积抵扣造成的。

## 3. 表二：GAE / return 独立复算与 next-value 一致性

数据：`gae_recompute.csv`、`next_value_consistency.csv`。

### 3.1 GAE 公式复算

| update | rank | n | max abs(adv_ref − adv_ind) | max abs(ret_ref − ret_ind) |
|---:|---:|---:|---:|---:|
| 381 | 0–3 | 128 | 2.3e-5 – 5.2e-5 | 3.8e-5 – 5.2e-5 |
| 387 | 0–3 | 128 | 2.0e-5 – 3.5e-5 | 2.8e-5 – 4.0e-5 |

独立实现逐 transition 复算，差异仅为 float32 舍入；**GAE 递推、done/truncated 边界正确**。

### 3.2 真正的错误：bootstrap 值与下一步 value 不一致

`next_value_t` 应该等于 `value_{t+1}`（非终止、非边界 transition）。修复前：

| update | phase | max abs gap | mean gap | p95 abs gap |
|---:|---|---:|---:|---:|
| 381 | before fix | **149.56** | −17.11 | 102.68 |
| 387 | before fix | **174.17** | −8.19 | 73.01 |
| 381 | **after fix** | **9.16e-5** | 3.8e-7 | 3.05e-5 |

这不是 critic 训练噪声；同一状态的两条前向路径输入不同，差值在一步内可达
150–174。GAE 的 `delta = r + γ·next_value − value` 因此被系统性污染。

## 4. 表三：B43 大 target 溯源

数据：`b43_large_targets.csv`、`b43_next_value_gaps.csv`。

### 4.1 最大 return（修复前）

| phase/update | rank | scale | episode/step | return | remaining MC from step | reward | value | next_value | next step value | done/trunc |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 381 replay | 3 | 13 | 6 / 20 | **+8.64** | **−10.33** | −0.959 | −4.47 | −5.05 | −8.70 | no/no |
| 381 replay | 3 | 13 | 6 / 21 | +6.72 | −9.38 | −0.751 | −8.70 | −5.07 | −9.09 | no/no |
| 387 replay | 3 | 12 | 44 / 15 | **+48.08** | **−8.12** | −1.049 | −9.63 | −11.32 | −28.64 | no/no |
| 387 replay | 0 | 19 | 42 / 31 | +46.31 | −8.86 | −0.782 | −40.56 | −38.51 | −53.95 | no/no |

中间步 `return` 不能与完整 episode 的 MC return 直接比较；这里对齐为**从该步起的
剩余 MC return**。即使按这个口径，`return` 仍为正而剩余 MC 为负，说明 bootstrap
值污染了该步的 return。

这些“大 target”不是失败惩罚，也不是 rollout 截断：它们都是**非终止、非边界**的
普通覆盖步；reward 为负，但 `return = advantage + value` 因 bootstrap 值不一致而变成正数。

`next_value` 与下一步 value 的最大 gap：

| update | rank | scale | episode/step | next_value | next step value | gap |
|---:|---:|---:|---|---:|---:|---:|
| 381 | 2 | 7 | 4 / 4 | −201.03 | −51.47 | **−149.56** |
| 381 | 2 | 6 | 6 / 2 | −199.25 | −59.10 | −140.16 |
| 387 | 1 | 5 | 45 / 2 | −226.04 | −51.87 | **−174.17** |
| 387 | 3 | 10 | 47 / 7 | −226.95 | −56.31 | −170.65 |

### 4.2 精确 44,326 的溯源限制

- 原始 B43 update 387 的 `critic_loss=44,326`、`return_std=253.8` 是训练日志事实。
- 但**没有 update 386 的 checkpoint**；从 380 replay 7 个 optimizer update 时，
  381/382 仍与原始日志几乎一致，383 起因 CUDA scatter/NCCL 非确定性逐步偏离；
  replay 的 387 `critic_loss=3,119`，不能声称逐位复现 44,326。
- 因此“44,326 的精确 buffer”不可恢复；但**可复现且与原始 381 日志接近的 update 381** 已经展示了同一机制：
  首步 return **−372.81** vs 完整 MC return **−39.45**（首步与完整 horizon 对齐），
  最大正 return **+8.64** 对应的剩余 MC 为 **−10.33**。
- 修复后同一 update 381，最大 return 变成 **−0.99**（终端步），`next_value` gap 降到
  **9.2e-5**。符号变化只是描述；正确性判据是状态一致性与独立 GAE 递推，不是 return 正负。

**结论：大 target 的来源是错误 bootstrap（缺 heading 的 env-only V(s_{t+1})），
不是失败惩罚、截断、GAE 公式或 reward 公式。** 44,326 与同一错误一致，但该峰值未能
逐位复现，不能把该特定峰值完整归因于它。输入错误与异常 target 的机制已由 381 replay 证实。

## 5. 表四：规模与统计账本

### 5.1 6 个正式 run 的累计账本

数据：`start_completion_ledger.csv`。

| run | transitions | starts | completed | starts−completed | 每 rank 差值 |
|---|---:|---:|---:|---:|---|
| A_seed42 | 204,800 | 9,524 | 7,924 | **1,600** | 400/400/400/400 |
| B_seed42 | 204,800 | 9,775 | 8,175 | **1,600** | 400/400/400/400 |
| A_seed43 | 204,800 | 9,622 | 8,022 | **1,600** | 400/400/400/400 |
| B_seed43 | 204,800 | 9,780 | 8,180 | **1,600** | 400/400/400/400 |
| A_seed44 | 204,800 | 9,640 | 8,040 | **1,600** | 400/400/400/400 |
| B_seed44 | 204,800 | 9,670 | 8,070 | **1,600** | 400/400/400/400 |

`starts−completed = 1,600 = 400 updates × 4 ranks × 1 个 rollout 尾片段`。
即每个 rank 每个 update 结束时恰好有 1 个**未完成 episode 被丢弃**（其最后 transition
被标 `truncated` 并 bootstrap，之后 env 在下一 update 开头 reset；不会跨 update 继续）。

- `episodes` / `field_counts` 只统计**完成**的 episode；1,600 个尾片段不在其中。
- 因此“5 地块 7.19% vs 20 地块 5.45%”是**完成 episode 的占比**，不是 reset 抽取比例，
  也不是 transition 占比。逐 reset 的规模抽取仍是 `U{5..20}` 均匀。
- 这解释了完成 episode 统计偏向短 episode，但**不能单独证明**小规模 transition 被过采样；
  逐 transition 规模计数在训练时没有记录。

### 5.2 审计 update 381 / 387 的逐 rank 账本

数据：`scale_ledger_audit.csv`、`scale_ledger_summary.csv`。

| update | starts | completed | transitions | tail fragments |
|---:|---:|---:|---:|---:|
| 381 | 23 | 20 | 512 | 3 |
| 387 | 25 | 22 | 512 | 3 |

- 每次 update 4 个 rank 合计 3 个尾片段（不是一个）；这是 4 个 rank 中 3 个有未完成 episode，
  另一个可能恰好以完成 episode 结束。
- 381 的尾片段规模为 6、16、17；387 为 10、13；**不是固定 20 地块**。
- 在这两个 update 中，逐规模 transition 占比与完成占比有波动，但样本太小（512 steps/update），
  没有足够证据支持“小规模长期过采样”或“需要平衡采样”。

### 5.3 `balanced_reward` 的定义（必须纠正）

代码在 `DistributedTrainer.train`：

- `episode_reward` 是当前 update 内 raw reward 的累加；
- 只有 `env_done` 时才写入 `count_emas[field_count]`（EMA α=0.1）；
- `balanced_reward = mean(count_emas.values())`；
- 该行只在 rank 0 打印/写 CSV；`count_emas` 是 rank 0 本地的；
- 它**不含 1,600 个尾片段**，只含完成 episode，且是 raw reward 的 per-field-count EMA 的
  简单平均。

因此：**`balanced_reward` 不是实际任务目标的直接测量，不是全局所有 rank 的指标，
不能用“它变好”推断训练性能改善。** 旧诊断里“训练 reward 变好、验证变差”的现象，
在这一定义下不能作为“目标与任务脱节”的证据；它可能是统计口径/实例混合/采样解码不同。

## 6. 哪些既有实验受影响

**确定受影响：**

- 6 个正式 A/B run：A/B × seed 42/43/44，SHA `869bdd4`；所有 GAE/critic target 使用了
  缺 heading 的 bootstrap value。
- 真实 DDP 门槛（3 updates × 128，time 模式）也使用了同一 bootstrap 路径；其
  环境/时间/返回/服务时间检查仍然有效，但“loss/GAE 目标正确”不成立。
- 单进程 `Trainer` 同样受影响（`value_for_graph` 之前没有 state），
  任何在 heading 特征接入后使用该路径的训练都受影响。

**不作为方法结论，但作为历史测量仍有效：**

- 最终测试 `50000-51599` 的 1,600 实例结果描述的是**已经训练出的旧权重**；
  greedy 评估、NN 对照、时间/覆盖/能耗指标没有使用 GAE，因此这些数值本身没有
  被 bootstrap bug 污染。
- 但它们是**旧训练目标的产物**，不能用于判断“修正后 time 目标是否更好”。
- 最终测试集已经被本轮确认性评价使用；若修复后重训，不能把同一测试集再当作
  sealed 确认集，需要新的测试种子/协议，或明确降级为历史诊断。

**需要重做：**

- 所有关于训练过程、critic、best checkpoint、采样占比的结论，都要在修复后的新 run 上重做。
- 旧诊断中的“B43 critic 发散”“5/6 显著后期退化”“A43 greedy 失败为主”“小规模过采样”
  需要分别改写为：价值输入错误、带选择偏差的描述、候选信号、仅完成 episode 统计偏差。

## 7. 结论与下一步

1. **这是实现错误，不是时间目标本身的失败。** 修复已提交 `c390951`，并有回归测试与
   一-update DDP 审计证据（next-value gap 149.6 → 9.2e-5）。
2. **6 个正式 A/B run 全部受影响，不能用于回答“time vs distance”的方法问题。**
   最终测试数字只作为旧权重的历史测量。
3. **不要先做 value clipping / reward 归一化 / greedy 解码 / 平衡采样。** 先在同一
   SHA `c390951` 上重跑 6 个正式 run，验证 GAE target 一致性、reward 闭合和
   规模账本；然后才讨论 critic 稳定性、解码或采样。
4. **不要用旧最终测试集给新 run 选 checkpoint 或做确认。** 若重训，更新协议并预留
   新的 sealed 测试种子；至少要把旧测试集明确标为“已使用”。
5. 本轮没有启动任何新训练；修复代码已推送，训练产物未改动。

## 8. 产物

- 审计报告：`docs/maneuver_time_value_audit.md`
- episode closure：`coverage_env/results/maneuver_time/diagnostics/episode_reward_closure.csv`
- GAE 复算：`gae_recompute.csv`
- next-value 一致性：`next_value_consistency.csv`
- B43 大 target：`b43_large_targets.csv`、`b43_next_value_gaps.csv`
- 规模账本：`start_completion_ledger.csv`、`scale_ledger_audit.csv`、`scale_ledger_summary.csv`
- 修复：`c390951`；训练/评估历史 SHA：`869bdd4`。

# 训练前正确性防护（value/return gate）

**2026-10-31 · 代码 `e382df3` · 不改变 reward、网络、采样或 PPO 更新公式**

## 0. 目标

把“状态用错、目标算错”从**训练结束后的事后诊断**前移为**每次 PPO 更新前的硬门禁**：

- 所有 rank 的异常一起汇总，任一 rank 不通过 → 所有 rank 停止该 update；
- 失败时写 SHA、配置、rank、update、state digest、reward/value/next_value/边界；
- 训练驱动看到非零退出后停止后续队列；
- 正常 update 的数值结果不被 gate 改变（只增加 eval/no-grad 检查）。

## 1. 每次更新前的检查

| 检查 | 内容 | 容差 |
|---|---|---|
| 相邻 value 一致性 | 非终止 `next_value_t` vs `value_{t+1}`，固定参数、eval 模式 | `1e-2 + 1e-4·scale` |
| rollout 尾部 bootstrap | 用保存的 `next_state` 独立重算 `V(s_T)`；终止则要求 0 | 同上 |
| 回放一致性 | 抽取 4 个 transition，用保存的 state/action 重算 log-prob/value | log-prob `1e-4`，value `1e-2` |
| 费用闭合 | 完整 episode：`Σreward` vs `−(T−A/80)/100 + 失败惩罚` | `1e-6` |
| 数值有效性 | 缓冲 reward/value/next_value、更新后 stats、actor/critic 参数 | 必须全部有限 |
| all-rank 汇总 | 以上指标 `MAX`/有限性 `MIN` all-reduce；任一 rank 失败则全体 abort | — |

失败时 `checkpoint_dir/gate_failures/` 写每个 rank 的 JSON + `last_state/last_next_state` `.pt`，
rank0 写 `update_N_summary.json`。所有 rank 在 barrier 后共同抛错，避免单卡异常、其余卡卡在通信。

`value_state_gap_max` 现在记录 **all-rank max**，不再是 rank0 单卡值。

## 2. API 收紧

`PPOAgent.value_for_graph(graph, environment=None, *, state)` 的 `state` 为必填；
缺失或传 `None` 直接 `ValueError`。训练调用必须传保存的 `next_state`。
诊断若要环境入口，应先由环境生成完整 observation，再走统一编码路径。

## 3. 故障注入与验收

- **2-rank Gloo**：rank1 注入 `value_state_gap_max=1.0`，两 rank 都拒绝 update，写 summary。
- **4-GPU NCCL**：rank3 注入 1.0，4/4 rank 都看到失败并 abort，summary 写出：
  `coverage_env/results/maneuver_time/gate_faults_nccl/report.json`。
- **真实 checkpoint 140 + 一 update**（A42，修复后代码、真实 4-GPU）：
  - `value_state_gap_max = 4.77e-6`（all-rank max）
  - `tail_value_error = 9.5e-7`
  - `replay_logprob_error = 4.8e-7`
  - `replay_value_error = 1.91e-6`
  - `episode_closure_error_max = 0.0`
  - `value_gate_passed = True`
  - 日志：`coverage_env/results/maneuver_time/gate_smoke_141/train2.log`
- **gate 不改变正常 update**：同一 ckpt140、同一 seed，gate on/off 的 update 141
  reward/loss/entropy 日志一致；actor 最大差 `5.0e-5`，同一 no-gate 重复运行差 `6.7e-5`，
  在 GPU 同任务非确定性包络内。gate 只做 eval/no-grad 前向，不消耗 RNG，不改优化器。

## 4. 暂停与恢复

- v2 A42 在 **ckpt_update_140** 正常保存后暂停，后续队列已停止。
- 旧 `formal_fixed/A_seed42/` 保留 1–140 的历史；新监控版本输出到
  `formal_fixed_gate/A_seed42/`：复制 20…140 checkpoint + `resolved_config_update_1.json`，
  从 140 exact resume；B42…B44 从随机初始化重新开始。
- 既有 A42 1–140 的 rank0 日志最大 gap `1.9e-5`；ckpt140 上做的真实一 update
  all-rank gate 通过。没有发现需要作废 1–140 权重的证据。
- **exact resume 的 `3.34e-6` 是 `value_state_gap_max`**（相邻 next-value 最大差），
  不是参数差；参数漂移另见 exact-resume gate。`2.86e-6` 同理。

## 5. 状态

- 代码：`e382df3`；训练尚未在新 gate 下继续（等待本次提交/预检）。
- 旧最终测试 `50000-51599` 是历史；v2 新 sealed 为 `60000-61599`。
- 阈值写在 `distributed.py` 常量中，失败时随 JSON 保存。

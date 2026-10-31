# 正确性门禁补充 v2（独立分支，尚未切入运行中队列）

**2026-10-31 · 分支 `fix/gate-hardening-v2` · 基于 `e382df3` 的审查补丁**

当前四卡 v2（A42 已过 update 180）继续在 `e382df3` 上运行；本文件描述的分支**没有**
在线覆盖正在使用的训练代码。按约定，在下一个安全 checkpoint 切换并记录版本边界。

## 1. 四项审查缺口的修正

### 1.1 回放 NaN

- 重算得到的 `log_probs`、`replay_values` 先用 `torch.isfinite(...).all()` 检查；
- 每个 `lp_error`/`val_error`、以及 `replay_logprob_error`/`replay_value_error`
  都检查有限性；
- 门禁汇总前，对所有标量指标再做一次有限性 `all_reduce MIN`；即使调用者误报
  `finite=True`，`replay_logprob_error=NaN` 也会让所有 rank 拒绝 update。
- 故障注入：2-rank Gloo 在 rank1 注入 `NaN`（同时保持 `finite=True`），两 rank 都 abort；
  另有单进程回放返回 NaN 的测试，断言 `finite=False` 且 `replay_bad_indices` 非空。

### 1.2 闭合误差日志

CSV 写 `gate_metrics["episode_closure_error_max"]`（all-rank max），不再写本地
`closure_error_max`。本地值只用于计算。

### 1.3 逐对相对容差

对每一对非终止相邻状态：

\[
q_i = \frac{|V_{\mathrm{next},i}-V_{i+1}|}
{\epsilon_{\mathrm{abs}}+\epsilon_{\mathrm{rel}}
 \max(|V_{\mathrm{next},i}|,|V_{i+1}|,1)}
\]

先逐对算 \(q_i\)，再跨 rank `MAX`；`q_i > 1` 才失败。尾部 \(q\) 同式单独计算。
绝对误差仍保留在 `value_state_gap_max`/`tail_value_error` 日志中。
单元测试：`0.02` 的误差在 value≈1000 时 \(q<1\)，在 value≈1 时 \(q>1\)。

### 1.4 故障快照与写入保护

- `_local_gate_metrics` 返回 `gap_index`、`gap_abs_index`、`replay_bad_indices`、
  `tail_index`；`gap_index` 已映射回原 transition 下标（不是压缩后的 pair 下标）。
- 失败时保存：最大 \(q\) 的前后两条、回放失败的 transition、尾部 transition；
  JSON 记录索引/数值/边界/state digest，`.pt` 保存对应 `state`、`next_state`、`action`。
- 所有诊断写入包在 `try/except` 中，`dist.barrier()` 放在 `finally`；即使磁盘写入
  失败，也不会在 barrier 前退出。
- NCCL `init_process_group` 增加 **10 min** 超时；驱动 `--task-timeout` 默认 **8 h**，
  超时向整个进程组发 TERM，再等 30 s 后 KILL，避免残留 rank。
- 故障注入测试验证 2-rank 下两 rank 都退出；4-GPU NCCL 脚本同时覆盖 gap 与 NaN，
  在本次审查期间未在运行中的四卡上重跑，留到安全切换点。

## 2. 验证结果

- 全量测试：**277 项，2 项 `5490b97` 基线失败**（`test_uav` 40 Wh、5-field NN 0/10）；
- 新增/更新：
  - 2-rank Gloo：rank1 注入 gap / NaN，均 2/2 abort；
  - 真实回放 NaN：`finite=False`；
  - 故障快照取实际最大 gap 索引（1/2），不是前三个 transition；
  - 逐对相对容差单测；
  - 单进程真实环境 + 1-rank DDP CPU gate（ckpt140，8 步）：gate pass，
    CPU 确定性下 gap/tail/replay 为 0，闭合 0；
- gate 仍只做 eval/no-grad 前向与 all-reduce，不消耗 RNG，不改优化器/网络/reward。

## 3. 对当前 v2 的处理

- **不中断、不作废**当前 A42；继续在 `e382df3` 上训练。
- 下一个安全 checkpoint：停止队列 → 切入本分支 → 用新门禁做一次真实一 update
  检查 → 从该 checkpoint exact resume；记录监控版本边界。
- 保留并纳入验证选择：A42 的 **20–140** checkpoint 与后续 160–400 一并扫描；
  不能只选 160–400。
- ckpt140 的一 update 只证明该 checkpoint 的状态编码一致，**不能追溯证明 1–140
  每一步都正确**；在没有错误证据时保留权重继续训练是合理的。
- 门禁通过只保证检查的计算约束，不等于策略性能一定更好；性能仍由最终实验回答。

## 4. 结论边界

本补丁关闭的是“检测/诊断机制”的四个缺口，不是“所有异常都能拦截”：
无法覆盖未建模的算法错误、数据分布问题和磁盘/内核级故障；这些仍需日志、验证和
最终测试共同判断。

# 机动时间改造 · 实验协议（2026-10-30 训练冻结版）

代码基线：`codex/maneuver-time`，包含 `27625b6` 与 `727350f`，以及本协议配套的门槛实现。
训练启动后**代码冻结**；发现 bug 则停止受影响任务并标记无效，不边改边混统计。
每次正式运行的准确 SHA、有效配置、命令、硬件与初始参数 hash 记录在
`coverage_env/results/maneuver_time/formal/preflight_and_manifest.json`。

## 1. 应用范围（冻结）

| 项 | 值 |
|---|---|
| 地块数 | **5–20 全部整数**；每个 reset 从 `U{5..20}` 均匀抽取 |
| 单块面积 | **0.5–2.0 ha**（`field_generator.field_size` 5000–20000 m²）|
| 面积与块数 | 独立采样 |
| 实例 | 持续重新生成；训练不重复使用固定验证/测试种子 |
| depot | 沿用现有生成规则 |
| 本轮不引入 | 新算法、新奖励项、新归一化、主动返航、排序预训练 |

## 2. 无人机控制情景（冻结）

| 项 | 值 |
|---|---|
| `ppo.randomize_uav` | **false** |
| `uav.speed` | **8.0 m/s** |
| `uav.coverage_width` | **10.0 m** |
| `uav.alpha` / `beta` | 0.08 / 0.02（原距离—面积能耗模型）|
| `rl.time.cruise_speed` | 8.0 m/s，与 `uav.speed` 一致 |
| `rl.reward.coverage_width` | 10.0 m，与 `uav.coverage_width` 一致 |

`CoverageEnvironment.__init__` 在 time 模式下对上述一致性做硬断言。

## 3. 机动时间代理（冻结；规划成本代理，非动力学模型）

```
Δt_turn(θ) = τ_π · (|θ|/π)^p        τ_π = 3.0 s, p = 2.0
```

只依赖方向；重复点、共线细分、转角拆分、直线拆分不改变成本。
`τ_π = 3.0 s` 是仿真参考情景，不是已标定的掉头时间。

## 4. 时间账本（冻结）

```
T_total = T_work_base + T_nonwork_base + T_maneuver_extra + T_service
```

- 所有飞行段（含补给返航与最终返航）统一经 `path_time` 结算；返回前保存真实航段与方向，到达后清方向。
- `time_service` 只挂在补给服务事件上。
- `summary` 暴露四项与 `time_closure_error`。
- 指标名称：**「模型估计的飞行时间，含机动损耗，不含未建模的补给服务及起降耗时」**。
- 不称为真实飞行测量；不声称机动时间已计入真实能耗。

## 5. Reward（冻结）

```
distance 模式: r = -Δd / total_area × reward_scale        （旧归一化，A 组）
time 模式:     r = -(ΔT - ΔA_unique/(coverage_width·nominal_work_speed)) / T_scale
```

- `T_scale = 100` 全局固定，不按实例面积/块数/参考策略归一化。
- 只抵扣本步新覆盖面积；重复覆盖抵扣 0。
- `gamma = 1.0`。
- `distance` 模式完整保留（A 组用它）。
- 本轮 A/B 比较的是**完整目标方案**（距离目标 vs 固定尺度时间目标），不声称只隔离转弯项或面积抵扣项。

## 6. 固定数据划分与 checkpoint 选择（冻结）

| 集合 | 种子 | 数量 |
|---|---|---|
| 历史已用（非 sealed）| 20000-20099、21000-21219、30000-30029、31000 | 仅参照 |
| **验证集** | **40000-40095** | **96**（每规模 6）|
| **最终测试集** | **50000-51599** | **1,600**（每规模 100）|

种子按规模连续分配：验证集规模 5 用 40000-40005，规模 6 用 40006-40011，……，规模 20 用 40090-40095；
测试集同理，规模 5 用 50000-50099，……，规模 20 用 51500-51599。
最终测试集不用于调参、不用于挑 checkpoint；每个训练种子只选**一个统一 checkpoint**，不按地块数分别选模型。

### 6.1 选择规则（实现于 `coverage_env/rl/evaluation/maneuver_eval.py`）

对每个候选 update（20, 40, …, 400）在 96 个固定验证实例上评估：

1. **完成数量更多的候选优先**。
2. **仅当候选完成全部 96 个实例时**，才在完整固定集上比较**规模等权平均 `total_time`**；最小者胜，平局取较早 update。
3. 若没有候选完成全部 96 个，报告**可靠性门槛未通过**，取最高完成数中**较早 update**；不比较不同成功子集的平均时间。
4. 完成数相同但未全部完成时，不允许用平均时间打破平局。

验证规则对 A/B 相同。

## 7. 训练设计（冻结）

| 项 | 值 |
|---|---|
| A 组 | `rl.reward.mode = distance` |
| B 组 | `rl.reward.mode = time` |
| 种子 | **42 / 43 / 44**（配对起点相同；同一 seed 的 A/B 初始参数 hash 必须相同）|
| 网络/输入 | 完全相同；UAV 全局输入 16 维（含方向特征）|
| `candidate_attention` | **false**（与历史冻结网络一致；显式写入有效配置）|
| 规模分布 | `U{5..20}` |
| 预算 | **400 updates**、`rollout_steps = 128`、每 **20 updates** 保存 |
| 并行 | **4 GPU、world_size = 4、6 个任务串行**，每个任务使用相同 GPU 数量，避免改变每 update 的全局采样量 |
| 顺序 | A42, B42, A43, B43, A44, B44 |
| 每任务全局 transitions | 400 × 128 × 4 = **204,800** |
| 每任务同步优化步 | 400 × (128/64) × 4 epochs = **3,200** |
| 记录 | 每卡/全局 transitions、episodes、优化步、墙钟、loss、KL、entropy、clip、梯度范数、规模分布、几何刷新数 |
| 初始权重 | 随机初始化；**不加载旧 13 维 checkpoint** |

## 8. 最终测试与报告（冻结）

- 每个整数规模 5–20 各 100 个**新实例**，共 **1,600** 个。
- 同一实例上评估 NN、A 三个种子、B 三个种子。
- 报告总量与四分项：`total_time`、`time_work_base`、`time_nonwork_base`、`time_maneuver_extra`、`time_service`。
- 固定符号：**baseline − candidate，正值代表 candidate 更好**。
  - NN 为 baseline，策略为 candidate；正值 = 策略模型时间更短。
  - A 为 baseline，B 为 candidate；正值 = B 模型时间更短。
- 配对差只在双方都完成的实例上计算；完成数、完成率单列。时间步不是独立样本；置信区间按实例重采样（逐规模分层，再规模等权平均）。
- 逐规模、逐训练种子报告；不以总体均值掩盖某规模退化。
- 同时报告：距离、覆盖率、完成率、能耗、资源违规、访问与补给次数、额外时间指标、推理与 mask 耗时。

## 9. 结论边界

允许：「固定控制情景 + 0.5–2.0 ha 范围下，统一模型在 5–20 地块新实例上的表现」。
不允许：任意农田适用、跨机型泛化、仿真机动成本 = 实测时间、每个实例都优于 NN。

## 10. 训练门槛状态（2026-10-30）

- 真实 4×RTX 5090 DDP 短链路（3 updates × 128 steps，time 模式）通过：
  - 参数更新、有限 loss/KL/entropy/梯度、reset 时间归零；
  - 160 次 reset 覆盖 5–20 全部 16 个规模，160 个不同几何；
  - 补给/最终返航机动费用非零且与 `path_time` 独立参考一致；
  - 非零服务时间下 reward 恒等式闭合（误差 < 1e-14）；
  - exact resume 的配置、update index、RNG/环境/优化器 payload 与日志指标一致。
- GPU 短链路实测 **36.2 s/update**，按 400 updates 预计 **4.0 h/任务**、6 任务约 **24 h**（不含验证/最终测试）。
- 该 GPU 栈不是逐位可复现：两个完全相同的全新任务在 3 updates 后已有约 `6.4e-4` 的参数差（CUDA scatter/NCCL 顺序）；因此 exact resume 不把逐位相等作为判据，而以配置/状态/日志一致且漂移不超过重复运行包络为准。正式 A/B 仍使用 3 个独立种子。

## 11. 审计后修正与正式重跑（v2，2026-10-31）

**v1（`869bdd4`）的 6 个 A/B run 已因 GAE bootstrap 实现错误被判定无效：**
`next_value` 未传 `next_state`，heading 三通道丢失；修复为 `c390951`，审计见
`docs/maneuver_time_value_audit.md`。v1 的最终测试 `50000-51599` 只描述旧权重，
不能再作为修正后方法的确认集。

### 11.1 修复验收（先于正式重跑）

- 单进程与真实 4-GPU DDP 均验证：正常飞行有效方向、补给后无效方向、
  rollout 截断前下一状态、真正终止 bootstrap=0、保存/恢复继续采样。
- 相邻 value 一致性在参数未更新、eval 模式下检查；修复后 DDP 短训练
  `value_state_gap_max` 为 `2.86e-6`（A/B），resume 为 `3.34e-6`。
- 配对短训练 A/B（seed 123，5 updates × 128 steps）通过：B 组 reward 闭合
  （最大误差 `2.13e-14`）、独立 GAE 复算最大差 `<5e-6`、保存/resume 正常。
- 详细报告：`coverage_env/results/maneuver_time/value_gate/value_gate_report.md`。
  短训练不要求胜 NN；它只验证实现。

### 11.2 重跑协议（与 v1 的差异仅限已发现的实现错误和测试集登记）

- 从随机初始化重新训练 A/B × 42/43/44；**不续训任何 v1 checkpoint**。
- 保持 v1 的网络、reward、预算（400 updates × 128 rollout）、U{5..20} 实例分布、
  固定 UAV 参数、`candidate_attention=false`；不新增 PopArt、value clipping、
  平衡采样或新解码策略。
- 输出目录：`coverage_env/results/maneuver_time/formal_fixed/`；旧 `formal/` 保留为历史。
- 验证集仍为 `40000-40095`（已反复用于开发/选择，不再称为 sealed）；选择规则不变。
- **新最终测试集：`60000-61599`，每个规模 100 个，共 1,600 个**；在 checkpoint 选择完成前
  不使用。旧 `50000-51599` 标为已使用的历史确认集，不参与 v2 选择或确认。
- 正式启动前重新记录 SHA、有效配置、硬件、初始 hash、value gate 报告与预算；
  训练期间代码冻结。

### 11.3 训练中防护版本变更（2026-10-31）

- v2 A42 在 `ckpt_update_140` 正常保存后暂停，后续队列停止；`formal_fixed/` 保留 1–140
  历史。新增 all-rank 正确性门禁 `e382df3` 后，1–140 checkpoint 复制到
  `formal_fixed_gate/A_seed42/`，从 140 exact resume；B42–B44 从随机初始化重新开始。
- 门禁只增加 eval/no-grad 检查与 all-reduce，不改变 reward、网络、采样或优化公式；
  真实 ckpt140 一 update 的 gate on/off 日志一致，参数差在同任务重复运行包络内。
- `value_state_gap_max` 为 **all-rank max**；失败时所有 rank 共同 abort 并写诊断，
  驱动停止后续任务。阈值与失败记录见 `docs/maneuver_time_gate_hardening.md`。
- `3.34e-6`/`2.86e-6` 是 `value_state_gap_max`，不是参数差。
- 启动 SHA `9fbae8b`；A42 从 140 resume 后 update141/142 gate 通过，B42–B44 重新初始化。

### 11.4 门禁补充 v2（待安全切换）

- 独立分支 `fix/gate-hardening-v2` 修正：回放 NaN、逐对相对容差、全局闭合日志、
  故障快照索引与写入保护、NCCL/驱动超时与进程组清理。
- 当前运行中的 v2 不切换到该分支；在下一个安全 checkpoint 停止队列、切入、
  用真实一 update 检查后从该 checkpoint exact resume，并记录监控版本边界。
- A42 的 20–140 checkpoint 保留，必须与 160–400 一起纳入验证选择。
- ckpt140 的一 update 只证明该 checkpoint 的状态编码一致，不能追溯证明 1–140
  每一步都正确；门禁通过不等于策略性能更好。

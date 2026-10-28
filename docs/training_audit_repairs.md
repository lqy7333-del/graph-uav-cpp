# 2026-10-28 训练审查修复

本次承接 b18463b 的 dropout、critic 全局初始化、CLI 优先级和重复入口修复。

- PPO update 本身固定 eval 模式，保留 autograd，避免重放 log probability 被 dropout 改写。
- 广播优势从逐步 reward 累计折扣回报，不能用末步的 GAE return 代替整局回报。critic 仍使用独立的 GAE 目标。
- 未完成的广播轨迹不参与 actor 更新；没有有效样本时跳过 optimizer.step，防止 Adam 动量继续改变参数。跨 rank 按有效样本数聚合。
- truncation 与 terminal 分离：时间限制保留 next value，但截断 GAE 跨轨迹递推。
- NN 基线复制当前环境且不 reset，检查快照指纹，使用相同 shaping 和折扣。未完成补全直接报错，不生成错误标签。
- UAV 随机化与每次 reset 是否换实例解耦；两类训练器通过 ppo.randomize_uav 使用相同规则。
- checkpoint 保存完整配置、模型、优化器、随机状态、环境和训练进度；DDP 保存每个 rank 的运行状态。exact resume 检查预算、配置和进程数一致；旧 checkpoint 只能显式 finetune，不能声称精确恢复。
- 评估入口优先使用 checkpoint 配置，只加载 actor，不恢复训练 RNG 和 optimizer。旧文件缺少配置时给出警告。
- 距离统计采用覆盖、转移、补给、最终返航四个互斥分项并检查闭合。完成局统计与当前未完成局分开。

## 验证与边界

新增测试实际调用 update 分发、验证完整回报、截断 bootstrap、无有效样本时参数不变、配置优先级和随机状态恢复。两进程 CPU/Gloo 测试验证不同样本数下 critic 初始化一致、广播有效样本不等时参数同步，以及连续执行与 checkpoint 恢复后参数逐张量相等。

本机只运行回归测试（含合成任务的有限优化步骤），没有启动正式训练。CPU/Gloo 验证不等于多 GPU/NCCL 验证。

历史广播与 NN 基线消融若经过上述缺陷路径，不能作为正确实现的算法对照；需重跑。已有冻结策略的实测距离不因代码修复自动消失，但不能用于证明修复后训练方法的效果。旧 checkpoint 没有完整配置时必须另行核对实验配置。

## 使用

DDP 新 checkpoint 默认 `--resume-mode exact`，保留原预算与进程数。更改训练预算或使用旧 checkpoint 时指定 `--resume-mode finetune` 并使用新的输出目录。

测试两进程功能：设置 `RUN_DDP_TESTS=1` 后执行 `python -m unittest coverage_env.tests.test_training_ddp_runtime`。checkpoint 包含 Python 环境对象，仅加载可信来源的文件。

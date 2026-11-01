# 固定 10 个 20 地块实例：已有模型微调诊断

目的：移除训练任务不断变化这一因素，测量当前 PPO/GNN 在有限训练集上还能获得多少收益。
这不是泛化测试，不是理论最优值估计。训练集上的最佳结果只代表本架构、初始化、优化器和预算
下已找到的策略；即使拟合失败，也不能直接证明没有优化空间。

## 固定设计

- 一份模型联合学习 **10 个固定实例，每个 20 地块**，不是每实例训练一份模型。
- 种子预先固定为 `70000–70009`；不根据 NN/PPO 好坏筛选、替换实例。
- `prepare` 从所选已有 checkpoint 读取完整配置；继承其 reward、网络和 PPO 参数，
  仅固定 field count、关闭 UAV 随机化并强制正确性门禁。不要在本轮同时换 reward/架构。
- 默认继续使用既有固定 UAV；位置、剩余资源、覆盖状态正常变化。
- 保存实际几何/网格/UAV/depot 的 pristine 快照与 SHA256。reset 从快照深拷贝恢复，
  不再次生成随机几何。实例编号只用于日志，不输入策略。
- 每 rank 用私有 RNG 打乱整份实例列表，每一轮各实例恰好抽取一次。
  固定步长 rollout 沿用既有语义，尾片段 bootstrap 后下一 update 重置；因此这不是
  每实例相等 transition 权重，必须报告每实例 starts/completions/transitions。
- 微调载入原 actor + critic；新建 optimizer，update 从 0 开始，原学习率/熵调度在新预算内重启。
  不把它称为原实验的 exact continuation。
- 一次主实验默认 **1000 updates × 128 steps/rank**；4 GPU 时为 512,000 transitions。
  每 20 updates 和最后一次保存并评估；不因看到测试结果临时改变预算。
- 起点使用 **B_seed42 按原验证规则选定的 update 240**，作为预先指定的 time 目标诊断，
  不从最终测试集挑最强种子。确认它是 v2 修复后 checkpoint；禁止用 v1 或旧 13 维输入权重。
  若该文件不可用，报告路径问题，不静默换成别的模型。入口本身也支持其他明确指定的权重。

## 评估与判读

每次评估完整的同一份 10 实例训练集，使用 greedy decoding：

1. update 0 保存/评估原模型；同时执行 NN，形成不可遗漏的两条基线。
2. 逐实例保存 total_time、四项时间、distance、coverage、energy、visits、resupplies、
   failed_steps、资源违规与完成状态；时间定义沿用现有角度机动代理，非实测飞行时间。
3. 统一差值符号：`NN − candidate`、`initial − candidate`，正值为改善。
   同时报绝对秒数、逐实例相对改善的均值，以及胜/平/负。
4. 任何实例未完成时，不计算只含成功子集的任务均值；checkpoint 先比完成数，全部完成后
   才按完整训练集平均时间选最优，平局选较早 update。update 0 也可被选中。
5. 主结论来自同一个 checkpoint 在全部十实例上的成绩。不能逐实例选不同 checkpoint
   后拼出一个并不存在的“最佳模型”。训练集反复选优不报告成 sealed 确认或理论上界。
6. 同时报告最佳与最后 checkpoint，不能仅展示最佳点。图使用真实评估点，不作边界填充值平滑。
7. 若固定集明显改善，说明现有架构/更新至少有拟合这批任务的能力，但不能直接把此前瓶颈
   全归因于泛化；若仍停滞，保留优化、探索、表征和可用改进空间等多种解释。

## 工作站运行

在独立分支/工作区运行，正式泛化实验不改动。先定位 v2 B42 选中权重，用真实路径替换下文
`<v2_B42_selected_update_240.pt>`。以下为 Linux 工作站命令，不在本机开长训练。

```bash
python -m coverage_env.rl.trainer.overfit prepare \
  --checkpoint <v2_B42_selected_update_240.pt> \
  --bank coverage_env/results/overfit20/bank \
  --count 10 --fields 20 --instance-seed 70000

# 独立 smoke 输出：真实 GPU、原训练门禁、完整十实例评估、2 updates。
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m coverage_env.rl.trainer.overfit train \
  --bank coverage_env/results/overfit20/bank \
  --output coverage_env/results/overfit20/smoke \
  --num-updates 2 --rollout-steps 128 --eval-every 1 --seed 42

# smoke 通过后，从同一原模型开始正式微调；不要从 smoke 权重接着训练。
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m coverage_env.rl.trainer.overfit train \
  --bank coverage_env/results/overfit20/bank \
  --output coverage_env/results/overfit20/run_seed42 \
  --num-updates 1000 --rollout-steps 128 --eval-every 20 --seed 42

python -m coverage_env.rl.trainer.overfit plot \
  --output coverage_env/results/overfit20/run_seed42
```

正式入口复用 `DistributedTrainer`、`PPOAgent` 和 all-rank value/tail/replay/closure/finite
门禁。没有另一套 PPO 实现。先在工作站通过真实 NCCL smoke，再开长训练；本地 CPU 冒烟不等于
4 GPU 验收。任一故障停止本次任务，不自动重试、更换实例或关闭门禁。

中断恢复：原命令增加 `--resume .../run_seed42/ckpt_update_N.pt`，保留原 output、bank、
seed、GPU 数量、总预算和 rollout_steps。恢复模型、optimizer、每卡 RNG、实例顺序/游标及
活动状态。保留 bank 文件和 manifest；checkpoint 不重复包含十份不可变实例模板。
原始预训练 checkpoint 在新训练启动时必须存在且哈希一致；exact resume 使用已保存权重。

## 必须返回的产物

- Git SHA、起点权重路径/哈希、`bank/manifest.json`、完整有效配置与预算。
- GPU smoke 及故障/门禁日志；每卡每实例曝光计数。
- `references.json`、`evaluation/update_*.json`、`learning_curve.csv/png`、`best.json`。
- update 0/100/200/400/600/800/1000 与最佳节点的完整十实例表；不足该预算的节点标未运行。
- 最佳与最后模型相对 NN、原模型的时间/距离/机动时间改善；十实例分别改善多少，是否由少数主导。
- 仅报告训练集拟合结果。本轮不访问旧 sealed 集，也不增加泛化实验或新的训练分支。

# 固定 10×20 实例：greedy vs 采样的有界诊断

**checkpoint：0、460、580、600、780、1000；每实例 16 次采样，不用 best-of-16。**

## 0. 结论先行

- **不是只有一个幸运点**：greedy 在 460–580 和 700–900 两个区间连续优于原模型；采样的平均成本也显示出同方向的阶段性改善与回退。
- **最终收益丢失不只是 greedy 解码**：
  - 780→1000，greedy 变慢 **130.7 s**；
  - 同一批固定随机种子下，采样平均也变慢 **67.8 s**（95% CI [+21.0, +119.8]）。
- **但 greedy 确实贡献了很大一部分最终退化**：
  - 采样平均从初始到 final 仍改善 **144.7 s**（CI [−183.3, −101.0]）；
  - greedy 从初始到 final 只差 **+1.8 s**（基本回到起点）。
  - 也就是说，final 的策略分布平均仍优于初始，但 greedy 的排名/选择没有把这份平均优势保留下来；同时分布本身也比 780 的最佳状态退步了约 68 s。
- **不是 critic 崩溃**：本区间实际 LR 恒为 `3e-4`，KL 约 `0.018–0.027`，clip 约 `0.15–0.17`，EV 约 `0.93–0.98`；没有 v1 式尖峰。600 的回退发生时 critic/EV 并未明显恶化。
- 下一步应优先区分 **greedy 排名/动作选择** 与 **actor 更新本身**；没有证据支持立即扩网络、提高转弯权重或单纯延长训练。

## 1. 设置

- 使用既有 `run_seed42` 的 6 个 checkpoint；不重新训练。
- 固定 bank：10 个实例，seed 70000–70009，全部 20 fields。
- 每个 checkpoint：每个实例 16 次完整采样任务，共 **160 episodes/checkpoint、960 episodes**；全部 **160/160 完成**。
- 每次 episode 使用固定任务随机种子（900000 + instance×16 + replicate），同一 checkpoint 内不挑最好结果；报告均值、SE 和按实例/重复的两层 bootstrap 95% CI。
- greedy 成本直接取同一 checkpoint 的既有 10 实例完整评估。
- 时间指标：**模型估计的飞行时间，含机动损耗，不含未建模的补给服务及起降耗时**。

## 2. greedy vs 采样平均

| update | greedy (s) | sampled mean (s) | sampled SE | 95% CI | sampled − greedy |
|---:|---:|---:|---:|---:|---:|
| 0 | 7089.3 | 7290.4 | 64.7 | [6786.1, 7808.5] | **+201.1** |
| 460 | 6964.0 | 7088.7 | 61.9 | [6593.8, 7580.6] | +124.7 |
| 580 | 7029.4 | 7164.5 | 64.5 | [6652.3, 7674.9] | +135.1 |
| 600 | 7205.8 | 7294.4 | 65.9 | [6777.0, 7815.3] | +88.6 |
| 780 | 6960.4 | 7077.9 | 63.4 | [6576.9, 7584.5] | +117.5 |
| 1000 | 7091.1 | 7145.7 | 63.3 | [6650.1, 7657.9] | +54.6 |

![sampled vs greedy](fixed_instance_overfit/sampled_vs_greedy.png)

- sampled 平均始终高于 greedy（即采样平均更慢），因为 greedy 选的是分布里的高概率/高排名动作；这不是异常。
- 关键是 **sampled − greedy 从 201.1 s 缩到 54.6 s**：final 的 greedy 相对其采样分布的领先变小。
- final 的 sampled−greedy 配对 95% CI 为 **[−14.0, +117.8]**，在这个 10 实例面板上不能断言最终 greedy 优势仍显著；best(780) 为 **[+64.7, +165.6]**。

## 3. 阶段变化：greedy 与采样同向，但幅度不同

| 阶段 | greedy 变化 | sampled 平均变化 (95% CI) |
|---|---:|---:|
| initial → 460 | −125.3 s | **−201.7 s** [−253.4, −156.9] |
| 460 → 580 | +65.5 s | +75.8 s [+24.9, +134.8] |
| 580 → 600 | +176.3 s | **+129.9 s** [+82.3, +175.2] |
| 780 → 1000 | +130.7 s | **+67.8 s** [+21.0, +119.8] |
| initial → final | **+1.8 s** | **−144.7 s** [−183.3, −101.0] |

- 580→600 的回退在 greedy 和采样平均上**同时出现**；这更像 actor/更新过程的变化，而不是单纯 greedy 排名。
- 780→1000 也是两者同时变慢，但采样平均只退化 67.8 s，greedy 退化 130.7 s；因此 final 的额外损失有一部分来自 greedy 选择/排名。
- initial→final：采样平均仍改善 144.7 s，greedy 基本持平；这是“期望表现 vs greedy 部署”的最直接分离。

## 4. 逐实例差异

![per-instance sampled minus greedy](fixed_instance_overfit/sampled_per_instance_delta.png)

- 大多数实例的 sampled−greedy 为正（采样更慢），但 final 的实例 70006 是明显例外：
  - final greedy 8776.1 s；
  - final sampled 平均 8589.6 s（比 greedy 快 186.5 s）。
  这直接显示 final 在该实例上的 greedy 选择并不代表分布的平均质量。
- 这不是“所有实例都被 greedy 解码毁掉”：final 仍有 70006 这类显著个例，但 overall sampled 平均仍比 greedy 慢 54.6 s。

## 5. 训练日志对齐

实际配置：`lr_schedule=none`，actor LR `3e-4`，`entropy_coef` 从 `0.01` 退火到 `0`。

| update | actor LR | entropy coef | approx KL | clip fraction | critic loss | EV |
|---:|---:|---:|---:|---:|---:|---:|
| 460 | 3e-4 | 0.00611 | 0.0178 | 0.162 | 3.53 | 0.980 |
| 580 | 3e-4 | 0.00478 | 0.0266 | 0.167 | 11.70 | 0.931 |
| 600 | 3e-4 | 0.00455 | 0.0219 | 0.169 | 8.04 | 0.950 |
| 780 | 3e-4 | 0.00255 | 0.0216 | 0.150 | 3.12 | 0.981 |
| 1000 | 3e-4 | 0.00011 | 0.0267 | 0.153 | 6.70 | 0.959 |

- **600 回退时** critic 11.7→8.0、EV 0.931→0.950，并未恶化；不支持“600 崩掉是 critic 失效”。
- **780→1000** critic 3.12→6.70、EV 0.981→0.959，有轻微价值拟合变差，但 EV 仍 0.96；不足以单独解释 greedy 的 130.7 s 回退。
- KL/clip 没有爆量；实际 LR 恒定，不是学习率调度切换造成的。
- 这仍只是相关性；本轮没有做干预实验，不能据此断言 critic 或学习率是原因。

## 6. 每卡/每实例曝光

`evaluation/update_1000.json` 的累计 usage（4 卡求和）：

- 每个实例 `starts` 都约 **1600**（所有实例均被抽到）；
- `transitions` 范围 **47,264–55,117**；
- 完成 episode 数 **1,174–1,255**；
- 总计 512,000 transitions、16,000 starts。

starts 并不偏少，但 transitions 因 episode 长度不同而相差约 16%；固定实例集合并不等于每次更新对十实例等权。完整逐 rank/实例表见
[rank_instance_usage.csv](fixed_instance_overfit/rank_instance_usage.csv)。

## 7. 判读

现在可以把“收益丢失”拆成两部分：

1. **actor/分布本身**：780→1000 sampled 平均变慢 67.8 s（CI 正），说明不只是 greedy；同时 580→600 的 sampled 回退也成立。
2. **greedy 部署/排名**：final sampled 平均仍比 initial 改善 144.7 s，而 greedy final 与 initial 持平；greedy 相对 sampled 的优势从 201 s 缩到 55 s，且 final 实例 70006 的 greedy 明显差于采样平均。

因此：

- 不能再说“final 的退化完全是随机噪声”；
- 也不能再说“只是 greedy 解码问题”；
- 更准确的定位是：**策略分布有阶段性学习，但最终 actor 分布相对 best 退步；greedy 排名/动作选择又额外丢失了一部分分布平均优势**。
- 没有证据把主因归到 critic、学习率、网络容量或转弯权重；当前最值得做的下一次有界诊断是**动作排名/top-k/温度与 greedy 的关系**，而不是直接扩网络或再加 1000 updates。

## 8. 产物

- [sampled_summary.json](fixed_instance_overfit/sampled_summary.json)
- [sampled_per_instance.csv](fixed_instance_overfit/sampled_per_instance.csv)
- [sampled_rows.csv](fixed_instance_overfit/sampled_rows.csv)
- [rank_instance_usage.csv](fixed_instance_overfit/rank_instance_usage.csv)
- [sampled_vs_greedy.png](fixed_instance_overfit/sampled_vs_greedy.png)
- [sampled_per_instance_delta.png](fixed_instance_overfit/sampled_per_instance_delta.png)

本诊断没有改 reward、网络、优化器或训练预算，也没有再训练。

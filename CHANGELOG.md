# 修改日志（Change Log）

本项目已纳入 git 仓库（`https://github.com/lqy7333-del/graph-uav-cpp.git`），本文件仍保留作为代码变更的可读说明。条目按时间倒序（最新在上）。
每次修改请追加一节，写明：**文件 / 变更（旧→新）/ 原因 / 回滚方法**。

---

## 2026-10-17

### 42. 场景分布改为「连片农田」：连片生长地块池 + 随机抽取（并重算全部位置尺度）
- **文件**：修改 `coverage_env/env/instance_generator.py`、
  `coverage_env/config/default.yaml`、`coverage_env/rl/decoder/global_state_encoder.py`、
  `coverage_env/rl/graph/graph_builder.py`、`coverage_env/rl/graph/relation_builder.py`、
  `coverage_env/tests/test_domain_randomization.py`、`test_graph_builder.py`；
  新增 `coverage_env/tests/test_contiguous_layout.py`。
- **原因（条目 41 的可视化暴露的问题）**：原生成器在整个工作区**均匀随机撒点**，导致地块
  又小又散。实测 20 地块：**边界到边界间隙中位 196.7 m**、块区 1328×1359 m、
  **路程 58.6% 是「飞过去」、只有 41.4% 是覆盖**。这更像 TSP 而不是农业覆盖作业 ——
  真实农田里相邻地块共用田埂，飞行只占零头。
- **变更（旧 → 新）**：`field_generator.layout` 新增配置；默认 `mode: contiguous`：
  1) **连片生长**：第一块放在原点，之后每块都**贴着已放置的某一块**、沿随机方向以
     `anchor_radius + self_radius + gap_m` 的距离放置；重叠用**外接半径**做廉价判定，
     失败就换邻居/方向重试（`growth_attempts`）。
  2) **随机抽取**：池子大小 = `num_fields × pool_oversample`（默认 1.5），再用
     `rng.sample` 抽出 `num_fields` 块当作本局实例。这样既保留相邻结构，又让每局实例不同，
     而且抽掉的那部分会留出真实的空档（农场不会每天把每块地都作业一遍）。
  3) 抽中的地块整体**平移到工作区中心**，并重新顺序编号为 `field-1..N`。
  4) **depot 放到块区外侧**（`depot_offset_m`，默认 150 m），像农场总部；原来在工作区里
     随机取点会重新引入多公里通勤。
  5) `mode: scattered` **完整保留**且能复现旧行为（同样的 RNG 抽取顺序），
     旧基线仍可复跑。
- **效果（20 地块，15 实例实测）**：

  | 指标 | scattered（旧） | **contiguous（新）** |
  | --- | --- | --- |
  | 边界到边界间隙（中位） | 196.7 m | **33.0 m** |
  | 块区尺寸 | 1328 × 1359 m | **832 × 723 m** |
  | depot → 最近地块边缘 | 70 m | **377 m**（块外） |
  | 路程：飞行 / 覆盖 | 58.6% / 41.4% | **43.5% / 56.5%** |

- **连带必须重算的位置尺度**（测试 `test_position_channels_are_well_scaled` 当场抓到了）：
  地块挤到 ~800 m 的块里以后，原来的 `position_scale_m=1500` 会把坐标压得过小。
  按新分布实测重定：`position_scale_m` **1500 → 800**（37 实例最大 |质心| 641 m）、
  `uav_distance_scale_m` **3000 → 1200**（最大 UAV→候选距离 902 m）、
  `global_distance_scale_m` **3000 → 1500**（= `d_near_m`，使该特征按构造恰好落在 [0,1]）。
  `DEFAULT_SCALES.position_scale_m` 同步改 800（有测试守着两者一致）。
- **测试**：
  - 新增 `test_contiguous_layout.py` 6 项：地块**相邻**（最近间隙中位 < 80 m 且 > 0）、
    块区紧凑（20 块 < 1400 m）、`scattered` 仍可复现旧布局、**depot 在块外**、
    不同种子给出不同实例、地块数与编号正确。
  - `test_domain_randomization` 里 3 个 patch `_sample_center` 的放置重试测试改为**显式指定
    scattered 模式**（连片模式不用 `_sample_center`，否则 patch 不生效）。
  - `test_position_channels_are_well_scaled` 的判据从「坐标均值绝对值」改为
    「**跨地块的坐标标准差**」—— 块区居中于原点后均值不再有代表性，而真正要看的是
    坐标通道有没有足够的变化幅度。
  - 全量 **134 项**，仍只有 2 个改动前即存在的失败。
- **重要影响**：这是**任务分布的改变**，此前所有 reward 数字（−32.99 / −33.35 / −33.86 /
  −40.00 / 启发式 −34.50）**全部作废、不可比**，需要重新建立基线与训练。
  reward 量级也会变化（路程/总面积下降）。
- **回滚方法**：`git revert <本提交 SHA>`，或把 `field_generator.layout.mode` 设为 `scattered`
  （后者只回滚分布，不回滚尺度）。

### 41. 新增训练后「贪心完整路径规划」可视化工具（固定场景、不同地块数） 新增训练后「贪心完整路径规划」可视化工具（固定场景、不同地块数）
- **文件**：新增 `coverage_env/visualization/policy_paths.py`。
- **用法**：
  ```
  PYTHONPATH=. .venv/bin/python -m coverage_env.visualization.policy_paths \
      --checkpoint coverage_env/results/<run>/ckpt_update_100.pt \
      --field-counts 5 10 20 --out coverage_env/results/<run>/paths.png
  ```
  同时在 stdout 打印一张表（reward / coverage / steps / resupplies / termination reason），
  所以不开图也能读结论。
- **功能**：在**固定的**测试场景（`num_fields ∈ {5,10,20}`，`randomize_each_reset=False`，
  seed 固定）上，用 **greedy 策略**跑完整 episode，并与**最近地块启发式在同一个实例上**并排绘制。
  每个面板画出：地块轮廓与编号、depot、UAV 起点、**按访问顺序着色的飞行路径**、
  entry 点的**访问序号**、**自动补给的回程**（红色点线）、**终局返回**（灰色虚线）。
- **目的**：把「策略是否合理」从数字变成一眼可见的判断。数值指标（配对 greedy）只能说明
  「好多少」，看不出**顺序是否绕路**；而 5 地块这种小场景的总路程对顺序极其敏感，
  画出来就能直接看出交叉和折返。
- **首次运行的读数（`run_uav_node_scale_fix/ckpt_update_100.pt`，见
  `coverage_env/results/run_uav_node_scale_fix/paths_newscale.png`）**：

  | 地块数 | 策略 | reward | coverage | steps | 补给 | 终止 |
  | --- | --- | --- | --- | --- | --- | --- |
  | 5 | greedy | −38.92 | 100% | 5 | 0 | mission_complete |
  | 5 | 最近地块 | −39.00 | 100% | 5 | 0 | mission_complete |
  | 10 | greedy | **−29.79** | 100% | 11 | 1 | mission_complete |
  | 10 | 最近地块 | −33.84 | 100% | 11 | 1 | mission_complete |
  | 20 | greedy | −32.26 | 100% | 24 | 4 | mission_complete |
  | 20 | 最近地块 | **−31.32** | 100% | 24 | 4 | mission_complete |

  **读出来的三件事**：
  1) 三个场景全部 **100% 覆盖、`mission_complete`** —— 策略确实完成了任务，不是靠"少干活"拿分。
  2) **5 地块场景暴露了顺序问题**：greedy 的访问顺序是 3→1→4→2→5，轨迹**先下、再上、再上、
     再大幅折回到最下方、最后向左**，肉眼可见地大量交叉。10 地块上策略明显优于启发式
     （−29.79 vs −33.84），20 地块上反而略差（−32.26 vs −31.32）—— 说明**两者都只是
     "近似贪心排序"，谁更好取决于实例**，而不是策略学会了更好的全局顺序。
  3) **补给回程（红色点线）在地图上很长且位置尴尬**：它由环境强制触发（`step()` 在
     所有动作都不可行时自动返回 depot），但**触发位置取决于策略的覆盖顺序**，
     而回程距离是计入 reward 的。所以顺序选择会间接决定这部分成本。
- **结论**：可视化把「数字上看不出差别」的问题变成了可见问题 —— **剩余的低效主要在
  "跨地块的全局顺序"，而不是"空间感知"**。这与条目 38/40 观察到的
  「最近地块率与收益不单调」以及「三种位置表示最终性能打平」是一致的。

### 40. 尺度校正后的 arm 1 重跑：训练轨迹明显变好，但**最终性能无法与其它方案区分** 尺度校正后的 arm 1 重跑：训练轨迹明显变好，但**最终性能无法与其它方案区分**
- **运行**：`run_uav_node_scale_fix/`，从零训练 100 轮（全新架构，旧 checkpoint 不兼容），
  `seed 42`、地块 5~20、4 卡 DDP。
- **训练日志（sampled reward，25 轮 block 均值，前 100 轮）—— 这是方差最低的比较**：

  | block | **新尺度** | 旧尺度 | 显式距离 |
  | --- | --- | --- | --- |
  | u1-25 | **−42.97** | −43.34 | −43.37 |
  | u26-50 | **−40.03** | −41.37 | −40.86 |
  | u51-75 | **−36.50** | −38.92 | −38.70 |
  | u76-100 | **−35.65** | −37.99 | −36.32 |

  **新尺度在每个 block 都最好**，末端领先旧尺度 **2.34**、领先显式距离 **0.67**。
  这说明「把本局机型信息从 0.02 宽的窄带恢复到 0.32 宽」确实带来了持续的训练收益。
- **20 个固定实例的 greedy 评估（ckpt_100）**：

  | 架构 | greedy 均值（绝对，可直接比较） | sd |
  | --- | --- | --- |
  | 旧尺度 arm 1 | **−32.99** | 3.79 |
  | 显式距离 | −33.35 | 4.65 |
  | 新尺度 arm 1 | −33.86 | 4.73 |
  | 坐标@1500 | −40.00 | 3.98 |

  前三者跨度只有 **0.87**，而 20 实例的配对 SE 约 1.0~1.4 —— **统计上无法区分**。
  行为指标上新尺度更好（实走/最近 **1.05×** vs 旧尺度 1.16×、显式距离 1.04×），
  但「选中最近地块」率反而略低（70.1% vs 74.4%）。
- **方法学更正（重要）**：本条目之前的报告里，我一直用「**各自架构未训练基线之上的配对 delta**」
  来横向比较不同架构（例如「显式距离 +10.32 vs UAV 节点 +9.63」）。**这个比较是无效的** ——
  未训练基线本身随架构与随机初始化波动很大：本次实测三个架构的 untrained greedy 分别是
  **−41.19 / −41.62 / −43.38**（两台随机策略之间就差 2.19，而三个训练后方案的差距只有 0.87）。
  基线噪声把配对 delta 抬高了 1~2 个点，足以颠倒排名。**跨架构比较必须用同一批实例上的
  `绝对 greedy 均值`，不要用配对 delta。**
- **结论（诚实版）**：
  1) **尺度校正本身是正确且有益的**：它把 `max_energy` 通道的动态范围放大 16×，
     训练日志轨迹在每个 block 都更好，末端领先 2.34。这条收益是稳的。
  2) 但在**最终策略性能**上，三种「让 actor 拿到位置信息」的方案
     （显式距离 / 旧尺度图节点 / 新尺度图节点）**落在同一个水平：约 −33.4 ± 1**。
     即**「用哪种表示」对最终性能影响不大；真正起作用的是「actor 到底有没有位置信息」**。
  3) 坐标路线（−40.00）**远在其后**，因为它没能真正让 actor 用上位置信息。
- **对后续方向的含义**：既然三种可用表示在最终性能上打平，**继续在这条表示轴上投入的边际收益很低**。
  剩余空间更可能在**更高层的决策**（跨地块全局顺序、能量/补给规划）—— 这也与条目 38 观察到的
  「最近地块率与收益不单调」一致。
- **回滚方法**：本条目仅为文档。

### 39. 校正 UAV 能力维度的归一化尺度（与 `uav.ranges` 对齐） 校正 UAV 能力维度的归一化尺度（与 `uav.ranges` 对齐）
- **文件**：修改 `coverage_env/config/default.yaml`（`rl.graph` 新增 4 个尺度键）、
  `coverage_env/rl/decoder/global_state_encoder.py`（`DEFAULT_SCALES`）、
  `coverage_env/rl/graph/feature_builder.py`（`UavNodeFeatureBuilder` 默认值）、
  `coverage_env/rl/graph/graph_builder.py`（配置读取默认值）、
  `coverage_env/tests/test_uav_graph_node.py`（新增 3 项测试）。
- **变更（旧 → 新）**：`max_energy_scale` **30000 → 1800**、`max_payload_scale` **100 → 60**、
  `speed_scale` **20 → 10**、`coverage_width_scale` **50 → 12**。
- **原因**：这四个尺度与 `uav.ranges`（域随机化范围：max_energy 1200–1800、
  max_payload 40–60、speed 6–10、coverage_width 8–12）严重不匹配。实测（训练设置
  `randomize_each_reset=True`）归一化后的跨度：

  | 维度 | 旧尺度 | 旧跨度 / sd | 新跨度 / sd | 改善 |
  | --- | --- | --- | --- | --- |
  | `max_energy` | 30000 | **0.020 / 0.006** | **0.323 / 0.099** | **16×** |
  | `coverage_width` | 50 | 0.072 / 0.020 | 0.302 / 0.083 | 4.2× |
  | `speed` | 20 | 0.200 / 0.056 | 0.393 / 0.112 | 2.0× |
  | `max_payload` | 100 | 0.183 / 0.056 | 0.305 / 0.093 | 1.7× |

  最严重的是 `max_energy`：30000 约为实际范围的 **20 倍**，把「本局电池多大」这一维压成
  0.02 宽的窄带（sd 0.006），而同一向量里 `payload_ratio` 的 sd 是 0.315 —— 相差 50 倍。
  这与已修的 `position_scale_m`（10000 → 1500）和 `global_distance_scale_m`（10000 → 3000）
  是**同一类欠尺度问题**。
- **取值原则**：每个尺度 = 对应 `uav.ranges` 的**上界** —— 这是**不截断任何采样值的前提下最小
  的尺度**，因此归一化跨度最大、条件数最好。四个尺度同时影响 **UAV 图节点特征**
  和 `build_uav_feature` 的 10 维 `x_UAV`（后者进 `z_G` 与 critic 输入），三处必须一致。
- **测试**：新增 3 项 —— (a) 每个尺度必须 ≥ 对应 `uav.ranges` 上界且 ≤ 1.5×（既不截断也不过压），
  直接与 `uav.ranges` 绑定，改范围就会报警；(b) `default.yaml` 与 `DEFAULT_SCALES` 四处尺度
  必须一致；(c) 域随机化下能力维的实测跨度必须 > 0.1（防再次被压平）。
  全量 **128 项**，仍只有 2 个改动前即存在的失败。
- **影响**：`x_UAV` 与 UAV 节点特征都变了，**旧 checkpoint 不再兼容，需从头训练**。
  条目 38 里 arm 1 的 +9.63/+7.04 是在**旧尺度**下得到的，如需评估本次校正的收益需重跑。
- **回滚方法**：`git revert <本提交 SHA>`（四处尺度一并回到 30000/100/20/50）。

### 38. UAV 图节点设计的 200 轮终局：**峰值追平手工特征，但长训练更差且不稳定** UAV 图节点设计的 200 轮终局：**峰值追平手工特征，但长训练更差且不稳定**
- **运行**：`run_uav_node/` 从 ckpt_100 续训到 200 轮（exit 0，4 rank checksum 一致 = 1005.017321）。
  续训的理由是「100 轮时仍在上升、尚未饱和」—— **这个判断被结果否定了**（见下）。
- **配对结果（20 实例逐实例，相对本架构未训练基线 −41.19 / 27.9%）**：

  | ckpt | UAV 节点（arm 1） | | | | 显式距离 `49249cc` |
  | --- | --- | --- | --- | --- | --- |
  | | 配对 delta | t | 最近地块 | 实走/最近 | 配对 delta |
  | 25 | +2.57 ± 1.27 | 2.0 | 34.1% | 1.69× | +2.00 |
  | 50 | +6.69 ± 1.03 | 6.5 | 55.9% | 1.23× | +9.79 |
  | 75 | +8.07 ± 1.50 | 5.4 | 63.1% | 1.15× | +9.13 |
  | 100 | **+9.63 ± 1.35** | 7.1 | 74.4% | 1.16× | +10.32 |
  | 150 | +7.30 ± 0.62 | 11.8 | **82.7%** | **1.03×** | +10.01 |
  | 200 | +7.04 ± 0.79 | 8.9 | 75.6% | 1.05× | **+10.33** |

  训练日志（25 轮 block 均值）：UAV 节点 −43.34 → **−36.82**（平台约 −36.2~−37.5）；
  显式距离 −43.37 → **−35.52**（平台约 −35.5）。
- **三条结论**：
  1) **本设计在 100 轮达到峰值 +9.63，与显式距离的 +10.32 基本持平（差 0.69）**，
     即「图原生版」在最佳点上**确实能追平手工特征**。
  2) **但它随后退化**（100 → 150 → 200：+9.63 → +7.30 → +7.04），而显式距离从 50 轮起
     稳定在 **+10.0 ~ +10.3**。所以**手工特征在长训练预算下明显更稳、更强**
     （200 轮处 +7.04 vs +10.33，合并 SE ≈ 1.5，t ≈ 2.2，边际显著）。
     **「100 轮仍在上升」的判断是错的**，它其实已经到顶。
  3) **一个反直觉的分离**：150 轮时「选中最近地块」率最高（**82.7%**）、实走/最近最低
     （**1.03×**，几乎完美贪心），但配对收益反而更低（+7.30）。说明**「永远去最近的地块」
     并不是全局最优** —— 收益还取决于能量/载荷/补给，过度承诺「单步最近」反而有害。
     这与条目 36 观察到的「坐标路线最近地块率 35~38% 时就饱和」是同一条线索的两面。
- **最终建议**：本设计**技术上是成功的**（相对根因、坐标路线都有数量级改进，且泛化性更好），
  但在**当前这个任务**上，**显式 per-action 距离特征仍是更实用、更稳的选择**（+10.33 且稳定）。
  若要继续沿坐标/图路线走，应把重点放在「为何图原生版本长训练会退化」而不是继续加机制。
- **回滚方法**：本条目仅为文档；如需回到显式距离版本，`git revert 5eaf410`。

### 37. 把 UAV 接入为图节点 + 关系边（方案 A：独立 Stage 3「agent-relative 读出」） 把 UAV 接入为图节点 + 关系边（方案 A：独立 Stage 3「agent-relative 读出」）
- **文件**：新增 `docs/uav_graph_node_design.md`、`coverage_env/tests/test_uav_graph_node.py`；
  修改 `coverage_env/rl/graph/feature_builder.py`、`relation_builder.py`、`graph_builder.py`、
  `coverage_env/rl/encoder/feature_embedding.py`、`hetero_encoder.py`、`batched_encoder.py`、
  `coverage_env/visualization/plot.py`、`coverage_env/config/default.yaml`、
  `coverage_env/tests/test_graph_builder.py`、`test_encoder.py`、
  `test_batched_encoder_equivalence.py`。
- **变更（旧 → 新）**：
  1) 统一节点空间由 `[Field, Boundary]` 扩为 **`[Field, Boundary, UAV]`**，UAV 排在最后，
     使 `edge_index - n_field` 这类既有表达式全部不变。UAV 节点 **1 个**（动作空间只能寻址一个 UAV）。
  2) 新增 `UavNodeFeatureBuilder`：6 维**只含状态**（`[energy_ratio, payload_ratio, speed,
     width, E_max, Q_max]`），**不含位置**；过独立的 `uav_mlp`。
  3) 新增关系 `EDGE_UF = 3`（`UAV→Field`）与 `EDGE_UB = 4`（`UAV→Boundary`），边特征均为
     `[dx, dy, dist]`（按 `uav_distance_scale_m` 归一化）。
  4) 新增 **Stage 3 `UavRelativeEncoder`**，在 Stage 2b 之后运行、**只吃 UAV 关系**，
     输出即 `ActionEmbedding` 读取的 agent-relative 嵌入；decoder 输入维度不变（仍 512）。
  5) 尺度：`global_distance_scale_m` **10000 → 3000**；新增 `uav_distance_scale_m: 3000`。
  6) 批处理：`BatchedGraph` 增加 `x_uav_node`/`uav_batch`/`uav_offsets`；`collate_graphs` 的端点
     重映射由二分类改为**三分类**（UAV 在最后，原 `< n_field` 判定会把它误判成 boundary）。
  7) `plot_graph` 补 UAV 边样式与 UAV 节点标记（原先遇到未知 `edge_type` 直接 `KeyError`）。
- **原因**：条目 31 定位的根因是「actor 拿不到 UAV→候选的距离」。两条既有修法里，显式距离
  有效（+10.32）但属人工特征；静态坐标无效（300 轮仅 +1.9~+3.1）。本条目把有效机制**泛化成
  图原生形式**：让**图自己把距离算出来并投递**，而不是在 decoder 上拼一个预计算特征。
- **关键机制（不变量）**：`RelationGATLayer` 的消息是 `source_h + relation_embedding(edge_attr)`，
  且注意力分数也含关系嵌入 —— 所以把 `dist` 放进边特征，距离会作为**显式加项**进入目标节点嵌入。
  而 `_destination_softmax` 是**对目标节点全部入边**做归一化，若把 UAV 边混进 Stage 1，
  一个 Boundary 的 UAV 消息权重会降到约 `1/(K+1)`（K 为跨地块邻居数，通常 5~15）→ 被稀释 6~16 倍。
  **独立 Stage 3 里每个目标恰好只有 1 条入边 → softmax 恒为 1 → 零稀释**，且残差保证
  Stage 1–2 信息不丢。这是选择「独立 Stage 3」而非并入 Stage 1 的决定性理由。
- **测试**：新增 `test_uav_graph_node.py` 6 项 —— UAV 边覆盖 `|F|+|B|` 且都从 UAV 出发；
  **UAV 边注意力恒为 1.0**（零稀释，实测通过）；只移动 UAV 时 **Stage 1–2 注意力逐位不变**
  而最终嵌入必须改变（map-relative / agent-relative 分离）；只把 UAV 边属性乘 0.5（拓扑与所有
  节点特征不变）必须改变 Field 嵌入（距离通道确实到达目的地）；距离通道尺度良好（防欠尺度回归）；
  UAV 节点特征不含位置。另更新 `test_graph_builder`（含 UAV 节点与边方向断言）、`test_encoder`
  （attention_debug 新增键）、`test_batched_encoder_equivalence`（含 UAV 的端点归属与注意力比对；
  单图批仍要求**逐位相等**，多图批因 GEMM 归约顺序放宽到 1e-5 并注明理由）。全量 **125 项**，
  仍只有 2 个改动前即存在的失败；4 卡 DDP 冒烟通过（checksum 一致）。
- **验证结果（`run_uav_node/`，100 轮 from scratch，seed 42，地块 5~20，4 卡 DDP，exit 0，
  4 rank checksum 一致 = 1065.993125）**。全部相对本架构自己的未训练基线
  （−41.19 / 最近地块 27.9%）做 20 实例逐实例配对：

  | ckpt | greedy | sd | 配对 delta | t | 选中最近地块 | 实走/最近 |
  | --- | --- | --- | --- | --- | --- | --- |
  | untrained | −41.19 | — | — | — | 27.9% | — |
  | 25 | −38.42 | 3.95 | +2.57 ± 1.27 | 2.0 | 34.1% | 1.69× |
  | 50 | −34.72 | 4.48 | +6.69 ± 1.03 | 6.5 | 55.9% | 1.23× |
  | 75 | −34.24 | 3.33 | +8.07 ± 1.50 | 5.4 | 63.1% | 1.15× |
  | **100** | **−32.99** | **3.79** | **+9.63 ± 1.35** | **7.1** | **74.4%** | **1.16×** |

  训练日志 block 均值：reward **−43.34 → −37.99**，`entropy` 4.378 → 3.731，
  `ratio_std` 0.131 → 0.215。
- **三条结论**：
  1) **设计成立且效果显著**：100 轮配对 **+9.63（t=7.1）**，最近地块率 27.9% → **74.4%**，
     实走/最近 2.2× → **1.16×**。
  2) **完胜静态坐标路线**：同期（100 轮）坐标路线只有 **+3.05（t=1.9，不显著）/ 28.1% / 1.74×**，
     而且在 ckpt_25/50 还**显著差于自己的未训练基线**（t=−2.8/−2.4）；本设计没有这个破坏性早期阶段。
  3) **基本追平显式距离特征**：显式距离 100 轮为 **+10.32（t=8.3）/ 79.9% / 1.04×**，
     本设计 **+9.63 / 74.4% / 1.16×**，差距仅 **0.69**（约效果的 7%）。
     **而且显式距离在 50 轮已达 +9.79、100 轮基本饱和；本设计 75→100 仍在上升
     （+8.07 → +9.63）**，即**尚未饱和**，继续训练仍有空间。
- **意义**：这与实现前的预期一致 —— 本设计是显式距离的**图原生泛化版**，在当前任务上
  **接近而非大幅超越**。它的价值不在这一项指标，而在于：(a) entry/exit 的选择也自动获得
  UAV 相对几何（显式特征只覆盖 entry，且无学习到的组合）；(b) 未来加 depot / 多 UAV / 障碍
  不需再手工造特征；(c) 距离同时进入**消息内容与注意力分数**，不只是拼上去的一个标量。
- **回滚方法**：`git revert <本提交 SHA>`（会同时恢复 `global_distance_scale_m` 为 10000）。

### 36. 坐标路线终局判定（300 轮）：**决定性劣于显式距离**，尺度不是主因 坐标路线终局判定（300 轮）：**决定性劣于显式距离**，尺度不是主因
- **目的**：条目 35 只跑到 100 轮且趋势仍在上升，不足以判「坐标路线行不行」。本次从
  `ckpt_update_100.pt` 续训到 **300 轮**（`run_coords_scale1500/`，exit 0，4 rank checksum
  一致 = 702.032333），每个检查点用同一套双重指标测量。
- **结果（全部相对各自架构的未训练基线 −41.65 / 24.2% 做配对）**：

  | ckpt | greedy | sd | 配对 delta | t | 选中最近地块 | 实走/最近 |
  | --- | --- | --- | --- | --- | --- | --- |
  | untrained | −41.65 | — | — | — | 24.2% | — |
  | 100 | −40.00 | 3.98 | +3.05 ± 1.61 | 1.9 | 28.1% | 1.74× |
  | 150 | −41.30 | 5.41 | +1.52 ± 1.02 | 1.5 | 32.8% | 1.66× |
  | 200 | −40.15 | 3.95 | +2.31 ± 1.22 | 1.9 | 35.0% | 1.63× |
  | 250 | −38.32 | 5.00 | **+3.10 ± 1.04** | **3.0** | **38.0%** | **1.50×** |
  | 300 | −39.35 | 5.04 | +1.90 ± 1.25 | 1.5 | 35.6% | 1.62× |

  训练日志（50 轮 block 均值）：reward **−43.20 → −40.85（300 轮共 +2.35）**，
  `entropy` 4.262 → 3.625，`ratio_std` 0.136 → 0.327（仍在缓慢上升，未见饱和）。
- **对照（条目 32，100 轮即饱和）**：显式距离 **100 轮就到 +10.32（t=8.3）/ 最近地块 79.9% /
  实走 1.04×**，200 轮 +10.33（已完全饱和）。

  | | 坐标@1500 | 显式距离 |
  | --- | --- | --- |
  | 配对收益 | **+1.9 ~ +3.1（300 轮）** | **+10.32（100 轮）** |
  | 训练日志 sampled reward | +2.35 / 300 轮 | **+8.33 / 200 轮** |
  | 最近地块率 | 35~38%（饱和） | **79.9%** |
  | 实走/最近 | 1.50~1.62× | **1.04×** |
- **终局结论：坐标路线决定性劣于显式距离。**
  1) 用 **3 倍**训练预算（300 vs 100 轮）只拿到 **约 1/3** 的改善；
  2) 行为指标**饱和在 35~38%**，不到显式距离（79.9%）的一半；
  3) 配对收益在 **+1.9 ~ +3.1 之间震荡**，从未接近 +10。
- **尺度缺陷的定位得以澄清**：修尺度确实带来可测收益（条目 35：+0.93 → +3.05，
  动力学也更好：`ratio_std` 0.097→0.183），**但它不是主因** —— 修好尺度并给足 3 倍预算
  仍然追不上。主因是**结构性的**：显式距离把「UAV→地块」的欧氏距离**算好了**，
  「去最近」只是对一列输入取单调函数；坐标路线要求网络**自己学会算欧氏距离**，还要穿过
  3 层 GAT 传下来，而**图上没有 UAV 节点**，UAV 坐标只在全局 `z_G` 里。
  这等于要求 decoder 独自把两个不同来源的量做非线性组合 —— 可表示，但学得极慢。
- **建议**：恢复并保留 `49249cc` 的显式距离；若要继续用坐标，应把它当**增量**而非替代
  （两者信息互补），或引入显式的「UAV 相对地块」关系边（把 UAV 作为图节点接入），
  让距离能被消息传递直接算出来，而不是要求 decoder 反推。
- **回滚方法**：本条目仅为文档；如需恢复显式距离，`git revert 5eaf410` 即可。

## 2026-10-16

### 35. 修正位置尺度（10000 → 1500 m）并复测静态坐标消融
- **文件**：修改 `coverage_env/config/default.yaml`、
  `coverage_env/rl/decoder/global_state_encoder.py`、
  `coverage_env/tests/test_graph_builder.py`。
- **变更（旧 → 新）**：
  1) `rl.graph.position_scale_m`：`10000.0` → **`1500.0`**（管 Field/Boundary 节点中心 +
     图的 global uav/depot 坐标）；
  2) `DEFAULT_SCALES["position_scale_m"]`：`10000.0` → **`1500.0`**（管 `z_G` 里的
     uav/depot 位置）。**两处必须一致**，否则节点中心与 UAV 位置不在同一坐标系；
  3) 新增 2 项测试：`test_position_channels_are_well_scaled`（坐标通道不得被同级特征淹没、
     且必须有实际变化）、`test_graph_and_global_position_scales_agree`（两处尺度一致）。
- **原因**：条目 34 实测静态坐标版本 100 轮完全没有学到空间决策（最近地块率 24.0% ≈
  未训练 25.1%，配对收益 +0.93 不显著），并量化出坐标通道被压缩了约 10 倍。
- **尺度怎么选的（实测，不是猜）**：60 个实例 / 790 个地块的真实 |中心坐标| 最大
  **1253 m**、工作区跨度约 2400 m。候选对比：

  | 尺度 | \|坐标\| 均值 | 被截断比例 |
  | --- | --- | --- |
  | 10000（原值） | 0.045 | 0.00% |
  | 1000 | 0.437 | **6.14%** |
  | **1500（采用）** | **0.290** | **0.00%** |
  | 2000 | 0.219 | 0.00% |
  | 3000 | 0.146 | 0.00% |

  取 **1500 m = 「最小且零截断」**。不用 1000 m 是因为被截断的 6.14% 恰好是最远的那批地块，
  它们的坐标会被压成同一个值 —— 这是**非线性**丢信息，比线性缩放更糟。
  改后实测：坐标 \|mean\| **0.045 → 0.290**（6.4×），per-column std **0.03 → 0.36**，
  两两坐标差中位数 0.0675 → **~0.45**，截断 **0.00%**。
- **一个必须说明的混淆**：本改动**同时**改变了 `z_G` 里 uav/depot 的位置尺度，而它是
  「UAV→地块」距离计算的另一半，所以若复测成功，**无法单独归因于节点坐标**。
  不过有一个反向论据：条目 32 的显式距离版本在 uav 位置尺度仍为 10000 时工作良好
  （+10.32，t=8.3），说明全局尺度不是「最近地块」决策的使能条件。
- **验证结果（100 轮 from scratch，`run_coords_scale1500/`）**：同配置、同评估方法。

  | ckpt | 坐标@10000 | 坐标@**1500** | 显式距离 `49249cc` |
  | --- | --- | --- | --- |
  | untrained | −42.99 / 25.1% | −41.65 / 24.2% | −43.38 / 28.2% |
  | ckpt_25 | −44.53（−0.64） | −45.46（**−3.72 ± 1.35，t=−2.8**） | −41.23（**+2.00**） |
  | ckpt_50 | −45.38（**−2.94**） | −44.50（**−2.96 ± 1.25，t=−2.4**） | −34.29（**+9.79，t=9.8**） |
  | ckpt_100 | −43.39（+0.93 ± 3.33，t=+0.3） | **−40.00（+3.05 ± 1.61，t=+1.9）** | **−33.35（+10.32，t=8.3）** |
  | 最近地块率 @100 | 24.0% | **28.1%** | **79.9%** |
  | 实走/最近 @100 | 1.89× | **1.74×** | **1.04×** |

  训练日志也一致：坐标@1500 的 `ratio_std` 从 0.097 涨到 0.183（坐标@10000 是
  0.064→0.110），`entropy` 降得更快（4.413→4.072 vs 4.444→4.301）—— **修正尺度确实改善了
  优化动力学**。
- **结论：尺度缺陷是真的（修它有可测收益），但它不是主因，也不足以让坐标路线追平显式距离。**
  1) 修尺度后配对收益从 **+0.93（不显著）升到 +3.05（t=1.9，仍不显著）**，最近地块率
     24.0% → 28.1%，实走/最近 1.89× → 1.74× —— 方向对、幅度小。
  2) 但同期显式距离是 **+10.32（t=8.3）/ 79.9% / 1.04×**，**领先约 7 个点、行为上差 2.9 倍**。
  3) 两条坐标路线在 **ckpt_25/50 都显著差于自己的未训练基线**（t=−2.8 / −2.4），
     说明存在一个**破坏性的早期阶段**；显式距离版本没有这个问题。
  4) 结构性差异仍在：显式距离**把「UAV→地块」的欧氏距离算好了**，「去最近」只是对一列输入
     取单调函数；坐标路线要求网络**自己学会算欧氏距离**，还要穿过 3 层 GAT（图上无 UAV 节点，
     UAV 坐标只在全局 `z_G` 里）。
- **保留的疑点（不够下终局结论）**：坐标@1500 在 50→100 轮之间从 −2.96 升到 +3.05，
  **仍在上升**，所以 100 轮可能不足以判定它「行不通」，只能判定它**在同等预算下远不如
  显式距离**。若要下终局结论需跑到 200–300 轮。
- **混淆说明的更新**：本改动同时改了节点中心与 `z_G` 的 uav/depot 位置尺度，故上面
  +0.93 → +3.05 的改善无法区分来自哪一处。
- **回滚方法**：`git revert <本提交 SHA>`（把两处尺度改回 10000 并删除 2 项新测试）。

### 34. 条目 33 消融实验的实测结果：静态节点坐标**未能学到空间决策**，与显式距离差距巨大 条目 33 消融实验的实测结果：静态节点坐标**未能学到空间决策**，与显式距离差距巨大
- **来源**：`git pull` 到 `5eaf410` 后，为条目 33 补做的验证（其 CHANGELOG 写了「运行结果记录在
  本次修改报告中」，但该提交里**没有报告文件**，只有 CHANGELOG，所以消融结论实际未被记录）。
- **方法**：同一套配置（`--num-updates 100 --seed 42`，地块 5~20 泛化，4 卡 DDP）从零训练，
  再用与条目 32 完全相同的双重指标评估：**20 个固定实例的逐实例配对 greedy 收益** +
  **「选中最近未完成地块」的比例**（随机基线 ≈ 1/未完成地块数）。
- **结果（两者都是各自架构的未训练基线之上配对计算）**：

  | ckpt | 静态坐标 `5eaf410` | 最近地块 | 实走/最近 | 显式距离 `49249cc` | 最近地块 | 实走/最近 |
  | --- | --- | --- | --- | --- | --- | --- |
  | untrained | −42.99 | 25.1% | — | −43.38 | 28.2% | — |
  | ckpt_25 | −44.53（−0.64 ± 3.00） | 21.9% | 2.05× | −41.23（**+2.00**） | 30.3% | 1.71× |
  | ckpt_50 | −45.38（**−2.94 ± 0.95，t=−3.1**） | 23.3% | 1.99× | −34.29（**+9.79，t=9.8**） | 68.5% | 1.08× |
  | ckpt_100 | −43.39（**+0.93 ± 3.33，t=+0.3**） | 24.0% | 1.89× | **−33.35（+10.32，t=8.3）** | **79.9%** | **1.04×** |

  训练日志自身的 sampled reward 也一致：静态坐标 100 轮 block 均值
  **−43.63 → −43.50（+0.13，完全平）**；显式距离 **−43.59 → −36.18（+7.41）**。
- **结论：条目 33 的消融是**负结果**。静态坐标版本在 100 轮内**
  1) 配对收益 **+0.93（t = +0.3）不显著**，ckpt_50 甚至**显著差于自己的未训练基线**（t = −3.1）；
  2) 「选中最近地块」比例 **24.0% ≈ 未训练 25.1%**，行为上**完全没有学到**该决策；
  3) 每步仍走最近距离的 **1.89 倍**（显式距离为 1.04 倍）。
  两者在 ckpt_100 相差 **约 10 个点**（−43.39 vs −33.35）。
- **一个可量化的缺陷（很可能是主因）**：坐标用 `rl.graph.position_scale_m = 10000` 归一化，
  但实测 60 个实例、790 个地块的真实坐标范围只有
  x ∈ [−1253, 1169] m、y ∈ [−1210, 1251] m（跨度约 **2400 m**）。于是归一化后
  |坐标| 均值仅 **0.045**、p95 仅 0.105，而同一 region 的其他 field 特征量级是
  **0.38~1.0**（area 0.377 / perimeter 0.436 / compactness 0.772 / remaining 1.0）
  —— **坐标通道比同级特征小约 11 倍**，而真正决定「谁最近」的地块两两坐标差中位数只有
  **0.0675**。没有任何值被截断（0.0%），所以不是正确性问题，而是**数值条件很差**。
  换成与工作区匹配的尺度（约 1000 m）会把坐标量级抬到 ~0.45，与其他特征可比。
- **另一层结构性原因**：显式距离把「UAV→地块」的欧氏距离**算好了**再交给 actor，
  「去最近」因此只是对某一列输入取单调递减函数；而静态坐标要求网络**自己学会计算欧氏距离**
  —— 并把信息从 3 层 GAT 传下来（图上没有 UAV 节点，UAV 坐标只在全局 `z_G` 里），
  这是一个难得多、也依赖消息传递保真的函数。
- **未做的事**：本条目只做验证与记录，**没有改代码**。是否回滚条目 33、或先修坐标尺度再复测，
  留给仓库维护者决定。
- **回滚方法**：本条目仅为文档，`git revert` 撤销即可（它不改变任何行为）。

---

## 2026-10-16

### 33. 用地块/边界静态坐标替代 Action 显式距离特征（消融实验）
- **文件**：修改 `coverage_env/rl/graph/feature_builder.py`、
  `coverage_env/rl/graph/graph_builder.py`、`coverage_env/rl/encoder/feature_embedding.py`、
  `coverage_env/rl/encoder/hetero_encoder.py`、`coverage_env/rl/decoder/action_decoder.py`、
  `coverage_env/tests/test_graph_builder.py`、`coverage_env/tests/test_policy_forward.py`；
  删除 `coverage_env/rl/decoder/action_spatial.py` 和
  `coverage_env/tests/test_action_spatial_features.py`。
- **变更（旧 → 新）**：Field 原始特征从 5 维扩为 7 维，追加归一化地块中心 `(x,y)`；
  Boundary 原始特征从 2 维扩为 4 维，追加归一化边界中心 `(x,y)`。坐标使用
  `rl.graph.position_scale_m`（默认 10,000 m）缩放并截断到 `[-1,1]`。Action Decoder 输入
  从 `384 + 5 + 128 = 517` 改为 `384 + 128 = 512`，不再拼接每动作 5 维显式距离。
- **原因**：验证策略能否从节点坐标和图消息传递中学习空间关系，减少人工构造的
  Action 距离特征；这是可回滚的输入表示消融，不代表已证明与显式距离性能等价。
- **兼容性**：旧 Field/Boundary 输入维度和带显式距离的 Decoder 权重均不兼容，需从头训练，
  不可直接加载此前 checkpoint。
- **验证**：更新图特征维度及 Policy 输入维度断言；运行结果记录在本次修改报告中。
- **回滚方法**：`git revert` 本条对应提交即可恢复 5/2 维节点特征及显式 Action 距离模块。

---

## 2026-10-15

### 32. 给每个候选动作加显式空间特征，让 actor 能表达「去最近的地块」
- **文件**：新增 `coverage_env/rl/decoder/action_spatial.py`、
  `coverage_env/tests/test_action_spatial_features.py`；修改
  `coverage_env/rl/decoder/action_decoder.py`、`coverage_env/tests/test_policy_forward.py`。
- **背景（条目 31 的根因）**：图的节点特征不含位置、边特征全是相对量（图平移不变），
  绝对位置只在 `z_G` 里被全局广播，对每个候选动作相同 —— 所以 actor 在结构上无法表达
  「去最近的地块」。实测策略选中最近地块的比例仅 18.2%（随机 ~11.7%），
  而一个平凡的最近地块启发式比它好 21%（−34.50 vs −43.76）。
- **变更（旧 → 新）**：`action_context` 由 `concat(action_vectors, z_G)` 改为
  `concat(action_vectors, x_action_spatial, z_G)`，decoder 输入由 512 变 **517**
  （384 + 5 + 128）。`x_action_spatial` 每行 5 维，全部归一化到 `[0, 1]`：
  `[UAV→地块质心, UAV→entry boundary, depot→地块质心, depot→entry boundary,
  entry→exit]`，尺度 `distance_scale_m = 2000`（与 site 摘要一致）。
  几何来源**优先取 transition 的 state**（与 `_global_sources` / `_site_source` 同理由，
  replay 时 environment 已经走到别处），缺失几何时退化为全零而不是抛异常。
- **影响**：decoder 首层输入维度变化，**旧 checkpoint 不兼容**，必须从头训练。
- **测试**：新增 4 项 —— 形状/取值范围/Environment 与 state 结果逐位一致；
  `UAV→entry` 这一列对候选的排序与真实几何排序完全相同（这是本改动的核心断言）；
  `action_context` 里确实嵌入了这块特征；几何缺失时退化为零。
  另把 `test_policy_forward` 里硬编码的 `4 * 128` 改为由模块维度推导（原写法会
  在维度变化时误报）。全量 121 项，仍只有 2 个改动前即存在的失败。
- **验证（已完成，`coverage_env/results/run_spatial_features/`，100 轮 from scratch）**：
  架构已变、旧 checkpoint 不兼容，故从零训练。注意**未训练基线本身也变了**
  （距离特征给了随机网络一个「偏好近处」的归纳偏置），所以参照系必须重测：
  新架构 untrained = −43.38 / 28.2%（旧架构是 −44.50 / 19.1%）。

  | checkpoint | greedy | sd | 配对 delta vs untrained | t | 选中最近地块 | 实走 vs 最近 |
  | --- | --- | --- | --- | --- | --- | --- |
  | untrained | −43.38 | — | — | — | 28.2% | — |
  | ckpt_25 | −41.23 | 6.40 | +2.00 ± 1.36 | 1.5 | 30.3% | 665 / 389 m（1.71×） |
  | ckpt_50 | −34.29 | 5.53 | +9.79 ± 1.00 | 9.8 | 68.5% | 425 / 394 m（1.08×） |
  | ckpt_75 | −34.53 | 5.00 | +9.13 ± 1.17 | 7.8 | 70.6% | 433 / 392 m（1.10×） |
  | ckpt_100 | −33.35 | 4.65 | +10.32 ± 1.25 | 8.3 | 79.9% | 403 / 386 m（1.04×） |
  | ckpt_150 | −33.58 | **4.24** | +10.01 ± 1.33 | 7.5 | **89.9%** | **402 / 400 m（1.005×）** |
  | ckpt_200 | −33.66 | 4.40 | +10.33 ± 1.32 | 7.8 | 70.8% | 412 / 383 m（1.08×） |
  | *最近地块启发式（参照）* | *−34.50* | *4.41* | | | | |

  **结论：根因诊断成立，修法有效且效果巨大。**
  1) 200 轮后 **配对改善 +10.33（t = 7.8）**，而旧架构跑满 200 轮只有 **+0.74（t = 0.43）**。
  2) **「选中最近地块」比例 28.2% → 70.8~89.9%**，每步实走距离从最近距离的 2.2 倍降到
     **1.005~1.08 倍** —— 行为层面确实学会了这个此前表达不出来的决策。
  3) **−33.66 优于启发式参照 −34.50**：学会「去最近地块」后还有超出。ckpt_150 的 89.9%
     降到 ckpt_200 的 70.8%，说明它会为能量/工作量等因素偏离纯最近，这是合理的。
  4) 连训练日志自身的 sampled reward 都一路上升（block 均值 −43.59 → **−35.26**，共 +8.33），
     而旧架构 200 轮该指标完全平；`mean_ep_fields` 全程 11.9~12.5，排除「抽到更简单地块」。
- **收益在 ~50–100 轮即饱和**：ckpt_100（−33.35 / +10.32）与 ckpt_200（−33.66 / +10.33）
  在噪声内完全相同，**100→200 轮零增益**，greedy 稳定在 ≈ −33.4。
  **所以剩下的空间不在这个杠杆上**，需换层面：工作量/entry-exit 选择、能量与补给、
  跨地块全局顺序（目前是贪心最近，未必全局最优）。
- **遗留问题（如实记录）**：后半程 critic 明显退化 —— 末 10 轮 `explained_variance`
  在 **−6.16 ~ +0.43** 乱跳，`value_mean` 在 −3 ~ −39 震荡，`adv_std` 有时与 `return_std`
  齐平（该轮基线失效）。可能原因：策略仍在快速变化（entropy 4.42 → 3.30）而 critic 追不上，
  或 critic 的 10× 学习率此时偏高。它**没有**影响本次策略结果（greedy 全程稳定），
  但很可能是 100 轮后停滞的原因之一，值得单独处理。

### 31. 根因定位：actor 的动作表示里**没有位置**，策略在结构上无法算「最近地块」
- **背景**：条目 26（修 critic，EV 0→0.92）、29（actor lr 10×）、30（λ=0）三个干预
  都没能让均值 reward 显著改善。于是先建立**参照系**，再查表示能力。
- **参照系（决定性）**：在 RL 动作空间里实现等价的「最近未完成地块」启发式
  （`/tmp/eval_nn.py`），用**同一个 reward 函数、同一批 20 个固定实例**：

  | policy | greedy 均值 | sd |
  | --- | --- | --- |
  | **最近地块启发式** | **−34.50** | 4.41 |
  | untrained 策略 | −44.50 | 7.46 |
  | 训练后 ckpt_200 | −43.76 | 4.62 |

  **任务远未饱和：一个平凡的启发式比训练后的策略好 9.26（21%）。**
- **策略连这个最基本的决策都没学会**（`/tmp/probe_choice.py`，10 个固定实例，
  greedy，统计「所选地块 == 质心最近的未完成地块」的比例）：

  | policy | 选中最近地块 | 随机基线 | 实走距离 vs 最近距离 |
  | --- | --- | --- | --- |
  | untrained | 19.1% | ~11.3% | — |
  | **trained-200** | **18.2%** | ~11.7% | **779.6 m vs 353.3 m** |

  训练前后**没有区别**（都在随机基线之上一点），且每步平均走的是最近距离的 **2.2 倍**。
- **根因（代码层确认）**：actor 的动作上下文是 `concat(action_vectors, z_G)`，其中
  `action_vectors` 由该动作的 field/boundary 嵌入拼成。而
  `coverage_env/rl/graph/feature_builder.py` 里：
  - `FieldFeatureBuilder` → `[area_norm, perimeter_norm, compactness, orientation, remaining_ratio]`
    —— **没有任何位置信息**；
  - `BoundaryFeatureBuilder` → `[length_norm, orientation]` —— **也没有位置**；
  - 位置只出现在 `build_uav_feature` 的 `x_u, y_u`（和 `GlobalFeatureBuilder`），
    而它们是**全局广播**进 `z_G` 的，对每个候选动作都相同。
  - 图上也没有 UAV 节点，所以不存在「UAV↔地块」的边来补偿。

  **结论：actor 完全拿不到「UAV 到候选地块」的距离，也就无法表达「去最近的地块」。
  这不是优化问题，是表示能力缺失。** 这也解释了为什么调 critic / lr / λ 全都无效
  —— 缺失的信息在 actor 的输入里，不在损失里。
- **λ 扫描结论（条目 30 的实验，两档都跑完）**：机制生效（`adv_std` 3.0→1.4、EV→0.97），
  但策略层面全部不显著，且点估计都为负、sd 随 λ 下降而变差：

  | λ | greedy 均值 | sd | 配对 delta vs ckpt_200 | t | 改善实例 |
  | --- | --- | --- | --- | --- | --- |
  | 0.95（基线 ckpt_200） | −43.76 | **4.62** | — | — | — |
  | 0.5（`run_gae_lam_0.5`） | −44.07 | 5.05 | −0.31 ± 1.25 | −0.25 | 12/20 |
  | 0.0（`run_gae_lam_0.0`） | −44.15 | 6.44 | −0.39 ± 1.47 | −0.27 | 11/20 |

  与根因诊断一致：优势估计变干净了，可 actor 依然没有位置信息可用。
- **至此优化侧的手段已穷尽**：critic（已修好，EV 0.92）、actor 学习率（10× 更差）、
  GAE λ（0.5 / 0 均无效）。三者都在改 loss 或优化，而缺失的信息在 actor 的**输入**里。
- **建议的下一步**：给**每个候选动作**加显式空间特征（UAV 当前位置到该动作 entry
  boundary / 地块质心的距离、到 depot 的距离、地块面积等），接进 `action_context`。
  这样「去最近地块」就变成输入的线性函数，应当立刻可学。

### 30. `--gae-lambda` 可命令行覆盖（优势估计的偏差/方差权衡）
- **文件**：修改 `coverage_env/rl/trainer/distributed.py`、
  `coverage_env/tests/test_training.py`。
- **变更（旧 → 新）**：`distributed.py` 新增 `--gae-lambda FLOAT`（缺省 `None` = 保留
  `default.yaml` 的 0.95），在 `_worker` 里覆盖 `config["ppo"]["gae_lambda"]`；
  原先 `--ppo-epochs` / `--batch-size` 的覆盖逻辑合并为 `ppo_overrides`。
- **原因**：条目 29 证明了「步长机制」对 actor 无效（actor lr 10× 实测更差），
  瓶颈是**policy-gradient 的信号质量**而非步长。`A(s,a)` 目前用 λ=0.95 的 GAE，
  几乎等于「该动作之后的全部回报」，其方差被 episode 结果噪声主导（实测同一实例内
  回报 sd ≈ 3.0，而动作质量能带来的差异只有 ~2）。λ→0 时 `A ≈ r + γV(s′) − V(s)`，
  是方差低得多的动作质量估计；**低 λ 引入的偏差只在 critic 不准时才有害，而当前
  EV = 0.92，正好满足使用它的前提** —— 这是修 critic 换来的、此前用不了的杠杆。
- **注意**：`epochs` 增大**不是**同类杠杆 —— 它只是在同一批 advantage 上多走几步，
  等价于增大步长，按条目 29 的结果预期无效；只有新的数据或更低方差的 advantage
  估计才治本。
- **测试**：新增 `AdvantageConfigTests`（`gae_lambda` 覆盖确实到达 agent；
  λ=0 时 advantage 的方差小于 λ=0.95）。
- **实验（进行中）**：从 `run_fields_5_20_final/ckpt_update_200.pt` 续训 50 轮，
  λ = 0.0 与 0.5 各一档，之后在 20 个固定实例上做逐实例配对评估。

### 29. actor 学习率独立可配（默认 10× = 3e-3）；更正「梯度裁剪限制步长」的误判
- **文件**：修改 `coverage_env/rl/ppo/ppo.py`、`coverage_env/config/default.yaml`、
  `coverage_env/tests/test_training.py`。
- **更正（重要）**：条目 28 里「`actor_grad_norm ≈ max_grad_norm`，说明 actor 步长受限」
  这个推断是**错的**。实测验证：`clip_grad_norm_` 只对梯度做**统一缩放**，而 Adam 对
  统一缩放**不变**（`m/√v` 不变），所以裁剪对 Adam 是 **no-op**，不构成任何步长限制。
  （实测：同一参数、同方向、梯度缩放 5 倍，8 步 Adam 后最大差 3e-8。）已把这条写进代码
  注释，避免以后再拿 `grad_norm` 当「受限」的证据。
- **变更（旧 → 新）**：`ppo` 新增 `actor_learning_rate`（缺省回落到 `learning_rate`），
  actor / critic 各自独立建 Adam；`default.yaml` 设 `actor_learning_rate: 0.003`（10×）。
- **保留下来的依据（比裁剪那条弱）**：Adam 每参数每步走 ≈ `lr`，故 `lr × steps` 限定一次
  训练中参数能走多远。共享 3e-4、200 轮 × 8 minibatch → actor 总位移仅约 0.48；
  critic 当初正是靠 10× 才够到 return 的量程。
  但 entropy 已经动过（4.40 → 3.76），说明 actor 并非完全走不动，所以这条依据不足以定论。
- **因此本次是实证而非推理**：用 `--resume ckpt_update_200.pt` 续训 50 轮（actor lr 10×），
  再在 **20 个固定实例上做逐实例配对评估**，与 ckpt_200 对比。
- **实测结果：候选 1 失败，方向为负。** 运行 `coverage_env/results/run_resume_actor_lr10x/`
  （201→250，exit 0，`ratio_std` 0.13→0.20、`entropy` 3.76→3.56 证明 actor 确实动得更快）：

  | policy | greedy 均值 | sd |
  | --- | --- | --- |
  | ckpt_200（actor lr 3e-4） | −43.76 | **4.62** |
  | ckpt_250（actor lr 3e-3） | −44.59 | **7.13** |

  **配对 delta = −0.83 ± 1.59，t = −0.52，8/20 改善，95% CI [−3.94, +2.29]**
  —— 不显著，点估计为负，且**一致性反而变差**（sd 4.62 → 7.13，把条目 28 里唯一的
  稳健收益部分抵消掉了）。
- **解释**：步长机制**不能从 critic 迁移到 actor**。critic 有明确的回归目标（return），
  步长大 → 更快逼近好的拟合；actor 的 policy-gradient 估计是**噪声主导**的，
  步长大 → 更快地在噪声上跑偏。这与条目 28 的观察一致：策略会变确定（entropy 下降）
  但均值不动。**瓶颈是信号而非步长。**
- **因此 `default.yaml` 把 `actor_learning_rate` 保持为基准 0.0003**（配置项保留，
  便于以后再试；`critic_learning_rate` 仍是实测有效的 0.003）。
- **测试**：`LearningRateTests` 断言 actor 保持基准、critic 保持 10×，以及两者可独立覆盖 /
  缺省回落基准。

### 28. 正式训练（200 轮，5~20 地块泛化）结果：critic 已修好，但均值仍无显著提升
- **运行**：`coverage_env/results/run_fields_5_20_final/`，`--num-updates 200 --seed 42`，
  未指定 `--num-fields`（=> `[5,20]` 每 episode 重采样），`critic_learning_rate=3e-3`。
  代码版本 `09b928d`。200/200 完成，exit 0，checkpoint 每 50 轮一份。
- **训练日志（`mean_ep_reward` = on-policy 采样回报）看起来完全平**：
  `OLS(reward ~ update)`：slope = **−0.00145 ± 0.00233，t = −0.62**；
  首/末三分之一 −43.47 vs −43.73（t≈−0.7）。20 轮 block 均值在 −43.2 ~ −43.9 之间震荡。
  同期内部指标却一路向好：`entropy` 4.40 → 3.76、`explained_variance` 0.67 → **0.92**、
  `critic_loss` 52 → **9.6**；`corr(reward, mean_ep_fields) = −0.020`（跨地块归一化仍有效）。
- **但把 checkpoint 拿出来在固定实例上做确定性（greedy）评估 —— 结论要打折，且必须以配对检验为准**：

  | checkpoint | greedy 均值 | sd | SE | **配对 delta vs untrained** | t | 改善实例数 |
  | --- | --- | --- | --- | --- | --- | --- |
  | untrained | −44.50 | 7.46 | 1.67 | — | — | — |
  | ckpt_50 | −44.59 | 5.84 | 1.31 | −0.09 ± 0.84 | −0.10 | 12/20 |
  | ckpt_100 | −44.54 | 6.18 | 1.38 | −0.04 ± 0.91 | −0.04 | 11/20 |
  | ckpt_150 | −43.76 | 4.16 | 0.93 | +0.74 ± 1.45 | 0.51 | 10/20 |
  | ckpt_200 | **−43.76** | **4.62** | 1.03 | **+0.74 ± 1.70** | **0.43** | 12/20 |

  **正确结论（在 20 个固定实例上做逐实例配对）**：
  1) **均值提升只有 +0.74，配对 t = 0.43，95% CI = [−2.60, +4.07] —— 不显著。**
     早先用 10 个实例得到 −44.44 → −42.11（+2.33）**是实例抽样造成的假象**
     （那 10 个里有一个 untrained 灾难样本 −73.76 → −44.24 的 +29.5 离群值）。
  2) **稳健成立的只有一个效应：方差显著下降**，sd 7.46 → **4.62**（约 −38%），
     SE 1.67 → 1.03；即策略变得更**一致**、不再出现灾难性 episode，但**平均水平没上去**。
  3) 因此「日志里 reward 不涨」**基本是真实的**，不是度量假象 —— 换成低方差的
     确定性评估后仍无显著改善。这一条更正了本节早先的判断。
- **修 critic 的价值仍然成立且必要**（EV 0 → 0.92、critic_loss 52 → 9.6、cross-field
  归一化 corr = −0.020 依然有效），但**它不是 reward 不涨的充分解释** ——
  修好 critic 只让策略变得更稳定，没有显著提升均值。
- **下一步的候选（尚未验证）**：`actor_grad_norm` 末段 ≈ 0.49 ≈ `max_grad_norm` 0.5，
  说明 actor 的梯度几乎每步都被裁剪，其步长同样受 `lr` 限制（与 critic 当初的问题同源）；
  `entropy_coef=0.01` 下 entropy 仍有 3.76（~100 候选均匀值 ≈ 4.6），探索成本约 1.2。
  可分别试「actor lr 提高」「rollout_steps / epochs 增大（更多在策略梯度步数）」
  「entropy_coef 降到 3e-3」。
- **建议的下一步**：给 trainer 加「固定实例上的定期确定性评估」并写入日志（把学习曲线
  的方差从 ~1.8 压到 <0.5），再用 `--resume` 续训对比。

### 27. DDP 训练支持从 checkpoint 续跑（`--resume`）
- **文件**：修改 `coverage_env/rl/trainer/distributed.py`、`coverage_env/rl/ppo/ppo.py`、
  `coverage_env/tests/test_training.py`。
- **变更（旧 → 新）**：
  1) `DistributedTrainer.train(num_updates, rollout_steps, *, start_update=1)` 新增
     `start_update`，循环从该 index 开始，`--num-updates` 仍表示**总目标轮数**。
  2) `_worker` 新增 `--resume PATH`：在 **DDP 包装之前**把 checkpoint 载入
     actor/critic/optimizer（这样 state_dict 键不带 `module.` 前缀），随后
     `start_update = load 返回的 update 索引 + 1`；各 rank 读同一个共享文件，
     载入后 `dist.barrier()` 再继续。
  3) `PPOAgent.load_checkpoint` 返回值由 `None` 改为 **checkpoint 的 update 索引**；
     并把 `_value_bias_initialized` 一并存取 —— 续跑时若不保留该标志，critic 的
     偏置热启动会重新触发，把已训练好的输出偏置覆盖掉。对旧格式 checkpoint
     该键缺省为 `True`（宁可不动，也不要覆盖）。
- **原因**：长训练一旦中断（本项目已在第 88 轮因地块摆放崩溃丢过一次 40 分钟）就只能从头再来；
  且「先跑 50 轮确认趋势再续跑」这种分阶段验证也需要它。
- **影响**：续跑**不是**与「一次跑完」逐位等价的轨迹 —— torch/环境 RNG 状态没有存进
  checkpoint，所以续跑后采样序列会分叉。对本用途（继续训练）无碍，但不要用它复现单次
  长跑的确切轨迹。
- **验证**：新增 2 项测试（载入返回 update 索引且保留热启动标志；
  `train(3, 4, start_update=4)` 不执行任何更新）；4 卡 DDP 端到端实测
  `--num-updates 1 --checkpoint-every 1` 存出 `ckpt_update_1.pt`，再
  `--resume ckpt_update_1.pt --num-updates 2` 打印 `next update 2` 且只跑了 update 2，
  actor checksum 由 870.000883 变为 869.665499（证明载入的权重确实被继续训练）。
- **回滚方法**：`git revert <本提交 SHA>`。

### 26. 定位并修复「critic 解释不了 return 方差」：critic 独立学习率（已验证 EV 0→0.88）
- **文件**：修改 `coverage_env/rl/ppo/ppo.py`、`coverage_env/config/default.yaml`、
  `coverage_env/tests/test_training.py`。
- **完整排查过程（含一次失败尝试，按时间顺序）**：
  1) `a684dde` **偏置热启动**：把 critic 输出偏置设为首个 return 批次的均值。
     `critic_loss` 立刻 342 → 68，`explained_variance` 一度升到 0.015，但随后衰减回 0；
     `value_std` 始终 ≤0.26 而 `return_std` 8~11 —— 说明只解决了「偏移」，没解决「尺度」。
  2) `a4b02fa` + `19bd340` **value 归一化（先运行尺度、后冻结尺度）**：**两版都不稳定**，
     都在 update 2 崩。`adv_std`/`return_std` 从 8.3 暴涨到 51.3，`critic_loss` 在
     0.4~138 之间振荡。根因：归一化把 critic 的输出乘上 `scale` 之后才喂回 GAE 的
     bootstrap，等于给 `returns → critic → bootstrap → returns` 这个环加了增益；
     原尺寸下该环是稳定的。
  3) `6975f12` **回退归一化**，恢复到「原始特征直给 critic + 偏置热启动」的稳定配置。
  4) `5d6f325` **critic 独立学习率（默认 10× = 3e-3）** —— 这一版成功。
- **根因（算术）**：return 偏移 ≈ −18、展布 ≈ 11，而 critic 最后一层权重初始化只有约
  ±0.06，要覆盖该范围需涨约 10 倍。Adam 对每个参数的步长 ≈ `lr`，8 minibatch/update
  → 约 2.4e-3/update → 需要 **200~400 轮**，而整个训练就是 200 轮。
  **所以 critic 是「步数预算不足」，不是「输入缺信息」。**
  这也解释了 `value_coef`（0.5）与 `max_grad_norm`（0.5）为何调不动它：Adam 按参数
  归一化，它们只改梯度**幅度**、不改**步长**；唯一能改步长的是 lr 或总步数。
- **改动**：`PPOAgent` 新增 `ppo.critic_learning_rate`（缺省回落到 `learning_rate`，
  向后兼容），actor / critic 分别建 Adam；`default.yaml` 设 `0.003`（10×）。
- **验证（15 轮，`--seed 42`，run 目录 `coverage_env/results/probe_critic_lr10x/`）**：

  | update | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 15 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | `explained_variance` | 0.036 | 0.082 | 0.253 | 0.565 | 0.734 | 0.862 | 0.694 | 0.856 | **0.908** | 0.891 |
  | `advantage_std` | 8.28 | 9.28 | 10.7 | 7.21 | 5.81 | 3.96 | 4.96 | 2.84 | 2.96 | 3.27 |
  | `return_std` | 8.27 | 9.82 | 11.2 | 10.4 | 10.6 | 10.5 | 11.3 | 10.6 | 10.6 | 11.2 |
  | `value_std` | 0.36 | 1.87 | 2.58 | 4.27 | 6.35 | 7.89 | 10.7 | 9.64 | 9.38 | 10.5 |

  前 3 轮 EV 均值 **0.124**，后 5 轮均值 **0.876**（max 0.908），且**全程稳定无爆炸**。
  关键旁证：`advantage_std` 从 ≈ `return_std` 降到约 **0.29 × return_std**
  —— 基线真正被减掉了；`value_std` 从 0.36 涨到 10.5，与 `return_std`（11.2）同量级。
  这已经接近探针测得的可预测性上限（0.88），即 critic 基本吃满了其输入中的信息。
- **尚未验证**：15 轮内 `mean_ep_reward` 仍在 −43 附近摆动（尚看不出趋势），
  `policy_loss` 仍 ≈ 0；需要完整 200 轮才能判断策略是否因此开始改进。
- **回滚方法**：`git revert 5d6f325`（或把 `default.yaml` 的 `critic_learning_rate`
  删掉即回落到共享 lr）。

### 25. 修法 1b：critic 改为直接消费「z_G + 原始 UAV/depot/布局特征」
- **文件**：修改 `coverage_env/rl/decoder/action_decoder.py`、`coverage_env/rl/ppo/ppo.py`、
  `coverage_env/rl/trainer/distributed.py`。
- **变更（旧 → 新）**：
  1) `CoveragePolicy._build_global_embedding` 改为返回 `(z_G, critic_input)`，其中
     `critic_input = concat(z_G, x_uav, x_depot, x_site)`；`forward` / `forward_batched`
     的输出新增 `"critic_input"`；`encode_graph` 返回值由 2 元组改为 3 元组
     `(embeddings, z_G, critic_input)`（`[1]` 语义不变）。
  2) `PPOAgent`：`ValueNetwork` 拆出 `input_dim`（原来把 `hidden_dim` 同时当输入与隐层宽度）；
     `PPOAgent.__init__` 按 `encoder.hidden_dim + uav_dim + depot_dim + site_dim` 推导出
     152 维 critic 输入；新增 `_critic_features(output)`（优先 `critic_input`，回退
     `global_embedding`）；`value_from_output` / `value_for_graph` / `_batched_minibatch_rows`
     统一走它。
  3) 诊断新增 `value_mean` / `value_std`（critic 预测的均值与标准差），用于区分
     「基线没学到」与「学到了均值但无状态依赖」。
  4) `DistributedTrainer` 的 next-value 编码同步改为 3 元组解包。
- **原因**：**修法 1（把布局特征接进 `GlobalStateEncoder`）不够**。重跑探针后，
  新 `z_G` 的线性可解码性只从 **0.140 升到 0.185**，而原始 13 维布局特征有 **0.458**。
  根因是架构性的：`GlobalStateEncoder` **只被 actor 的损失训练**，且在进入 critic 前被
  `.detach()` —— 布局信息若只经由它就等于穿过一个 critic 无法塑造的表征。
  另一个独立问题：48 轮里 `explained_variance ≈ 0`，**连线性探针在同样输入上都能拿到
  0.14~0.19**，说明 critic 自身也没拟合好。
- **验证**：critic 输入 152 维（128+10+2+12）、`encode_graph` 3 元组、`value_for_graph`
  正常；新增 24 项相关测试通过；全量 109 项仍只有 2 个改动前即存在的失败。
  效果（`explained_variance` 是否从 0 涨起来）在紧随其后的训练里验证。
- **回滚方法**：`git revert <本提交 SHA>`（恢复 `ValueNetwork(hidden_dim)` 与 2 元组
  `encode_graph`，并把 `value_from_output` 改回用 `global_embedding`）。

### 24. 修法 1：给 `z_G` 加入显式地块布局特征，修复 critic 无法预测 return
- **文件**：修改 `coverage_env/rl/decoder/global_state_encoder.py`、
  `coverage_env/rl/decoder/action_decoder.py`、
  `coverage_env/tests/test_global_context_and_role_embedding.py`。
- **变更（旧 → 新）**：
  1) `global_state_encoder.py` 新增 `build_site_feature(source)`（12 维）与
     `_site_tables()`：从 **Environment 或序列化 state dict** 计算
     `[num_fields, total_area, depot↔地块距离的 mean/max/min/std,
     地块两两距离的 mean/max/min/std, 地块面积 mean/std]`；新增 4 个归一化尺度
     （`max_fields_scale` / `max_total_area_scale` / `site_distance_scale_m` /
     `max_field_area_scale`）。
  2) `GlobalStateEncoder` 新增 `site_dim`（默认 12）与 `site_feature` 入参，
     `input_dim = graph_dim + uav_dim + depot_dim + site_dim`；`forward` 与
     `forward_batched` 都经由 `_assemble()` 按固定顺序拼接。
  3) `CoveragePolicy` 新增 `_site_source()`（**优先用 transition 的 state**，与
     `_global_sources` 同一理由），并在 `_build_global_embedding` 与 `forward_batched`
     里计算并传入 `x_site`。
- **原因**：`explained_variance` 48 轮恒为 0、`advantage_std ≈ return_std`，说明 critic
  没减掉任何基线。**探针实验**（50 instance × 3 局，嵌套交叉验证）定位了根因：
  可预测性上限 R²=0.864，但 `z_G` 只拿到 **0.140**，而 13 维显式空间特征拿到 **0.537**
  —— `MeanPool` 对地块的**数量、顺序、位置**全不变，而 return（单位面积路程成本）恰恰
  取决于 depot↔地块距离与地块散布。`num_fields` 单独仅 −0.054，说明归一化有效、缺的是几何。
- **注意**：`z_G` 同时馈入 critic 与 actor 的每个动作上下文，所以这一改动**同时**让 actor
  能看到布局摘要；网络首层输入由 268 维变为 280 维，**旧 checkpoint 不再兼容**。
- **验证**：新增 3 项测试 —— `build_site_feature` 12 维且 Environment 与 state dict 逐位一致、
  对地块顺序不变但对实例敏感、以及「整体平移地块必须改变 `z_G`」（因为池化对位置不变，
  变化只能来自 `x_site`）。全量 109 项，仍只有 2 个改动前即存在的失败。
- **回滚方法**：`git revert <本提交 SHA>`（并把 `GlobalStateEncoder` 的 `site_dim` 设回 0）。

### 23. PPO 训练诊断指标：定位「critic 收敛但 reward 不涨」
- **文件**：修改 `coverage_env/rl/ppo/ppo.py`、`coverage_env/rl/trainer/distributed.py`。
- **变更（旧 → 新）**：`PPOAgent.update()` 的返回字典新增
  `policy_loss`（advantage 加权 surrogate，与 entropy bonus 分离）、
  `advantage_std`（**归一化前**的原始 advantage 尺度）、`return_mean` / `return_std`、
  `explained_variance`（critic 对 return 方差的解释度 `1 - Var(target-pred)/Var(target)`）、
  `ratio_std`、`clip_fraction`、`approx_kl`、`actor_grad_norm`
  （`clip_grad_norm_` 在裁剪**前**返回的总范数）；`DistributedTrainer` 把这些写入 CSV，
  并在控制台第二行打印其中最关键几项。空 buffer 时返回全零，保证日志 schema 稳定。
  **只增加统计与记录，不改变任何训练数值**（advantage 归一化、loss、梯度裁剪均原样）。
- **原因**：88 轮正式训练显示 `mean_ep_reward` 从 −42.65 漂到 −43.90（块均值标准误 ≈0.26，
  即约 6 个标准误的**反向**漂移），与最初「`critic_loss` 收敛但 reward 不涨」同一症状；
  而 `entropy` 虽已算出却未记录，导致只能靠推理判断是「熵主导」还是「critic 太弱」。
- **首测（3 轮）已给出关键证据**：

  | update | entropy | policy_loss | explained_var | advantage_std | return_std | grad_norm |
  | --- | --- | --- | --- | --- | --- | --- |
  | 2 | 4.348 | −0.00088 | **+0.000** | 8.540 | 8.542 | 0.1343 |
  | 3 | 4.574 | −0.01005 | **+0.002** | 8.099 | 8.106 | 0.1692 |

  两点：① `explained_variance ≈ 0` 且 `advantage_std ≈ return_std` —— **critic 没有减掉任何
  基线**，advantage 几乎等于原始 return，信号被 return 的方差（sd ≈ 8.5）淹没；
  ② `policy_loss ≈ 0` 而 entropy bonus 约 `0.01 × 4.4 = 0.044`，且 `entropy` 在**上升**
  （4.35 → 4.57）—— 熵正则可能正在把策略推回均匀分布。
- **回滚方法**：`git revert <本提交 SHA>`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。

### 22. 修复跨 episode 泛化的地块摆放崩溃：`sample_components` 增加有限重试
- **文件**：修改 `coverage_env/env/instance_generator.py`、
  `coverage_env/tests/test_domain_randomization.py`。
- **变更（旧 → 新）**：`sample_components` 拆为「重试壳 + `_sample_components_once`」；
  前者最多尝试 `attempts`（默认 8）次，捕获摆放失败（`_sample_center` 抛出的
  `RuntimeError`）后整轮重抽，全部失败才抛出带尝试次数的 `RuntimeError`。
- **原因**：**这是条目 15 引入的回归。** 地块摆放是拒绝采样（`_sample_center` 只试 5000 个
  随机位置），密集 workspace 下即使「通常放得下」的地块数量也可能偶尔失败。旧代码整个训练
  只在启动时生成一次实例，所以几乎碰不到；改为**每 episode 重采样**后，一次正式训练要重采样
  数千次，稀有失败就变成了必然。实测一次 200 轮 4 卡训练在**第 88 轮**崩溃：
  ```
  RuntimeError: Unable to place non-overlapping fields in configured workspace
    environment.reset() → _resample_instance() → sample_components() → _sample_center()
  ```
  单独测量 5~20 地块各 120 次（共 1920 次）摆放失败率为 **0**，即 p 很小（约 1e-4 量级），
  与「3000 次重采样中出现 1 次」一致；按 p=1e-3 估计，重试 8 次后失败概率约 1e-24。
- **不影响采样分布**：重试只是多消耗 RNG，成功路径与原来逐位一致（未触发重试时，
  重抽序列完全相同——所以重启训练会复现前 88 轮的数据）。
- **验证**：新增 3 项测试 —— 前两次摆放失败仍能成功返回、尝试耗尽后抛出且信息含尝试次数、
  以及走完整 `reset()` 路径能恢复。全量 106 项，仍只有 2 个改动前即存在的失败。
- **回滚方法**：`git revert <本提交 SHA>`（把重试壳去掉，`_sample_components_once` 改回
  `sample_components`）。

### 21. DDP 训练日志增加「按 episode」指标，使跨地块数量泛化的趋势可评估
- **文件**：修改 `coverage_env/rl/trainer/distributed.py`。
- **变更（旧 → 新）**：`DistributedTrainer.train()` 的 rollout 循环新增逐 episode 累计
  `episode_reward`，在 `env_done` 时（此刻 `environment` 仍描述刚结束的那一集）记录该集
  reward 与地块数；rank 0 的 CSV 新增 `episodes` / `mean_ep_reward` / `mean_ep_fields`
  三列，控制台行追加 `eps=… mean_ep_reward=… mean_ep_fields=…`。
  **只增加记录，不改动任何训练逻辑**（梯度、rollout 长度、DDP 同步均不变）。
- **原因**：地块数量泛化后，128 步 rollout 会跨越 7~10 集、每集地块数不同，`reward` 只是
  rollout 求和，被实例方差主导（实测 −304 ~ −486）；而 `coverage_ratio` / `total_distance`
  是 rollout 结束瞬间的快照，经常为 0（如第 8 轮）。两者都无法支撑验收标准里的
  「reward 上升趋势」与「不同地块数量之间量级可比」两条。
- **效果（前 7 轮实测）**：`mean_ep_reward` 稳定在 **−41.0 ~ −44.3**，而同期
  `mean_ep_fields` 在 9.2 ~ 14.9 之间变化、`reward` 在 −304 ~ −452 之间摆动。
  即归一化后的「单位面积路程成本」与地块数量无关，验收标准第 1 条的量级可比性获得直接证据。
- **回滚方法**：`git revert <本提交 SHA>`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。

### 20. 可见性图连接器优化第二步：按 region 缓存多边形内部可见性子图
- **文件**：修改 `coverage_env/planner/coverage_path.py`、
  `coverage_env/planner/boundary_aware_sweep.py`、
  `coverage_env/tests/test_visibility_connector.py`。
- **变更（旧 → 新）**：
  1) `_shortest_path_inside_polygon` 新增 keyword-only 参数 `pair_cache`（默认 `None`，
     行为与原来完全一致）；对 `first >= 2` 的顶点对查表/写表，
     `start`/`end` 参与的顶点对永不缓存。
  2) `BoundaryAwareSweepPlanner.plan` 每次调用建一个 `visibility_pairs` 字典，透传给
     `_materialize_strokes` → `_connector`；字典生命周期与 `region_geometry` 一致，
     因此天然有界，不存在全局缓存增长或跨几何体下标串味的问题。
  3) `_connector` 由 `@staticmethod` 改为实例方法以携带该字典（仓库内无其他调用点）。
- **原因**：一次 `plan()` 只构造一个 `region_geometry`，却要为每个候选方向 / `reverse_lanes`
  反复调用 `_connector`；实测同一几何体平均被查 **12.7~19.4 次**。而
  `_shortest_path_inside_polygon` 的 O(V²) 谓词扫描里约 **88%** 是「多边形自身顶点两两之间」
  ——这部分只取决于几何体，与 `start`/`end` 无关，是纯重复计算。
- **不改变算法**：顶点集合、候选对的遍历顺序、边插入顺序（决定 Dijkstra 的并列择优）
  全部保持原样，只是跳过重复的 `covers` 谓词。
- **实测（固定权重与动作序列的 8 步 rollout，三次测量）**：
  改前 3.92s → 改后 **1.34~1.36s（2.90×）**；输出完全一致：
  `distance=10921.693648 coverage=0.397438`，
  trajectory `sha256=18599effde4501b3`（含/不含该步骤三次运行全部相同）。
- **回滚方法**：`git revert a689c83`；或删除 `pair_cache` 参数、`plan()` 里的
  `visibility_pairs`、`_materialize_strokes`/`_connector` 的透传参数，并把 `_connector`
  还原为 `@staticmethod`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。
- **测试**：`test_visibility_connector` 增至 6 项，新增「真实输入上 cached vs uncached
  逐点一致」与「只缓存多边形顶点对、不缓存 start/end」；planner/validation 相关 33 项通过。

### 19. 可见性图连接器优化第一步：prepared geometry 加速重复 covers 谓词
- **文件**：修改 `coverage_env/planner/coverage_path.py`；新增
  `coverage_env/tests/test_visibility_connector.py`。
- **变更（旧 → 新）**：`_shortest_path_inside_polygon` 里把 `polygon.buffer(1e-7)` 提出来
  只算一次，并用 `shapely.prepared.prep(...).covers()` 替换裸 `.covers()`：
  ```python
  visible = prep(polygon.buffer(1e-7))
  if visible.covers(direct): ...
  ```
- **原因**：该函数是 rollout 的 CPU 热点（约占 72%），成本来自约 **23 万次** shapely
  `covers` 谓词与约 **23 万次** `LineString` 构造。prepared geometry 会缓存内部空间索引，
  重复谓词快约 2 倍。（顺带发现：`polygon.buffer(1e-7)` 本身只要 0.025 ms，不是收益来源，
  原先算了两次也只是顺手合并。）
- **不改变算法**：顶点集合、O(V²) 候选对、遍历顺序、边插入顺序、Dijkstra 全部不变，
  只替换谓词实现。
- **实测（20 地块、`seed=42`，同一批 296 次 connector 调用逐输入 A/B）**：
  改前 8.38 ms/call → 改后 4.26 ms/call，**1.97×**，输出逐点完全一致；
  端到端 8 步 rollout：7.58s → 3.88s（约 **1.95×**，重复测稳定）。
- **回滚方法**：`git revert cef3fc2`（把 `prep(...)` 还原为 `polygon.buffer(1e-7).covers(...)`），
  删除 `test_visibility_connector.py`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。
- **测试**：新增 `test_visibility_connector.py` 4 项 —— 凹多边形黄金路径逐点断言（4 用例）、
  端点与包含性、prepared 谓词在合成几何与真实区域几何上与标量 `covers` 逐对一致（>500 对）。
  另跑 planner 相关 30 项测试全通过。

### 18. 修复 PPO replay 用错全局状态：`_global_sources` 改为优先 transition 的 state
- **文件**：修改 `coverage_env/rl/decoder/action_decoder.py`（`_global_sources`）、
  `docs/training_performance_notes.md`；新增
  `coverage_env/tests/test_ppo_replay_state.py`。
- **变更（旧 → 新）**：`CoveragePolicy._global_sources(environment, state)` 原先**先看
  environment**——只要 `environment.uavs` 非空就直接返回 `environment.uavs[0]` /
  `environment.depot`，`state` 只在没有 environment 时才用。现改为**先看 `state`**：
  从 `state["uavs"]` / `state["depot"]` 取 UAV 与 depot 特征，缺哪一项才回落到
  environment（`state=None` 或没有 `uavs` 时行为不变，例如 rollout 期的
  `value_for_graph`）。
- **原因**：`PPOAgent.update()` 调 actor 时不传 `environment`，`forward` 回落到
  trainer 绑定的活环境；而 replay 发生在整段 rollout **之后**，环境早已推进，于是重算出的
  `z_G`（进而 log-prob 与 value）与采集时记录的不一致。这直接破坏 on-policy 假设：
  首个 epoch 的 `ratio ≠ 1`，clipped surrogate 从一开始就在优化错误目标，critic 的
  回归目标也自相矛盾。属于条目 17 记录时顺带发现、当时未修的既有问题。
- **实测（10 条 transition，dropout=0，把活环境改到远离 rollout 终点后重算 log-prob）**：

  | | max abs Δlog-prob | ratio 范围 |
  | --- | --- | --- |
  | 修复前（env 优先） | 4.73e-03 | [0.9953, 1.0024] |
  | 修复后（state 优先） | **0.00e+00** | **[1.0000, 1.0000]** |

  即修复后 replay 与采集**逐位一致**，`ratio ≡ 1`。
- **为什么不影响 rollout**：rollout 时传入的 `state` 就是当前状态，`build_uav_feature` /
  `build_depot_feature` 对「UAV 对象」与「state dict」两条分支产生**完全相同**的向量
  （`get_state()` 已包含 `position/max_energy/max_payload/speed/coverage_width/
  remaining_energy/remaining_payload`）。新增测试
  `test_uav_and_depot_features_agree_between_object_and_state` 与
  `test_rollout_encoding_is_unchanged` 守住这一点。
- **影响**：会改变训练行为（这是修复本身的目的——首个 epoch 的 ratio 从 ≠1 变成 =1）。
  若需复现旧行为（不推荐），把 `_global_sources` 的优先级改回 environment 优先即可。
- **回滚方法**：`git revert <本提交 SHA>`（或恢复 `_global_sources` 的 environment 优先
  逻辑），删除 `test_ppo_replay_state.py`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。
- **测试**：新增 `test_ppo_replay_state.py` 6 项（对象/state 特征一致、`_global_sources`
  优先级与兜底、stale 环境不影响 replay、逐条与批量 replay 均复现 stored log-prob、
  rollout 编码不变）。

### 17. 图 batch 化接入 update()：action embedding 切片 + 批量策略前向（默认关闭）
- **文件**：修改 `coverage_env/rl/encoder/batched_encoder.py`、
  `coverage_env/rl/decoder/action_embedding.py`、
  `coverage_env/rl/decoder/global_state_encoder.py`、
  `coverage_env/rl/decoder/action_decoder.py`、`coverage_env/rl/ppo/ppo.py`、
  `docs/training_performance_notes.md`；新增
  `coverage_env/tests/test_batched_policy_update.py`。
- **变更（旧 → 新）**：
  1) `batched_encoder.py` 把批量前向抽成自由函数 `encode_graph_batch(encoder, batch)`，
     任意 `CoverageGraphEncoder` 实例可直接批量化（不再依赖子类或复制权重）。
  2) `ActionEmbedding.forward` 新增 `field_offset`/`boundary_offset`（默认 0）：把本图
     局部索引重定位到批 embedding 的行区间。
  3) `GlobalStateEncoder.forward_batched` 新增：原 `forward` 的 `reshape(-1)` 只服务单图，
     批量版保持 `[G, ...]` 前导维，输出 `[G, hidden]`。
  4) `CoveragePolicy.forward_batched(contexts)` 新增：**1 次 GNN** over 批图 + 逐
     transition 解码，返回与 `forward` 同构的输出（含 `action_distribution`/`mask`/
     `global_embedding`），两条路径可互换。
  5) `PPOAgent.update(buffer, *, batched=False)` 新增开关：批量路径用
     `_batched_minibatch_rows`（1 次编码 + critic 按 minibatch 一次求值），默认路径抽成
     `_per_transition_minibatch_rows`；**默认 `batched=False`，训练行为完全不变**。
  6) 新增 7 项测试：批量前向 vs 逐条前向（绑定环境 / 仅 state 两种）、分布合法性、
     空输入、update 两条路径损失一致、批量路径确实更新参数、空 buffer。
- **原因**：承接条目 16 的原型，用户要求「接着做 action embedding 切片，把只 batch encoder
  这条打通到可以真正接进 update()」。
- **实测结果**：
  - 策略前向等价（12 transition / 93~123 候选）：mask 完全相同，
    distribution 1.5e-08、log_prob 2.4e-07、entropy 9.5e-07、`global_embedding` 8.9e-07。
  - `update` 端到端（64 transition / epochs=4 / batch=64）：CPU 6.45s → 1.78s（3.62×），
    **CUDA 2.25s → 0.35s（6.39×）**；`actor_loss`/`critic_loss`/`entropy` 相对误差 < 1e-4。
  - 全量测试 90 项，仅剩 2 个改动前即存在的失败（见条目 15 的「已知失败」）。
- **注意事项**：
  1) 默认关闭；`DistributedTrainer`/`Trainer` 本次未接，避免静默改变训练语义。
  2) `dropout > 0` 时两条路径的 dropout mask 排布不同（统计等价、逐位不等价），故等价性
     测试用 `dropout=0`；默认 `dropout=0.1` 下属两种都合法的正则化实现。
  3) `places=5` 之类的**绝对**容差在这里不适用（`critic_loss` ~484，逐次前向差 ~1e-7 经多个
     optimizer step 放大到 ~1e-5），测试改用相对容差 1e-4。
- **顺带发现的既有问题**：`update()` 调 actor 时不传 `environment`，
  `CoveragePolicy.forward` 回落到 `self.environment`，于是 `_global_sources` 优先取
  **当前**环境的 UAV/depot 而非 transition 存下的 `state`——replay 首个 epoch 的 ratio
  因此 ≠ 1。**已由条目 18 修复**（本条提交时批量路径忠实复现了同一行为，故两者仍一致）。
- **回滚方法**：`git revert <本提交 SHA>`；或删除 `test_batched_policy_update.py`、
  `encode_graph_batch`、`ActionEmbedding` 的两个 offset 参数、`GlobalStateEncoder.forward_batched`、
  `CoveragePolicy.forward_batched`、`PPOAgent.update` 的 `batched` 参数与两个 helper
  （恢复内联的逐 transition 循环）。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。

### 16. 图 batch 化原型：disjoint-union 编码器 + 等价性验证（不接入训练）
- **文件**：新增 `coverage_env/rl/encoder/batched_encoder.py`、
  `coverage_env/tests/test_batched_encoder_equivalence.py`；修改
  `docs/training_performance_notes.md`（P1 补充实测结论）。**未改任何训练路径代码。**
- **变更（旧 → 新）**：
  1) `batched_encoder.py` 新增 `collate_graphs()`（把 N 张单图拼成一张不连通大图：
     field 全部在前、boundary 全部在后，边端点重映射到批的统一索引空间，并记录
     `field/boundary/edge_batch` 与 per-graph offsets）、`segment_mean()`（按 graph 分段
     均值）、`batched_graph_context()`（替代全局 mean 池化）、`BatchedGraph` 容器与
     `BatchedCoverageGraphEncoder`（继承 `CoverageGraphEncoder`，共享权重，新增
     `forward_batched()`；原 `forward()` 保持不变）。
  2) 新增 8 项等价性与不变量测试：真实重采样实例逐位一致、跨字段边全空的退化场景、
     单图批、`global_feature` 按图堆叠、边不跨分量、boundary 局部索引不越界（回归守卫）、
     空批拒绝、CUDA 舍入级一致。
  3) `docs/training_performance_notes.md` 的 P1 增补实测小节。
- **原因**：用户要求先做一个「不接入训练、只验证大图拼接前后 encoder 输出逐位一致」的
  原型，用来决定 P1（图 batch 化）是否值得落地。
- **关键结论**：
  1) 由于 batch 采用类型分离布局，单图里的 `edge_index - n_field` **原样成立**（换成总
     field 数即可），`FieldBoundaryEncoder` 无需改动；真正要改的只有「全局 mean 池化 →
     按图分段 mean」一处，比条目 15 文档里估计的工作量小很多。
  2) CPU 上 `field_embedding`/`boundary_embedding`/三类 attention **逐位一致（diff=0.0）**；
     仅 pooled `graph_context` 差 ~2e-7~5e-7（`index_add_` 与 `mean` 累加顺序不同）。
     CUDA 上全部差异 ~1e-6（GEMM tiling）。
  3) 实测加速（64 图 / 4013 节点 / 纯 encoder forward）：CPU 1.22×，**CUDA 14.9×**
     （per-graph 98.1 ms → batched 6.6 ms），印证 GPU kernel launch 开销是主因。
- **踩坑记录**：初版把每条边的 boundary 索引减成「本图内 `[0, n_boundary_k)`」，导致不同
  图的 boundary 特征行**重叠**，输出误差约 1.0 却不抛异常；改为批级统一索引
  （`edge_index - n_field_total`）后归零。该不变量已由
  `test_boundary_indices_stay_inside_their_owning_graph` 守住。
- **未完成（接入训练前仍需做）**：action embedding 的 per-transition 切片、rollout 图快照
  的按 minibatch collate、`attention_debug` 多图语义、critic 侧 `z_G` 的 per-graph 消费。
- **回滚方法**：删除新增的 2 个文件；把 `docs/training_performance_notes.md` 的
  「P1 补充」小节删去即可。不影响任何既有训练/推理路径。
- **验证**：`python -m unittest coverage_env.tests.test_batched_encoder_equivalence` 8/8 通过；
  连同 `test_encoder`/`test_policy_forward`/`test_graph_builder` 共 16 项通过；
  直接运行 `python coverage_env/tests/test_batched_encoder_equivalence.py` 打印完整对比表。

### 15. 跨 episode 泛化 + 总路程 reward 重设计 + 训练效率调研
- **文件**：修改 `coverage_env/env/instance_generator.py`、`coverage_env/env/environment.py`、
  `coverage_env/rl/reward.py`、`coverage_env/config/default.yaml`、
  `coverage_env/rl/trainer/distributed.py`、`coverage_env/rl/trainer/trainer.py`、
  `coverage_env/rl/evaluation/evaluator.py`、
  `coverage_env/tests/test_boundary_guided_planner.py`、`coverage_env/tests/test_rl_interface.py`；
  新增 `coverage_env/tests/test_domain_randomization.py`、`docs/training_performance_notes.md`。
- **变更（旧 → 新）**：
  1) **跨 episode 泛化（任务 1）**：`InstanceGenerator` 拆出 `sample_num_fields()` /
     `sample_components()`（生成 `fields/depot/uavs/resolution`，不建环境、不离散化），
     `generate()` 增加 `randomize_each_reset`（显式 `num_fields` 时默认 `False`，
     省略时默认 `True`）。`CoverageEnvironment.__init__` 新增 `config`/`rng`/
     `randomize_each_reset`；`reset()` 在首次之后每轮调用 `_resample_instance()`，
     用环境持有的 `random.Random` 重新抽地块数量（`environment.num_fields` 的 `[min,max]`）、
     地块几何、depot 和 UAV 参数；新增 `_compute_total_area()` 并在每轮 reset 缓存
     `self.total_area`（离散化可覆盖面积之和）。`default.yaml` 新增 `uav.ranges`
     （`max_energy/max_payload/speed/coverage_width/alpha/beta` 的 min/max），
     `distributed.py` 不再固定 `num_fields=20`（`--num-fields` 默认改为 `None`，给了才固定），
     `Trainer`/`Evaluator` 的 `max_steps` 每轮按当前地块数重算。
  2) **reward 重设计（任务 2）**：`calculate_reward` 由
     `-λ1·d/ds - λ2·e/es + λ3·a/cs - λ4·frag/fs` 改为只保留
     `-distance / total_area * reward_scale`（step-level shaping，episode 累加即
     `-总路程/总面积`）；`environment.step()` 删除 fragmentation 前后计算与
     `_reward_scales()`/`_characteristic_distance()`/`_fragmentation_summary()`，
     改用当前 episode 的 `self.total_area`；`default.yaml` 删除 `rl.reward_weights`、
     `rl.reward_normalization`、`rl.fragmentation_small_component_max_points`，新增
     `rl.reward_scale: 100.0`。
  3) **训练效率调研（任务 3）**：新增 `docs/training_performance_notes.md`，只总结
     逐 transition 前向、DDP all-reduce、rollout planner 长尾的根因、优化方向、
     预期收益/工作量/风险，**未改任何性能代码**。
- **原因**：① 训练期间地块数量与 UAV 参数完全固定，策略过拟合单一场景；② 旧 reward
  由距离/能耗/覆盖/碎片化四项加权、信号混杂，且未对齐「最小化总路程」目标，跨地块数量
  不可比；③ update 的逐 transition 前向与高频 all-reduce 缺少书面调研。
- **取舍说明**：归一化用 `total_area`（离散化可覆盖面积）而非 `num_fields`，可同时消除
  面积差异；实测 5/10/20 地块的单次作业 `总路程/总面积 = 0.55/0.44/0.43`，跨地块数量可比
  （20 地块不再系统性更负）。`reward_scale` 取 **100**（而非示例的 1000）：实测比值约
  0.43~0.55，1000 会把 episode return 抬到 −430~−550、critic MSE ~2.5e5；100 使 episode
  return 落在 O(10~100)（−38~−55），critic MSE ~4e2，更易拟合。scale 与地块数量无关，
  不破坏归一化。task 3 未对 planner 的 CPU 长尾做实现。
- **回滚方法**：`git revert <本提交 SHA>`；或手动恢复 `InstanceGenerator.generate()`/
  `CoverageEnvironment`（删除 `config`/`rng`/`randomize_each_reset`/`_resample_instance`/
  `_replace_instance`/`_compute_total_area`/`total_area`）、`calculate_reward` 旧四项目标、
  `default.yaml` 的 `reward_weights`/`reward_normalization`/`fragmentation_small_component_max_points`
  与 `uav.ranges`、`distributed.py` 的 `num_fields` 固定逻辑、`Trainer`/`Evaluator` 的
  `max_steps`；删除 `test_domain_randomization.py` 与 `docs/`。之后运行
  `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。
- **验证**：单卡 `python -m unittest coverage_env.tests.test_training` 通过；新增
  `test_domain_randomization`（重采样/可复现/total_area/step reward）4 项通过；4 卡 DDP
  `--num-updates 2 --rollout-steps 8 --ppo-epochs 1 --batch-size 4` 跑通，4 rank
  actor 参数 checksum 一致（928.721299），无死锁、无 unused-parameter 报错；单卡泛化
  训练 6 episode 跑通（reward −38~−49，critic_loss ~4e2~7e2）。
- **已知失败（与本改动无关，改动前已存在）**：`test_uav.UAVSimulationTests.
  test_depot_and_manual_path_execution` 仍硬编码 `energy==40.0`（alpha 0.08 下应为 16.0，
  见条目 14）；`test_nearest_neighbor_episode` 5 地块基线因药量耗尽无法 10/10 完成
  （见条目 13 的「已知影响」）。两者在 `git stash` 回退到改动前代码后同样失败，故未在
  本次修复。

---

## 2026-10-14

### 14. 缩小 workspace + 调低 alpha，使 payload 成为主约束、电量宽裕
- **文件**：修改 `coverage_env/config/default.yaml`。
- **变更（旧 → 新）**：
  1) `environment.workspace`：`min_lat 31.20→31.218`、`max_lat 31.25→31.232`、
     `min_lon 121.42→121.462`、`max_lon 121.52→121.478`（workspace 由约 5.5×9.5 km 缩到
     约 1.5×1.5 km）。
  2) `field_generator.min_field_spacing`：`40.0 → 15.0`。
  3) `uav.energy.alpha`：`0.20 → 0.08`。
- **原因**：workspace 过大导致 20 个小地块散布（实测地块最近邻间距均值约 900 m、最大 3.8 km，
  depot 到地块均值约 3.3 km、最大 7.6 km），`alpha·distance` 的转移能耗把能量吃光，能量先于药量
  耗尽。缩小 workspace 后地块紧凑（间距均值约 280 m、depot 约 700 m），并调低 alpha 降低移动
  能耗，使 payload 成为主约束、电量宽裕。实测（20 地块、不补给 greedy、8 个 seed）：一次作业
  平均覆盖 4.2 个地块，药量先耗尽（剩余约 0.4%），能量还剩约 45.7%。
- **回滚方法**：恢复 `default.yaml` 的 workspace 四个值、`min_field_spacing: 40.0`、
  `alpha: 0.20`。之后运行 `python -m unittest discover -s coverage_env/tests -p "test_*.py"` 验证。

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

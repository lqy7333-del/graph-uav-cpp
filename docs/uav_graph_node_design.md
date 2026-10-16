# UAV 图节点设计说明（方案 A：独立 Stage 3「agent-relative 读出」）

## 1. 要解决的问题

策略要最小化路程，因此「**哪个候选地块/边界离我最近**」是首要决策变量。但原架构里
actor 拿不到它：

- `FieldFeatureBuilder` / `BoundaryFeatureBuilder` 产出的节点特征**不含位置**；
- 所有边特征都是**相对量**（例如 `E_FB` 用 `(boundary.center − field.centroid)/field_scale`），
  整张图**平移不变**；
- 唯一的绝对位置是 `build_uav_feature` 的 `x_u, y_u`，而它在 `z_G` 里被**广播给所有候选动作**，
  对每个动作完全相同；
- 图上没有 UAV 节点，所以也不存在「UAV↔地块」的边可作补偿。

实测后果：策略选中「最近未完成地块」的比例只有 18.2%（随机基线 11.7%），每步平均走
最近距离的 **2.2 倍**；而一个平凡的「去最近地块」启发式比它好 **21%**（−34.50 vs −43.76）。

两条修法都验证过：
- **显式 per-action 距离特征**（`49249cc`）：**有效**，100 轮配对改善 **+10.32（t=8.3）**，
  最近地块率 **79.9%**，实走/最近 **1.04×**。
- **静态节点坐标**（`5eaf410`，含后续尺度修正）：**无效**，300 轮也只有 +1.9~+3.1，
  最近地块率饱和在 35~38%，实走/最近 1.50~1.62×。

本设计是第三条路：**把 UAV 做成图节点、用关系边把几何交给消息传递**，也就是把「显式距离」
那条有效机制**泛化成图原生形式**，而不是在 decoder 上拼一个人工特征。

## 2. 关键机制：为什么这条路线能行

`RelationGATLayer` 的消息是

```
messages = source_h + relation_embedding(edge_attr)
score    = <source,·> + <target,·> + <relation_embedding,·>
output   = LN( W_out(注意力加权聚合) + W_res(node_features) )      # 残差
```

**边特征直接进入消息内容，同时调制注意力。** 因此只要把**距离放进边特征**，距离就会以
`MLP([dx, dy, dist])` 的形式作为一个**显式加项**落到目标节点的嵌入里 —— 是**图把距离算出来
投递过去**，而不是要求 decoder 从原始坐标反推。

## 3. 设计

### 3.1 节点与边

统一索引空间扩展为 **`[所有 Field, 所有 Boundary, UAV]`**，UAV 排在**最后**（这样
`edge_index - n_field` 这类既有表达式全部不变）。

| 项 | 内容 |
| --- | --- |
| 节点 | `UAV` × 1（动作空间 `CoverageAction(field, entry, exit)` 只能寻址一个 UAV） |
| UAV 节点特征 | 6 维，**只有状态、不含位置**：`[energy_ratio, payload_ratio, speed, width, E_max, Q_max]`；过独立的 `uav_mlp` |
| `EDGE_UF = 3` | `UAV → Field`，边特征 `[dx, dy, dist]`（按 `uav_distance_scale_m` 归一化） |
| `EDGE_UB = 4` | `UAV → Boundary`，边特征 `[dx, dy, dist]` |

**UAV 节点为什么不放绝对坐标**：其他关系全是相对量，放绝对坐标会破坏平移不变性，而且
`5eaf410` 已经证明坐标会被尺度问题折磨。几何全部放边特征，既保住平移不变性，又让 GAT
直接使用它。

**边特征为什么带 `dist` 标量**：不能只给 `dx, dy` 让网络自己开方 —— 这是从坐标路线学到的
最直接教训（`MLP` 能看到 `dist` 本身，代价低得多）。

### 3.2 放在哪：独立 Stage 3

```
FeatureEmbedding(x_field, x_boundary, x_uav_node)
   ↓ Stage 1  E_BB_inter        跨地块转移结构   （不变）
   ↓ Stage 2a E_BB_intra        地块内覆盖结构   （不变）
   ↓ Stage 2b E_FB              地块↔边界聚合    （不变）
   ↓ Stage 3  E_UF / E_UB       agent-relative 读出   ← 新增
z_F', z_B' → ActionEmbedding(z_F', z_B'(entry)+r_entry, z_B'(exit)+r_exit) → ActionDecoder
```

**语义**：Stage 1–2 建立的是 **map-relative**（不随 UAV 位置变化）的节点嵌入；Stage 3 把
它们转成动作嵌入真正要读的 **agent-relative** 版本。把「地图结构」和「我在哪」在表示上
分开，既保留了原 two-stage 的 coarse-to-fine 逻辑，又给「我在哪」一个专门的落点。
Stage 3 是残差层，所以 Stage 1–2 的信息一点不丢，只是被增广。

### 3.3 为什么不能塞进 Stage 1（本设计最关键的论证）

`_destination_softmax` 是**对该目标节点的所有入边**做 softmax。若把 UAV 边混进 Stage 1，
一个 Boundary 节点在 Stage 1 的入边 = 若干 `E_BB_inter` + 1 条 `EDGE_UB`，分数相近时
UAV 那条边的权重 ≈ **1/(K+1)**（K 通常 5~15）→ **UAV 消息被稀释 6~16 倍**。

而独立 Stage 3 里每个目标节点**恰好只有 1 条入边** → softmax 恒等于 1 → **零稀释**。

这不是「顺手这么放」，而是唯一能让距离不被竞争掉的放法；它同时等价于「给 UAV 关系配一组
专属权重」，而现有 GAT 本来就是每个 relation family 一个独立 layer，所以这在架构上是自洽的。

**实测确认**：`test_uav_message_attention_is_exactly_one` 断言 UAV 边注意力**恒等于 1.0**，
通过。

## 4. 一并修正的尺度问题

| 配置 | 旧值 | 新值 | 理由 |
| --- | --- | --- | --- |
| `global_distance_scale_m` | 10000 | **3000** | `E_BB_inter` 的 `distance/global_scale`；该关系只连接 `d_near_m=1500` 以内的边界，用 10000 时特征只能落在 0~0.15，与已修的坐标欠尺度问题同类 |
| `uav_distance_scale_m` | 新增 | **3000** | UAV 可在约 2.4 km 工作区内任意位置，3000 m 让对角最远情况仍落在 [0,1] 内且不截断 |

## 5. 工程细节

- **批处理**：`BatchedGraph` 增加 `x_uav_node` / `uav_batch` / `uav_offsets`；
  `collate_graphs` 的端点重映射从「field / boundary 二分类」改为**三分类**（UAV 排在最后，
  原来的 `< n_field` 判定会把它误判成 boundary）。
- **replay 一致性**：UAV 节点是**存储图的一部分**，`encode_graph` 走存储图，因此回放天然一致；
  绝不能改成从 environment 现取 UAV 位置（那会重蹈 `_global_sources` 的 bug）。
- **可视化**：`plot_graph` 原来对未知 `edge_type` 直接 `KeyError`，已补 UAV 边样式与 UAV 节点标记
  （这是本次改动暴露出的既有缺口）。
- **空边安全**：`RelationGATLayer` 对 `E == 0` 有残差兜底；UAV 边永不为空。
- **检查点不兼容**：新增 `uav_mlp` 与 Stage 3 参数，必须从头训练。

## 6. 测试覆盖（`coverage_env/tests/test_uav_graph_node.py`）

1. UAV 边覆盖每个 Field 与 Boundary，数量为 `|F| + |B|`。
2. UAV 边的距离通道**尺度良好**（落在 [0,1]，`max > 0.1`，offset 通道非全零）—— 防止欠尺度回归。
3. **UAV 边注意力恒为 1.0**（零稀释）。
4. **Stage 1–2 保持 map-relative**：只移动 UAV 时，Stage 1/2 的注意力逐位不变；而最终嵌入
   必须有响应 → 分离成立。
5. **距离通道确实到达目的地嵌入**：只把 UAV 边属性乘 0.5（拓扑与所有节点特征不变），
   Field 嵌入必须改变。
6. UAV 节点特征只含状态：移动 UAV 不改变 `x_uav_node`。

此外 `test_batched_encoder_equivalence` 覆盖了含 UAV 节点的批处理等价（单图批仍要求**逐位相等**；
多图批因 GEMM 归约顺序不同放宽到 1e-5，并注明「若节点布局错误会是 O(1) 量级」）。

## 7. 评估方案与成功标准

基线（均为 `seed 42`、地块 5~20、4 卡 DDP、100 轮 from scratch、20 个固定实例配对 greedy）：

| 变体 | 配对收益 | 最近地块率 | 实走/最近 |
| --- | --- | --- | --- |
| 静态坐标（尺度修正后） | +3.05（t=1.9） | 28.1% | 1.74× |
| 显式距离 `49249cc` | **+10.32（t=8.3）** | **79.9%** | **1.04×** |

**成功标准：达到或超过 +10.32。** 需要诚实预期：本设计是被验证有效的显式距离的
**图原生泛化版**，在当前任务上更可能是**追平**而非大幅超越。

它的真正价值在三点泛化：
1. **entry/exit 的选择也自动获得 UAV 相对几何** —— 显式特征只覆盖了 entry，既没有 exit，
   也没有学出来的组合；
2. 未来加 depot / 多 UAV / 障碍 / 禁飞区时**不需要再手工造特征**；
3. 距离由**注意力**参与调制（既进消息内容又进注意力分数），而不只是一个拼上去的标量。

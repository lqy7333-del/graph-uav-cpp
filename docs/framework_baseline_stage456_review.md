# 第四至六段：代码交付与测试边界

日期：2026-11-11。第三段基线 `11b8fdf`；本轮未训练 RL，未运行正式 10/20 地块搜索。

## 实现

- P→D：访问顺序与正权重先确定，再按前后访问质心生成投影区域；每个离散点恰好归属一次。
  零方向使用地块长轴。区域按连通性和满资源可行性有界拆分，不跨原访问重分配。
  编码补齐缺失地块和长度截断另记 encoding_repairs，不与几何修复混合。
- 每个原访问一个门 key，修复子访问继承该 key；所有区域使用共享扫描/执行/补给与重放。
  失败划分也计一次候选评估；每次候选重新划分和探测，不隐藏缓存或免费 CC 补全。
- 共同面板使用已有可信本地 pickle 快照与 SHA256 清单，不重新 reset/生成地图。
  三种框架使用同一快照、物理配置和预算；历史 CC 复用必须满足文件哈希、快照哈希、完整预算及重放校验。
- 队列按完整搜索边界恢复，核对代码、面板、配置、预算；异常停止并写诊断。
  正常 wall-limit 保留完整 incumbent；无解保留为失败，不删除或补采。
- 输出四分项、扣 work 成本、完成率、配对实例 bootstrap、搜索/修复审计、首实例分区图及墙钟搜索曲线。
  任一方法有失败时不报告其成功子集均值为主成绩；完整配对子集明确标注为辅助。

## 实测

- 共享执行、CC/D→P/P→D、队列、补给、机动时间和转弯账本相关回归：98 passed。
- 之后新增邻居上下文改变区域的测试，并补正式面积范围检查，P→D/队列专项测试 11 passed。
- 联合 smoke：seeds 1900200–1900202，3 张新 10 地块地图，每种方法最多 5 次候选、60 秒。
  三框架共 9 次搜索均完成 5 次评估，任务完成且独立重放一致。
  CC 搜索 1.83–2.48 s，D→P 2.81–3.59 s，P→D 15.63–22.90 s。
  P→D 每候选重新做可行性探测，明显更贵；正式验收必须检查其预算利用。
- smoke 面积为 500–800 m²，**不是正式 5,000–20,000 m² 分布**，不支持质量或性能优势结论。
  receipt 的 domain_verified/passed 均为 false，不能用它启动正式队列。
- smoke 运行后完善了报告及正式面积校验；保留原运行代码指纹，不将该记录冒充最终版本正式验收。
- 没有运行完整仓库测试、GPU 训练、正式面积联合验收或正式共同面板实验。

## 工作站入口

使用工作站保存的初始快照，不能把诊断/开发实例当正式面板，也不要加载不可信 pickle。

```bash
# 从 3–5 个独立 10 地块正式面积快照生成验收清单
python -m coverage_env.rl.evaluation.framework_panel \
  --snapshots <snapshot1.pkl> <snapshot2.pkl> <snapshot3.pkl> \
  --output <acceptance_manifest.json>

# 完整预算开发验收；先审查用时，再决定正式队列
python -m coverage_env.rl.evaluation.framework_comparison \
  --manifest <acceptance_manifest.json> --output <acceptance_output> \
  --stage acceptance --max-evaluations 385 --wall-seconds 1800 --seed 42

# 共同 10/20 地块面板另建 manifest；与验收几何不重叠
python -m coverage_env.rl.evaluation.framework_comparison \
  --manifest <formal_manifest.json> --output <formal_output> \
  --stage formal --receipt <acceptance_output>/acceptance_receipt.json \
  --max-evaluations 385 --wall-seconds 1800 --seed 42
```

验收凭证仅检查完成/重放、正式面积和基本搜索执行；不替代对编码多样性、修复程度和计时的人工审查。
正式配置应在验收后冻结。恢复时加 `--resume` 并保持原参数；不支持中断候选内精确恢复。
若旧 CC JSON 缺少必要快照身份或预算字段，明确拒绝直接复用，应先独立审计转换或重新搜索。
本轮没有宣称旧 CC 已经兼容复用，没有把搜索平台当最优性证明。

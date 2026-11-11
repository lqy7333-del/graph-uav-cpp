# 第二段：共享规划与执行接口

日期：2026-11-11。基线：476c1fc。未运行训练、GA 搜索或正式对照。

## 实际实现

- `ConfigurationExecutor.plan_configuration`：纯 CC 规划，保留原区域与扫描逻辑。
- `ConfigurationExecutor.plan_region` / `Environment.plan_region`：指定 RegionTask 的点集与门，不调用 CC 区域生成器。
- `CoveragePlan`：版本 1、不可变元组几何、区域 ID、实例/配置/状态指纹；JSON 往返支持。
- `execute_plan`：执行保存的三段路径，不重新规划。整段资源预检、指定区域返航储备、段间故障回滚。
- 环境共享 `_settle_coverage_result`、`_accumulate_time`、`_perform_resupply`：保留相同物理执行及成本结算。
- 事务补齐 heading、时间、补给计数、终止状态、失败次数、缓存，以及已有 UAV/覆盖/轨迹/记录/日志。
- `framework_execution.replay_regions`：从初始快照重放保存路径，不运行解码器或搜索器；比较实际成本与完成状态。

## 两个执行入口的区别

```python
plan = env.plan_region(task, entry_id, exit_id)
transition = env.step_region_plan(plan)
```

严格执行已保存路径；状态变化、资源不足时抛错，不偷偷补给或重规划。

```python
transition = env.step_region(task, entry_id, exit_id, allow_resupply=True)
trace_row = transition[3]['framework_trace']
```

固定区域任务调度：当前资源不足时，在副本上先验证补给后同区域、同门的规划；
只有验证通过才真实补给并执行。补给后路径变化明确保存在 trace 中。
没有缩减工作量、改分区或调用 CC fallback。补给、覆盖、最终返航及结算任一异常均回滚。
异常由后续搜索器记录为失败候选，本接口不生成虚假的低成本完成结果。

所有纯规划入口要求已 reset/discretize 的单 UAV 实例（显式区域接口）。
旧 CC execute 包装仍保留原有懒初始化兼容行为。

## 验收证据

1. 修改前保存 `fixtures/framework_cc_476c1fc.json`：矩形任务、4 次访问、3 次补给、最终返航。
   重构后逐步路径、区域 ID 和完整 summary 与该独立基准精确一致；reward 容差 1e-10。
2. 新生成的 10 地块地图上，同一区域与门的 CC/显式入口路径及 summary 一致。
3. 显式区域规划期间屏蔽 CC region generator，仍可成功执行，确认没有区域替换。
4. 独立检查已分配点到喷洒扫描段的距离不超过半扫幅；虚报未扫过点被拒绝。
   同时检查扫描段/连接段重建与实际覆盖路径一致、端点属于声明边界、内部路径不出地块。
5. 能量、朝向、覆盖、配置、几何变化均使旧规划失效。
6. 三段执行后注入异常、结算后注入异常、补给后注入异常均恢复原状态。
7. 满资源也不可行、关闭补给、返航储备不足、未知/已覆盖点均拒绝。
8. 完整固定区域任务包含多次补给，独立从全轨迹求距离与航向代理（补给停机不算连续转弯），
   验证总时间=距离/速度+机动+服务；JSON 保存后重放全部匹配，篡改预期成本被拒绝。

## 实际测试

```text
python -m pytest -q coverage_env/tests/test_framework_contracts.py coverage_env/tests/test_framework_execution.py coverage_env/tests/test_configuration_executor.py coverage_env/tests/test_nested_ga.py coverage_env/tests/test_nested_ga_controls.py coverage_env/tests/test_boundary_guided_planner.py coverage_env/tests/test_auto_resupply.py coverage_env/tests/test_failed_action_transition.py coverage_env/tests/test_turn_ledger.py coverage_env/tests/test_maneuver_time.py coverage_env/tests/test_maneuver_eval.py coverage_env/tests/validation/test_energy_consistency.py coverage_env/tests/validation/test_configuration_chain.py --tb=short
107 passed in 43.59s
```

其中新执行测试 19 项。没有运行全量套件、GPU 测试或性能基准。
开发过程发现并修正测试中的零长段处理；最终独立全轨迹航向核算通过。
一次绘图测试遇到 Windows 图片文件占用，后续完整相关测试通过；临时图片目录已清理。

## 边界与第三段交接

- 覆盖审计基于现有离散点完成语义，不声称连续多边形每一平方米均无遗漏。
- CC 保留原有区域/补给决策；新增区域入口按指定任务触发补给，区别仍需在方法描述中公开。
- 成本公式、RL、GA 搜索规则未改；共享事务现在对异常提供更完整的回滚。
- 新指纹与检查有计算开销，未声称在线运行时间与旧版相同。第五段需实测。
- 小型旧 CC oracle 与生成地图回归通过，不代替正式面板旧轨迹的兼容性核查。
- D→P/P→D 算法尚未实现；下一段先预划分并冻结 RegionTask，再接 GA，不要绕过新执行接口。
- 搜索器必须把规划/补给预检耗时计入预算，并将异常计为失败尝试，不能无限重试。

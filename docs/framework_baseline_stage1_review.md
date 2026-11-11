# 第一段审查与交接

基线 af12bca；2026-11-11。正式训练/搜索均未启动。

## 已核实的代码事实

| 模块 | 事实 | 下一段处理 |
|---|---|---|
| planner/configuration_executor.py::execute | 同时生成区域、规划三段路径、执行 UAV、标记点已覆盖 | 拆纯规划与原子执行，旧 execute 保留兼容包装 |
| planner/boundary_aware_sweep.py::plan | 已接收 region 参数，用点对应方块并集裁剪到地块后生成扫描路径 | 三组共用；区域是否被实际扫过需独立检查 |
| env/environment.py::step | CoverageAction 会 estimate/refine workload；CoverageConfiguration 也受 CC 最大 workload 检查 | 指定区域不能通过这两条路径接入 |
| env/environment.py::step_configuration | 写轨迹、heading、执行记录、时间反馈；与 configuration 字段耦合 | 提取共享结算，不复制一份时间模型 |
| env/environment.py::step 末尾 | 无可行 CC action 才自动补给，最终累计 _time_totals | 新基线的下一固定任务可能已不可行但其他 CC action 仍可行；需共享物理补给接口与独立触发判断 |
| env/environment.py::_capture_physical_state | 保存 UAV、覆盖标记、轨迹、记录/日志长度；没有 heading-valid/time totals 等全任务字段 | 新接口事务需补齐回滚范围，并用注入异常验证；不据此声称旧结果已损坏 |
| baseline/nested_ga.py | evaluate 耦合 CC candidates/mask/step；replay 也按 action 三元组重放 | 保留 CC 适配器；新框架需要区域/路径轨迹 schema，不能只保存旧 action 三元组 |
| baseline/nested_ga.py::run | 定时从 run 开始，初始化计入预算；构造 deepcopy 未计入 | 保留旧 CC 口径；共享 setup 单列，新方法特有预处理计时 |

## 第一段实际代码

`planner/framework_contracts.py`：
- 不可变 `RegionTask`，使用 field-scoped point ID，避免 deepcopy 后引用旧对象；
- `validate_partition` 检查离散任务精确、无重叠、无遗漏分配；
- `normalize_visit_weights` 检查 P→D 正权重并归一化，拒绝 NaN/Inf/零/不可表示范围。

这不是几何/可行性验证器，不能直接用来标记任务物理完成。
未加入生产 import，未改变 CC、GA、PPO、reward 或模型输入。

## 测试

Windows Python 3.13：

```text
python -m pytest -q coverage_env/tests/test_framework_contracts.py coverage_env/tests/test_nested_ga.py coverage_env/tests/test_configuration_executor.py
30 passed in 22.23s
```

本次没有跑全量套件，没有 GPU 测试，没有性能或框架质量实验。
尚不能称 D→P/P→D 实现完成或接口已支持任意区域执行。

## 第二段必做验收

1. 抽离前后同 CC 动作序列：覆盖 ID、路径、时间、资源、返航完全对应（浮点容差明示）。
2. 指定区域点集不被 CC region generator 改写；实际扫描覆盖独立核验。
3. 当前资源不足、满资源足够：共享补给后执行；满资源不够：原子拒绝。
4. 三段中任一段异常：覆盖、UAV、heading、时间、记录、日志完整回滚。
5. 状态改变后拒绝过期规划；不可用保存的成本冒充重新执行测量。
6. 分段路径距离、完整航向与分项时间闭合，包含补给和最终返航。
7. 新旧轨迹格式明确版本；新格式包含区域 ID、路径、实例/配置指纹。

上述通过后再开始第三段。中途若必须更改物理语义，应单独记录并重新判断旧 CC 结果可复用性。

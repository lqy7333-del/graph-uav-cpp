import json
import time
from copy import deepcopy
from unittest.mock import patch
import pytest

from coverage_env.tests.test_framework_execution import make_env, state
from coverage_env.baseline.divide_then_plan import DivideThenPlanGA, components
from coverage_env.baseline.nested_ga import GAConfig
from coverage_env.planner.framework_contracts import validate_partition
from coverage_env.planner.framework_execution import replay_regions


def config(**kwargs):
    return GAConfig(**dict(outer_population=2, outer_generations=2,
        inner_population=2, inner_generations=1, max_evaluations=9,
        wall_seconds=60, **kwargs))


def test_real_partition_search_and_replay():
    env = make_env(); before = state(env)
    search = DivideThenPlanGA(env, config())
    result = search.run()
    assert state(env) == before
    assert result['evaluations'] == 9
    assert len(result['outer_records']) == 2
    assert all(r['completed_round'] for r in result['outer_records'])
    assert result['best']['completed'] and result['best']['fallback_steps'] == 0
    assert result['best']['cost'] <= result['baseline']['cost']
    validate_partition(search.tasks, {f.field_id: [p.id for p in f.grid_points] for f in env.fields})
    assert len(search.tasks) > 1  # payload forces actual division
    for t in search.tasks:
        points = {p.id: p for p in env.get_field(t.field_id).grid_points}
        assert len(components(t.point_ids, points)) == 1
    frozen = deepcopy(search.tasks)
    for _ in range(30):
        a = (tuple(t.task_id for t in search.tasks), tuple(.5 for _ in search.tasks))
        b = search.mutate_outer(a, a)
        assert sorted(b[0]) == sorted(a[0])
    assert search.tasks == frozen
    restored = json.loads(json.dumps(result, allow_nan=False))
    assert replay_regions(env, restored['best']['actions'])['total_time'] == result['best']['cost']
    again = DivideThenPlanGA(env, config()).run()
    assert again['best']['cost'] == result['best']['cost']
    assert again['tasks'] == result['tasks']


def test_infeasible_partition_is_reported_not_dropped():
    env = make_env(); env.uavs[0].max_payload = env.uavs[0].remaining_payload = 1
    result = DivideThenPlanGA(env, config()).run()
    assert result['termination'] == 'partition_failure'
    assert result['best'] is None and result['evaluations'] == 0


def test_preprocessing_deadline_is_part_of_budget():
    search = DivideThenPlanGA(make_env(), config())
    with patch.object(search, 'expired', return_value=True): result = search.run()
    assert result['termination'] == 'wall_limit'
    assert result['evaluations'] == 0 and result['best'] is None


def test_invalid_permutation_is_not_repaired_by_fallback():
    search = DivideThenPlanGA(make_env(), config())
    search.started = time.monotonic(); search.partition()
    with pytest.raises(ValueError, match='every frozen task'):
        search.evaluate((), (), 'bad')


def test_programming_error_propagates():
    search = DivideThenPlanGA(make_env(), config())
    with patch('coverage_env.planner.configuration_executor.ConfigurationExecutor.plan_region',
               side_effect=RuntimeError('injected')):
        with pytest.raises(RuntimeError, match='injected'): search.run()

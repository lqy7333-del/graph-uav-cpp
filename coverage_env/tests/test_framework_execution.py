"""Stage 2: independent legacy oracle, exact geometry and transaction boundaries."""
from copy import deepcopy
from dataclasses import replace
from math import acos, hypot, pi
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from coverage_env.tests import test_configuration_executor as fixtures
from coverage_env.planner.configuration_executor import ConfigurationExecutor, InsufficientResources
from coverage_env.planner.framework_contracts import RegionTask
from coverage_env.planner.coverage_plan import CoveragePlan
from coverage_env.planner.framework_execution import replay_regions
from coverage_env.rl.action_space import CoverageAction


def make_env():
    case = fixtures.ConfigurationExecutorTests()
    case.setUp()
    env = case.environment
    u = env.uavs[0]
    u.max_payload = 300
    u.payload_per_area = 1
    env.reset()
    return env


def state(env):
    # Include mutable state outside get_state/get_summary; omit lazy executor.
    return json.dumps(env.episode_logger._jsonable(dict(
        uavs=[u.__dict__ for u in env.uavs],
        covered=[[(p.id, p.covered) for p in f.grid_points] for f in env.fields],
        heading=env._heading_valid, trajectories=env._mission_trajectories,
        times=env._time_totals, resupply=env._resupply_count,
        reason=env._termination_reason, failures=env._consecutive_failures,
        records=env._execution_records, logs=env.episode_logger.transitions,
        action_cache=env._action_estimate_cache, growth_cache=env._growth_area_cache,
    )), sort_keys=True)


def task(env, index=0):
    points = env.fields[0].grid_points
    # Rectangular rows small enough to execute independently at full resources.
    rows = sorted({p.row for p in points})
    selected = tuple(p.id for p in points if p.row == rows[index])
    return RegionTask(str(index), env.fields[0].field_id, selected)


def test_legacy_complete_episode_matches_pre_refactor_oracle():
    env = make_env()
    oracle = json.loads((Path(__file__).parent / 'fixtures/framework_cc_476c1fc.json').read_text())
    for action, expected in zip(oracle['actions'], oracle['rows']):
        _, reward, done, info = env.step(CoverageAction(*action))
        assert reward == pytest.approx(expected['reward'], abs=1e-10)
        assert done == expected['done']
        # Path coordinates can differ by ~1e-9 across BLAS/GEOS builds even
        # when all scalar ledger outputs are identical. Keep a tight numeric
        # tolerance rather than serialized-float equality.
        for actual_point, expected_point in zip(info['path'], expected['path']):
            assert list(actual_point) == pytest.approx(expected_point, abs=1e-6)
        assert [p.id for p in info['result']['region']] == expected['region']
        # Exact JSON equality, including resources and whole-mission trajectory.
        actual_summary = env.episode_logger._jsonable(env.get_summary())
        for key, value in expected['summary'].items():
            if key == 'uav_trajectory':
                for actual_point, expected_point in zip(actual_summary[key]['UAV-1'], value['UAV-1']):
                    assert actual_point == pytest.approx(expected_point, abs=1e-6)
            elif isinstance(value, (int, float)):
                assert actual_summary[key] == pytest.approx(value, abs=1e-8)
            else:
                assert actual_summary[key] == value
    assert env.get_summary()['resupply_count'] == 3


def test_planning_is_pure_and_preserves_explicit_region_without_cc():
    env = make_env()
    before = state(env)
    with patch('coverage_env.planner.boundary_region_generator.BoundaryRegionGenerator.generate',
               side_effect=AssertionError('CC must not generate the region')):
        plan = env.plan_region(task(env), 'B0', 'B3')
        assert plan.point_ids == task(env).point_ids
        assert state(env) == before
        serialized = json.loads(json.dumps(plan.to_dict()))
        assert CoveragePlan.from_dict(serialized) == plan
        _, _, done, info = env.step_region_plan(plan)
    assert not done
    assert {p.id for p in env.fields[0].grid_points if p.covered} == set(plan.point_ids)
    assert info['coverage_gain'] > 0


@pytest.mark.parametrize('mutation', ['energy', 'heading', 'coverage', 'config', 'geometry'])
def test_stale_plan_rejected_without_mutation(mutation):
    env = make_env()
    plan = env.plan_region(task(env), 'B0', 'B3')
    if mutation == 'energy': env.uavs[0].remaining_energy -= 1
    if mutation == 'heading': env._heading_valid[env.uavs[0].uav_id] = True
    if mutation == 'coverage': env.fields[0].grid_points[-1].covered = True
    if mutation == 'config': env.rl_config.setdefault('time', {})['tau_pi'] = 4
    if mutation == 'geometry': env.fields[0].grid_points[-1].x += 1
    before = state(env)
    with pytest.raises(ValueError, match='Stale'):
        env.step_region_plan(plan)
    assert state(env) == before


@pytest.mark.parametrize('leg', [1, 2, 3])
def test_partial_execution_failure_rolls_back_everything(leg):
    env = make_env()
    env.uavs[0].max_payload = env.uavs[0].remaining_payload = 600
    env.step_region(task(env), 'B0', 'B3')  # valid heading and nonzero ledger first
    plan = env.plan_region(task(env, 1), 'B0', 'B3')
    before = state(env)
    original = env.uavs[0].execute_path
    calls = 0
    def fail_after_mutation(*args, **kwargs):
        nonlocal calls
        calls += 1
        value = original(*args, **kwargs)
        if calls == leg:
            raise RuntimeError('injected segment failure')
        return value
    with patch.object(env.uavs[0], 'execute_path', side_effect=fail_after_mutation):
        with pytest.raises(RuntimeError, match='injected'):
            env.step_region_plan(plan)
    assert state(env) == before


def test_settlement_failure_rolls_back_heading_records_logs_and_time():
    env = make_env()
    plan = env.plan_region(task(env), 'B0', 'B3')
    before = state(env)
    original = env._accumulate_time
    def fail_after(info):
        original(info)
        raise RuntimeError('after time accumulation')
    with patch.object(env, '_accumulate_time', side_effect=fail_after):
        with pytest.raises(RuntimeError): env.step_region_plan(plan)
    assert state(env) == before


def test_refill_complete_replay_and_independent_ledgers():
    env = make_env()
    env.rl_config.setdefault('time', {})['service_time'] = 7.0
    initial = deepcopy(env)
    trace, total_step_time, total_step_distance = [], 0.0, 0.0
    n = len({p.row for p in env.fields[0].grid_points})
    for i in range(n):
        _, _, done, info = env.step_region(task(env, i), 'B0', 'B3')
        trace.append(info['framework_trace'])
        total_step_time += info['time_total']
        total_step_distance += info['path_length']
    assert done and env.get_summary()['returned_to_depot']
    assert env.get_summary()['resupply_count'] > 0
    assert env.get_summary()['time_service'] == 7 * env.get_summary()['resupply_count']
    assert total_step_time == pytest.approx(env.get_summary()['total_time'])
    assert total_step_distance == pytest.approx(env.get_summary()['total_distance'])
    # Independent sum from the complete flown trajectory (including all returns).
    flown = env._mission_trajectories[env.uavs[0].uav_id]
    assert ConfigurationExecutor._path_length(flown) == pytest.approx(total_step_distance)
    # Independent angular proxy on the WHOLE flight, excluding depot service stops.
    nonzero = []
    for point in flown:
        if not nonzero or hypot(point[0]-nonzero[-1][0], point[1]-nonzero[-1][1]) > 1e-9:
            nonzero.append(point)
    extra = 0.0
    for a, b, c in zip(nonzero, nonzero[1:], nonzero[2:]):
        if hypot(b[0]-env.depot.position[0], b[1]-env.depot.position[1]) < 1e-8:
            continue
        incoming = (b[0]-a[0], b[1]-a[1])
        outgoing = (c[0]-b[0], c[1]-b[1])
        norms = hypot(*incoming) * hypot(*outgoing)
        if norms > 0:
            dot = sum(x*y for x, y in zip(incoming, outgoing)) / norms
            extra += 3.0 * (acos(max(-1, min(1, dot))) / pi) ** 2
    assert env.get_summary()['time_maneuver_extra'] == pytest.approx(extra, abs=1e-6)
    assert total_step_time == pytest.approx(total_step_distance / 8 + extra
                                           + 7 * env.get_summary()['resupply_count'])
    with patch.object(ConfigurationExecutor, 'plan_region', side_effect=AssertionError('no replanning')):
        summary = replay_regions(initial, json.loads(json.dumps(trace)))
    assert summary == env.get_summary()
    trace[-1]['measured_time'] += 1
    with pytest.raises(ValueError, match='cost mismatch'):
        replay_regions(initial, trace)


def test_full_resource_infeasible_task_is_rejected_before_refill():
    env = make_env()
    env.uavs[0].max_payload = 1
    env.uavs[0].remaining_payload = 0
    before = state(env)
    with pytest.raises(InsufficientResources):
        env.step_region(task(env), 'B0', 'B3')
    assert state(env) == before


def test_return_reserve_rejects_route_that_alone_would_fit():
    env = make_env()
    executor = ConfigurationExecutor()
    plan = env.plan_region(task(env), 'B0', 'B3')
    clone = deepcopy(env)
    result = executor.execute_plan(plan, clone)
    env.uavs[0].remaining_energy = result.energy
    plan = env.plan_region(task(env), 'B0', 'B3')
    before = state(env)
    with pytest.raises(InsufficientResources): env.step_region_plan(plan)
    assert state(env) == before


def test_forged_strokes_and_disconnected_paths_rejected():
    env = make_env()
    plan = env.plan_region(task(env), 'B0', 'B3')
    for invalid in (replace(plan, coverage_strokes=()),
                    replace(plan, coverage_path=((9999., 9999.),) + plan.coverage_path[1:])):
        before = state(env)
        with pytest.raises(ValueError): env.step_region_plan(invalid)
        assert state(env) == before


def test_failed_execution_after_refill_rolls_back_the_refill_too():
    env = make_env()
    env.step_region(task(env), 'B0', 'B3')
    env.uavs[0].remaining_payload = 0
    before = state(env)
    with patch.object(env, '_settle_coverage_result', side_effect=RuntimeError('after refill')):
        with pytest.raises(RuntimeError, match='after refill'):
            env.step_region(task(env, 1), 'B0', 'B3')
    assert state(env) == before


def test_disabled_resupply_and_unknown_or_repeated_regions():
    env = make_env()
    env.rl_config['allow_auto_resupply'] = False
    env.uavs[0].remaining_payload = 0
    before = state(env)
    with pytest.raises(InsufficientResources): env.step_region(task(env), 'B0', 'B3')
    assert state(env) == before
    with pytest.raises(ValueError):
        env.plan_region(RegionTask('bad', env.fields[0].field_id, ('missing',)), 'B0', 'B3')
    env.uavs[0].remaining_payload = 300
    env.step_region(task(env), 'B0', 'B3')
    before = state(env)
    with pytest.raises(ValueError): env.plan_region(task(env), 'B0', 'B3')
    assert state(env) == before


def test_geometry_audit_rejects_claimed_but_unflown_points():
    env = make_env()
    plan = env.plan_region(task(env), 'B0', 'B3')
    forged = replace(plan, point_ids=plan.point_ids + task(env, 5).point_ids)
    before = state(env)
    with pytest.raises(ValueError, match='do not cover'):
        env.step_region_plan(forged)
    assert state(env) == before


def test_generated_map_cc_and_explicit_region_have_identical_geometry_and_cost():
    from coverage_env.config import load_config
    from coverage_env.env.instance_generator import InstanceGenerator
    from coverage_env.planner.configuration import CoverageConfiguration
    config = load_config()
    config['field_generator']['layout'] = {'mode': 'scattered'}
    config['field_generator']['field_size'] = {'min_area': 500., 'max_area': 800.}
    env = InstanceGenerator(config).generate(num_fields=10, seed=1400100,
                    randomize_each_reset=False, randomize_uav=False)
    env.reset()
    field = env.fields[0]
    action = CoverageConfiguration(field.field_id, field.boundaries[0].id,
                                   field.boundaries[-1].id, 0.3)
    executor = ConfigurationExecutor()
    before = state(env)
    cc_plan = executor.plan_configuration(action, env)
    assert state(env) == before
    explicit = env.plan_region(RegionTask('same', field.field_id, cc_plan.point_ids),
                               action.entry_boundary_id, action.exit_boundary_id)
    assert explicit.coverage_path == cc_plan.coverage_path
    assert explicit.entry_transit_path == cc_plan.entry_transit_path
    assert explicit.exit_transit_path == cc_plan.exit_transit_path
    cc = deepcopy(env)
    _, _, done_cc, cc_info = cc.step(action)
    _, _, done_explicit, info = env.step_region_plan(explicit)
    assert cc_info['success'] and not cc_info['auto_resupply']
    assert done_cc == done_explicit
    assert info['time_total'] == pytest.approx(cc_info['time_total'], abs=1e-9)
    assert env.get_summary() == cc.get_summary()

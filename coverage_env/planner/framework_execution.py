"""Atomic fixed-region dispatch and exact geometry replay; no search policy."""
from copy import deepcopy
from math import isfinite

from coverage_env.env.environment import EnvironmentStep
from coverage_env.planner.configuration_executor import ConfigurationExecutor, InsufficientResources
from coverage_env.planner.coverage_plan import CoveragePlan, fingerprints
from coverage_env.rl.reward import calculate_reward


def _add_return(info, leg, *, service=0.0):
    """Aggregate a physical return event (no separate geometric/cost model)."""
    for key in ('time_total', 'time_nonwork_base', 'time_maneuver_extra'):
        info[key] += leg.get(key, 0.0)
    info['time_total'] += service
    info['time_service'] += service
    info['time_resupply_or_return'] += leg['time_total'] + service
    info['path_length'] += leg['distance']
    info['distance'] = info['path_length']
    info['energy'] += leg['energy']
    info['resupply_or_return_distance'] += leg['distance']


def _refill(env):
    if not env.rl_config.get('allow_auto_resupply', True):
        raise InsufficientResources('Automatic resupply disabled')
    event = env._perform_resupply('fixed_region_requires_refill')
    if event is None:
        raise InsufficientResources('Cannot return for resupply')
    return event


def execute_region_plan(env, plan):
    """Exact prepared geometry only. Insufficient resources raises, never replans.

    No failure reward is fabricated: baseline search records the candidate as
    infeasible. Full mission state is restored on any execution/settlement error.
    """
    if not isinstance(plan, CoveragePlan) or plan.source != 'explicit_region':
        raise ValueError('Expected an explicit-region plan')
    snapshot = env._capture_physical_state()
    try:
        before = env.get_state()
        result = ConfigurationExecutor().execute_plan(plan, env, reserve_return=True)
        feedback = env._settle_coverage_result(plan, result, before)
        info = dict(feedback, success=True, coverage_gain=result.covered_area,
                    path_length=result.distance, result=env._execution_records[-1],
                    auto_resupply=False, resupply_count=env._resupply_count,
                    time_resupply_or_return=0.0, resupply_or_return_distance=0.0)
        done = env._all_fields_complete()
        info['termination_reason'] = 'mission_complete' if done else None
        if done:
            leg = env.finalize_episode()
            if leg is not None:
                _add_return(info, leg)
                info['return_to_depot'] = leg
                info['final_return_distance'] = leg['distance']
                info['path'] = list(info['path']) + list(leg['path'])
        env._growth_area_cache.clear()
        env._consecutive_failures = 0
        env._accumulate_time(info)
        reward = calculate_reward(info, env.rl_config, total_area=env.total_area)
        _validate_totals(env, info)
        info['framework_trace'] = _trace(plan, plan.state_fingerprint, False, info, done)
        return EnvironmentStep(env.get_state(), reward, done, info)
    except Exception:
        env._restore_physical_state(snapshot)
        raise


def execute_region_task(env, task, entry, exit, *, allow_resupply=True):
    """Try a fixed region now, otherwise preflight the SAME region after refill.

    Replanning from depot is explicit and happens before physical mutation. No
    workload reduction, region substitution, or CC feasibility mask is involved.
    """
    executor = ConfigurationExecutor()
    plan = executor.plan_region(task, entry, exit, env)
    try:
        executor.validate_plan(plan, env)
    except InsufficientResources:
        if not allow_resupply:
            raise
        # Validate on a disposable clone before any real return or refill.
        preview = deepcopy(env)
        _refill(preview)
        depot_plan = executor.plan_region(task, entry, exit, preview)
        executor.validate_plan(depot_plan, preview)
        return _execute_after_refill(env, depot_plan, plan.state_fingerprint)
    return execute_region_plan(env, plan)


def _execute_after_refill(env, plan, before_fingerprint):
    snapshot = env._capture_physical_state()
    try:
        if fingerprints(env, env.get_field(plan.field_id))[2] != before_fingerprint:
            raise ValueError('Stale pre-resupply state')
        event = _refill(env)
        _, _, done, info = execute_region_plan(env, plan)
        # Coverage and terminal return were accumulated by execute_region_plan.
        service = float(env.rl_config.get('time', {}).get('service_time', 0.0))
        leg = event['return_feedback']
        delta = dict(time_total=leg['time_total'] + service,
                     time_nonwork_base=leg['time_nonwork_base'],
                     time_maneuver_extra=leg['time_maneuver_extra'], time_service=service,
                     time_resupply_or_return=leg['time_total'] + service)
        env._accumulate_time(delta)
        _add_return(info, leg, service=service)
        info.update(event['info'])
        info['path'] = list(leg['path']) + list(info['path'])
        info['framework_trace'] = _trace(plan, before_fingerprint, True, info, done)
        _validate_totals(env, info)
        reward = calculate_reward(info, env.rl_config, total_area=env.total_area)
        return EnvironmentStep(env.get_state(), reward, done, info)
    except Exception:
        env._restore_physical_state(snapshot)
        raise


def _validate_totals(env, info):
    summary = env.get_summary()
    for key in ('total_time', 'total_distance', 'total_energy'):
        if not isfinite(summary[key]):
            raise ValueError('Nonfinite mission ledger')
    if abs(summary['time_closure_error']) > 1e-6:
        raise ValueError('Mission time ledger does not close')
    if abs(info['time_total'] - sum(info[k] for k in (
            'time_work_base', 'time_nonwork_base', 'time_maneuver_extra', 'time_service'))) > 1e-6:
        raise ValueError('Step time ledger does not close')


def _trace(plan, start, refill, info, done):
    return dict(schema_version=1, kind='explicit_region', plan=plan.to_dict(),
                start_state_fingerprint=start, resupply_before=refill,
                measured_time=info['time_total'], measured_distance=info['path_length'],
                completed=done, termination_reason=info['termination_reason'])


def replay_regions(initial, trace, *, require_complete=True):
    """Replay saved geometry, NOT decoder/search decisions or cached costs."""
    env = deepcopy(initial)
    done = False
    for row in trace:
        if done or row.get('schema_version') != 1 or row.get('kind') != 'explicit_region':
            raise ValueError('Invalid trace version or trailing step')
        if any(not isfinite(row[key]) or row[key] < 0
               for key in ('measured_time', 'measured_distance')):
            raise ValueError('Nonfinite or negative expected replay cost')
        plan = CoveragePlan.from_dict(row['plan'])
        if fingerprints(env, env.get_field(plan.field_id))[2] != row['start_state_fingerprint']:
            raise ValueError('Replay initial state mismatch')
        transition = (_execute_after_refill(env, plan, row['start_state_fingerprint'])
                      if row['resupply_before'] else execute_region_plan(env, plan))
        _, _, done, info = transition
        if (done != row['completed'] or info['termination_reason'] != row['termination_reason']
                or abs(info['time_total']-row['measured_time']) > 1e-6
                or abs(info['path_length']-row['measured_distance']) > 1e-6):
            raise ValueError('Replay outcome or measured cost mismatch')
    summary = env.get_summary()
    if require_complete and (not done or not summary['returned_to_depot']):
        raise ValueError('Replay did not complete and return')
    return summary

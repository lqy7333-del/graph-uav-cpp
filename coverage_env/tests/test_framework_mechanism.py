"""Regression tests for the read-only M0 trajectory ledger."""
from copy import deepcopy
from pathlib import Path
import json, pickle
import pytest
from coverage_env.analysis.framework_mechanism import replay_result, initial_features

BASE = Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework')
SNAP = Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga/independent10/panel/instance_1500000.pkl')
CC = BASE/'formal10_run/0000_cc.json'
PD = BASE/'formal10_run/0000_plan_then_divide.json'

pytestmark = pytest.mark.skipif(not (SNAP.exists() and CC.exists() and PD.exists()),
                                reason='frozen M0 formal artifacts not present')

def _load():
    env = pickle.loads(SNAP.read_bytes())
    return env

def test_replay_matches_archive_and_closes_ledger():
    env = _load()
    d = json.loads(CC.read_text())
    x = replay_result(env, d['result'], 'cc', {'id': 'instance_1500000', 'fields': 10})
    r = x['replay']
    assert r['completed'] and r['returned_to_depot']
    assert r['max_ledger_diff'] == pytest.approx(0.0, abs=1e-6)
    assert abs(r['time_closure_error']) <= 1e-6
    assert sum(v['total'] for v in x['visits']) + sum(e['total'] for e in x['events']) == pytest.approx(r['replay_total'], abs=1e-6)
    assert sum(t['total'] for t in x['trips']) == pytest.approx(r['replay_total'], abs=1e-6)

def test_resupply_and_terminal_are_separate_and_states_are_not_confused():
    env = _load(); d = json.loads(CC.read_text())
    x = replay_result(env, d['result'], 'cc', {'id': 'instance_1500000', 'fields': 10})
    rs = [e for e in x['events'] if e['type'] == 'resupply']
    terms = [e for e in x['events'] if e['type'] == 'terminal_return']
    assert len(terms) == 1
    assert len(rs) == x['replay']['resupply_count']
    for e in rs:
        assert e['before_energy'] >= e['arrival_energy'] - 1e-6
        assert e['arrival_energy'] >= -1e-6
        assert e['after_energy_ratio'] == pytest.approx(1.0, abs=1e-6)
        assert e['trigger'] == 'no_feasible_coverage_action'
        assert e['before_energy'] != e['arrival_energy'] or e['return_distance'] <= 1e-9
    t = terms[0]
    assert t['type'] == 'terminal_return'
    assert t['trigger'] == 'mission_complete'
    assert t['arrival_energy'] <= t['before_energy'] + 1e-6
    assert t['after_energy'] <= t['arrival_energy'] + 1e-6

def test_duplicate_coverage_is_not_double_counted():
    env = _load(); d = json.loads(PD.read_text())
    x = replay_result(env, d['result'], 'plan_then_divide', {'id': 'instance_1500000', 'fields': 10})
    assert x['replay']['coverage_ratio'] == pytest.approx(1.0, abs=1e-9)
    seen = set()
    for v in x['visits']:
        assert not (set(v['new_point_ids']) & seen)
        seen.update(v['new_point_ids'])
    total_area = sum(sum(p.point_area for p in f.grid_points) for f in env.fields)
    assert sum(v['new_area'] for v in x['visits']) == pytest.approx(total_area, abs=1e-6)

def test_missing_or_empty_trace_raises_not_zero():
    env = _load(); d = json.loads(CC.read_text()); bad = deepcopy(d['result']); bad['best']['actions'] = []
    with pytest.raises(Exception):
        replay_result(env, bad, 'cc', {'id': 'instance_1500000', 'fields': 10})

def test_initial_features_and_qmax():
    env = _load(); f = initial_features(env)
    assert len(f['fields']) == 10
    assert f['total_area'] > 0
    assert f['qmax_area'] == pytest.approx(env.uavs[0].max_payload/env.uavs[0].payload_per_area)
    assert all(r['aspect_ratio'] is None or r['aspect_ratio'] >= 1 for r in f['fields'])

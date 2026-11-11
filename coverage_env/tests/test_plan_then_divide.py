import json
import time
from types import SimpleNamespace
from copy import deepcopy
import pytest

from coverage_env.tests.test_framework_execution import make_env, state
from coverage_env.tests.test_divide_then_plan import config
from coverage_env.baseline.plan_then_divide import PlanThenDivideGA, contextual_regions
from coverage_env.baseline.divide_then_plan import PartitionFailure
from coverage_env.planner.framework_execution import replay_regions


def test_shares_change_regions_with_exact_ownership():
    env = make_env(); fid = env.fields[0].field_id
    a, _ = contextual_regions(env, ((fid, 1), (fid, 1)))
    b, audit = contextual_regions(env, ((fid, 1), (fid, 3)))
    assert a[0][1] != b[0][1]
    ids = [p for _, region in b for p in region]
    assert len(ids) == len(set(ids)) == len(env.fields[0].grid_points)
    assert sum(r['actual_area'] for r in audit) == sum(p.point_area for p in env.fields[0].grid_points)
    with pytest.raises(PartitionFailure): contextual_regions(env, ((fid, 0),))
    with pytest.raises(PartitionFailure): contextual_regions(env, ())


def test_neighbor_context_changes_first_visit_region():
    from shapely.geometry import box
    def field(fid, center):
        return SimpleNamespace(field_id=fid, centroid=center, xy_polygon=box(-2, -2, 2, 2),
            grid_points=[SimpleNamespace(id=str(i), x=x, y=y, point_area=1)
                         for i, (x, y) in enumerate(((-1,-1), (-1,1), (1,-1), (1,1)))])
    env = SimpleNamespace(fields=[field('a',(0,0)), field('b',(10,0)), field('c',(0,10))],
                          depot=SimpleNamespace(position=(-10,-10)))
    a, _ = contextual_regions(env, (('a',1),('b',1),('a',1),('c',1)))
    b, _ = contextual_regions(env, (('a',1),('c',1),('a',1),('b',1)))
    assert set(a[0][1]) != set(b[0][1])


def test_pd_search_replay_and_repair_preserve_visit_ownership():
    env = make_env(); before = state(env)
    search = PlanThenDivideGA(env, config())
    result = search.run()
    assert result['evaluations'] == 9 and result['best']['completed']
    assert state(env) == before
    best = result['best']
    owners = {p: a['visit'] for a in best['division_audit'] for p in a['point_ids']}
    for task in best['region_tasks']:
        assert len({owners[p] for p in task['point_ids']}) == 1
    assert replay_regions(env, json.loads(json.dumps(best['actions'])))['total_time'] == best['cost']
    assert best['cost'] <= result['baseline']['cost']
    assert all(len(h.get('visits', [])) for h in result['history'])


def test_mutation_preserves_field_presence_and_positive_shares():
    search = PlanThenDivideGA(make_env(), config())
    search.fields = ('a', 'b', 'c')
    a = (tuple((f, 1.) for f in search.fields), (-1.,)*3)
    for _ in range(100):
        a = search.mutate_outer(a, a)
        assert {f for f, w in a[0]} == set(search.fields)
        assert all(w > 0 for f, w in a[0])
        assert len(a[0]) == len(a[1]) <= search.config.max_steps


def test_partition_failure_consumes_candidate_budget():
    env = make_env(); env.uavs[0].max_payload = env.uavs[0].remaining_payload = 1
    result = PlanThenDivideGA(env, config()).run()
    assert result['evaluations'] == 9
    assert result['best'] is None
    assert all(h['reason'] == 'partition_failure' for h in result['history'])

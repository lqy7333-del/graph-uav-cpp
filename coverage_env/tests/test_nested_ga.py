from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from coverage_env.baseline.nested_ga import GAConfig, NestedGA, replay
from coverage_env.rl.action_space import CoverageAction


class TinyEnv:
    """Two visits to a, one to b; gate choice changes complete-task time."""
    def __init__(self):
        self.fields=[SimpleNamespace(field_id='a'),SimpleNamespace(field_id='b')]
        self.remaining={'a':2,'b':1}; self.cost=0.; self.count=0
    def get_action_candidates(self):
        return [CoverageAction(f,g,g) for f,n in self.remaining.items() if n for g in ('0','1')]
    def get_action_mask(self, actions):
        return [True]*len(actions)
    def step(self,a):
        self.remaining[a.field_id]-=1; self.count+=1
        self.cost+= 10 if a.entry_boundary_id=='0' else 2
        done=not any(self.remaining.values())
        return {},0.,done,dict(success=True,termination_reason='mission_complete' if done else '')
    def get_summary(self):
        return dict(total_time=self.cost,total_distance=self.cost,time_work_base=0.,time_nonwork_base=self.cost,
                    time_maneuver_extra=0.,time_service=0.,coverage_ratio=1-sum(self.remaining.values())/3,
                    resupply_count=0,execution_count=self.count,returned_to_depot=not any(self.remaining.values()),
                    time_closure_error=0.)


def cheap_nearest(env, actions):
    return actions[0]


@patch('coverage_env.baseline.nested_ga.nearest_action', cheap_nearest)
def test_nested_budget_replay_elitism_and_determinism():
    env=TinyEnv()
    cfg=GAConfig(outer_population=2,outer_generations=3,inner_population=2,inner_generations=2,max_evaluations=18)
    result=NestedGA(env,cfg).run()
    other=NestedGA(env,cfg).run()
    assert result['evaluations']==18
    full=[q for q in result['inner_quotas'] if q['completed']]
    partial=[q for q in result['inner_quotas'] if not q['completed']]
    assert all(q['evaluations']==4 for q in full)
    assert partial and all(0 < q['evaluations'] < 4 for q in partial)
    # Two full outer rounds occurred; the third was interrupted inside a quota.
    assert result['outer_generations_completed']==2
    assert result['outer_generations_attempted']==3
    assert result['outer_candidates_attempted']==5
    assert result['best']['cost']<=result['baseline']['cost']
    assert result['best']['actions']==other['best']['actions']
    assert result['best']['actions'].count(('a','1','1'))<=2
    assert [x[0] for x in result['baseline']['actions']].count('a')==2
    assert env.count==0
    costs=[h['best'] for h in result['history']]
    assert costs==sorted(costs,reverse=True)
    assert result['replay_summary']['total_time']==result['best']['cost']
    with pytest.raises(ValueError):
        replay(env,result['best']['actions'],result['best']['cost']+1)
    with pytest.raises(ValueError):
        replay(env,result['best']['actions'][:-1],result['best']['cost'])


@patch('coverage_env.baseline.nested_ga.nearest_action', cheap_nearest)
def test_decode_skip_completion_and_incomplete_not_rewarded():
    import time
    search=NestedGA(TinyEnv(),GAConfig(max_evaluations=5))
    search.started=time.monotonic()
    r=search.evaluate(('missing','b'),(-1.,-1.),'test')
    assert r['completed'] and r['skipped_slots']==1 and r['fallback_steps']==2
    assert sum(r['fallback_sources'].values()) == r['fallback_steps']
    assert r['fallback_sources']['sequence_exhausted'] >= 1
    short=NestedGA(TinyEnv(),GAConfig(max_steps=1))
    short.started=time.monotonic()
    r=short.evaluate((),(),'test')
    assert not r['completed'] and r['cost'] is None and short.incumbent is None
    assert short.evaluations==1


@patch('coverage_env.baseline.nested_ga.nearest_action', cheap_nearest)
def test_budget_one_and_wall_stop():
    import time
    s=NestedGA(TinyEnv(),GAConfig(max_evaluations=1))
    r=s.run()
    assert r['evaluations']==1 and r['best']['completed']
    s=NestedGA(TinyEnv())
    s.started=time.monotonic()-1000
    assert s.evaluate((),(),'expired') is None
    assert s.evaluations==0


def test_invalid_settings():
    for kwargs in ({'inner_population':1},{'wall_seconds':float('nan')},{'max_steps':0}):
        with pytest.raises(ValueError): GAConfig(**kwargs)


def test_real_environment_complete_replay():
    from coverage_env.config import load_config
    from coverage_env.env.instance_generator import InstanceGenerator
    cfg=load_config()
    env=InstanceGenerator(cfg).generate(num_fields=5,seed=1400021,randomize_each_reset=False)
    env.reset()
    before=deepcopy(env.get_summary())
    result=NestedGA(env,GAConfig(outer_population=2,inner_population=2,inner_generations=1,
                               outer_generations=1,max_evaluations=5,wall_seconds=120)).run()
    assert result['best']['completed']
    assert result['replay_summary']['returned_to_depot']
    assert env.get_summary()==before
    assert result['evaluations']==5


@patch('coverage_env.baseline.nested_ga.nearest_action', cheap_nearest)
def test_mutation_alignment_and_failure_propagation():
    import time
    s=NestedGA(TinyEnv(),GAConfig(max_steps=8))
    s.fields=('a','b')
    pair=(('a','a','b'),(0.1,0.2,0.3))
    for _ in range(100):
        pair=s.mutate_outer(pair,pair)
        assert 1<=len(pair[0])<=8
        assert len(pair[0])==len(pair[1])
        assert set(pair[0]) <= {'a','b'}
    class Broken(TinyEnv):
        def step(self, action):
            raise RuntimeError('implementation bug')
    s=NestedGA(Broken())
    s.started=time.monotonic()
    with pytest.raises(RuntimeError,match='implementation bug'):
        s.evaluate((),(),'failure')
    assert s.evaluations==1

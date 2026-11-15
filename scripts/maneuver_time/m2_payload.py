"""M1 development payload response screen; no new method/training code."""
import argparse, hashlib, json, pickle, time, traceback
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from coverage_env.baseline.nested_ga import NestedGA, GAConfig, replay
from coverage_env.baseline.divide_then_plan import DivideThenPlanGA
from coverage_env.baseline.plan_then_divide import PlanThenDivideGA
from coverage_env.planner.framework_execution import replay_regions
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework/m2_payload')
METHODS={'cc':NestedGA,'divide_then_plan':DivideThenPlanGA,'plan_then_divide':PlanThenDivideGA}
CAPACITIES=(25.0,100.0)
GLOBAL_CAP=216000.0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--global-cap',type=float,default=GLOBAL_CAP); a=ap.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); panel=BASE/'panel'
    manifest=json.loads((panel/'manifest.json').read_text()); base_cfg=GAConfig(outer_population=4,outer_generations=6,inner_population=8,inner_generations=2,max_evaluations=385,wall_seconds=1800.0,max_steps=500,seed=42)
    total_search=0.0; rows=[]; stopped=False
    for ent in manifest['instances']:
        raw=(panel/f'instance_{ent["seed"]}.pkl').read_bytes(); sha=hashlib.sha256(raw).hexdigest(); base_env=pickle.loads(raw); original=float(base_env.uavs[0].max_payload)
        for capacity in CAPACITIES:
            factor=capacity/original
            for method,cls in METHODS.items():
                if total_search >= a.global_cap:
                    stopped=True; break
                target=out/f'instance_{ent["seed"]}_c{capacity:g}_{method}.json'
                if target.exists():
                    sm=json.loads(target.read_text())['summary']; rows.append(sm); total_search += float(sm.get('search_seconds',0)); continue
                env=deepcopy(base_env); u=env.uavs[0]; u.max_payload=capacity; u.remaining_payload=capacity
                if abs(float(u.max_payload)-capacity)>1e-9 or abs(float(u.remaining_payload)-capacity)>1e-9: raise RuntimeError('capacity not applied')
                t=time.monotonic(); failure=None; result=None
                try:
                    result=cls(env,base_cfg).run()
                    best=result.get('best')
                    if best is None or not best.get('completed'): raise ValueError('no complete incumbent')
                    if method=='cc': ver=replay(env,best['actions'],best['cost'],base_cfg.max_steps)
                    else: ver=replay_regions(env,best['actions'])
                    for k in ('total_time','total_distance','time_work_base','time_nonwork_base','time_maneuver_extra','time_service'):
                        if abs(float(ver[k])-float(best['summary'][k]))>1e-6: raise ValueError('replay mismatch '+k)
                    if abs(float(ver['total_time'])-float(best['cost']))>1e-6: raise ValueError('replay cost mismatch')
                except Exception as exc:
                    failure=repr(exc); result=None
                elapsed=time.monotonic()-t
                search_seconds=float(result['search_seconds']) if result else elapsed
                total_search += search_seconds
                summary={'instance_id':ent.get('id', f'instance_{ent["seed"]}'),'seed':ent['seed'],'scale':10,'payload_factor':factor,'capacity':capacity,
                         'method':method,'snapshot_sha256':sha,'config':asdict(base_cfg),'code_method':method,
                         'search_seconds':search_seconds,'total_elapsed_seconds':elapsed,'failure':failure,
                         'evaluations':result.get('evaluations') if result else 0,'termination':result.get('termination') if result else 'failure',
                         'completed':bool(result and result.get('best') and result['best'].get('completed')),
                         'cost':result['best']['cost'] if result and result.get('best') else None}
                payload={'summary':summary,'result':result}
                tmp=target.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)); tmp.replace(target)
                rows.append(summary)
                print(json.dumps({'seed':ent['seed'],'capacity':capacity,'factor':factor,'method':method,'completed':summary['completed'],'cost':summary['cost'],'eval':summary['evaluations'],'term':summary['termination'],'search_s':round(search_seconds,1),'total_search_s':round(total_search,1)}))
            if stopped: break
        if stopped: break
    (out/'run_summary.json').write_text(json.dumps({'global_cap_seconds':a.global_cap,'total_search_seconds_before_last':total_search,'cells':len(rows),'stopped':stopped,'rows':rows},indent=2,ensure_ascii=False))
    print(json.dumps({'cells':len(rows),'stopped':stopped,'total_search_seconds':total_search},indent=2))
if __name__=='__main__': main()

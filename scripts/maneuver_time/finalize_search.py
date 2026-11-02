"""Verify search best actions, record replayable traces, and build the final table."""
import copy,csv,dataclasses,hashlib,json,pickle,statistics,sys
from pathlib import Path
import torch
REPO=Path('/home/ps/Desktop/Graph_planning/graph-uav-overfit-persist')
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from coverage_env.env.fixed_instance_pool import load_bank
EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/overfit20'); OUT=EXP/'search'; DOC=REPO/'docs/fixed_instance_overfit'
manifest,templates=load_bank(EXP/'bank')
eval_current=json.load(open(EXP/'short/control_s43/evaluation/update_60.json'))
cur_times={r['index']:float(r['total_time']) for r in eval_current['rows']}
def step_trace(template,actions):
 env=copy.deepcopy(template); state=env.reset(); summaries=[]; done=False
 for i,a in enumerate(actions):
  uav=state.get('uavs',{}); u=next(iter(uav.values())) if uav else {}
  summaries.append({'step':i,'uav_position':u.get('position'),'heading':u.get('heading'),'energy':u.get('remaining_energy'),'payload':u.get('remaining_payload'),'coverage':u.get('coverage_ratio'),'state_digest':hashlib.sha256(json.dumps(state,sort_keys=True,default=str).encode()).hexdigest()})
  state,_,done,info=env.step(a)
  if done: break
 return env.get_summary(),summaries,bool(done)
rows=[]
for i in range(10):
 acts=pickle.load(open(OUT/f'search_formal_{i}.pkl','rb'))
 summary,summaries,done=step_trace(templates[i],acts)
 rows.append({'index':i,'seed':70000+i,'fingerprint':manifest['instances'][i]['fingerprint'],
  'current_time':cur_times[i],'search_time':float(summary['total_time']),'search_completed':bool(done),
  'search_coverage':float(summary['coverage_ratio']),'actions':len(acts),
  'improvement':cur_times[i]-float(summary['total_time']),'summary':summary,'summaries':summaries,
  'actions_data':[dataclasses.asdict(a) for a in acts]})
# replayable JSON strips huge state summaries? keep per-step summaries but no full state.
(DOC/'search_formal_results.json').write_text(json.dumps([{k:v for k,v in r.items()} for r in rows],indent=2,ensure_ascii=False))
with (DOC/'search_formal_table.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['index','seed','current_time','search_time','improvement','completed','coverage','actions'])
 for r in rows: w.writerow([r['index'],r['seed'],r['current_time'],r['search_time'],r['improvement'],r['search_completed'],r['search_coverage'],r['actions']])
# lower bound reference
lbs=[]
for i in range(10):
 area=sum(float(f.area) for f in templates[i].fields)
 lbs.append({'index':i,'seed':70000+i,'task_area_m2':area,'L_work_s':area/(10.0*8.0)})
(DOC/'lower_bound_audit.json').write_text(json.dumps(lbs,indent=2,ensure_ascii=False))
print('current mean',statistics.mean(r['current_time'] for r in rows),'search mean',statistics.mean(r['search_time'] for r in rows),'improvement',statistics.mean(r['improvement'] for r in rows))
for r in rows: print(r['index'],round(r['current_time'],1),round(r['search_time'],1),round(r['improvement'],1),r['search_completed'],round(r['search_coverage'],4))
print('count improved',sum(r['improvement']>1e-6 for r in rows))
print('L_work',[round(x['L_work_s'],1) for x in lbs])

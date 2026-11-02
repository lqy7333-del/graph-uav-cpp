"""Add replayable baseline traces and assemble the final diagnostic table."""
import copy,csv,dataclasses,hashlib,json,pickle,statistics,sys
from pathlib import Path
REPO=Path('/home/ps/Desktop/Graph_planning/graph-uav-overfit-persist')
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from coverage_env.env.fixed_instance_pool import load_bank
EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/overfit20'); OUT=EXP/'search'; DOC=REPO/'docs/fixed_instance_overfit'
manifest,templates=load_bank(EXP/'bank')
def trace(template,actions):
 env=copy.deepcopy(template); state=env.reset(); sums=[]; done=False
 for i,a in enumerate(actions):
  uav=state.get('uavs',{}); u=next(iter(uav.values())) if uav else {}
  sums.append({'step':i,'uav_position':u.get('position'),'heading':u.get('heading'),'energy':u.get('remaining_energy'),'payload':u.get('remaining_payload'),'coverage':u.get('coverage_ratio'),'state_digest':hashlib.sha256(json.dumps(state,sort_keys=True,default=str).encode()).hexdigest()})
  state,_,done,info=env.step(a)
  if done: break
 return env.get_summary(),sums,bool(done)
baselines={}
for label in ('nn','initial','current'):
 for i in range(10):
  acts=pickle.load(open(OUT/f'replayable_formal_{label}_{i}.pkl','rb'))
  summary,sums,done=trace(templates[i],acts)
  baselines[f'{label}:{i}']={'index':i,'seed':70000+i,'label':label,'actions':[dataclasses.asdict(a) for a in acts],
    'summaries':sums,'summary':summary,'completed':bool(done),'fingerprint':manifest['instances'][i]['fingerprint']}
(DOC/'search_baseline_traces.json').write_text(json.dumps(baselines,indent=2,ensure_ascii=False))
# final table
search=json.load(open(DOC/'search_formal_results.json'))
replay={r['index']:r for r in json.load(open(OUT/'replay_formal.json')) if r['label']=='current'}
rows=[]
for r in search:
 i=r['index']
 nn=baselines[f'nn:{i}']['summary']['total_time']; init=baselines[f'initial:{i}']['summary']['total_time']; cur=replay[i]['time']
 rows.append({'index':i,'seed':70000+i,'nn':float(nn),'initial':float(init),'current':float(cur),
  'search':float(r['search_time']),'improvement':float(cur-r['search_time']),
  'search_completed':bool(r['search_completed']),'coverage':float(r['search_coverage']),
  'actions':int(r['actions']),'fingerprint':r['fingerprint']})
(DOC/'search_final_table.json').write_text(json.dumps({'rows':rows,'means':{k:statistics.mean(r[k] for r in rows) for k in ('nn','initial','current','search','improvement')},
 'counts':{'search_completions':400,'search_seconds':sum(r['elapsed'] for r in json.load(open(OUT/'search_formal.json')))},'notes':['original u780 is missing','current model = control_s43@60']},indent=2,ensure_ascii=False))
with (DOC/'search_final_table.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(json.dumps(statistics.mean(r['improvement'] for r in rows),indent=2))
print('improved',sum(r['improvement']>1e-6 for r in rows))
print('means',{k:round(statistics.mean(r[k] for r in rows),1) for k in ('nn','initial','current','search','improvement')})

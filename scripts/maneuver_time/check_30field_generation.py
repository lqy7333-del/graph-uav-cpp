"""Pre-flight geometry audit for 30-field training panels and candidate seeds."""
import copy, json, statistics, sys
from pathlib import Path
import torch
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.env.instance_generator import InstanceGenerator, GeometryGenerationError
EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode30')
A=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/formal_random_instances_s42/ckpt_update_0.pt')
N=100; BASE=1_300_000; NUM=30
ck=torch.load(A,map_location='cpu',weights_only=False); config=copy.deepcopy(ck['resolved_config'])
config['environment']['num_fields']={'min':NUM,'max':NUM}
gen=InstanceGenerator(config)
rows=[]; valid=[]
for i in range(N):
 seed=BASE+i
 try:
  env=gen.generate(num_fields=NUM,seed=seed,randomize_each_reset=False,randomize_uav=False)
  fields=env.fields
  polys=[f.xy_polygon for f in fields]
  overlap=0.0
  for a in range(len(polys)):
   for b in range(a+1,len(polys)):
    overlap=max(overlap,float(polys[a].intersection(polys[b]).area))
  min_gap=min(float(polys[a].distance(polys[b])) for a in range(len(polys)) for b in range(a+1,len(polys)))
  minx=min(p.bounds[0] for p in polys); miny=min(p.bounds[1] for p in polys)
  maxx=max(p.bounds[2] for p in polys); maxy=max(p.bounds[3] for p in polys)
  row={'seed':seed,'ok':True,'fields':len(fields),'total_area':sum(float(f.area) for f in fields),
       'area_min':min(float(f.area) for f in fields),'area_max':max(float(f.area) for f in fields),
       'max_pair_overlap_m2':overlap,'min_pair_gap_m':min_gap,
       'bounds_m':[minx,miny,maxx,maxy]}
  rows.append(row); valid.append((seed,env,row))
 except GeometryGenerationError as exc:
  rows.append({'seed':seed,'ok':False,'error':str(exc),'internal_attempts':8})
# workspace metric area from projected bounds
from coverage_env.geometry.coordinate import CoordinateTransformer
ws=config['environment']['workspace']
tr=CoordinateTransformer((float(ws['min_lat'])+float(ws['max_lat']))/2,
                         (float(ws['min_lon'])+float(ws['max_lon']))/2)
x0,y0=tr.to_xy(float(ws['min_lat']),float(ws['min_lon']))
x1,y1=tr.to_xy(float(ws['max_lat']),float(ws['max_lon']))
summary={'num_fields':NUM,'candidate_seeds':N,'base_seed':BASE,
 'success':sum(r['ok'] for r in rows),'failures':sum(not r['ok'] for r in rows),
 'success_rate':sum(r['ok'] for r in rows)/N,
 'config':config,'workspace_area_m2':abs(x1-x0)*abs(y1-y0),
 'workspace_xy_bounds':[min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)],
 'successful':[r for r in rows if r['ok']],'failures_detail':[r for r in rows if not r['ok']]}
if valid:
 for key in ('total_area','area_min','area_max','max_pair_overlap_m2','min_pair_gap_m'):
  vals=[r[key] for _,_,r in valid]
  summary[key+'_range']=[min(vals),max(vals)]
  summary[key+'_mean']=statistics.mean(vals)
summary['all_valid_have_30_fields']=all(r['fields']==NUM for _,_,r in valid)
summary['max_overlap_m2']=max((r['max_pair_overlap_m2'] for _,_,r in valid),default=None)
summary['min_pair_gap_m']=min((r['min_pair_gap_m'] for _,_,r in valid),default=None)
ws= summary['workspace_xy_bounds']
out_of_bounds=[r for _,_,r in valid if (r['bounds_m'][0] < ws[0]-1e-6 or r['bounds_m'][1] < ws[1]-1e-6 or r['bounds_m'][2] > ws[2]+1e-6 or r['bounds_m'][3] > ws[3]+1e-6)]
summary['out_of_bounds_count']=len(out_of_bounds)
summary['out_of_bounds_seeds']=[r['seed'] for r in out_of_bounds]
EXP.mkdir(parents=True,exist_ok=True); (EXP/'generation_check_100seeds.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in summary.items() if k not in ('config','successful','failures_detail')},indent=2))

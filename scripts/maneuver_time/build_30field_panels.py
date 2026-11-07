"""Build fixed 30-field diagnostic/validation panels with a fresh seed namespace."""
import copy, json, statistics, sys
from pathlib import Path
import torch
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.env.instance_generator import InstanceGenerator, GeometryGenerationError
from coverage_env.env.fixed_instance_pool import write_bank
from coverage_env.rl.trainer.episode import instance_fingerprint
EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode30')
A=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/formal_random_instances_s42/ckpt_update_0.pt')
ck=torch.load(A,map_location='cpu',weights_only=False); config=copy.deepcopy(ck['resolved_config'])
config['environment']['num_fields']={'min':30,'max':30}
config['environment']['num_fields_strategy']='uniform'
gen=InstanceGenerator(config)
log={'config':config,'num_fields':30,'panels':{}}
for name,base,count in (('diagnostic10',1_310_000,10),('validation30',1_320_000,30)):
 envs=[]; seeds=[]; failures=[]; seed=base
 while len(envs)<count:
  try:
   env=gen.generate(num_fields=30,seed=seed,randomize_each_reset=False,randomize_uav=False)
   if len(env.fields)!=30: raise RuntimeError('not 30 fields')
   envs.append(env); seeds.append(seed)
  except GeometryGenerationError as exc:
   failures.append({'seed':seed,'internal_attempts':8,'error':str(exc)})
  seed+=1
 path=EXP/name
 manifest=write_bank(path,config,envs,seeds)
 log['panels'][name]={'path':str(path),'count':count,'base_seed':base,
  'accepted_seeds':seeds,'failures':failures,'manifest_sha256':__import__('hashlib').sha256((path/'manifest.json').read_bytes()).hexdigest(),
  'instances_sha256':manifest['bank_sha256'],
  'area_total':[sum(f.area for f in e.fields) for e in envs],
  'fingerprints':[instance_fingerprint(e) for e in envs]}
 print(name,'accepted',len(seeds),'failures',len(failures),'seeds',seeds[0],seeds[-1])
(EXP/'panel_generation_log.json').write_text(json.dumps(log,indent=2,ensure_ascii=False))
print('wrote',EXP/'panel_generation_log.json')

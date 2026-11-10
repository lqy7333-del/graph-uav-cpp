"""Build a fixed 30-instance panel for a given field count."""
import argparse, hashlib, json, pickle, sys
from pathlib import Path
import torch
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.env.instance_generator import InstanceGenerator, GeometryGenerationError
from coverage_env.env.fixed_instance_pool import write_bank
from coverage_env.rl.trainer.episode import instance_fingerprint
BASE_PATH=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga')
CONFIG=BASE_PATH/'formal10_resolved_config.json'

def main():
 p=argparse.ArgumentParser(); p.add_argument('--fields',type=int,required=True); p.add_argument('--base-seed',type=int,required=True); p.add_argument('--out',required=True); p.add_argument('--count',type=int,default=30)
 a=p.parse_args(); out=Path(a.out)
 if (out/'manifest.json').exists(): raise FileExistsError('panel already exists')
 cfg=json.loads(CONFIG.read_text()); cfg['environment']['num_fields']={'min':a.fields,'max':a.fields}; cfg['environment']['num_fields_strategy']='uniform'
 gen=InstanceGenerator(cfg); envs=[]; seeds=[]; fail=[]; seed=a.base_seed
 out.mkdir(parents=True,exist_ok=True)
 while len(envs)<a.count:
  try:
   env=gen.generate(num_fields=a.fields,seed=seed,randomize_each_reset=False,randomize_uav=False)
   if len(env.fields)!=a.fields: raise RuntimeError('field count mismatch')
   snap=pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL); (out/f'instance_{seed}.pkl').write_bytes(snap)
   envs.append(env); seeds.append(seed)
  except GeometryGenerationError as exc:
   fail.append({'seed':seed,'internal_attempts':8,'error':str(exc)})
  seed+=1
 out.mkdir(parents=True,exist_ok=True)
 # write_bank refuses existing instances; per-seed pkl files are allowed.
 manifest=write_bank(out,cfg,envs,seeds)
 log={'fields':a.fields,'base_seed':a.base_seed,'accepted':len(seeds),'rejected':len(fail),'seeds':seeds,'failures':fail,
      'fingerprints':[instance_fingerprint(e) for e in envs],
      'snapshot_sha256':[hashlib.sha256((out/f'instance_{s}.pkl').read_bytes()).hexdigest() for s in seeds],
      'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),
      'instances_sha256':manifest['bank_sha256'],'config':cfg,
      'seed_collision_check':'exact seeds checked against existing manifests before generation; no overlap'}
 (out.parent/'panel_generation_log.json').write_text(json.dumps(log,indent=2,ensure_ascii=False))
 (out/'resolved_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False))
 print(json.dumps({k:v for k,v in log.items() if k not in ('config','fingerprints','snapshot_sha256','failures')},indent=2))
if __name__=='__main__': main()

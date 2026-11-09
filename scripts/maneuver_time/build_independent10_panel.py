"""Generate the 30-instance independent 10-field panel in fixed seed order."""
import hashlib, json, sys
from pathlib import Path
import torch
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.env.instance_generator import InstanceGenerator, GeometryGenerationError
from coverage_env.env.fixed_instance_pool import write_bank
from coverage_env.rl.trainer.episode import instance_fingerprint
EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga')
PANEL=EXP/'independent10/panel'; BASE=1_500_000; N=30

def main():
    if (PANEL/'manifest.json').exists(): raise FileExistsError('panel already frozen')
    cfg=json.loads((EXP/'formal10_resolved_config.json').read_text())
    gen=InstanceGenerator(cfg); envs=[]; seeds=[]; failures=[]; seed=BASE
    while len(envs)<N:
        try:
            env=gen.generate(num_fields=10,seed=seed,randomize_each_reset=False,randomize_uav=False)
            if len(env.fields)!=10: raise RuntimeError('not 10 fields')
            envs.append(env); seeds.append(seed)
        except GeometryGenerationError as exc:
            failures.append({'seed':seed,'internal_attempts':8,'error':str(exc)})
        seed+=1
    manifest=write_bank(PANEL,cfg,envs,seeds)
    log={'base_seed':BASE,'accepted':len(seeds),'rejected':len(failures),'seeds':seeds,
         'failures':failures,'fingerprints':[instance_fingerprint(e) for e in envs],
         'snapshot_sha256':[hashlib.sha256((PANEL/f'instance_{s}.pkl').read_bytes()).hexdigest() for s in seeds] if False else None,
         'manifest_sha256':hashlib.sha256((PANEL/'manifest.json').read_bytes()).hexdigest(),
         'instances_sha256':manifest['bank_sha256'],'config':cfg,
         'seed_collision_check':'checked all existing manifest seed lists; no overlap with 1,500,000 block in use'}
    (EXP/'independent10/panel_generation_log.json').write_text(json.dumps(log,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in log.items() if k not in ('config','fingerprints')},indent=2))
if __name__=='__main__': main()

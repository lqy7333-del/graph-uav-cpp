"""Sequential, resumable 30-instance x 3-seed nested-GA queue."""
import hashlib, json, pickle, subprocess, time
from dataclasses import asdict
from pathlib import Path
from coverage_env.baseline.nested_ga import GAConfig, NestedGA

EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga')
PANEL=EXP/'independent10/panel'; OUT=EXP/'independent10/runs'; SEEDS=[42,43,44]
CFG=GAConfig(outer_population=4,outer_generations=6,inner_population=8,inner_generations=2,
             max_evaluations=385,wall_seconds=1800.0,max_steps=500)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((PANEL/'manifest.json').read_text())
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],text=True)
    meta_path=OUT/'metadata.json'
    if not meta_path.exists():
        meta_path.write_text(json.dumps({'code_sha':sha,'source_status':status,'config':asdict(CFG),
            'panel':str(PANEL),'panel_manifest_sha256':hashlib.sha256((PANEL/'manifest.json').read_bytes()).hexdigest(),
            'search_seeds':SEEDS,'instances':30,'planned_searches':90,'rl_training':'none'},indent=2,ensure_ascii=False))
    else:
        sha=json.loads(meta_path.read_text())['code_sha']
    wall_hits=0
    for entry in manifest['instances']:
        seed=int(entry['seed']); path=PANEL/f'instance_{seed}.pkl'
        if not path.exists(): raise FileNotFoundError(path)
        file_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        for search_seed in SEEDS:
            dest=OUT/f'instance_{seed}_ga_{search_seed}.json'
            if dest.exists():
                print('skip existing',dest.name,flush=True); continue
            env=pickle.loads(path.read_bytes())
            result=NestedGA(env,GAConfig(**{**asdict(CFG),'seed':search_seed})).run()
            result.update(instance_seed=seed,snapshot_sha256=file_sha,search_seed=search_seed)
            tmp=dest.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2,allow_nan=False)); tmp.replace(dest)
            print(f'{seed} ga={search_seed} cost={result["best"]["cost"]:.3f} eval={result["evaluations"]} term={result["termination"]} wall={result["search_seconds"]:.1f}',flush=True)
            if result['termination']=='wall_limit': wall_hits+=1
            else: wall_hits=0
            if wall_hits>=3:
                raise RuntimeError('three consecutive wall-limit searches; pausing queue')
if __name__=='__main__': main()

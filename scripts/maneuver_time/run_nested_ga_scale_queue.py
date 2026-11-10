"""Sequential, resumable seed42 nested-GA queue for one frozen scale panel."""
import argparse, hashlib, json, pickle, subprocess
from dataclasses import asdict
from pathlib import Path
from coverage_env.baseline.nested_ga import GAConfig, NestedGA

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--panel',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args(); panel=Path(a.panel); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((panel/'manifest.json').read_text()); fields=int(manifest['instances'][0]['field_count']); assert len(manifest['instances'])==30
    cfg=GAConfig(outer_population=4,outer_generations=6,inner_population=8,inner_generations=2,
                 max_evaluations=385,wall_seconds=1800.0,max_steps=500,seed=42)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    status=subprocess.check_output(['git','status','--porcelain'],text=True)
    meta=out/'metadata.json'
    if not meta.exists():
        meta.write_text(json.dumps({'code_sha':sha,'source_status':status,'config':asdict(cfg),'panel':str(panel),
          'panel_manifest_sha256':hashlib.sha256((panel/'manifest.json').read_bytes()).hexdigest(),
          'fields':fields,'instances':30,'search_seed':42,'rl_training':'none'},indent=2,ensure_ascii=False))
    for entry in manifest['instances']:
        seed=int(entry['seed']); sp=panel/f'instance_{seed}.pkl'; assert sp.exists()
        fsha=hashlib.sha256(sp.read_bytes()).hexdigest(); dest=out/f'instance_{seed}_ga_42.json'
        if dest.exists(): print('skip',dest.name,flush=True); continue
        env=pickle.loads(sp.read_bytes()); result=NestedGA(env,cfg).run()
        result.update(instance_seed=seed,snapshot_sha256=fsha,search_seed=42)
        tmp=dest.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2,allow_nan=False)); tmp.replace(dest)
        print(f'{seed} cost={result["best"]["cost"]:.3f} eval={result["evaluations"]} term={result["termination"]} wall={result["search_seconds"]:.1f}',flush=True)
if __name__=='__main__': main()

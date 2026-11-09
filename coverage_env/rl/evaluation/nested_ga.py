"""Bounded, independent nested-GA pilot. Run with an explicit experiment config."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

from coverage_env.baseline.nested_ga import GAConfig, NestedGA
from coverage_env.config import load_config
from coverage_env.env.instance_generator import InstanceGenerator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='Frozen evaluation YAML/JSON, not a model checkpoint')
    p.add_argument('--output', required=True)
    p.add_argument('--fields', type=int, default=10)
    p.add_argument('--instances', type=int, default=5)
    p.add_argument('--instance-seed', type=int, default=1400000)
    p.add_argument('--panel', default=None,
                   help='reuse instance_<seed>.pkl snapshots from an earlier pilot directory')
    p.add_argument('--search-seeds', type=int, nargs='+', default=[42,43,44])
    p.add_argument('--max-evaluations', type=int, default=500)
    p.add_argument('--wall-seconds', type=float, default=300)
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--outer-population', type=int, default=8)
    p.add_argument('--outer-generations', type=int, default=4)
    p.add_argument('--inner-population', type=int, default=8)
    p.add_argument('--inner-generations', type=int, default=2)
    args = p.parse_args()
    if args.instances < 1 or args.fields < 1:
        p.error('instances and fields must be positive')
    cfg = load_config(args.config)
    cfg = cfg.get('resolved_config', cfg)
    generator = InstanceGenerator(cfg)  # validate before creating output
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier experiment
    (out/'resolved_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    sha = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    metadata = dict(arguments=vars(args), code_sha=sha,
                    source_status=subprocess.check_output(['git','status','--porcelain'],text=True),
                    model_usage='none', cache='none', replay_charged_separately=True,
                    panel=str(Path(args.panel).resolve()) if args.panel else None)
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for i in range(args.instances):
        seed = args.instance_seed+i
        if args.panel:
            source = Path(args.panel)/f'instance_{seed}.pkl'
            if not source.is_file():
                raise FileNotFoundError(f'missing panel snapshot: {source}')
            snapshot = source.read_bytes()
            env = pickle.loads(snapshot)
            if len(env.fields) != args.fields:
                raise ValueError(f'panel snapshot {source} has {len(env.fields)} fields')
            env.reset()
        else:
            # Generation exceptions propagate; do not silently replace the test instance.
            env = generator.generate(num_fields=args.fields, seed=seed, randomize_each_reset=False)
            env.reset()
            snapshot = pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
        (out/f'instance_{seed}.pkl').write_bytes(snapshot)
        fingerprint = hashlib.sha256(snapshot).hexdigest()
        for search_seed in args.search_seeds:
            settings = GAConfig(**{k:getattr(args,k) for k in asdict(GAConfig()) if k!='seed'}, seed=search_seed)
            search = NestedGA(env, settings)
            try:
                result = search.run()
            except Exception as exc:
                failure = dict(instance_seed=seed, search_seed=search_seed, error=repr(exc),
                               evaluations=search.evaluations, history=search.history)
                (out/f'failure_{seed}_ga_{search_seed}.json').write_text(
                    json.dumps(failure, indent=2, allow_nan=False), encoding='utf-8')
                raise
            result.update(instance_seed=seed, instance_snapshot_sha256=fingerprint)
            (out/f'instance_{seed}_ga_{search_seed}.json').write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            print(f'instance={seed} ga={search_seed} time={result["best"]["cost"]:.3f} evaluations={result["evaluations"]}', flush=True)


if __name__ == '__main__':
    main()

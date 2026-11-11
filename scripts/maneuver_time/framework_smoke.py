"""Bounded engineering smoke; never grants formal-domain acceptance."""
import argparse
import pickle
from pathlib import Path

from coverage_env.config import load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.baseline.nested_ga import GAConfig
from coverage_env.rl.evaluation.framework_comparison import prepare_manifest, run_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    cfg = load_config()
    cfg['field_generator']['layout'] = {'mode': 'scattered'}
    cfg['field_generator']['field_size'] = {'min_area': 500., 'max_area': 800.}
    snapshots = []
    for seed in range(1900200, 1900203):
        env = InstanceGenerator(cfg).generate(num_fields=10, seed=seed,
            randomize_each_reset=False, randomize_uav=False)
        env.reset()
        path = out/f'instance_{seed}.pkl'
        path.write_bytes(pickle.dumps(env)); snapshots.append(path)
    prepare_manifest(snapshots, out/'manifest.json')
    settings = GAConfig(outer_population=2, outer_generations=1, inner_population=2,
                        inner_generations=1, max_evaluations=5, wall_seconds=60, seed=42)
    run_panel(out/'manifest.json', out/'comparison', settings)


if __name__ == '__main__': main()

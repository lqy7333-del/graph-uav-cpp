"""Same-snapshot controls for the bounded nested-GA pilot.

Original NN and the GA action-space initializer are intentionally separate:
the former uses the legacy nearest-field agent/workload implementation; the
latter repeatedly chooses the nearest feasible action, exactly like the GA
baseline. CHANet checkpoints are frozen and never used to initialize the GA.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import time

import torch

from coverage_env.baseline.nested_ga import nearest_action
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.evaluation.maneuver_eval import (
    _nn_choice, _policy_choice_factory, replay_recorded, run_episode,
)
from coverage_env.rl.ppo.ppo import PPOAgent


def initializer_choice(environment, _state):
    actions = environment.get_action_candidates()
    mask = environment.get_action_mask(actions)
    feasible = [a for a, ok in zip(actions, mask) if bool(ok)]
    if not feasible:
        return None, 0.0, 0.0
    return nearest_action(environment, feasible), 0.0, 0.0


def _load_policy(path, template):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = copy.deepcopy(checkpoint['resolved_config'])
    actor = CoveragePolicy(template, config=config).cuda()
    agent = PPOAgent(actor, config, device='cuda')
    agent.load_checkpoint(path, map_location='cuda', restore_rng=False, load_optimizers=False)
    agent.actor.eval()
    return _policy_choice_factory(agent.actor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True, help='GA pilot directory with instance_*.pkl')
    ap.add_argument('--output', required=True)
    ap.add_argument('--new-best', default='')
    ap.add_argument('--new-final', default='')
    ap.add_argument('--a10-best', required=True)
    ap.add_argument('--a10-final', required=True)
    args = ap.parse_args()
    panel = Path(args.panel); out = Path(args.output)
    if out.exists(): raise SystemExit('control output already exists')
    out.mkdir(parents=True)
    models = {
        'A10_u1060': args.a10_best,
        'A10_u1200': args.a10_final,
    }
    rows = []; actions_all = {}
    snapshots = sorted(panel.glob('instance_*.pkl'))
    if len(snapshots) < 1: raise RuntimeError('no snapshots found')
    template = pickle.loads(snapshots[0].read_bytes())
    policies = {label: _load_policy(path, template) for label, path in models.items()}
    controls = {'original_NN': _nn_choice, 'ga_initializer': initializer_choice}
    controls.update(policies)
    for snap_path in snapshots:
        env0 = pickle.loads(snap_path.read_bytes())
        seed = int(snap_path.stem.split('_')[-1])
        sha = hashlib.sha256(snap_path.read_bytes()).hexdigest()
        for label, choose in controls.items():
            started = time.perf_counter(); result = run_episode(copy.deepcopy(env0), choose, record=True); wall = time.perf_counter() - started
            replay = replay_recorded(copy.deepcopy(env0), result['actions'])
            match = (bool(replay['completed']) == bool(result['completed'])
                     and replay['termination_reason'] == result['termination_reason']
                     and abs(float(replay['total_time']) - float(result['total_time'])) < 1e-6)
            row = {k: result[k] for k in ('completed','termination_reason','total_time','time_work_base',
                'time_nonwork_base','time_maneuver_extra','time_service','total_distance','coverage_ratio',
                'steps','failed_steps','visits','resupplies','time_closure_error')}
            row.update(instance_seed=seed, snapshot_sha256=sha, model=label, replay_ok=match,
                       replay_total_time=replay['total_time'], eval_wall_seconds=wall)
            rows.append(row); actions_all[f'{seed}/{label}'] = result['actions']
            print(seed, label, round(row['total_time'], 3), row['completed'], match, flush=True)
    import csv
    keys = ['instance_seed','snapshot_sha256','model','completed','termination_reason','total_time',
            'time_work_base','time_nonwork_base','time_maneuver_extra','time_service','total_distance',
            'coverage_ratio','steps','failed_steps','visits','resupplies','time_closure_error',
            'replay_ok','replay_total_time','eval_wall_seconds']
    with (out/'rows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for row in rows: w.writerow({k: row.get(k) for k in keys})
    (out/'actions.json').write_text(json.dumps(actions_all, ensure_ascii=False))
    summary = {'panel': str(panel), 'output': str(out), 'checkpoints': {}, 'models': {}, 'rows': len(rows)}
    for label, path in models.items():
        summary['checkpoints'][label] = {'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    for label in controls:
        rr = [r for r in rows if r['model'] == label]
        summary['models'][label] = {
            'completed': sum(bool(r['completed']) for r in rr),
            'mean_total_time': sum(float(r['total_time']) for r in rr) / len(rr),
            'mean_components': {k: sum(float(r[k]) for r in rr)/len(rr) for k in
                ('time_work_base','time_nonwork_base','time_maneuver_extra','time_service')},
            'mean_distance': sum(float(r['total_distance']) for r in rr)/len(rr),
            'mean_visits': sum(float(r['visits']) for r in rr)/len(rr),
            'mean_resupplies': sum(float(r['resupplies']) for r in rr)/len(rr),
            'replay_all': all(bool(r['replay_ok']) for r in rr),
            'mean_eval_wall_seconds': sum(float(r['eval_wall_seconds']) for r in rr)/len(rr),
        }
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

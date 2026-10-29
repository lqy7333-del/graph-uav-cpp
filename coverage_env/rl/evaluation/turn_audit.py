"""Frozen-policy turn diagnostics; geometric angles are not physical turn costs."""
from copy import deepcopy
from math import atan2, degrees, hypot, sin, cos
import argparse
import json
from pathlib import Path


def path_angles(path, threshold=5.0):
    """Ignore duplicate/collinear points; count unsigned heading changes in degrees."""
    if not 0 < threshold <= 180:
        raise ValueError('threshold must be in (0, 180]')
    headings = [atan2(b[1]-a[1], b[0]-a[0]) for a,b in zip(path,path[1:])
                if hypot(b[0]-a[0],b[1]-a[1]) > 1e-8]
    angles = [abs(degrees(atan2(sin(b-a), cos(b-a))))
              for a,b in zip(headings,headings[1:])]
    return {'geometric_corners':sum(a >= threshold for a in angles),
            'heading_change_deg':sum(angles)}


def stroke_reversals(strokes):
    """Count opposite directed consecutive work strokes, not connector vertices.

    A 180-degree reversal between strokes is one sweep reversal even when its
    connector has two 90-degree corners. Fragmented same-direction strokes do
    not count. This metric does not claim a dynamically feasible U-turn.
    """
    vectors=[(b[0]-a[0],b[1]-a[1]) for a,b in strokes if hypot(b[0]-a[0],b[1]-a[1])>1e-8]
    return sum((a[0]*b[0]+a[1]*b[1])/(hypot(*a)*hypot(*b)) < -0.99
               for a,b in zip(vectors,vectors[1:]))


def mission_metrics(environment, threshold=5.0):
    """Measure work trajectories separately; never connect separate executions."""
    records=environment.execution_records
    angles=[path_angles(r['coverage_path'],threshold) for r in records]
    summary=environment.get_summary()
    return {'distance_m':summary['total_distance'], 'coverage_ratio':summary['coverage_ratio'],
            'covered_area':summary['covered_area'], 'visits':len(records),
            'sweep_reversals':sum(stroke_reversals(r['coverage_strokes']) for r in records),
            'coverage_corners':sum(r['geometric_corners'] for r in angles),
            'coverage_heading_change_deg':sum(r['heading_change_deg'] for r in angles)}


def main():
    """Compare frozen checkpoint and NN on identical maps and write raw paired rows."""
    import torch
    from coverage_env.config import load_config
    from coverage_env.env.instance_generator import InstanceGenerator
    from coverage_env.rl.checkpoint import evaluation_config
    from coverage_env.rl.decoder.action_decoder import CoveragePolicy
    from coverage_env.rl.evaluation.validation_suite import _run, _nearest_field_choice, _policy_choice
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--instances',type=int,default=5)
    parser.add_argument('--seed',type=int,default=30000)
    parser.add_argument('--fields',type=int,default=5)
    parser.add_argument('--angle-threshold',type=float,default=5.)
    parser.add_argument('--output',default='coverage_env/results/turn_audit.json')
    args=parser.parse_args()
    if args.instances<1 or args.fields<1:
        parser.error('instances and fields must be positive')
    path_angles([],args.angle_threshold)
    torch.set_num_threads(1)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=evaluation_config(saved,load_config())
    config['environment']['num_fields']={'min':args.fields,'max':args.fields}
    rows=[]
    for seed in range(args.seed,args.seed+args.instances):
        env=InstanceGenerator(config).generate(num_fields=args.fields,seed=seed,randomize_each_reset=False)
        baseline=deepcopy(env)
        actor=CoveragePolicy(env,config=config)
        actor.load_state_dict(saved['actor'])
        actor.eval()
        with torch.no_grad():
            outcomes=[_run(env,_policy_choice(actor)),_run(baseline,_nearest_field_choice)]
        rows.append({'seed':seed,'policy':mission_metrics(env,args.angle_threshold),
                     'nn':mission_metrics(baseline,args.angle_threshold),
                     'completed':[o['completed'] for o in outcomes]})
        print(json.dumps(rows[-1]))
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'checkpoint':args.checkpoint,'angle_threshold':args.angle_threshold,
        'config':config,'rows':rows},indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

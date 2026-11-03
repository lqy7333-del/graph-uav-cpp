"""One-shot paired greedy evaluation of frozen u1060 on fresh 10/20/30-field sets."""
import argparse, copy, datetime, json, math, random, socket, sys
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.evaluation.maneuver_eval import run_episode, _nn_choice, _policy_choice_factory
from coverage_env.rl.ppo.ppo import PPOAgent

SCALES=(10,20,30)
PER_SCALE=20
SEED_BASE={10:1000000,20:1010000,30:1020000}


def _worker(rank, world, path, out):
    device=torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda': torch.cuda.set_device(rank)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    ck=torch.load(saved['checkpoint'],map_location='cpu',weights_only=False)
    config=copy.deepcopy(ck['resolved_config'])
    template=saved['instances'][10][0]
    actor=CoveragePolicy(template,config=config).to(device)
    agent=PPOAgent(actor,config,device=device)
    agent.load_checkpoint(saved['checkpoint'],map_location=device,restore_rng=False,load_optimizers=False)
    agent.actor.eval()
    policy_choice=_policy_choice_factory(agent.actor)
    rows=[]
    for scale in SCALES:
        for i,env in enumerate(saved['instances'][scale]):
            if i%world!=rank: continue
            seed=saved['seeds'][scale][i]
            fp=saved['fingerprints'][scale][i]
            p=run_episode(copy.deepcopy(env),policy_choice)
            n=run_episode(copy.deepcopy(env),_nn_choice)
            p.update(scale=scale,index=i,seed=seed,fingerprint=fp,method='u1060')
            n.update(scale=scale,index=i,seed=seed,fingerprint=fp,method='NN')
            rows.extend([p,n])
    (Path(out)/f'rows_rank{rank}.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out',required=True)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=copy.deepcopy(ck['resolved_config'])
    instances={}; seeds={}; fingerprints={}
    from coverage_env.rl.trainer.episode import instance_fingerprint
    generation_log={}
    for scale in SCALES:
        cc=copy.deepcopy(config); cc['environment']['num_fields']={'min':scale,'max':scale}
        envs=[]; ss=[]; fps=[]; failures=[]
        seed=SEED_BASE[scale]
        # Fixed rule: scan the declared seed block in order and keep the first
        # PER_SCALE valid instances. Rejection is based only on geometry
        # validity, never on policy performance. Failed seeds are recorded.
        while len(envs)<PER_SCALE:
            if seed>=SEED_BASE[scale]+10000:
                raise RuntimeError(f'not enough valid {scale}-field instances in seed block')
            try:
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                env=InstanceGenerator(cc).generate(num_fields=scale,seed=seed,randomize_each_reset=False,randomize_uav=False)
                if len(env.fields)!=scale:
                    raise RuntimeError(f'generator returned {len(env.fields)} fields')
                envs.append(env); ss.append(seed); fps.append(instance_fingerprint(env))
            except Exception as exc:
                failures.append({'seed':seed,'error':f'{type(exc).__name__}: {exc}'})
            seed+=1
        instances[scale]=envs; seeds[scale]=ss; fingerprints[scale]=fps
        generation_log[scale]={'accepted':len(envs),'rejected':len(failures),'failures':failures}
        print(f'{scale} fields: {PER_SCALE} instances, area '
              f'{min(sum(f.area for f in e.fields) for e in envs):.0f}..'
              f'{max(sum(f.area for f in e.fields) for e in envs):.0f} m2, '
              f'rejected seeds={len(failures)}',flush=True)
    torch.save({'checkpoint':str(Path(args.checkpoint).resolve()),'config':config,
                'instances':instances,'seeds':seeds,'fingerprints':fingerprints,
                'generation_log':generation_log},
               out/'instances.pt')
    world=torch.cuda.device_count() if torch.cuda.is_available() else 1
    mp.spawn(_worker,args=(world,out/'instances.pt',out),nprocs=world,join=True)
    rows=[]
    for p in out.glob('rows_rank*.json'): rows+=json.loads(p.read_text())
    rows.sort(key=lambda r:(r['scale'],r['method'],r['index']))
    with (out/'paired_rows.csv').open('w',newline='') as f:
        import csv
        keys=['scale','index','seed','fingerprint','method','completed','termination_reason','total_time','total_distance','coverage_ratio','time_work_base','time_nonwork_base','time_maneuver_extra','failed_steps','steps']
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in keys})
    summary={'checkpoint':str(Path(args.checkpoint).resolve()),'checkpoint_update':int(ck['update_count']),
             'checkpoint_sha256':__import__('hashlib').sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
             'generation_log':{},'scales':{}}
    rng=np.random.default_rng(20261103)
    for scale in SCALES:
        summary['generation_log'][str(scale)]=generation_log[scale]
        p={r['index']:r for r in rows if r['scale']==scale and r['method']=='u1060'}
        n={r['index']:r for r in rows if r['scale']==scale and r['method']=='NN'}
        both=[i for i in p if p[i]['completed'] and n[i]['completed']]
        diffs=[n[i]['total_time']-p[i]['total_time'] for i in both]
        boot=[float(np.mean(rng.choice(diffs,len(diffs),replace=True))) for _ in range(10000)] if diffs else []
        summary['scales'][str(scale)]={
          'instances':PER_SCALE,'policy_completed':sum(p[i]['completed'] for i in p),
          'nn_completed':sum(n[i]['completed'] for i in n),
          'paired_complete':len(both),'policy_mean_all':sum(p[i]['total_time'] for i in p)/PER_SCALE,
          'nn_mean_all':sum(n[i]['total_time'] for i in n)/PER_SCALE,
          'policy_mean_complete':float(np.mean([p[i]['total_time'] for i in both])) if both else None,
          'nn_mean_complete':float(np.mean([n[i]['total_time'] for i in both])) if both else None,
          'nn_minus_policy_mean':float(np.mean(diffs)) if diffs else None,
          'nn_minus_policy_ci95':[float(x) for x in np.percentile(boot,[2.5,97.5])] if boot else None,
          'policy_wins':sum(d>1e-6 for d in diffs),'policy_losses':sum(d<-1e-6 for d in diffs),
          'policy_ties':sum(abs(d)<=1e-6 for d in diffs),
          'seeds':seeds[scale],'fingerprints':fingerprints[scale],
          'per_instance':[{'index':i,'seed':seeds[scale][i],'fingerprint':fingerprints[scale][i],
                           'policy_time':p[i]['total_time'],'nn_time':n[i]['total_time'],
                           'policy_completed':p[i]['completed'],'nn_completed':n[i]['completed'],
                           'nn_minus_policy':n[i]['total_time']-p[i]['total_time']} for i in range(PER_SCALE)]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ('seeds','fingerprints','per_instance')} for k,v in summary['scales'].items()},indent=2))


if __name__=='__main__': main()

"""Two-hour bounded replayable rollout search on the ten fixed instances."""
from __future__ import annotations
import argparse, copy, csv, dataclasses, hashlib, json, math, pickle, statistics, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch

REPO=Path('/home/ps/Desktop/Graph_planning/graph-uav-overfit-persist')
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from coverage_env.env.fixed_instance_pool import load_bank
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.evaluation.maneuver_eval import _nn_choice, _policy_choice_factory
from coverage_env.rl.ppo.ppo import PPOAgent

EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/overfit20')
BANK=EXP/'bank'
OUT=EXP/'search'
INITIAL=EXP/'initial/B_seed42_update_240.pt'
CURRENT=EXP/'short/control_s43/ckpt_update_60.pt'
WIDTH,SPEED=10.0,8.0
MAX_STEPS=400
SMOKE_INSTANCES=2
SMOKE_BUDGET=10
FORMAL_BUDGET=40
PER_INSTANCE_SECONDS=300
TOTAL_SECONDS=3000


def digest(obj)->str:
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()


def load_policy(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    config=copy.deepcopy(ck['resolved_config'])
    return ck,config


def make_agent(path,template):
    ck,config=load_policy(path)
    agent=PPOAgent(CoveragePolicy(template,config=config),config,device='cpu')
    agent.load_checkpoint(path,map_location='cpu',restore_rng=False,load_optimizers=False)
    agent.actor.eval(); agent.critic.eval()
    return agent.actor,ck


def state_summary(state,step):
    uavs=state.get('uavs',{})
    u=next(iter(uavs.values())) if uavs else {}
    return {'step':step,'uav_position':u.get('position'),'heading':u.get('heading'),
            'heading_valid':u.get('heading_valid'),'energy':u.get('remaining_energy'),
            'payload':u.get('remaining_payload'),'coverage':u.get('coverage_ratio'),
            'state_digest':digest(state)}


def run_actions(env,choose,record=True):
    state=env.reset(); actions=[]; summaries=[]; failed=0; done=False
    for step in range(MAX_STEPS):
        action,_,_=choose(env,state)
        if action is None: break
        if record: summaries.append(state_summary(state,step))
        state,_,done,info=env.step(action)
        actions.append(action)
        if info.get('failed'): failed+=1
        if done: break
    summary=env.get_summary()
    return {'actions':actions,'summaries':summaries,'summary':summary,'failed':failed,
            'completed':bool(done)}


def replay(template,actions):
    env=copy.deepcopy(template); state=env.reset(); done=False
    for a in actions:
        state,_,done,info=env.step(a)
        if done: break
    return env.get_summary(),bool(done)


def pick_uniform(items,n):
    if len(items)<=n: return list(items)
    if n<=0: return []
    if n==1: return [items[len(items)//2]]
    idx=sorted(set(round(i*(len(items)-1)/(n-1)) for i in range(n)))
    return [items[i] for i in idx]


def candidates_at(env,base,n_gate,n_field):
    actions=env.get_action_candidates(); mask=env.get_action_mask(actions)
    feasible=[a for a,m in zip(actions,mask) if m]
    same=[a for a in feasible if a.field_id==base.field_id and a!=base]
    gate=pick_uniform(same,n_gate)
    by={}
    for a in feasible:
        if a.field_id!=base.field_id: by.setdefault(a.field_id,[]).append(a)
    fields=pick_uniform(sorted(by,key=lambda x:str(x)),n_field)
    field=[pick_uniform(by[f],1)[0] for f in fields if by.get(f)]
    return gate,field


def complete_from(prefix_env,candidate,policy_choice=None):
    env=copy.deepcopy(prefix_env)
    _s,_r,done,_i=env.step(candidate)
    # NN completion is the fixed baseline used by the search.
    for _ in range(MAX_STEPS):
        if done: break
        action,_,_=_nn_choice(env,env.get_state())
        if action is None: break
        _s,_r,done,_i=env.step(action)
    summary=env.get_summary(); return summary,bool(done)


def search_instance(index,template,baseline_actions,baseline_digests,policy_choice,budget,deadline):
    start=time.perf_counter(); candidates=[]; current_best=list(baseline_actions); current_time=None
    prefix_mismatches=0; candidate_errors=0
    # Baseline time via replay.
    env=copy.deepcopy(template); env.reset(); done=False
    for a in current_best:
        _s,_r,done,_i=env.step(a)
        if done: break
    current_time=float(env.get_summary().get('total_time',float('inf')))
    positions_frac=(.15,.30,.50,.70,.85)
    positions=[]
    for f in positions_frac:
        p=min(len(current_best)-1,max(0,round(f*(len(current_best)-1))))
        if p not in positions: positions.append(p)
    rounds=2 if budget>=20 else 1
    per_round=budget//rounds
    per_prefix=max(1,per_round//max(1,len(positions)))
    gate_n=per_prefix//2; field_n=per_prefix-gate_n
    for rnd in range(rounds):
        if time.perf_counter()-start>PER_INSTANCE_SECONDS or time.perf_counter()>deadline: break
        for pos in positions:
            if time.perf_counter()-start>PER_INSTANCE_SECONDS or time.perf_counter()>deadline: break
            env=copy.deepcopy(template); env.reset()
            for a in current_best[:pos]:
                _s,_r,_d,_i=env.step(a)
                if _d: break
            # Verify restored prefix state against the recorded baseline only in round 0.
            if rnd==0 and pos<len(baseline_digests) and state_summary(env.get_state(),pos)['state_digest']!=baseline_digests[pos]['state_digest']:
                prefix_mismatches+=1
            base=current_best[pos]
            try:
                gates,fields=candidates_at(env,base,gate_n,field_n)
            except Exception as exc:
                candidate_errors+=1
                print(f'[search] candidate gen error instance={index} pos={pos}: {exc}',flush=True)
                continue
            for kind,cands in (('gate',gates),('field',fields)):
                for cand in cands:
                    if time.perf_counter()-start>PER_INSTANCE_SECONDS or time.perf_counter()>deadline: break
                    try:
                        summary,completed=complete_from(env,cand)
                    except Exception as exc:
                        candidate_errors+=1
                        continue
                    t=float(summary.get('total_time',float('inf')))
                    candidates.append({'round':rnd,'prefix':pos,'kind':kind,'candidate':dataclasses.asdict(cand),
                                       'time':t,'completed':bool(completed),'steps':len(current_best)})
                    if completed and t<current_time:
                        current_time=t
                        current_best=list(current_best[:pos])+[cand]
                        # completion actions are not recorded; use prefix+candidate only for next round?
                        # For a full replayable solution, rerun from candidate with NN and capture actions.
                        env2=copy.deepcopy(env); env2.step(cand); acts=list(current_best[:pos])+[cand]
                        for _ in range(MAX_STEPS):
                            a,_,_=_nn_choice(env2,env2.get_state())
                            if a is None: break
                            env2.step(a); acts.append(a)
                            if env2.get_summary().get('termination_reason')=='mission_complete': break
                        current_best=acts
    return {'index':index,'baseline_time':float(current_time if rnd==rounds-1 else current_time),
            'best_time':float(current_time),'candidates':candidates,'best_actions':current_best,
            'elapsed':time.perf_counter()-start,'budget':budget,
            'prefix_mismatches':prefix_mismatches,'candidate_errors':candidate_errors}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=('smoke','formal'),required=True)
    args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    manifest,templates=load_bank(BANK)
    policy_choice=None
    # Load current best saved model (control_s43@60); original u780 is missing.
    current_actor,_=make_agent(CURRENT,templates[0])
    current_choice=_policy_choice_factory(current_actor)
    initial_actor,_=make_agent(INITIAL,templates[0])
    initial_choice=_policy_choice_factory(initial_actor)
    budget=SMOKE_BUDGET if args.phase=='smoke' else FORMAL_BUDGET
    n=SMOKE_INSTANCES if args.phase=='smoke' else 10
    deadline=time.perf_counter()+TOTAL_SECONDS
    all_results=[]; replay_rows=[]
    for i in range(n):
        template=templates[i]
        # Baseline current best and initial, plus NN, with replay verification.
        cb=run_actions(copy.deepcopy(template),current_choice)
        ib=run_actions(copy.deepcopy(template),initial_choice)
        nb=run_actions(copy.deepcopy(template),_nn_choice)
        for label,b in (('nn',nb),('initial',ib),('current',cb)):
            r,replay_done=replay(template,b['actions'])
            ok=(bool(replay_done) and abs(float(r['total_time'])-float(b['summary']['total_time']))<1e-6)
            replay_rows.append({'index':i,'seed':70000+i,'label':label,'time':float(b['summary']['total_time']),
                                'completed':bool(b['completed']),'replay_ok':bool(ok),
                                'actions':len(b['actions']),'state_digest':b['summaries'][-1]['state_digest'] if b['summaries'] else None})
            with (OUT/f'replayable_{args.phase}_{label}_{i}.pkl').open('wb') as f: pickle.dump(b['actions'],f)
        sr=search_instance(i,template,cb['actions'],cb['summaries'],current_choice,budget,deadline)
        all_results.append(sr)
        with (OUT/f'search_{args.phase}_{i}.pkl').open('wb') as f: pickle.dump(sr['best_actions'],f)
        print(f'[{args.phase}] instance {i}: baseline {sr["baseline_time"]:.1f} best {sr["best_time"]:.1f} candidates {len(sr["candidates"])} elapsed {sr["elapsed"]:.1f}s',flush=True)
    (OUT/f'replay_{args.phase}.json').write_text(json.dumps(replay_rows,indent=2,ensure_ascii=False))
    (OUT/f'search_{args.phase}.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='best_actions'} for r in all_results],indent=2,ensure_ascii=False))
    # lower bound reference per instance
    lbs=[]
    for i in range(n):
        area=sum(float(f.area) for f in templates[i].fields)
        lbs.append({'index':i,'seed':70000+i,'task_area_m2':area,'L_work_s':area/(WIDTH*SPEED)})
    (OUT/f'lower_bound_{args.phase}.json').write_text(json.dumps(lbs,indent=2,ensure_ascii=False))
    print('done',args.phase)


if __name__=='__main__': main()

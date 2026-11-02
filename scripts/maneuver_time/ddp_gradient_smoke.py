"""Two-rank Gloo gradient comparison for unequal complete-episode batches.

Rank 0 has 3 valid rows, rank 1 has 6; with batch_size=4 the second local
mini-batch on rank 0 is pure padding. DDP gradients (no optimizer step) are
compared with a centralized cumulative-gradient reference.
"""
import datetime, json, socket, sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.tests.test_training_runtime import ToyPolicy, toy_config


def make_rows(agent, n, seed=123):
    torch.manual_seed(seed)
    rows=[]
    raw=agent.actor.module if hasattr(agent.actor,'module') else agent.actor
    for i in range(n):
        state={'features':[0.2+0.01*i, 0.1+0.02*i]}
        with torch.no_grad():
            out=raw(state)
            lp=torch.log(out['action_distribution'][0].clamp_min(1e-12)).item()
        context={'observation':state,'graph':out['graph'],'actions':[0,1],'mask':[True,True]}
        rows.append(SimpleNamespace(state=context, action=0, log_prob=lp,
                                    value=0., next_value=0., done=False, truncated=False,
                                    reward=-1.))
    return rows


def losses(agent, rows, indices, mask, global_count, scale_world=2):
    logp, ent, values = agent._per_transition_minibatch_rows(rows, indices)
    old=torch.as_tensor([rows[i].log_prob for i in indices],dtype=torch.float32)
    ratios=torch.exp(logp-old)
    weights=torch.as_tensor(mask,dtype=torch.float32)
    batch_count=float(global_count)
    scale=scale_world/batch_count
    surrogate=(ratios*weights).sum()*scale
    entropy=(ent*weights).sum()*scale
    actor_loss=-surrogate-agent.entropy_coef*entropy
    critic_loss=(values.square()*weights).sum()*scale
    return actor_loss, critic_loss


def worker(rank, world, port, out):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=world,
                            timeout=datetime.timedelta(seconds=120))
    try:
        torch.manual_seed(42)
        agent=PPOAgent(ToyPolicy(),toy_config())
        agent.actor=torch.nn.parallel.DistributedDataParallel(agent.actor)
        agent.critic=torch.nn.parallel.DistributedDataParallel(agent.critic)
        agent.batch_size=4; agent.epochs=1; agent.max_grad_norm=1e9
        agent.actor.eval(); agent.critic.eval()
        rows=make_rows(agent,3 if rank==0 else 6)
        mask=[True]*(3 if rank==0 else 6)
        # all ranks use the same identity minibatch order
        torch.randperm=lambda n: torch.arange(n)
        zg_tracker=[]; step_tracker=[]
        agent.actor_optimizer.zero_grad=lambda set_to_none=True: zg_tracker.append('a')
        agent.critic_optimizer.zero_grad=lambda set_to_none=True: zg_tracker.append('c')
        agent.actor_optimizer.step=lambda: step_tracker.append('a')
        agent.critic_optimizer.step=lambda: step_tracker.append('c')
        adv=torch.ones(len(rows)); ret=torch.zeros(len(rows))
        stats=agent.update_masked(rows,mask,global_valid_count=9,global_steps=6,
                                  advantages=adv,returns=ret)
        actor_grads={k:v.grad.detach().clone() for k,v in agent.actor.module.named_parameters() if v.grad is not None}
        critic_grads={k:v.grad.detach().clone() for k,v in agent.critic.module.named_parameters() if v.grad is not None}
        if rank==0:
            ref=PPOAgent(ToyPolicy(),toy_config())
            ref.actor.load_state_dict(agent.actor.module.state_dict())
            ref.critic.load_state_dict(agent.critic.module.state_dict())
            ref.actor.eval(); ref.critic.eval(); ref.entropy_coef=agent.entropy_coef
            torch.manual_seed(42)
            # Manual central reference: average the two ranks' local cumulative
            # gradients; batch 0 has 3+4 valid rows, batch 1 has 0+2.
            for local_rows, local_mask, counts in (
                (rows, mask, (7,0)),):
                pass
            # Build both rank buffers explicitly (rank0 has 3 padded rows,
            # rank1 has 6 real rows).
            r0=make_rows(agent,3); r1=make_rows(agent,6)
            r0pad=r0+[r0[-1]]*3
            r1pad=r1
            # identity permutation: batch starts 0 and 4
            ratio_checks=[]
            for batch_start,batch_count in ((0,7),(4,2)):
                # rank 0 local contribution
                idx0=[i for i in range(batch_start,min(batch_start+4,6))]
                m0=[i < 3 for i in idx0]
                # rank 1 local contribution
                idx1=[i for i in range(batch_start,min(batch_start+4,6))]
                m1=[True]*len(idx1)
                if any(m0):
                    a,c=losses(ref,r0pad,idx0,m0,batch_count)
                    ratio_checks.append(float(a.detach()))
                    (a+c*0.5).backward()
                if any(m1):
                    a,c=losses(ref,r1pad,idx1,m1,batch_count)
                    ratio_checks.append(float(a.detach()))
                    (a+c*0.5).backward()
            for p in ref.actor.parameters(): p.grad.div_(2.0)
            for p in ref.critic.parameters(): p.grad.div_(2.0)
            diffs_a=[]; diffs_c=[]
            for k,p in ref.actor.named_parameters():
                if p.grad is not None: diffs_a.append(float((p.grad-actor_grads[k]).abs().max().item()))
            for k,p in ref.critic.named_parameters():
                if p.grad is not None: diffs_c.append(float((p.grad-critic_grads[k]).abs().max().item()))
            def gn(mod): return float(sum(float(p.grad.norm())**2 for p in mod.parameters() if p.grad is not None)**0.5)
            payload={'update_steps':stats['updates'],'ratio_std':stats['ratio_std'],'actor_max_grad_diff':max(diffs_a) if diffs_a else None,
                     'critic_max_grad_diff':max(diffs_c) if diffs_c else None,
                     'ddp_actor_gn':gn(agent.actor.module), 'ref_actor_gn':gn(ref.actor),
                     'ddp_critic_gn':gn(agent.critic.module), 'ref_critic_gn':gn(ref.critic),
                     'ref_losses':ratio_checks,'zero_grad_calls':zg_tracker,'optimizer_steps':step_tracker,
                     'ddp_actor_loss':stats['actor_loss'],'ddp_policy_loss':stats['policy_loss'],'ddp_entropy':stats['entropy'],
                     'rank0_rows':3,'rank1_rows':6,'batch_size':4,'global_steps':6,
                     'padding_batch_rank0':True}
            Path(out).write_text(json.dumps(payload,indent=2))
        dist.barrier()
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


def main():
    out=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/ddp_gradient.json')
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    mp.spawn(worker,args=(2,port,out),nprocs=2,join=True)
    print(out.read_text())


if __name__=='__main__': main()

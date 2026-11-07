"""GPU latency benchmark for frozen A/B/C/D policies on one validation instance."""
import copy, json, statistics, time
from pathlib import Path
import torch
from coverage_env.env.fixed_instance_pool import load_bank
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.evaluation.maneuver_eval import _policy_choice_factory

EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10')
MODELS={
 'A':EXP/'formal_random_instances_s42/ckpt_update_1060.pt',
 'B':EXP/'ablation_node_mlp_s42/ckpt_update_760.pt',
 'C':EXP/'ablation_no_inter_s42/ckpt_update_1020.pt',
 'D':EXP/'ablation_no_intra_s42/ckpt_update_900.pt',
}
out={'device':'cuda:0','warmup':20,'iterations':200,'models':{}}
_,envs=load_bank(EXP/'valid30'); env=copy.deepcopy(envs[0]); state=env.reset()
torch.cuda.set_device(0)
for label,path in MODELS.items():
 ck=torch.load(path,map_location='cpu',weights_only=False); config=copy.deepcopy(ck['resolved_config'])
 template=env
 actor=CoveragePolicy(template,config=config).cuda()
 agent=PPOAgent(actor,config,device='cuda')
 agent.load_checkpoint(path,map_location='cuda',restore_rng=False,load_optimizers=False)
 agent.actor.eval(); choose=_policy_choice_factory(agent.actor)
 with torch.no_grad():
  for _ in range(20):
   torch.cuda.synchronize(); choose(env,state); torch.cuda.synchronize()
  times=[]
  for _ in range(200):
   torch.cuda.synchronize(); t=time.perf_counter(); choose(env,state); torch.cuda.synchronize(); times.append(time.perf_counter()-t)
 out['models'][label]={'params':sum(p.numel() for p in agent.actor.parameters()),
  'mean_ms':statistics.mean(times)*1e3,'median_ms':statistics.median(times)*1e3,
  'p95_ms':sorted(times)[int(.95*len(times))-1]*1e3}
 print(label,out['models'][label],flush=True)
(EXP/'ablation_inference_benchmark.json').write_text(json.dumps(out,indent=2))

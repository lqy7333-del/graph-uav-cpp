"""Bounded synthetic PPO tests: no farm training or long optimization runs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from coverage_env.config import load_config
from coverage_env.rl.buffer.rollout_buffer import RolloutBuffer
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.ppo.loss import compute_advantage
from coverage_env.rl.trainer.distributed import resolve_config, validate_exact_resume
from coverage_env.rl.trainer.trainer import Trainer
from coverage_env.rl.checkpoint import capture_rng_state, restore_rng_state, evaluation_config


class ToyGraphBuilder:
    """Tensor graphs keep checkpoint/rollout tests independent of expensive geometry."""
    def build_graph(self, environment):
        return torch.tensor(environment.get_state()['features'])


class ToyPolicy(nn.Module):
    """Tiny dropout policy exposing the same actor contract as CoveragePolicy."""
    def __init__(self):
        super().__init__()
        self.encoder = SimpleNamespace(hidden_dim=8)
        self.features = nn.Sequential(nn.Linear(2,8), nn.Dropout(.5))
        self.head = nn.Linear(8,2)
        self.graph_builder = ToyGraphBuilder()

    def forward(self, state, **kwargs):
        graph = torch.tensor(state['features'],dtype=torch.float32)
        z = self.features(graph)
        return {'action_distribution':self.head(z).softmax(-1), 'global_embedding':z,
                'mask':torch.tensor([True,True]),'actions':[0,1], 'graph':graph}

    def encode_graph(self, graph, **kwargs):
        z=self.features(graph.float())
        return {},z,z


class ToyEnvironment:
    """Two-step deterministic episodes with a serializable private random stream."""
    def __init__(self, seed=0):
        self.rng=random.Random(seed)
        self.fields=[0,1]
        self.total_area=100
        self.reset()

    def reset(self):
        self.steps=0
        self.distance=0.
        self.features=[self.rng.random(), self.rng.random()]
        return self.get_state()

    def get_state(self):
        return {'features':list(self.features)}

    def step(self, action):
        self.steps+=1
        cost=1.+float(action)+self.features[0]
        self.distance+=cost
        self.features=[self.features[1],self.features[0]/2]
        done=self.steps==2
        return self.get_state(),-cost,done,{'termination_reason':'mission_complete' if done else None}

    def get_summary(self):
        return {'coverage_ratio':self.steps/2,'total_distance':self.distance,
                'total_energy':self.distance,'remaining_energy':100-self.distance,
                'completed_fields':self.fields if self.steps==2 else []}


def toy_config(mode='gae'):
    return {'ppo':{'gamma':1.,'gae_lambda':.95,'epochs':1,'batch_size':4,
                   'advantage_mode':mode,'entropy_coef':.01,'entropy_coef_end':0.},
            'shaping':{'enable':False}}


def toy_buffer(agent, complete=True):
    buffer=RolloutBuffer()
    for i,reward in enumerate([-1.,-2.,-3.]):
        state={'features':[.25,float(i)/3]}
        with torch.no_grad():
            output=agent.actor(state)
        context={'observation':state,'graph':output['graph'],'actions':[0,1],'mask':[True,True]}
        buffer.add(context,0,reward,context,complete and i==2,
                   output['action_distribution'][0].log().item(),0.,0.)
    return buffer


class RuntimeTests(unittest.TestCase):
    """Exercise the update, configuration and resume paths rather than source strings."""

    def test_actual_update_replays_dropout_deterministically(self):
        a=PPOAgent(ToyPolicy(),toy_config())
        a.actor.eval()
        buffer=toy_buffer(a)
        a.actor.train()  # update must enforce deterministic replay itself
        with patch.object(a.actor_optimizer,'step'), patch.object(a.critic_optimizer,'step'):
            stats=a.update(buffer)
        self.assertLess(stats['ratio_std'],1e-7)
        self.assertEqual(stats['clip_fraction'],0.)
        self.assertFalse(a.actor.training)

    def test_actual_update_uses_complete_episode_returns(self):
        a=PPOAgent(ToyPolicy(),toy_config('broadcast'))
        a.actor.eval()
        buffer=toy_buffer(a)
        original=a._select_advantages
        seen=[]
        def record(*args):
            result=original(*args)
            seen.append(result.clone())
            return result
        with patch.object(a,'_select_advantages',side_effect=record):
            stats=a.update(buffer)
        self.assertEqual(seen[0].tolist(),[-6.]*3)
        self.assertEqual(stats['actor_samples'],3)
        self.assertEqual(a.last_advantage_mode,'broadcast')

    def test_incomplete_broadcast_does_not_move_actor_or_momentum(self):
        a=PPOAgent(ToyPolicy(),toy_config('broadcast'))
        a.actor.eval()
        a.update(toy_buffer(a))  # initialize Adam moments
        before=deepcopy(a.actor.state_dict())
        optimizer_before=deepcopy(a.actor_optimizer.state_dict())
        stats=a.update(toy_buffer(a,complete=False))
        self.assertEqual(stats['actor_samples'],0)
        for key,value in before.items():
            self.assertTrue(torch.equal(value,a.actor.state_dict()[key]))
        for index,values in optimizer_before['state'].items():
            for key,value in values.items():
                self.assertTrue(torch.equal(value,a.actor_optimizer.state_dict()['state'][index][key]))

    def test_hand_computed_baselines_discount_and_truncation(self):
        a=PPOAgent(ToyPolicy(),toy_config('broadcast_nn'))
        rows=[SimpleNamespace(reward=-1,done=True,value=-8,truncated=False),
              SimpleNamespace(reward=-100,done=False,value=0,truncated=True),
              SimpleNamespace(reward=-2,done=True,value=-4,truncated=False)]
        a.nn_baselines=[-3,-999,-5]
        self.assertEqual(a._select_advantages(rows,None,None,a.device).tolist(),[2,0,3])
        a.gamma=.5
        rows[0].done=False
        rows[1].truncated=False
        rows[1].done=True
        a.config['advantage_mode']='broadcast'
        self.assertEqual(a._select_advantages(rows,None,None,a.device).tolist(),[-51,-51,-2])

    def test_time_limit_bootstraps_without_cross_episode_gae(self):
        advantage,targets=compute_advantage([-1,-100],[2,0],[5,0],[False,True],1,1,
                                             truncated=[True,False])
        self.assertEqual(advantage.tolist(),[2,-100])
        self.assertEqual(targets.tolist(),[4,-100])

    def test_cli_priority_uses_effective_configuration(self):
        args=SimpleNamespace(set=['ppo.epochs=1','ppo.batch_size=7'],num_fields=None,
                             ppo_epochs=None,batch_size=None,gae_lambda=None)
        config=resolve_config(args)
        self.assertEqual((config['ppo']['epochs'],config['ppo']['batch_size']),(1,7))
        args.ppo_epochs=2
        self.assertEqual(resolve_config(args)['ppo']['epochs'],2)

    def test_rng_roundtrip(self):
        state=capture_rng_state()
        expected=(random.random(),np.random.rand(),torch.rand(3))
        restore_rng_state(state)
        self.assertEqual(expected[0],random.random())
        self.assertEqual(expected[1],np.random.rand())
        self.assertTrue(torch.equal(expected[2],torch.rand(3)))

    def test_resume_validation_rejects_legacy_or_changed_budget(self):
        args=SimpleNamespace(num_updates=5,rollout_steps=4,seed=42)
        config=toy_config()
        with self.assertRaisesRegex(ValueError,'finetune'):
            validate_exact_resume({},config,args,1)
        saved={'rank_runtime':[{}],'resolved_config':config,
               'training_metadata':dict(num_updates=5,rollout_steps=4,seed=42,world_size=1)}
        validate_exact_resume(saved,config,args,1)
        args.num_updates=10
        with self.assertRaisesRegex(ValueError,'num_updates'):
            validate_exact_resume(saved,config,args,1)

    def test_evaluation_prefers_saved_config(self):
        config=load_config()
        config['rl']['reward_scale']=1234.
        actual=evaluation_config({'resolved_config':config,'actor':{}},load_config())
        self.assertEqual(actual['rl']['reward_scale'],1234.)

    def test_single_trainer_resume_matches_uninterrupted(self):
        torch.manual_seed(42)
        config=toy_config()
        a=PPOAgent(ToyPolicy(),config)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            t=Trainer(ToyEnvironment(42),a,config=config,checkpoint_dir=directory)
            t.train(2)
            b=PPOAgent(ToyPolicy(),config)
            resumed=Trainer(ToyEnvironment(99),b,config=config)
            resumed.load_checkpoint(Path(directory)/'ppo_episode_1.pt')
            resumed.train(2)
            for left,right in zip(a.actor.parameters(),b.actor.parameters()):
                self.assertTrue(torch.equal(left,right))
            for left,right in zip(a.critic.parameters(),b.critic.parameters()):
                self.assertTrue(torch.equal(left,right))
            self.assertEqual(t.history,resumed.history)

    def test_single_trainer_time_limit_and_schedule_reach_update(self):
        config=toy_config()
        a=PPOAgent(ToyPolicy(),config)
        trainer=Trainer(ToyEnvironment(),a,config=config,max_steps=1)
        observed=[]
        def inspect(buffer):
            observed.append(deepcopy(buffer.transitions[-1]))
            return a._empty_update_stats()
        with patch.object(a,'update',side_effect=inspect), patch.object(a,'value_for_graph',return_value=12.):
            trainer.train(2)
        self.assertFalse(observed[0].done)
        self.assertTrue(observed[0].truncated)
        self.assertEqual(observed[0].next_value,12.)
        self.assertEqual(a.entropy_coef,0.)

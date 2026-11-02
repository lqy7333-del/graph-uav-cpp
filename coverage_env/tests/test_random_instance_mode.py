"""Tests for the random 10-field training source and fixed evaluation panels."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import torch
from coverage_env.config import load_config
from coverage_env.env.fixed_instance_pool import load_bank, write_bank
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.episode_random import (
    RandomInstanceEpisodeTrainer, RandomInstanceStream,
)


class RandomInstanceModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=load_config()
        cls.config['environment']['num_fields']={'min':10,'max':10}
        cls.config['field_generator']['layout']={'mode':'scattered'}
        cls.config['field_generator']['field_size']={'min_area':500.0,'max_area':800.0}
        cls.config['ppo'].update(epochs=1,batch_size=2,value_gate=True,
                                 randomize_uav=False,gamma=1.0,actor_learning_rate=1e-4,
                                 critic_learning_rate=3e-3)
        cls.config['rl']['reward']['mode']='time'
        cls.config['rl']['decoder']['candidate_attention']=False

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        envs=[InstanceGenerator(self.config).generate(num_fields=10,seed=70000+i,
            randomize_each_reset=False,randomize_uav=False) for i in range(2)]
        self.bank=self.root/'panel'
        write_bank(self.bank,self.config,envs,[70000,70001])

    def test_rank_seed_blocks_are_disjoint_and_reproducible(self):
        a=RandomInstanceStream(900_000_000,0)
        b=RandomInstanceStream(900_000_000,1)
        a_seeds=[a.candidate(a.next_episode()) for _ in range(20)]
        b_seeds=[b.candidate(b.next_episode()) for _ in range(20)]
        self.assertTrue(set(a_seeds).isdisjoint(b_seeds))
        rebuilt=RandomInstanceStream(900_000_000,0)
        again=[rebuilt.candidate(rebuilt.next_episode()) for _ in range(20)]
        self.assertEqual(a_seeds,again)

    def test_stream_counter_is_independent_of_episode_length(self):
        stream=RandomInstanceStream(900_000_000,0)
        first=stream.candidate(stream.next_episode())
        # Consume arbitrary action/other randomness between episode starts.
        torch.manual_seed(999)
        _=torch.randn(100)
        second=stream.candidate(stream.next_episode())
        self.assertNotEqual(first,second)
        self.assertEqual(second,900_000_064)

    def test_new_instances_change_seed_and_fingerprint_and_are_10_fields(self):
        pool,_=load_bank(self.bank)
        env0=load_bank(self.bank)[1][0]
        agent=PPOAgent(CoveragePolicy(env0,config=self.config),self.config,device='cpu')
        trainer=RandomInstanceEpisodeTrainer(
            env0,agent,rank=0,world_size=1,checkpoint_dir=None,
            config=self.config,diagnostic_bank=self.bank,validation_bank=self.bank,
            instance_seed_base=900_000_000,num_fields=10,
            episodes_per_rank=2,max_episode_steps=100)
        trainer.train(1,rollout_steps=1)
        records=trainer._episode_records
        self.assertEqual(len(records),2)
        self.assertTrue(all(r['field_count']==10 for r in records))
        self.assertEqual(len({r['instance_seed'] for r in records}),2)
        self.assertEqual(len({r['fingerprint'] for r in records}),2)
        self.assertEqual(trainer._episode_seq,2)
        self.assertFalse(trainer._episode_open)

    def test_fixed_panel_evaluation_does_not_advance_instance_stream(self):
        env0=load_bank(self.bank)[1][0]
        agent=PPOAgent(CoveragePolicy(env0,config=self.config),self.config,device='cpu')
        out=self.root/'evalrun'
        trainer=RandomInstanceEpisodeTrainer(
            env0,agent,rank=0,world_size=1,checkpoint_dir=out,
            config=self.config,diagnostic_bank=self.bank,validation_bank=self.bank,
            instance_seed_base=900_000_000,num_fields=10,
            episodes_per_rank=1,max_episode_steps=100)
        counter=trainer.stream.counter
        torch_state=torch.random.get_rng_state().clone()
        trainer._evaluate_panel(self.bank,'diagnostic',0)
        self.assertEqual(trainer.stream.counter,counter)
        self.assertTrue(torch.equal(torch.random.get_rng_state(),torch_state))
        self.assertTrue((out/'diagnostic/evaluation/update_0.json').exists())

    def test_episode_records_are_rank_local_and_10_fields(self):
        env0=load_bank(self.bank)[1][0]
        agent=PPOAgent(CoveragePolicy(env0,config=self.config),self.config,device='cpu')
        out=self.root/'records'
        trainer=RandomInstanceEpisodeTrainer(
            env0,agent,rank=0,world_size=1,checkpoint_dir=out,
            config=self.config,diagnostic_bank=self.bank,validation_bank=self.bank,
            instance_seed_base=900_000_000,num_fields=10,
            episodes_per_rank=1,max_episode_steps=100)
        trainer.train(1,rollout_steps=1)
        rows=list(csv.DictReader((out/'train_episodes_rank_0.csv').open()))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['field_count'],'10')
        self.assertIn('instance_seed',rows[0])


if __name__=='__main__': unittest.main()

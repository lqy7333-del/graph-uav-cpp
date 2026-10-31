"""Opt-in two-process CPU/Gloo regression tests; no UAV training is started."""
from copy import deepcopy
import datetime
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from coverage_env.rl.ppo.ppo import PPOAgent, ValueNetwork
from coverage_env.rl.trainer.distributed import DistributedTrainer
from coverage_env.rl.checkpoint import restore_rng_state, assert_rank_parameters_equal
from coverage_env.tests.test_training_runtime import ToyPolicy, ToyEnvironment, toy_config, toy_buffer


def _worker(rank, port, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2,
                            timeout=datetime.timedelta(seconds=90))
    try:
        torch.manual_seed(42)
        probe=PPOAgent.__new__(PPOAgent)
        probe.critic=DDP(ValueNetwork(4,8)).eval()
        probe._value_bias_initialized=False
        # Unequal sample counts test the global sum/count, not the mean of means.
        returns=torch.tensor([-4.,-6.]) if rank==0 else torch.tensor([-8.])
        probe._warm_start_value_bias(returns)
        assert probe.critic.module.network[-1].bias.item()==-6.
        probe.critic(torch.ones(1,4)).square().mean().backward()
        assert_rank_parameters_equal(probe.critic)

        config=toy_config()
        torch.manual_seed(42)
        agent=PPOAgent(ToyPolicy(),config)
        agent.actor=DDP(agent.actor)
        agent.critic=DDP(agent.critic)
        trainer=DistributedTrainer(ToyEnvironment(42+rank),agent,rank=rank,world_size=2,
                                   config=config,checkpoint_dir=directory,checkpoint_every=1)
        trainer.run_metadata={'seed':42}
        torch.manual_seed(42+rank)
        trainer.train(2,4)
        assert trainer._rank_stats["transitions"] == 8
        assert trainer._rank_stats["updates"] == 2
        if rank == 0:
            assert (Path(directory)/'run_summary.json').exists()
        actor_end=deepcopy(agent.actor.module.state_dict())
        critic_end=deepcopy(agent.critic.module.state_dict())
        saved=torch.load(Path(directory)/'ckpt_update_1.pt',map_location='cpu',weights_only=False)
        resumed=PPOAgent(ToyPolicy(),config)
        resumed.load_checkpoint(Path(directory)/'ckpt_update_1.pt',restore_rng=False)
        resumed.actor=DDP(resumed.actor)
        resumed.critic=DDP(resumed.critic)
        runtime=saved['rank_runtime'][rank]
        continuation=DistributedTrainer(runtime['environment'],resumed,rank=rank,world_size=2,config=config)
        continuation._count_emas=runtime['count_emas']
        restore_rng_state(runtime['rng'])
        continuation.train(2,4,start_update=2)
        for key,value in actor_end.items():
            assert torch.equal(value,resumed.actor.module.state_dict()[key]),key
        for key,value in critic_end.items():
            assert torch.equal(value,resumed.critic.module.state_dict()[key]),key
        assert_rank_parameters_equal(resumed.actor,resumed.critic)

        # Different ranks can have different eligible broadcast sample counts.
        # All still issue the same backward calls and stay synchronized.
        resumed.config['advantage_mode']='broadcast'
        resumed.actor.eval()
        buffer=toy_buffer(resumed,complete=rank==0)
        stats=resumed.update(buffer)
        assert stats['actor_samples']==(3 if rank==0 else 0)
        assert_rank_parameters_equal(resumed.actor,resumed.critic)
    finally:
        dist.destroy_process_group()


def _gate_failure_worker(rank, port, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=60))
    try:
        config=toy_config()
        torch.manual_seed(42)
        agent=PPOAgent(ToyPolicy(),config)
        trainer=DistributedTrainer(ToyEnvironment(42+rank),agent,rank=rank,
                                   world_size=2,config=config,
                                   checkpoint_dir=directory)
        good={
            'value_state_gap_max':0.0,'value_state_scale':1.0,
            'pair_q_max':0.1,'tail_value_error':0.0,'tail_value_scale':1.0,
            'tail_q':0.1,'replay_logprob_error':0.0,
            'replay_value_error':0.0,'episode_closure_error_max':0.0,
            'finite':True,
        }
        trainer._gate_or_abort(good, 1)
        bad=dict(good)
        bad['value_state_gap_max']=1.0
        bad['pair_q_max']=2.0 if rank==1 else 0.0
        saw_gap=False
        try:
            trainer._gate_or_abort(bad, 2)
        except RuntimeError:
            saw_gap=True
        nan_bad=dict(good)
        # Keep finite=True: the explicit finite check on the metric itself
        # (not the caller's flag) must catch the injected NaN.
        nan_bad['replay_logprob_error']=float('nan') if rank==1 else 0.0
        saw_nan=False
        try:
            trainer._gate_or_abort(nan_bad, 3)
        except RuntimeError:
            saw_nan=True
        Path(directory, f'rank{rank}_saw_failure.json').write_text(
            json.dumps({'saw_gap':saw_gap,'saw_nan':saw_nan,'rank':rank}))
        assert saw_gap and saw_nan
    finally:
        dist.destroy_process_group()


class GateMetricTests(unittest.TestCase):
    """Local replay output NaN must be a finite-flag failure, not a no-op max."""

    def test_replay_nan_marks_local_gate_nonfinite(self):
        config=toy_config()
        agent=PPOAgent(ToyPolicy(),config)
        agent.actor.eval(); agent.critic.eval()
        buffer=toy_buffer(agent)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            trainer=DistributedTrainer(ToyEnvironment(3),agent,rank=0,world_size=1,
                                       config=config,checkpoint_dir=directory)
            original=trainer.agent._per_transition_minibatch_rows
            def poisoned(transitions,indices):
                count=len(indices)
                return (torch.full((count,),float('nan')),
                        torch.zeros(count),
                        torch.full((count,),float('nan')))
            trainer.agent._per_transition_minibatch_rows=poisoned
            try:
                metrics=trainer._local_gate_metrics(buffer.transitions,0.0)
            finally:
                trainer.agent._per_transition_minibatch_rows=original
            self.assertFalse(metrics['finite'])
            self.assertTrue(metrics['replay_bad_indices'])
            self.assertFalse(math.isfinite(metrics['replay_logprob_error']))

    def test_failure_snapshot_uses_max_gap_index(self):
        config=toy_config()
        agent=PPOAgent(ToyPolicy(),config)
        agent.actor.eval(); agent.critic.eval()
        buffer=toy_buffer(agent)
        buffer.transitions[1].next_value=100.0
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            trainer=DistributedTrainer(ToyEnvironment(4),agent,rank=0,world_size=1,
                                       config=config,checkpoint_dir=directory)
            trainer.buffer=buffer
            metrics=trainer._local_gate_metrics(buffer.transitions,0.0)
            self.assertEqual(metrics['gap_index'],1)
            trainer._write_gate_failure(7,'test maximum gap',metrics,{})
            payload=json.loads(
                Path(directory,'gate_failures/update_7_rank_0.json').read_text())
            self.assertIn(1,payload['bad_indices'])
            self.assertIn(2,payload['bad_indices'])
            saved=torch.load(
                Path(directory,'gate_failures/update_7_rank_0.pt'),
                map_location='cpu',weights_only=False)
            self.assertIn(1,saved['samples'])
            self.assertIn(2,saved['samples'])


@unittest.skipUnless(os.environ.get('RUN_DDP_TESTS')=='1','Set RUN_DDP_TESTS=1 for local loopback/Gloo tests.')
class DistributedGateFailureTests(unittest.TestCase):
    """One rank's bad metric must make every rank reject the update."""

    def test_one_bad_rank_aborts_all_ranks(self):
        with socket.socket() as connection:
            connection.bind(('127.0.0.1',0))
            port=connection.getsockname()[1]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            mp.spawn(_gate_failure_worker,args=(port,directory),nprocs=2,join=True)
            for rank in (0,1):
                payload=json.loads(Path(directory,f'rank{rank}_saw_failure.json').read_text())
                self.assertTrue(payload['saw_gap'],payload)
                self.assertTrue(payload['saw_nan'],payload)
            self.assertTrue(Path(directory,'gate_failures/update_2_summary.json').exists())
            self.assertTrue(Path(directory,'gate_failures/update_3_summary.json').exists())

    def test_pair_relative_tolerance_is_not_global(self):
        from coverage_env.rl.trainer.distributed import VALUE_GAP_ABS, VALUE_GAP_REL
        small = DistributedTrainer._normalized_error(
            0.02, 1000.0, 1000.0, VALUE_GAP_ABS, VALUE_GAP_REL)
        ordinary = DistributedTrainer._normalized_error(
            0.02, 1.0, 1.0, VALUE_GAP_ABS, VALUE_GAP_REL)
        self.assertLess(small, 1.0)
        self.assertGreater(ordinary, 1.0)


@unittest.skipUnless(os.environ.get('RUN_DDP_TESTS')=='1','Set RUN_DDP_TESTS=1 for local loopback/Gloo tests.')
class DistributedRuntimeTests(unittest.TestCase):
    """Cross-rank initialization, uneven eligibility, and exact restart parity."""
    def test_two_rank_runtime(self):
        with socket.socket() as connection:
            connection.bind(('127.0.0.1',0))
            port=connection.getsockname()[1]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            mp.spawn(_worker,args=(port,directory),nprocs=2,join=True)


if __name__=='__main__':
    unittest.main()

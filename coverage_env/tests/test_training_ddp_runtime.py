"""Opt-in two-process CPU/Gloo regression tests; no UAV training is started."""
from copy import deepcopy
import datetime
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

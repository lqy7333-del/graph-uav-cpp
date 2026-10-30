"""Baselines share geometry/resources and the complete reward convention."""
import unittest
from unittest.mock import patch
from coverage_env.config import load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.trainer.episode import (instance_fingerprint, nearest_neighbor_return,
                                           current_potential, learning_reward)
from coverage_env.rl.trainer.rollout_baseline import _build_instance


class EpisodePairingTests(unittest.TestCase):
    """Catch reset-triggered resampling and capability distribution mismatches."""

    def test_nn_keeps_snapshot_and_does_not_mutate_live_environment(self):
        config = load_config()
        config['environment']['num_fields'] = {'min':5,'max':5}
        env = InstanceGenerator(config).generate(seed=42)
        env.reset()
        before = instance_fingerprint(env)
        from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice
        seen = []
        def choose(trial, state):
            if not seen:
                seen.append(instance_fingerprint(trial))
            return _nearest_field_choice(trial,state)
        with patch('coverage_env.rl.evaluation.validation_suite._nearest_field_choice',choose):
            value, fingerprint = nearest_neighbor_return(env,config,1.)
        self.assertEqual(seen,[before])
        self.assertEqual(fingerprint,before)
        self.assertEqual(instance_fingerprint(env),before)
        self.assertLess(value,0)

    def test_incomplete_baseline_is_rejected(self):
        env = InstanceGenerator(load_config()).generate(num_fields=5,seed=42)
        with self.assertRaises(RuntimeError):
            nearest_neighbor_return(env,{},1,max_steps=0)

    def test_fixed_instance_can_have_random_capabilities(self):
        config = load_config()
        # The project default is now a FIXED control scenario (randomize_uav=false);
        # this test exercises the randomized path explicitly.
        config["ppo"]["randomize_uav"] = True
        a = InstanceGenerator(config).generate(num_fields=5,seed=42,randomize_each_reset=False,randomize_uav=True)
        b = _build_instance(config,5,42)
        self.assertEqual(instance_fingerprint(a),instance_fingerprint(b))
        fingerprint = instance_fingerprint(b)
        b.reset()
        self.assertEqual(instance_fingerprint(b),fingerprint)
        self.assertNotEqual(b.uavs[0].max_energy,config['uav']['max_energy'])

    def test_capability_randomization_can_be_disabled_independently(self):
        config=load_config()
        config['environment']['num_fields']={'min':5,'max':5}
        env=InstanceGenerator(config).generate(seed=42,randomize_each_reset=True,randomize_uav=False)
        first=env.fields[0].area
        env.reset()
        self.assertNotEqual(env.fields[0].area,first)
        self.assertEqual(env.uavs[0].max_energy,config['uav']['max_energy'])

    def test_shaping_uses_current_reset_and_zero_terminal_potential(self):
        config=load_config()
        config['shaping']={'enable':True,'strength':2.}
        env=InstanceGenerator(config).generate(seed=42)
        env.reset()
        before=current_potential(env,config)
        self.assertNotEqual(before,0)
        self.assertAlmostEqual(learning_reward(env,-4,before,True,config,1),-4-before)

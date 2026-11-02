"""Bounded regression tests for fixed-bank isolation, evaluation and PPO wiring."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import torch

from coverage_env.config import load_config
from coverage_env.env.fixed_instance_pool import FixedInstancePool, write_bank, load_bank
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice
from coverage_env.rl.trainer.episode import instance_fingerprint
from coverage_env.rl.trainer.overfit import aggregate_rows, candidate_key, prepare, plot_curve, _worker


class FixedInstanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = load_config()
        cls.config["environment"]["num_fields"] = {"min": 2, "max": 2}
        cls.config["field_generator"]["layout"] = {"mode": "scattered"}
        cls.config["field_generator"]["field_size"] = {"min_area": 500.0, "max_area": 800.0}
        cls.config["ppo"].update(epochs=1, batch_size=2, value_gate=True,
                                 randomize_uav=False, gamma=1.0)
        cls.config["rl"]["reward"]["mode"] = "time"
        cls.config["rl"]["decoder"]["candidate_attention"] = False
        cls.envs = [InstanceGenerator(cls.config).generate(
            num_fields=2, seed=seed, randomize_each_reset=False, randomize_uav=False)
            for seed in (70000, 70001)]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bank = self.root / "bank"
        self.manifest = write_bank(self.bank, self.config, self.envs, [70000, 70001])

    def test_bank_roundtrip_preserves_exact_instances(self):
        manifest, envs = load_bank(self.bank)
        self.assertEqual(manifest["config"], self.config)
        self.assertEqual([instance_fingerprint(e) for e in envs],
                         [e["fingerprint"] for e in manifest["instances"]])
        self.assertNotEqual(manifest["instances"][0]["fingerprint"],
                            manifest["instances"][1]["fingerprint"])

    def test_bank_cannot_be_overwritten_or_tampered(self):
        with self.assertRaises(FileExistsError):
            write_bank(self.bank, self.config, self.envs, [70000, 70001])
        with (self.bank / "instances.pt").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_bank(self.bank)

    def test_every_cycle_visits_exactly_the_bank_without_observed_ids(self):
        pool = FixedInstancePool(self.bank, seed=43)
        for _cycle in range(3):
            seen = []
            for _ in range(2):
                state = pool.reset()
                seen.append(pool.current_index)
                self.assertNotIn("instance_id", state)
                self.assertNotIn("current_index", state)
                self.assertEqual(instance_fingerprint(pool._active),
                                 self.manifest["instances"][pool.current_index]["fingerprint"])
            self.assertEqual(sorted(seen), [0, 1])
        self.assertEqual(pool.starts, [3, 3])

    def test_real_step_then_reset_restores_coverage_resources_and_heading(self):
        pool = FixedInstancePool(self.bank, seed=42)
        pool.reset()
        index = pool.current_index
        _state, _reward, _done, info = pool.step(_nearest_field_choice(pool, pool.get_state()))
        self.assertGreater(info["coverage_gain"], 0)
        self.assertEqual(pool.transitions[index], 1)
        self.assertNotEqual(instance_fingerprint(pool._active),
                            self.manifest["instances"][index]["fingerprint"])
        pool.reset()
        pool.reset()  # Starts another cycle; identity must always be pristine.
        self.assertEqual(instance_fingerprint(pool._active),
                         self.manifest["instances"][pool.current_index]["fingerprint"])
        # Executing an episode never mutates the template bank.
        _, templates = load_bank(self.bank)
        self.assertEqual(instance_fingerprint(templates[index]),
                         self.manifest["instances"][index]["fingerprint"])

    def test_checkpoint_roundtrip_preserves_schedule_and_active_state(self):
        pool = FixedInstancePool(self.bank, seed=44)
        pool.reset()
        pool.step(_nearest_field_choice(pool, pool.get_state()))
        path = self.root / "runtime.pt"
        torch.save(pool, path)
        restored = torch.load(path, weights_only=False)
        self.assertIsNone(restored._templates)
        self.assertEqual(instance_fingerprint(restored._active), instance_fingerprint(pool._active))
        for _ in range(5):
            self.assertEqual(pool.reset(), restored.reset())
            self.assertEqual(pool.current_index, restored.current_index)
            self.assertEqual(pool.usage(), restored.usage())

    def test_bank_draws_do_not_advance_global_random_stream(self):
        pool = FixedInstancePool(self.bank)
        random.seed(123)
        before = random.getstate()
        pool.reset()
        self.assertEqual(before, random.getstate())

    def test_prepare_inherits_existing_checkpoint_objective_and_parameters(self):
        checkpoint = self.root / "initial.pt"
        torch.save({"resolved_config": self.config}, checkpoint)
        target = self.root / "prepared"
        prepare(SimpleNamespace(checkpoint=str(checkpoint), count=2, fields=2,
                                instance_seed=70002, bank=str(target)))
        manifest, _ = load_bank(target)
        self.assertEqual(manifest["config"]["rl"], self.config["rl"])
        self.assertEqual(manifest["config"]["uav"], self.config["uav"])
        self.assertEqual(manifest["config"]["ppo"], self.config["ppo"])
        self.assertEqual(manifest["initialization"]["checkpoint"], str(checkpoint.resolve()))

    def test_prepare_rejects_previous_test_seed_blocks(self):
        with self.assertRaisesRegex(ValueError, "prior validation"):
            prepare(SimpleNamespace(checkpoint=None, count=10, fields=20,
                                    instance_seed=60000, bank=str(self.root / "forbidden")))

    def test_finetune_update_zero_is_exactly_the_supplied_actor_and_critic(self):
        initial = PPOAgent(CoveragePolicy(None, config=self.config), self.config)
        initial.update_count = 600
        checkpoint = self.root / "pretrained.pt"
        initial.save_checkpoint(checkpoint)
        bank = self.root / "finetune_bank"
        prepare(SimpleNamespace(checkpoint=str(checkpoint), count=2, fields=2,
                                instance_seed=70002, bank=str(bank)))
        output = self.root / "finetune"
        output.mkdir()
        _worker(0, 1, SimpleNamespace(bank=str(bank), output=str(output), seed=42,
                num_updates=1, rollout_steps=2, eval_every=1, resume=None, device="cpu"))
        zero = torch.load(output / "ckpt_update_0.pt", weights_only=False)
        original = torch.load(checkpoint, weights_only=False)
        for module in ("actor", "critic"):
            for key, value in original[module].items():
                torch.testing.assert_close(value, zero[module][key], rtol=0, atol=0)
        self.assertEqual(zero["update_count"], 0)
        self.assertFalse(zero["actor_optimizer"]["state"])
        self.assertEqual(zero["resolved_config"]["rl"], original["resolved_config"]["rl"])

    def test_branch_changes_only_actor_lr_and_stop_at_is_bounded(self):
        # Tiny CPU run: create a source checkpoint at update 1, branch with a
        # lower actor LR, and separately verify --stop-at leaves no later file.
        source = self.root / "source"
        source.mkdir()
        base = SimpleNamespace(bank=str(self.bank), output=str(source), seed=42,
                               num_updates=2, rollout_steps=2, eval_every=1,
                               resume=None, device="cpu", branch_from=None,
                               actor_lr=None, branch_label=None, stop_at=None)
        _worker(0, 1, base)
        branch = self.root / "branch"
        branch.mkdir()
        branched = SimpleNamespace(bank=str(self.bank), output=str(branch), seed=42,
                                   num_updates=2, rollout_steps=2, eval_every=1,
                                   resume=None, device="cpu",
                                   branch_from=str(source / "ckpt_update_1.pt"),
                                   actor_lr=1e-4, branch_label="unit",
                                   stop_at=None)
        _worker(0, 1, branched)
        checkpoint = torch.load(branch / "ckpt_update_2.pt", weights_only=False)
        self.assertEqual(checkpoint["update_count"], 2)
        self.assertEqual(checkpoint["actor_optimizer"]["param_groups"][0]["lr"], 1e-4)
        self.assertEqual(checkpoint["training_metadata"]["branch_actor_lr"], 1e-4)
        self.assertEqual(checkpoint["training_metadata"]["branch_source_update"], 1)
        stopped = self.root / "stopped"
        stopped.mkdir()
        bounded = SimpleNamespace(bank=str(self.bank), output=str(stopped), seed=42,
                                  num_updates=2, rollout_steps=2, eval_every=1,
                                  resume=None, device="cpu", branch_from=None,
                                  actor_lr=None, branch_label=None, stop_at=1)
        _worker(0, 1, bounded)
        self.assertTrue((stopped / "ckpt_update_1.pt").exists())
        self.assertFalse((stopped / "ckpt_update_2.pt").exists())

    def test_checkpoint_rejects_modified_manifest(self):
        restored = deepcopy(FixedInstancePool(self.bank))
        path = self.bank / "manifest.json"
        payload = json.loads(path.read_text())
        payload["purpose"] = "changed"
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            restored.reset()

    def test_real_short_training_has_gate_baseline_update_and_resume(self):
        # Two tiny updates (4 transitions total), not a farm-training experiment.
        output = self.root / "smoke"
        output.mkdir()
        args = SimpleNamespace(bank=str(self.bank), output=str(output), seed=42,
                               num_updates=2, rollout_steps=2, eval_every=1,
                               resume=None, device="cpu")
        _worker(0, 1, args)
        first = torch.load(output / "ckpt_update_0.pt", weights_only=False)
        last = torch.load(output / "ckpt_update_2.pt", weights_only=False)
        self.assertTrue(any(not torch.equal(first["actor"][k], last["actor"][k])
                            for k in first["actor"]))
        self.assertTrue(last["actor_optimizer"]["state"])
        self.assertIn("True", (output / "train_log.csv").read_text())
        refs = json.loads((output / "references.json").read_text())
        self.assertEqual(len(refs["nn"]), 2)
        self.assertTrue(all(r["completed"] for r in refs["nn"]))
        # Resume at 1 within the same declared 2-update budget; compare parameters.
        args.resume = str(output / "ckpt_update_1.pt")
        _worker(0, 1, args)
        resumed = torch.load(output / "ckpt_update_2.pt", weights_only=False)
        for module in ("actor", "critic"):
            for key, value in last[module].items():
                torch.testing.assert_close(value, resumed[module][key], rtol=0, atol=0)
        self.assertEqual(last["rank_runtime"][0]["environment"].usage(),
                         resumed["rank_runtime"][0]["environment"].usage())
        plot_curve(output)
        self.assertTrue((output / "learning_curve.png").is_file())


class OverfitMetricTests(unittest.TestCase):
    @staticmethod
    def row(index, time, complete=True):
        return dict(index=index, completed=complete, total_time=time, failed_steps=0,
                    total_distance=time * 8, total_energy=time, coverage_ratio=1,
                    visits=2, resupplies=0, time_work_base=time, time_nonwork_base=0,
                    time_maneuver_extra=0, time_service=0)

    def test_improvement_is_baseline_minus_candidate_with_per_instance_ratios(self):
        baseline = [self.row(0, 100), self.row(1, 200)]
        candidate = [self.row(0, 90), self.row(1, 160)]
        stats = aggregate_rows(candidate, {"initial": baseline, "nn": baseline})
        self.assertEqual(stats["nn_minus_candidate_total_time"], 25)
        self.assertEqual(stats["nn_time_improvement_percent"], 15)
        self.assertEqual(stats["nn_wins"], 2)

    def test_incomplete_candidates_do_not_win_by_cheap_failure(self):
        failed = aggregate_rows([self.row(0, 1, False), self.row(1, 1)])
        complete = aggregate_rows([self.row(0, 100), self.row(1, 200)])
        failed["update"], complete["update"] = 20, 0
        self.assertIsNone(failed["total_time"])
        self.assertLess(candidate_key(complete), candidate_key(failed))

    def test_mismatched_reference_instances_are_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_rows([self.row(0, 100)], {"nn": [self.row(1, 100)]})


if __name__ == "__main__":
    unittest.main()

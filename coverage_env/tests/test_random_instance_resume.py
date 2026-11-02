"""Regression tests for strict random-instance resume and generation failures."""
import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from coverage_env.env.instance_generator import InstanceGenerator, GeometryGenerationError
from coverage_env.rl.trainer.episode_random import RandomInstanceEpisodeTrainer, RandomInstanceStream
from coverage_env.rl.trainer.episode_random_worker import (
    _RUN_DEFAULTS, resolve_run_arguments, restore_panel_history, validate_optimizer_lrs,
)


def saved_config():
    return {'training_metadata': {**_RUN_DEFAULTS, 'collection_mode': 'complete_episodes'},
            'resolved_config': {'ppo': {'actor_learning_rate': 1e-4,
                                       'critic_learning_rate': .003, 'lr_schedule': 'none'}}}


def args(**kwargs):
    return argparse.Namespace(**{**dict.fromkeys(_RUN_DEFAULTS), **kwargs})


def test_resume_inherits_and_rejects_each_explicit_override():
    saved = saved_config()
    omitted = args()
    resolve_run_arguments(omitted, saved)
    assert vars(omitted) == _RUN_DEFAULTS
    for name, value in _RUN_DEFAULTS.items():
        with pytest.raises(ValueError, match=name):
            resolve_run_arguments(args(**{name: value + 1}), saved)
    fresh = args()
    resolve_run_arguments(fresh)
    assert vars(fresh) == _RUN_DEFAULTS


def test_loaded_optimizer_lr_must_match_config_and_logged_lr():
    config = saved_config()['resolved_config']
    agent = SimpleNamespace(actor_optimizer=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-4),
                            critic_optimizer=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=.003))
    validate_optimizer_lrs(agent, config)
    agent.actor_optimizer.param_groups[0]['lr'] = 3e-4
    with pytest.raises(ValueError, match='actor optimizer LR'):
        validate_optimizer_lrs(agent, config)


def panel_history(root):
    for panel in ('diagnostic', 'validation'):
        folder = root / panel
        (folder / 'evaluation').mkdir(parents=True)
        (folder / 'references.json').write_text('{}')
        for update, cost in [(0, 12), (10, 8), (20, 11), (30, 1)]:
            summary = dict(update=update, total_time=cost, completed=2, instances=2)
            (folder / 'evaluation' / f'update_{update}.json').write_text(json.dumps({'summary': summary}))
            (root / f'ckpt_update_{update}.pt').write_bytes(f'weights{update}'.encode())


def test_cross_directory_restore_retains_historical_best_and_weights(tmp_path):
    source, out = tmp_path / 'old', tmp_path / 'new'
    panel_history(source)
    restore_panel_history(source, out, 20)
    for panel in ('diagnostic', 'validation'):
        best = json.loads((out / panel / 'best.json').read_text())
        assert best['update'] == 10
        assert not (out / panel / 'evaluation/update_30.json').exists()
    assert (out / 'ckpt_update_10.pt').read_bytes() == b'weights10'
    # Simulate a worse post-resume evaluation. Selection must still include history.
    for panel in ('diagnostic', 'validation'):
        path = out / panel / 'evaluation/update_40.json'
        path.write_text(json.dumps({'summary': dict(update=40, total_time=10, completed=2, instances=2)}))
    restore_panel_history(out, out, 40)
    assert json.loads((out / 'validation/best.json').read_text())['update'] == 10


def test_same_directory_rewind_rejected(tmp_path):
    panel_history(tmp_path)
    with pytest.raises(ValueError, match='newer than checkpoint'):
        restore_panel_history(tmp_path, tmp_path, 20)


def test_missing_selected_weights_rejected(tmp_path):
    source = tmp_path / 'old'
    panel_history(source)
    (source / 'ckpt_update_10.pt').unlink()
    with pytest.raises(ValueError, match='missing historical best'):
        restore_panel_history(source, tmp_path / 'new', 20)


def trainer_stub(tmp_path):
    trainer = object.__new__(RandomInstanceEpisodeTrainer)
    trainer.num_fields = 10
    trainer.stream = RandomInstanceStream(900_000_000, 0)
    trainer.rank = 0
    trainer._current_update = 1
    trainer._episode_seq = 0
    trainer.checkpoint_dir = tmp_path
    trainer.generator = Mock()
    return trainer


def test_retryable_generation_records_seed_reason_and_recovers(tmp_path, monkeypatch):
    trainer = trainer_stub(tmp_path)
    env = SimpleNamespace(fields=[SimpleNamespace(area=6000)] * 10)
    trainer.generator.generate.side_effect = [GeometryGenerationError('crowded geometry'), env]
    monkeypatch.setattr('coverage_env.rl.trainer.episode_random.instance_fingerprint', lambda e: 'fingerprint')
    assert trainer._new_environment() is env
    assert trainer._episode_meta['instance_seed'] == 900_000_001
    row = json.loads((tmp_path / 'generation_retries_rank_0.jsonl').read_text())
    assert row['instance_seed'] == 900_000_000
    assert row['errors'][0] == {'type': 'GeometryGenerationError', 'message': 'crowded geometry'}
    assert trainer.stream.counter == 1


@pytest.mark.parametrize('error', [ValueError('bad config'), RuntimeError('program error'), KeyError('missing')])
def test_programming_errors_are_not_retried(tmp_path, error):
    trainer = trainer_stub(tmp_path)
    trainer.generator.generate.side_effect = error
    with pytest.raises(type(error)):
        trainer._new_environment()
    assert trainer.generator.generate.call_count == 1
    assert trainer.stream.retries == 0


def test_generator_internal_retry_does_not_swallow_runtime_bug():
    generator = object.__new__(InstanceGenerator)
    generator._sample_components_once = Mock(side_effect=RuntimeError('bug'))
    with pytest.raises(RuntimeError, match='bug'):
        generator.sample_components(None, 10)
    assert generator._sample_components_once.call_count == 1

def test_cpu_worker_cross_directory_resume_matches_continuous(tmp_path, monkeypatch):
    from coverage_env.config import load_config
    from coverage_env.env.fixed_instance_pool import write_bank
    from coverage_env.rl.trainer.episode_random_worker import _worker
    config = load_config()
    config['environment']['num_fields'] = {'min': 10, 'max': 10}
    config['field_generator']['layout'] = {'mode': 'scattered'}
    config['field_generator']['field_size'] = {'min_area': 500., 'max_area': 800.}
    config['ppo'].update(epochs=1, batch_size=64, gamma=1., lr_schedule='none', randomize_uav=False)
    config['rl']['reward']['mode'] = 'time'
    config['rl']['decoder']['candidate_attention'] = False
    env = InstanceGenerator(config).generate(num_fields=10, seed=81000, randomize_each_reset=False, randomize_uav=False)
    bank = tmp_path / 'bank'
    write_bank(bank, config, [env], [81000])
    def run_args(output, resume=None):
        return argparse.Namespace(**{**_RUN_DEFAULTS, 'num_updates': 2, 'episodes_per_rank': 1,
            'device': 'cpu', 'output': str(output), 'resume': str(resume) if resume else None,
            'diagnostic_bank': str(bank), 'validation_bank': str(bank), 'checkpoint_every': 1})
    full, partial, resumed = (tmp_path / x for x in ('full', 'partial', 'resumed'))
    _worker(0, 1, run_args(full))
    original = RandomInstanceEpisodeTrainer.train
    def stop_first(self, *args, **kwargs):
        return original(self, *args, **kwargs, stop_at=1)
    with monkeypatch.context() as patch:
        patch.setattr(RandomInstanceEpisodeTrainer, 'train', stop_first)
        _worker(0, 1, run_args(partial))
    _worker(0, 1, run_args(resumed, partial / 'ckpt_update_1.pt'))
    a = torch.load(full / 'ckpt_update_2.pt', weights_only=False)
    b = torch.load(resumed / 'ckpt_update_2.pt', weights_only=False)
    for model in ('actor', 'critic'):
        assert all(torch.equal(a[model][key], b[model][key]) for key in a[model])
    assert a['rank_runtime'][0]['instance_stream'] == b['rank_runtime'][0]['instance_stream']
    assert a['training_metadata']['initial_actor_sha256'] == b['training_metadata']['initial_actor_sha256']
    for panel in ('diagnostic', 'validation'):
        assert json.loads((full / panel / 'best.json').read_text()) == json.loads((resumed / panel / 'best.json').read_text())
    assert b['actor_optimizer']['param_groups'][0]['lr'] == 1e-4
    assert b['critic_optimizer']['param_groups'][0]['lr'] == .003

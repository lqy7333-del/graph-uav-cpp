"""Strict paired initialization for the CHANet vs node-MLP experiment."""
import copy
from pathlib import Path
import torch
from coverage_env.env.fixed_instance_pool import file_sha256


def initialize_from_a_zero(agent, path, config, expected):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('update_idx') != 0 or saved.get('update_count') != 0:
        raise ValueError('paired initialization requires A update 0, never trained weights')
    if saved['training_metadata'].get('initialization') is not None:
        raise ValueError('A must start from random initialization')
    for key, value in expected.items():
        if saved['training_metadata'].get(key) != value:
            raise ValueError(f'A reference metadata mismatch: {key}')
    a, b = copy.deepcopy(saved['resolved_config']), copy.deepcopy(config)
    if a['rl']['encoder'].get('variant', 'chanet') != 'chanet':
        raise ValueError('reference encoder must be CHANet')
    variant = b['rl']['encoder'].get('variant', 'chanet')
    for cfg in (a, b):
        cfg['rl']['encoder'].pop('variant', None)
    if a != b:
        raise ValueError('A/B resolved configs differ beyond encoder variant')
    target = agent.actor.state_dict()
    if variant == 'node_mlp':
        prefixes = ('encoder.feature_embedding.', 'action_embedding.',
                    'global_state_encoder.', 'hierarchical_decoder.', 'action_decoder.')
        shared = [key for key in target if key.startswith(prefixes)]
        replaced = ('encoder',)
    elif variant == 'no_cross_field_boundary':
        replaced = ('encoder.boundary_inter_encoder.',)
        shared = [key for key in target if not key.startswith(replaced)]
    elif variant == 'no_intra_boundary':
        replaced = ('encoder.boundary_intra_encoder.',)
        shared = [key for key in target if not key.startswith(replaced)]
    else:
        raise ValueError(f'unsupported paired variant: {variant}')
    for key in shared:
        if key not in saved['actor'] or target[key].shape != saved['actor'][key].shape:
            raise ValueError(f'common weight incompatible: {key}')
        target[key] = saved['actor'][key]
    # Strict load catches omitted/extra target keys; replaced modules are the
    # only target tensors intentionally not copied from A.
    agent.actor.load_state_dict(target, strict=True)
    agent.critic.load_state_dict(saved['critic'], strict=True)
    if agent.actor_optimizer.state or agent.critic_optimizer.state:
        raise ValueError('paired scratch initialization requires new optimizers')
    return dict(reference_path=str(Path(path).resolve()), reference_sha256=file_sha256(path),
                copied_actor_tensors=len(shared), copied_critic=True,
                source_update=0, encoder_variant=variant,
                replaced_modules=list(replaced),
                reference_actor_parameters=sum(v.numel() for v in saved['actor'].values()),
                actor_parameters=sum(p.numel() for p in agent.actor.parameters()),
                encoder_parameters=sum(p.numel() for p in agent.actor.encoder.parameters()),
                reference_encoder_parameters=sum(v.numel() for k, v in saved['actor'].items()
                                                 if k.startswith('encoder.')))

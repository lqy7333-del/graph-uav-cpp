"""Small checks for the same-snapshot nested-GA control adapter."""
import copy
from coverage_env.config import load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.evaluation.nested_ga_controls import initializer_choice


def test_initializer_choice_is_a_feasible_action():
    c = load_config()
    c['environment']['num_fields'] = {'min': 10, 'max': 10}
    c['field_generator']['layout'] = {'mode': 'scattered'}
    c['field_generator']['field_size'] = {'min_area': 500.0, 'max_area': 800.0}
    env = InstanceGenerator(c).generate(num_fields=10, seed=1400100,
                                        randomize_each_reset=False, randomize_uav=False)
    env.reset()
    candidates = env.get_action_candidates()
    mask = env.get_action_mask(candidates)
    action, mask_s, infer_s = initializer_choice(env, env.get_state())
    assert action is not None
    assert action in [a for a, ok in zip(candidates, mask) if ok]
    assert mask_s == 0.0 and infer_s == 0.0
    state, _, done, _ = env.step(action)
    assert state is not None

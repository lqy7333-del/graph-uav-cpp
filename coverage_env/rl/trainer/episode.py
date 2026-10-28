"""Episode pairing and reward semantics shared by training entry points."""
from copy import deepcopy
import hashlib
import pickle

from coverage_env.rl.shaping import potential, shaped_reward


def instance_fingerprint(environment):
    """Identify geometry, resources, coverage and depot of the current snapshot."""
    geometry = [(f.field_id, f.xy_polygon.wkb) for f in environment.fields]
    payload = (geometry, environment.get_state())
    return hashlib.sha256(pickle.dumps(payload, protocol=4)).hexdigest()


def learning_reward(environment, reward, before, done, config, gamma):
    """Shape with zero terminal potential, including unsuccessful terminal states."""
    shaping = config.get("shaping", {})
    if not shaping.get("enable", False):
        return float(reward)
    after = 0.0 if done else potential(environment, float(shaping.get("strength", 1.0)), environment.total_area)
    return shaped_reward(reward, before, after, gamma)


def current_potential(environment, config):
    """Compute potential from the current (possibly newly reset) episode."""
    shaping = config.get("shaping", {})
    return (potential(environment, float(shaping.get("strength", 1.0)), environment.total_area)
            if shaping.get("enable", False) else 0.0)


def nearest_neighbor_return(environment, config, gamma, *, max_steps=300):
    """Complete the *current* snapshot without resetting or touching the live env.

    A failed or truncated baseline has no valid complete-return label and raises.
    """
    from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice
    trial = deepcopy(environment)
    fingerprint = instance_fingerprint(environment)
    if instance_fingerprint(trial) != fingerprint:
        raise RuntimeError("Nearest-neighbor baseline snapshot differs from the sampled episode.")
    total, discount = 0.0, 1.0
    for _ in range(max_steps):
        action = _nearest_field_choice(trial, trial.get_state())
        if action is None:
            raise RuntimeError("Nearest-neighbor baseline has no feasible completion.")
        before = current_potential(trial, config)
        _, reward, done, info = trial.step(action)
        total += discount * learning_reward(trial, reward, before, done, config, gamma)
        discount *= gamma
        if done:
            if info.get("termination_reason") != "mission_complete" or info.get("final_return_failed"):
                raise RuntimeError("Nearest-neighbor baseline did not complete the mission.")
            return total, fingerprint
    raise RuntimeError("Nearest-neighbor baseline reached its step limit without completing.")

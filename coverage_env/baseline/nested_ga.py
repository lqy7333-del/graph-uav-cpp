"""Nested GA: repeated field visits outside, feasible gate random keys inside.

Independent of learned policies. All fitness evaluations use the real executor.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from math import hypot, isfinite
import random
import time



@dataclass(frozen=True)
class GAConfig:
    outer_population: int = 8
    outer_generations: int = 4
    inner_population: int = 8
    inner_generations: int = 2
    max_evaluations: int = 500
    wall_seconds: float = 300.0
    max_steps: int = 500
    seed: int = 42

    def __post_init__(self):
        for key in ('outer_population', 'inner_population'):
            if getattr(self, key) < 2:
                raise ValueError(f'{key} must be >= 2')
        for key in ('outer_generations', 'inner_generations', 'max_evaluations', 'max_steps'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if not isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError('wall_seconds must be finite and positive')


def action_key(action):
    return (action.field_id, action.entry_boundary_id, action.exit_boundary_id)


def nearest_action(env, actions):
    """Action-space nearest heuristic; NOT the legacy NN workload heuristic."""
    from shapely.geometry import LineString, Point
    position = env.uavs[0].current_position
    fields = {f.field_id: f for f in env.fields}
    def score(a):
        f = fields[a.field_id]
        boundary = next(b for b in f.boundaries if b.id == a.entry_boundary_id)
        return (hypot(f.centroid[0]-position[0], f.centroid[1]-position[1]),
                LineString([boundary.start_point, boundary.end_point]).distance(Point(position)),
                a.entry_boundary_id != a.exit_boundary_id, action_key(a))
    return min(actions, key=score)


def summary_values(env):
    s = env.get_summary()
    keys = ('total_time', 'total_distance', 'time_work_base', 'time_nonwork_base',
            'time_maneuver_extra', 'time_service', 'coverage_ratio', 'resupply_count',
            'execution_count', 'returned_to_depot', 'time_closure_error')
    result = {k: s[k] for k in keys}
    if any(not isfinite(float(v)) for v in result.values()):
        raise ValueError('Non-finite execution ledger')
    if abs(result['time_closure_error']) > 1e-6:
        raise ValueError('Time ledger does not close')
    return result


def replay(initial, trace, expected_time, max_steps=500):
    """Replay exact actions, without GA repair or completion."""
    env = deepcopy(initial)
    done = False
    info = {}
    if len(trace) > max_steps:
        raise ValueError('Trace exceeds step limit')
    for index, key in enumerate(trace):
        actions = env.get_action_candidates()
        mask = env.get_action_mask(actions)
        matches = [a for a, valid in zip(actions, mask) if bool(valid) and action_key(a) == tuple(key)]
        if not matches:
            raise ValueError(f'Invalid replay action at {index}')
        _, _, done, info = env.step(matches[0])
        if not info.get('success', False):
            raise ValueError(f'Replay execution failed at {index}')
        if done and index != len(trace)-1:
            raise ValueError('Trailing actions after terminal state')
    s = summary_values(env)
    if not done or info.get('termination_reason') != 'mission_complete' or not s['returned_to_depot']:
        raise ValueError('Replay did not complete and return')
    if abs(s['total_time']-expected_time) > 1e-6:
        raise ValueError('Replay cost mismatch')
    return s


class NestedGA:
    def __init__(self, initial, config=GAConfig()):
        # Caller supplies a reset, fixed instance. Never reset/re-generate during search.
        if initial.get_summary()['execution_count'] != 0:
            raise ValueError('Expected initial, unexecuted environment')
        self.initial = deepcopy(initial)
        self.config = config
        self.rng = random.Random(config.seed)
        self.evaluations = 0
        self.transitions = 0
        self.history = []
        self.incumbent = None
        self.started = None
        self.inner_records = []

    def expired(self):
        return time.monotonic()-self.started >= self.config.wall_seconds

    def evaluate(self, order, gates, label):
        if self.evaluations >= self.config.max_evaluations or self.expired():
            return None
        self.evaluations += 1  # failed/partial candidates consume budget too
        env = deepcopy(self.initial)
        trace, used_keys = [], []
        cursor = 0
        skipped = fallback = 0
        reason = 'step_limit'
        complete = False
        for _ in range(self.config.max_steps):
            if self.expired():
                reason = 'wall_limit'
                break
            actions = env.get_action_candidates()
            mask = env.get_action_mask(actions)
            feasible = sorted([a for a, ok in zip(actions, mask) if bool(ok)], key=action_key)
            if not feasible:
                reason = 'no_feasible_action'
                break
            chosen = None
            while cursor < len(order):
                slot = cursor
                field = order[cursor]
                cursor += 1
                choices = [a for a in feasible if a.field_id == field]
                if not choices:
                    skipped += 1
                    continue
                key = gates[slot]
                chosen = nearest_action(env, choices) if key < 0 else choices[min(int(key*len(choices)), len(choices)-1)]
                break
            if chosen is None:
                chosen = nearest_action(env, feasible)
                choices = [a for a in feasible if a.field_id == chosen.field_id]
                fallback += 1
            # Encode exact selected gate as a bin midpoint for warm-start replay.
            used_keys.append((choices.index(chosen)+0.5)/len(choices))
            trace.append(action_key(chosen))
            _, _, done, info = env.step(chosen)
            self.transitions += 1
            if not info.get('success', False):
                reason = info.get('termination_reason', 'execution_failure')
                break
            if done:
                reason = info.get('termination_reason', 'unknown_terminal')
                complete = reason == 'mission_complete'
                break
        s = summary_values(env)
        complete = complete and s['returned_to_depot']
        result = dict(completed=bool(complete), reason=reason, cost=s['total_time'] if complete else None,
                      actions=trace, gate_keys=used_keys, chromosome_order=list(order),
                      chromosome_gates=list(gates), summary=s, skipped_slots=skipped,
                      fallback_steps=fallback, label=label)
        if complete and (self.incumbent is None or result['cost'] < self.incumbent['cost']):
            self.incumbent = deepcopy(result)
        self.history.append(dict(evaluation=self.evaluations, seconds=time.monotonic()-self.started,
                                 completed=bool(complete), cost=result['cost'], reason=reason,
                                 fallback_steps=fallback, skipped_slots=skipped, steps=len(trace),
                                 label=label, best=self.incumbent['cost'] if self.incumbent else None))
        return result

    @staticmethod
    def fitness(result):
        return result['cost'] if result and result['completed'] else float('inf')

    def inner(self, order, seed_gates):
        p = self.config.inner_population
        population = [tuple(seed_gates), tuple(-1.0 for _ in order)]
        population += [tuple(self.rng.random() for _ in order) for _ in range(p-2)]
        champion, champion_gates = None, None
        count_before = self.evaluations
        for generation in range(self.config.inner_generations):
            scored = []
            for gates in population:
                result = self.evaluate(order, gates, 'inner')
                if result is None:
                    return None  # incomplete quotas never enter outer selection
                scored.append((self.fitness(result), gates))
                if champion is None or self.fitness(result) < self.fitness(champion):
                    champion, champion_gates = result, gates
            scored.sort(key=lambda item: item[0])
            if generation+1 < self.config.inner_generations:
                population = [scored[0][1]]
                while len(population) < p:
                    a, b = (self.rng.choice(scored[:max(2, p//2)])[1] for _ in range(2))
                    child = [x if self.rng.random()<0.5 else y for x,y in zip(a,b)]
                    child[self.rng.randrange(len(child))] = self.rng.random()
                    population.append(tuple(child))
        self.inner_records.append(dict(evaluations=self.evaluations-count_before,
                                       cost=champion['cost'], completed=champion['completed']))
        return order, champion_gates, champion

    def mutate_outer(self, a, b):
        # One-point splice supports repeated fields; carry aligned gate seeds.
        cut_a = self.rng.randrange(len(a[0])+1)
        cut_b = self.rng.randrange(len(b[0])+1)
        order = list(a[0][:cut_a]+b[0][cut_b:])
        gates = list(a[1][:cut_a]+b[1][cut_b:])
        if not order:
            order, gates = [self.rng.choice(self.fields)], [-1.0]
        op = self.rng.choice(('swap', 'insert', 'delete', 'replace'))
        i = self.rng.randrange(len(order))
        if op == 'swap':
            j = self.rng.randrange(len(order))
            order[i], order[j] = order[j], order[i]
            gates[i], gates[j] = gates[j], gates[i]
        elif op == 'insert':
            order.insert(i, self.rng.choice(self.fields)); gates.insert(i, self.rng.random())
        elif op == 'delete' and len(order)>1:
            order.pop(i); gates.pop(i)
        elif op == 'replace':
            order[i] = self.rng.choice(self.fields); gates[i] = self.rng.random()
        return tuple(order[:self.config.max_steps]), tuple(gates[:self.config.max_steps])

    def run(self):
        if self.started is not None:
            raise RuntimeError('Search objects are single-use')
        self.started = time.monotonic()
        self.fields = tuple(sorted(f.field_id for f in self.initial.fields))
        baseline = self.evaluate((), (), 'action_space_nearest_initialization')
        if not baseline or not baseline['completed']:
            raise RuntimeError('Nearest initialization failed; no complete incumbent')
        root = (tuple(a[0] for a in baseline['actions']), tuple(baseline['gate_keys']))
        population = [root]
        while len(population) < self.config.outer_population:
            population.append(self.mutate_outer(root, root))
        for _ in range(self.config.outer_generations):
            scored = []
            for order, gates in population:
                result = self.inner(order, gates)
                if result is None:
                    break
                scored.append(result)
            if len(scored) != self.config.outer_population or self.expired():
                break
            scored.sort(key=lambda item: self.fitness(item[2]))
            population = [(scored[0][0], scored[0][1])]
            while len(population) < self.config.outer_population:
                a,b = (self.rng.choice(scored[:max(2,len(scored)//2)]) for _ in range(2))
                population.append(self.mutate_outer(a,b))
        elapsed = time.monotonic()-self.started
        verified = replay(self.initial, self.incumbent['actions'], self.incumbent['cost'], self.config.max_steps)
        return dict(config=asdict(self.config), baseline=baseline, best=self.incumbent,
                    evaluations=self.evaluations, transitions=self.transitions, history=self.history,
                    inner_quotas=self.inner_records, search_seconds=elapsed,
                    replay_seconds=time.monotonic()-self.started-elapsed, replay_summary=verified,
                    termination=('wall_limit' if elapsed>=self.config.wall_seconds else
                                 'evaluation_limit' if self.evaluations>=self.config.max_evaluations else 'generation_limit'))

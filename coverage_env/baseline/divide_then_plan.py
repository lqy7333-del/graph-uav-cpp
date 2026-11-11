"""Resource-feasible frozen partition followed by nested permutation/gate GA."""
from copy import deepcopy
from dataclasses import asdict
from math import hypot
import time

from coverage_env.baseline.nested_ga import GAConfig, NestedGA, summary_values
from coverage_env.planner.configuration_executor import ConfigurationExecutor, InsufficientResources
from coverage_env.planner.framework_contracts import RegionTask, validate_partition
from coverage_env.planner.framework_execution import replay_regions


class PartitionFailure(ValueError):
    pass


class BudgetExpired(Exception):
    pass


def strip_order(field):
    corners = list(field.xy_polygon.minimum_rotated_rectangle.exterior.coords)
    axes = []
    for a, b in zip(corners, corners[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = hypot(dx, dy)
        if length == 0:
            continue
        if dx < 0 or (dx == 0 and dy < 0): dx, dy = -dx, -dy
        axes.append((-round(length, 8), dx/length, dy/length))
    if not axes: raise PartitionFailure('Degenerate field axis')
    _, x, y = min(axes)
    return tuple(p.id for p in sorted(field.grid_points,
        key=lambda p: (-y*p.x+x*p.y, x*p.x+y*p.y, p.id)))


def components(ids, points):
    """Deterministic connected components, preserving projection order."""
    remaining = set(ids)
    result = []
    for root in ids:
        if root not in remaining: continue
        found, pending = set(), [root]
        while pending:
            pid = pending.pop()
            if pid not in remaining: continue
            remaining.remove(pid); found.add(pid)
            pending.extend(p for p in points[pid].neighbors if p in remaining)
        result.append(tuple(p for p in ids if p in found))
    return result


class DivideThenPlanGA(NestedGA):
    """Reuse inner breeding/fitness; replace only task encoding and evaluation.

    All partition work is inside run's wall budget. No CC fallback or free retry.
    """
    def __init__(self, initial, config=GAConfig()):
        super().__init__(initial, config)
        if len(initial.uavs) != 1 or any(p.covered for f in initial.fields for p in f.grid_points):
            raise ValueError('Expected a fresh single-UAV snapshot')
        u = initial.uavs[0]
        if (u.current_position != initial.depot.position or u.remaining_energy != u.max_energy
                or u.remaining_payload != u.max_payload):
            raise ValueError('Expected full resources at depot')
        self.tasks = ()
        self.gate_options = {}
        self.partition_records = []
        self.probes = 0

    def check_wall(self):
        if self.expired(): raise BudgetExpired()

    def partition(self):
        executor = ConfigurationExecutor()
        tasks = []
        for field in self.initial.fields:
            points = {p.id: p for p in field.grid_points}
            if not points: raise PartitionFailure('Empty grid')
            pending = list(reversed(components(strip_order(field), points)))
            while pending:
                self.check_wall()
                ids = pending.pop()
                candidate = RegionTask(f'task_{len(tasks)}', field.field_id, ids)
                options, errors = [], []
                # Payload is an exact necessary condition; avoid redundant geometry probes.
                area = sum(points[p].point_area for p in ids)
                if area*self.initial.uavs[0].payload_per_area <= self.initial.uavs[0].max_payload:
                    for entry in sorted(b.id for b in field.boundaries):
                        for exit in sorted(b.id for b in field.boundaries):
                            self.check_wall(); self.probes += 1
                            try:
                                plan = executor.plan_region(candidate, entry, exit, self.initial)
                                executor.validate_plan(plan, self.initial)
                                options.append((entry, exit))
                            except ValueError as exc:
                                errors.append(str(exc))
                else:
                    errors.append('payload necessary bound')
                self.partition_records.append(dict(field_id=field.field_id, point_ids=ids,
                    feasible_gates=options, rejected_gate_reasons=errors))
                self.check_wall()
                if options:
                    tasks.append(candidate)
                    self.gate_options[candidate.task_id] = tuple(options)
                    if len(tasks) > self.config.max_steps:
                        raise PartitionFailure('Partition exceeds step limit')
                    continue
                if len(ids) == 1:
                    raise PartitionFailure(f'Infeasible singleton {field.field_id}/{ids[0]}')
                cumulative = 0.; cuts = []
                for index, pid in enumerate(ids[:-1], 1):
                    cumulative += points[pid].point_area
                    cuts.append((abs(cumulative-area/2), index))
                cut = min(cuts)[1]
                children = components(ids[:cut], points)+components(ids[cut:], points)
                pending.extend(reversed(children))
        validate_partition(tasks, {f.field_id: tuple(p.id for p in f.grid_points) for f in self.initial.fields})
        self.tasks = tuple(tasks)
        self.by_task = {t.task_id: t for t in tasks}

    def nearest_gate(self, env, task_id):
        from shapely.geometry import LineString, Point
        field = env.get_field(self.by_task[task_id].field_id)
        def key(pair):
            edge = field.get_boundary(pair[0])
            return (LineString([edge.start_point, edge.end_point]).distance(Point(env.uavs[0].current_position)), pair)
        return min(self.gate_options[task_id], key=key)

    def evaluate(self, order, gates, label):
        if self.evaluations >= self.config.max_evaluations or self.expired(): return None
        if len(order) != len(self.tasks) or set(order) != set(self.by_task) or len(gates) != len(order):
            raise ValueError('Chromosome must contain every frozen task exactly once')
        self.evaluations += 1
        env = deepcopy(self.initial)
        trace, used = [], []
        complete, reason = False, 'step_limit'
        for task_id, key in zip(order, gates):
            if self.expired(): reason = 'wall_limit'; break
            choices = self.gate_options[task_id]
            pair = self.nearest_gate(env, task_id) if key < 0 else choices[min(int(key*len(choices)), len(choices)-1)]
            used.append((choices.index(pair)+.5)/len(choices))
            try:
                _, _, done, info = env.step_region(self.by_task[task_id], *pair)
            except InsufficientResources as exc:
                reason = 'resource_infeasible: '+str(exc); break
            # Geometry/programming/ledger errors propagate: don't hide implementation defects.
            self.transitions += 1
            trace.append(info['framework_trace'])
            if done:
                complete = len(trace) == len(order) and env.get_summary()['returned_to_depot']
                reason = 'mission_complete' if complete else 'premature_terminal'
                break
        summary = summary_values(env)
        result = dict(completed=bool(complete), reason=reason, cost=summary['total_time'] if complete else None,
                      actions=trace, gate_keys=used, chromosome_order=list(order),
                      chromosome_gates=list(gates), summary=summary, label=label, fallback_steps=0)
        improved = complete and (self.incumbent is None or result['cost'] < self.incumbent['cost'])
        if improved: self.incumbent = deepcopy(result)
        self.history.append(dict(evaluation=self.evaluations, seconds=time.monotonic()-self.started,
            completed=bool(complete), cost=result['cost'], reason=reason, improved=improved,
            best=self.incumbent['cost'] if self.incumbent else None,
            order=list(order), gates=list(gates), generation=self._generation, candidate=self._candidate))
        return result

    def mutate_outer(self, a, b):
        # Ordered prefix crossover, with task-aligned gate keys; then swap/insertion.
        cut = self.rng.randrange(len(a[0])+1)
        pairs = list(zip(a[0][:cut], a[1][:cut]))
        seen = {t for t, _ in pairs}
        pairs.extend((t, g) for t, g in zip(b[0], b[1]) if t not in seen)
        i, j = self.rng.randrange(len(pairs)), self.rng.randrange(len(pairs))
        if self.rng.random() < .5: pairs[i], pairs[j] = pairs[j], pairs[i]
        else: pairs.insert(j, pairs.pop(i))
        return tuple(t for t, _ in pairs), tuple(g for _, g in pairs)

    def inner(self, *args, **kwargs):
        result = super().inner(*args, **kwargs)
        if result is not None and self.history and self.history[-1]['reason'] == 'wall_limit':
            self.inner_records[-1].update(completed=False, interruption='wall_limit')
            return None
        return result

    def run(self):
        if self.started is not None: raise RuntimeError('Search objects are single-use')
        self.started = time.monotonic()
        baseline = None
        termination = None
        try:
            self.partition()
        except (BudgetExpired, PartitionFailure) as exc:
            termination = 'wall_limit' if isinstance(exc, BudgetExpired) else 'partition_failure'
            self.partition_records.append(dict(failure=str(exc), reason=termination))
        preprocessing = time.monotonic()-self.started
        if termination is None:
            # Deterministic nearest-region centroid order, independent of learned policies.
            position = self.initial.depot.position
            remaining, order = list(self.tasks), []
            centers = {}
            for task in remaining:
                pts = {p.id: p for p in self.initial.get_field(task.field_id).grid_points}
                centers[task.task_id] = tuple(sum(getattr(pts[i], xy) for i in task.point_ids)/len(task.point_ids)
                                              for xy in ('x', 'y'))
            while remaining:
                chosen = min(remaining, key=lambda t: (hypot(centers[t.task_id][0]-position[0],
                                                          centers[t.task_id][1]-position[1]), t.task_id))
                order.append(chosen.task_id); position = centers[chosen.task_id]; remaining.remove(chosen)
            root = (tuple(order), tuple(-1. for _ in order))
            baseline = self.evaluate(*root, 'fixed_partition_nearest_initialization')
            population = [root]
            while len(population) < self.config.outer_population: population.append(self.mutate_outer(root, root))
            for generation in range(self.config.outer_generations):
                scored = []
                attempted = 0
                for candidate, (order, gates) in enumerate(population):
                    attempted += 1
                    result = self.inner(order, gates, generation=generation, candidate=candidate)
                    if result is None: break
                    scored.append(result)
                full = len(scored) == len(population) and not self.expired()
                self.outer_records.append(dict(generation=generation, attempted=attempted,
                    completed_quotas=len(scored), completed_round=full))
                if not full: break
                scored.sort(key=lambda x: self.fitness(x[2]))
                population = [(scored[0][0], scored[0][1])]
                while len(population) < self.config.outer_population:
                    parents = [self.rng.choice(scored[:max(2, len(scored)//2)]) for _ in range(2)]
                    population.append(self.mutate_outer(parents[0], parents[1]))
        elapsed = time.monotonic()-self.started
        verified = None
        if self.incumbent:
            verified = replay_regions(self.initial, self.incumbent['actions'])
            if abs(verified['total_time']-self.incumbent['cost']) > 1e-6:
                raise ValueError('Incumbent replay mismatch')
        return dict(method='divide_then_plan', config=asdict(self.config), baseline=baseline,
            best=self.incumbent, tasks=[asdict(t) for t in self.tasks], gate_options=self.gate_options,
            partition_records=self.partition_records, preprocessing_seconds=preprocessing,
            preprocessing_probes=self.probes, evaluations=self.evaluations, transitions=self.transitions,
            history=self.history, inner_quotas=self.inner_records, outer_records=self.outer_records,
            search_seconds=elapsed, replay_seconds=time.monotonic()-self.started-elapsed,
            replay_summary=verified, termination=termination or ('wall_limit' if elapsed >= self.config.wall_seconds else
                'evaluation_limit' if self.evaluations >= self.config.max_evaluations else 'generation_limit'))

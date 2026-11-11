"""Visit order/area shares first, contextual discrete division second."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from math import hypot
import time

from coverage_env.baseline.divide_then_plan import (
    DivideThenPlanGA, PartitionFailure, BudgetExpired, long_axis,
)
from coverage_env.planner.framework_contracts import normalize_visit_weights
from coverage_env.planner.framework_execution import replay_regions


def contextual_regions(initial, visits):
    """Exact ownership in visit order; no resource repair or environment mutation."""
    fields = {f.field_id: f for f in initial.fields}
    counts = {fid: sum(v[0] == fid for v in visits) for fid in fields}
    if any(v[0] not in fields for v in visits) or any(n == 0 for n in counts.values()):
        raise PartitionFailure('Visits must include every field and no unknown field')
    shares = {}
    for fid, field in fields.items():
        indices = [i for i, v in enumerate(visits) if v[0] == fid]
        if len(indices) > len(field.grid_points):
            raise PartitionFailure('More visits than discrete points')
        try:
            fractions = normalize_visit_weights([visits[i][1] for i in indices])
        except ValueError as exc:
            raise PartitionFailure(str(exc)) from exc
        shares.update(zip(indices, fractions))
    remaining = {fid: list(f.grid_points) for fid, f in fields.items()}
    total = {fid: sum(p.point_area for p in f.grid_points) for fid, f in fields.items()}
    regions, audit = [], []
    for index, (fid, _) in enumerate(visits):
        field = fields[fid]
        prev = initial.depot.position if index == 0 else fields[visits[index-1][0]].centroid
        nxt = initial.depot.position if index == len(visits)-1 else fields[visits[index+1][0]].centroid
        dx, dy = nxt[0]-prev[0], nxt[1]-prev[1]
        if hypot(dx, dy) <= 1e-12:
            dx, dy = long_axis(field)
        length = hypot(dx, dy); dx, dy = dx/length, dy/length
        ordered = sorted(remaining[fid], key=lambda p: (p.x*dx+p.y*dy, -p.x*dy+p.y*dx, p.id))
        counts[fid] -= 1
        target = shares[index]*total[fid]
        if counts[fid] == 0:
            selected = ordered
        else:
            area = 0.; candidates = []
            for k, point in enumerate(ordered[:len(ordered)-counts[fid]], 1):
                area += point.point_area
                candidates.append((abs(area-target), k))
            selected = ordered[:min(candidates)[1]]
        selected_ids = {p.id for p in selected}
        remaining[fid] = [p for p in ordered if p.id not in selected_ids]
        regions.append((field, tuple(p.id for p in selected)))
        actual = sum(p.point_area for p in selected)
        audit.append(dict(visit=index, field_id=fid, target_area=target,
                          actual_area=actual, area_error=actual-target, point_ids=regions[-1][1]))
    return regions, audit


class PlanThenDivideGA(DivideThenPlanGA):
    """One gate key per original visit, inherited by its repaired subvisits.

    The bounded split repair cannot move points between original visits.
    Decoding/repair is charged for every candidate; there is no hidden cache.
    """
    def evaluate(self, visits, gates, label):
        if self.evaluations >= self.config.max_evaluations or self.expired(): return None
        self.evaluations += 1  # includes candidates that fail during division
        if len(gates) != len(visits): raise ValueError('Visit/key length mismatch')
        start = time.monotonic()
        audit = []
        self.partition_records = []
        try:
            self.check_wall()
            if len(visits) > self.config.max_steps: raise PartitionFailure('Visit limit exceeded')
            regions, audit = contextual_regions(self.initial, visits)
            self.partition_regions(regions)
            owner = {(f.field_id, pid): i for i, (f, ids) in enumerate(regions) for pid in ids}
            decoded_gates = tuple(gates[owner[t.field_id, t.point_ids[0]]] for t in self.tasks)
            result = self._evaluate_decoded(tuple(t.task_id for t in self.tasks),
                                            decoded_gates, label, reserved=True)
            result['region_tasks'] = [asdict(t) for t in self.tasks]
            result['repair_added_visits'] = len(self.tasks)-len(visits)
            result['partition_signature'] = sha256(json.dumps(result['region_tasks'], sort_keys=True).encode()).hexdigest()
        except (PartitionFailure, BudgetExpired) as exc:
            reason = 'wall_limit' if isinstance(exc, BudgetExpired) else 'partition_failure'
            result = dict(completed=False, cost=None, reason=reason, error=str(exc), label=label,
                          actions=[], gate_keys=list(gates), fallback_steps=0)
            self.history.append(dict(evaluation=self.evaluations, seconds=time.monotonic()-self.started,
                completed=False, cost=None, reason=reason, best=self.incumbent['cost'] if self.incumbent else None,
                improved=False, generation=self._generation, candidate=self._candidate))
        result.update(chromosome_order=list(visits), chromosome_gates=list(gates),
                      division_audit=audit, repair_records=deepcopy(self.partition_records),
                      decode_and_execution_seconds=time.monotonic()-start)
        self.history[-1].update(visits=list(visits),
            repair_added_visits=result.get('repair_added_visits'),
            partition_signature=result.get('partition_signature'),
            area_errors=[a['area_error'] for a in audit])
        # Parent evaluator records the incumbent before these PD-specific fields.
        if self.history[-1]['improved']: self.incumbent = deepcopy(result)
        return result

    def mutate_outer(self, a, b):
        cut_a = self.rng.randrange(len(a[0])+1)
        cut_b = self.rng.randrange(len(b[0])+1)
        pairs = list(zip(a[0][:cut_a], a[1][:cut_a])) + list(zip(b[0][cut_b:], b[1][cut_b:]))
        if not pairs: pairs = [((self.fields[0], 1.), -1.)]
        i = self.rng.randrange(len(pairs))
        op = self.rng.choice(('swap', 'insert', 'delete', 'weight'))
        if op == 'swap':
            j = self.rng.randrange(len(pairs)); pairs[i], pairs[j] = pairs[j], pairs[i]
        elif op == 'insert': pairs.insert(i, ((self.rng.choice(self.fields), 1.), self.rng.random()))
        elif op == 'delete': pairs.pop(i)
        else:
            (fid, weight), gate = pairs[i]
            pairs[i] = ((fid, min(1e6, max(1e-6, weight*self.rng.choice((.5, 2.))))), gate)
        # Preserve at least one visit per field under the common step limit.
        original_length = len(pairs)
        pairs = pairs[:max(0, self.config.max_steps-len(self.fields))]
        present = {v[0] for v, _ in pairs}
        missing = [fid for fid in self.fields if fid not in present]
        if not hasattr(self, 'encoding_repairs'): self.encoding_repairs = []
        self.encoding_repairs.append(dict(truncated_visits=original_length-len(pairs),
                                          appended_fields=missing))
        for fid in self.fields:
            if fid not in present: pairs.append(((fid, 1.), -1.))
        return tuple(v for v, _ in pairs), tuple(g for _, g in pairs)

    def run(self):
        if self.started is not None: raise RuntimeError('Search objects are single-use')
        self.started = time.monotonic()
        self.fields = tuple(f.field_id for f in sorted(self.initial.fields, key=lambda f: str(f.field_id)))
        if len(self.fields) > self.config.max_steps: raise ValueError('Field count exceeds max_steps')
        remaining = list(self.initial.fields); position = self.initial.depot.position; order = []
        while remaining:
            f = min(remaining, key=lambda f: (hypot(f.centroid[0]-position[0], f.centroid[1]-position[1]), str(f.field_id)))
            order.append((f.field_id, 1.)); position = f.centroid; remaining.remove(f)
        root = (tuple(order), tuple(-1. for _ in order))
        baseline = self.evaluate(*root, 'visit_nearest_initialization')
        population = [root]
        while len(population) < self.config.outer_population: population.append(self.mutate_outer(root, root))
        for generation in range(self.config.outer_generations):
            scored = []; attempted = 0
            for candidate, (visits, gates) in enumerate(population):
                attempted += 1
                item = self.inner(visits, gates, generation=generation, candidate=candidate)
                if item is None: break
                scored.append(item)
            full = len(scored) == len(population) and not self.expired()
            self.outer_records.append(dict(generation=generation, attempted=attempted,
                completed_quotas=len(scored), completed_round=full))
            if not full: break
            scored.sort(key=lambda x: self.fitness(x[2]))
            population = [(scored[0][0], scored[0][1])]
            while len(population) < self.config.outer_population:
                a, b = (self.rng.choice(scored[:max(2, len(scored)//2)]) for _ in range(2))
                population.append(self.mutate_outer(a, b))
        elapsed = time.monotonic()-self.started
        verified = replay_regions(self.initial, self.incumbent['actions']) if self.incumbent else None
        if verified and abs(verified['total_time']-self.incumbent['cost']) > 1e-6:
            raise ValueError('Incumbent replay mismatch')
        return dict(method='plan_then_divide', config=asdict(self.config), baseline=baseline,
            best=self.incumbent, evaluations=self.evaluations, transitions=self.transitions,
            repair_probes=self.probes, encoding_repairs=getattr(self, 'encoding_repairs', []),
            history=self.history, inner_quotas=self.inner_records,
            outer_records=self.outer_records, search_seconds=elapsed,
            replay_seconds=time.monotonic()-self.started-elapsed, replay_summary=verified,
            termination='wall_limit' if elapsed >= self.config.wall_seconds else
            'evaluation_limit' if self.evaluations >= self.config.max_evaluations else 'generation_limit')

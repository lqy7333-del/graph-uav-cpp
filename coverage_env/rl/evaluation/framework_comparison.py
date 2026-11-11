"""Shared-snapshot acceptance/formal queue. Never generates replacement instances.

Snapshots are trusted local pickle files; do not use untrusted downloaded pickles.
Formal execution requires a successful acceptance receipt from the same code and
physical configuration. Resume is between searches, not within a timed GA run.
"""
import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import pickle
import time

import numpy as np

from coverage_env.baseline.nested_ga import GAConfig, NestedGA, replay, summary_values
from coverage_env.baseline.divide_then_plan import DivideThenPlanGA
from coverage_env.baseline.plan_then_divide import PlanThenDivideGA
from coverage_env.planner.coverage_plan import fingerprints
from coverage_env.planner.framework_execution import replay_regions

METHODS = {'cc': NestedGA, 'divide_then_plan': DivideThenPlanGA, 'plan_then_divide': PlanThenDivideGA}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_digest():
    root = Path(__file__).resolve().parents[2]
    # Include every Python source affecting environment, geometry, planning or search.
    files = sorted(p for p in root.rglob('*.py') if 'tests' not in p.parts)
    return digest([(str(p.relative_to(root)), sha256(p.read_bytes()).hexdigest()) for p in files])


def write_json(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(20):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 19: raise
            time.sleep(.05)  # Windows indexer/antivirus can briefly hold a new file.


def load_panel(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or not manifest.get('instances'):
        raise ValueError('Invalid or empty panel manifest')
    panel, ids, seen_instances = [], set(), set()
    for row in manifest['instances']:
        if row['id'] in ids: raise ValueError('Duplicate panel ID')
        ids.add(row['id'])
        path = (manifest_path.parent / row['snapshot']).resolve()
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != row['sha256']: raise ValueError('Snapshot hash mismatch')
        env = pickle.loads(raw)  # trusted local input only; NEVER call reset here
        if len(env.fields) != row['fields'] or env.get_summary()['execution_count'] != 0:
            raise ValueError('Snapshot scale or initial-state mismatch')
        if any(p.covered for f in env.fields for p in f.grid_points):
            raise ValueError('Snapshot already has covered points')
        u = env.uavs[0]
        if len(env.uavs) != 1 or u.current_position != env.depot.position or u.remaining_energy != u.max_energy or u.remaining_payload != u.max_payload:
            raise ValueError('Snapshot must have one full-resource UAV at depot')
        instance, config, _ = fingerprints(env, env.fields[0])
        if instance in seen_instances: raise ValueError('Duplicate geometry in panel')
        seen_instances.add(instance)
        if config != manifest['physical_config_fingerprint']:
            raise ValueError('Panel physical configuration mismatch')
        panel.append((row, env, instance))
    return manifest, panel


def prepare_manifest(snapshot_paths, output, *, cc_results=None):
    """Freeze existing trusted initial snapshots; does not regenerate or reset them."""
    import os
    out = Path(output).resolve()
    if out.exists(): raise FileExistsError(out)
    entries, physical = [], None
    for source in snapshot_paths:
        path = Path(source).resolve(); raw = path.read_bytes(); env = pickle.loads(raw)
        _, config, _ = fingerprints(env, env.fields[0])
        if physical is not None and physical != config: raise ValueError('Mixed physical configurations')
        physical = config
        item = dict(id=path.stem, fields=len(env.fields),
                    snapshot=os.path.relpath(path, out.parent), sha256=sha256(raw).hexdigest())
        if cc_results:
            prior = Path(cc_results).resolve() / (path.stem+'_ga_42.json')
            item.update(cc_result=os.path.relpath(prior, out.parent),
                        cc_result_sha256=sha256(prior.read_bytes()).hexdigest())
        entries.append(item)
    if not entries: raise ValueError('Empty snapshot list')
    out.parent.mkdir(parents=True, exist_ok=True)
    # Only publish a manifest after validation succeeds.
    candidate = out.with_suffix('.validation.tmp')
    if candidate.exists(): raise FileExistsError(candidate)
    try:
        write_json(candidate, dict(schema_version=1, physical_config_fingerprint=physical, instances=entries))
        load_panel(candidate)
        candidate.replace(out)
    finally:
        if candidate.exists(): candidate.unlink()


def verify_result(env, method, result):
    best = result.get('best')
    if best is None:
        if result.get('replay_summary') is not None: raise ValueError('Failure has fabricated replay')
        return False
    if not best.get('completed'): raise ValueError('Incumbent is incomplete')
    s = (replay(env, best['actions'], best['cost'], result['config']['max_steps']) if method == 'cc'
         else replay_regions(env, best['actions']))
    if abs(s['total_time']-best['cost']) > 1e-6: raise ValueError('Stored cost is not reproducible')
    for key in ('total_time', 'total_distance', 'time_work_base', 'time_nonwork_base',
                'time_maneuver_extra', 'time_service'):
        if abs(s[key]-best['summary'][key]) > 1e-6:
            raise ValueError('Stored component differs from replay')
    return True


def paired_summary(rows, seed=12345):
    """Instance-paired bootstrap; failures retained, complete pairs auxiliary only."""
    rng = np.random.default_rng(seed)
    report = {}
    for scale in sorted({r['fields'] for r in rows}):
        subset = [r for r in rows if r['fields'] == scale]
        grouped = {m: {r['id']: r for r in subset if r['method'] == m} for m in METHODS}
        scale_report = {'methods': {}, 'comparisons': {}}
        for m, values in grouped.items():
            good = [r for r in values.values() if r['completed']]
            scale_report['methods'][m] = dict(n=len(values), completed=len(good),
                mean_total_time=float(np.mean([r['total_time'] for r in good])) if good and len(good)==len(values) else None,
                mean_search_seconds=float(np.mean([r['search_seconds'] for r in values.values()])) if values else None)
        for m in ('divide_then_plan', 'plan_then_divide'):
            common = sorted(set(grouped[m]) & set(grouped['cc']))
            pairs = [grouped[m][i]['total_time']-grouped['cc'][i]['total_time'] for i in common
                     if grouped[m][i]['completed'] and grouped['cc'][i]['completed']]
            values = np.asarray(pairs)
            ci = None
            if len(values) >= 2:
                draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
                ci = np.quantile(draws, [.025, .975]).tolist()
            scale_report['comparisons'][m+' minus cc'] = dict(n_pairs=len(pairs), panel_n=len(common),
                scope='all paired instances' if len(pairs)==len(common) else 'complete-pair subset; auxiliary only',
                mean=float(values.mean()) if len(values) else None, ci95=ci,
                cc_wins=int((values>0).sum()), ties=int((values==0).sum()), other_wins=int((values<0).sum()))
        report[str(scale)] = scale_report
    return report


def export_report(out, rows, panel):
    with (out/'rows.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = paired_summary(rows)
    lines = ['# Framework comparison', '',
        'Positive paired differences mean CC is faster. Instance bootstrap intervals do not quantify search-seed uncertainty.', '',
        '| Fields | Method | Completed | Mean total time (s) | Mean search (s) |',
        '|---|---|---|---|---|']
    for scale, data in report.items():
        for m, item in data['methods'].items():
            lines.append(f"| {scale} | {m} | {item['completed']}/{item['n']} | {item['mean_total_time']} | {item['mean_search_seconds']} |")
    lines += ['', 'Failures are retained; method mean time is omitted if any task failed.',
              'All decomposed costs and paired statistics are in rows.csv / summary.json.',
              'Search times include method-specific preprocessing. Verification, plots and replay are outside search time.',
              'This is a bounded search comparison, not proof of optimality or statistical equivalence.']
    (out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    audit = []
    for index, (item, _, _) in enumerate(panel):
        for method in METHODS:
            result = json.loads((out/f'{index:04d}_{method}.json').read_text())['result']
            history = result['history']
            signatures = {h['partition_signature'] for h in history if h.get('partition_signature')}
            chromosomes = {digest(h.get('visits', h.get('order'))) for h in history
                           if h.get('visits') or h.get('order')}
            repair = [h['repair_added_visits'] for h in history if h.get('repair_added_visits') is not None]
            audit.append(dict(id=item['id'], method=method,
                distinct_complete_costs=len({h['cost'] for h in history if h.get('completed')}),
                distinct_chromosomes=len(chromosomes) if chromosomes else None,
                distinct_partitions=len(signatures) if signatures else None,
                maximum_added_visits=max(repair) if repair else None,
                local_geometry_probes=result.get('repair_probes', result.get('preprocessing_probes')),
                preprocessing_seconds=result.get('preprocessing_seconds'),
                fallback_steps=(result.get('best') or {}).get('fallback_steps'),
                notes='Null means not recorded/applicable, not zero. Cost diversity is not chromosome diversity.'))
    write_json(out/'search_audit.json', audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Predeclared first snapshot only; never choose an attractive case after results.
    item, initial, _ = panel[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, method in zip(axes, METHODS):
        result = json.loads((out/f'0000_{method}.json').read_text())['result']
        for field in initial.fields:
            x, y = field.xy_polygon.exterior.xy; ax.plot(x, y, color='black', linewidth=.7)
        best = result.get('best')
        if best:
            if method == 'cc':
                from copy import deepcopy
                from coverage_env.rl.action_space import CoverageAction
                clone = deepcopy(initial)
                for action in best['actions']: clone.step(CoverageAction(*action))
                regions = [(r['field_id'], [p.id for p in r['region']]) for r in clone.execution_records]
            else:
                regions = [(r['plan']['field_id'], r['plan']['point_ids']) for r in best['actions']]
            for index, (fid, ids) in enumerate(regions):
                points = {p.id: p for p in initial.get_field(fid).grid_points}
                ax.scatter([points[p].x for p in ids], [points[p].y for p in ids],
                           s=6, color=plt.get_cmap('tab20')(index % 20))
        ax.set_title(method + (' (failed)' if not best else ''))
        ax.set_aspect('equal'); ax.set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    fig.suptitle('First manifest instance: executed region assignments (colors = visits)')
    fig.tight_layout(); fig.savefig(out/'first_instance_regions.png', dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        result = json.loads((out/f'0000_{method}.json').read_text())['result']
        history = [h for h in result['history'] if h.get('best') is not None]
        ax.step([h['seconds'] for h in history], [h['best'] for h in history], where='post', label=method)
    ax.set_xlabel('Search wall time (s), including preprocessing'); ax.set_ylabel('Best complete task time (s)')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'first_instance_progress.png', dpi=140); plt.close(fig)


def run_panel(manifest_path, output, settings, *, stage='acceptance', receipt=None, resume=False):
    manifest, panel = load_panel(manifest_path)
    code = code_digest()
    contract = dict(schema_version=1, stage=stage, manifest_digest=digest(manifest),
                    code_digest=code, config=asdict(settings), methods=list(METHODS))
    if stage not in ('acceptance', 'formal'): raise ValueError('Unknown stage')
    if stage == 'formal':
        if receipt is None: raise ValueError('Formal run requires acceptance receipt')
        accepted = json.loads(Path(receipt).read_text(encoding='utf-8'))
        if (not accepted.get('passed') or not accepted.get('domain_verified') or accepted['code_digest'] != code
                or accepted['physical_config_fingerprint'] != manifest['physical_config_fingerprint']):
            raise ValueError('Acceptance receipt is not compatible with this run')
        if set(accepted['instance_fingerprints']) & {p[2] for p in panel}:
            raise ValueError('Acceptance and formal panels overlap')
        if any(row['fields'] not in (10, 20) for row, _, _ in panel):
            raise ValueError('Formal stage supports the frozen 10/20-field protocol only')
        if any(not 4999.999 <= f.area <= 20000.001 for _, env, _ in panel for f in env.fields):
            raise ValueError('Formal field areas are outside the frozen domain')
        contract['receipt_digest'] = digest(accepted)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=resume)
    meta = out/'run_contract.json'
    if resume:
        if not meta.exists() or json.loads(meta.read_text()) != contract:
            raise ValueError('Resume contract mismatch')
    else: write_json(meta, contract)
    # Exclusive create prevents two live queues from racing on the same results.
    lock = out/'queue.lock'
    with lock.open('x') as handle: handle.write(str(time.time()))
    rows = []
    try:
        for index, (item, env, instance) in enumerate(panel):
            for method, cls in METHODS.items():
                target = out/f'{index:04d}_{method}.json'
                identity = dict(id=item['id'], snapshot_sha256=item['sha256'], method=method,
                                contract_digest=digest(contract))
                if target.exists():
                    saved = json.loads(target.read_text(encoding='utf-8'))
                    if saved['identity'] != identity: raise ValueError('Existing result identity mismatch')
                    result = saved['result']
                else:
                    try:
                        if method == 'cc' and 'cc_result' in item:
                            source = (Path(manifest_path).resolve().parent/item['cc_result']).read_bytes()
                            if sha256(source).hexdigest() != item['cc_result_sha256']:
                                raise ValueError('Historical CC file hash mismatch')
                            result = json.loads(source)
                            if (result['config'] != asdict(settings)
                                    or result['instance_snapshot_sha256'] != item['sha256']):
                                raise ValueError('Historical CC snapshot or budget mismatch')
                        else:
                            result = cls(env, settings).run()
                        verify_result(env, method, result)
                    except Exception as exc:
                        write_json(out/'failure.json', dict(identity=identity, error=repr(exc)))
                        raise  # unexpected faults stop the queue
                    write_json(target, dict(identity=identity, result=result))
                completed = verify_result(env, method, result)
                best = result.get('best'); s = best['summary'] if completed else {}
                rows.append(dict(id=item['id'], fields=item['fields'], method=method, completed=completed,
                    total_time=s.get('total_time'), work=s.get('time_work_base'), nonwork=s.get('time_nonwork_base'),
                    maneuver=s.get('time_maneuver_extra'), service=s.get('time_service'),
                    excess_time=s['total_time']-s['time_work_base'] if completed else None,
                    search_seconds=result['search_seconds'], evaluations=result['evaluations'],
                    full_outer_rounds=sum(bool(r.get('completed_round')) for r in result['outer_records']),
                    termination=result['termination']))
                write_json(out/'progress.json', dict(completed_searches=len(rows), total_searches=len(panel)*3))
        write_json(out/'rows.json', rows)
        write_json(out/'summary.json', paired_summary(rows))
        export_report(out, rows, panel)
        if stage == 'acceptance':
            domain_verified = all(row['fields'] == 10 and all(4999.999 <= f.area <= 20000.001 for f in env.fields)
                                  for row, env, _ in panel)
            search_exercised = all(r['evaluations'] > 1 and r['full_outer_rounds'] >= 1 for r in rows)
            passed = 3 <= len(panel) <= 5 and all(r['completed'] for r in rows) and domain_verified and search_exercised
            write_json(out/'acceptance_receipt.json', dict(passed=passed, code_digest=code,
                domain_verified=domain_verified,
                search_exercised=search_exercised,
                physical_config_fingerprint=manifest['physical_config_fingerprint'],
                instance_fingerprints=[p[2] for p in panel],
                scope='engineering completion/replay gate only; inspect search diversity and timing separately',
                contract_digest=digest(contract)))
        return rows
    finally:
        lock.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=('acceptance', 'formal'), default='acceptance')
    p.add_argument('--receipt')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-evaluations', type=int, default=385)
    p.add_argument('--wall-seconds', type=float, default=1800)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    settings = GAConfig(outer_population=4, outer_generations=6, inner_population=8,
        inner_generations=2, max_evaluations=args.max_evaluations, wall_seconds=args.wall_seconds, seed=args.seed)
    run_panel(args.manifest, args.output, settings, stage=args.stage, receipt=args.receipt, resume=args.resume)


if __name__ == '__main__': main()

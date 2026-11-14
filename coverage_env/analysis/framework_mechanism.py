"""Read-only trajectory/mechanism ledger for the frozen framework experiment.

This module never selects candidates or changes a plan. It replays archived
actions on a copy of the initial snapshot and records execution events.
"""
from __future__ import annotations
import json, math, pickle
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from shapely.geometry import Point

from coverage_env.baseline.nested_ga import action_key
from coverage_env.baseline.divide_then_plan import components
from coverage_env.planner.coverage_plan import CoveragePlan
from coverage_env.planner.framework_execution import execute_region_plan, _execute_after_refill

TOL = 1e-6
LEDGER_KEYS = ('total_time', 'time_work_base', 'time_nonwork_base',
               'time_maneuver_extra', 'time_service', 'total_distance',
               'resupply_count', 'execution_count', 'coverage_ratio')

def _state(env):
    u = env.uavs[0]
    s = env.get_summary()
    return dict(position=None if u.current_position is None else list(u.current_position),
                energy=float(u.remaining_energy), payload=float(u.remaining_payload),
                energy_ratio=float(u.remaining_energy/u.max_energy),
                payload_ratio=float(u.remaining_payload/u.max_payload),
                heading_valid=bool(env._heading_valid.get(u.uav_id, False)),
                time=float(s['total_time']), distance=float(s['total_distance']))

def _coverage(env):
    return {f.field_id: {p.id for p in f.grid_points if p.covered} for f in env.fields}

def _field_areas(env):
    return {f.field_id: {p.id: float(p.point_area) for p in f.grid_points} for f in env.fields}

def _area(ids, field_areas, field_id):
    return sum(field_areas[field_id].get(pid, 0.) for pid in ids)

def _components_for(ids, env, field_id):
    f = next(f for f in env.fields if f.field_id == field_id)
    by = {p.id: p for p in f.grid_points}
    if not ids: return 0
    return len(components(tuple(ids), by))

def _add_event(out, event):
    for k in ('work','nonwork','maneuver','service','total','distance'):
        out[k] = out.get(k,0.) + float(event.get(k,0.) or 0.)

def replay_result(initial, result, method, identity):
    """Replay one archived best trace and return visit/trip/event ledgers.

    The archived result is not mutated. Unknown values are None, never coerced to 0.
    """
    env = deepcopy(initial)
    captures = []
    original_resupply = env._perform_resupply
    def captured_resupply(reason):
        pre = _state(env)
        event = original_resupply(reason)
        post = _state(env)
        if event is not None:
            captures.append(dict(reason=reason, pre=pre, post=post, event=event))
        return event
    env._perform_resupply = captured_resupply
    best = result['best']; rows = best['actions']
    visits=[]; events=[]; trips=[]; current_trip=None; trip_no=0; visit_no=0; event_no=0
    covered = _coverage(env); areas = _field_areas(env)
    def new_trip(start_time, start_event=None):
        nonlocal trip_no
        trip_no += 1
        return dict(instance_id=identity['id'], scale=identity['fields'], method=method,
                    trip_id=trip_no, start_time=float(start_time), end_time=None,
                    visits=0, coverage=0., work=0., nonwork=0., maneuver=0., service=0.,
                    total=0., distance=0., fields=[], resupply_count=0, end_event=None,
                    start_event=start_event)
    def close_trip(trip, end_time, end_event):
        if trip is None: return
        trip['end_time']=float(end_time); trip['end_event']=end_event
        trip['duration']=float(end_time)-trip['start_time']
        # An initial refill before any visit is not a depot-to-depot outing.
        if trip['visits'] > 0 or trip['start_event'] != 'initial':
            trips.append(trip)
    current_trip = new_trip(0.0, 'initial')
    for step_index,row in enumerate(rows):
        before = _state(env); step_start=before['time']; before_cov = {fid:set(v) for fid,v in covered.items()}
        captures.clear()
        if method == 'cc':
            actions = env.get_action_candidates()
            matches = [a for a in actions if action_key(a) == tuple(row)]
            if len(matches) != 1: raise ValueError(f'CC replay action mismatch at {step_index}')
            _,_,done,info = env.step(matches[0])
            action=None
        else:
            plan = CoveragePlan.from_dict(row['plan'])
            if row.get('resupply_before'):
                _,_,done,info = _execute_after_refill(env, plan, row['start_state_fingerprint'])
            else:
                _,_,done,info = execute_region_plan(env, plan)
            action=row
        after = _state(env); after_cov = _coverage(env)
        step_events=[]
        # Build exact return/resupply events captured inside the step.
        for cap in captures:
            leg=cap['event']['return_feedback']; svc=float(env.rl_config.get('time',{}).get('service_time',0.0))
            ev=dict(instance_id=identity['id'], scale=identity['fields'], method=method,
                    event_index=event_no, type='resupply', trigger=cap['reason'],
                    trip_id=current_trip['trip_id'] if current_trip else None,
                    before_energy=cap['pre']['energy'], before_payload=cap['pre']['payload'],
                    before_energy_ratio=cap['pre']['energy_ratio'], before_payload_ratio=cap['pre']['payload_ratio'],
                    arrival_energy=cap['pre']['energy']-float(leg.get('energy',0.)),
                    arrival_payload=cap['pre']['payload'],
                    after_energy=cap['post']['energy'], after_payload=cap['post']['payload'],
                    after_energy_ratio=cap['post']['energy_ratio'], after_payload_ratio=cap['post']['payload_ratio'],
                    return_distance=float(leg.get('distance',0.)), return_energy=float(leg.get('energy',0.)),
                    return_nominal_time=float(leg.get('time_total',0.)), service_time=svc,
                    work=0., nonwork=float(leg.get('time_nonwork_base',0.)),
                    maneuver=float(leg.get('time_maneuver_extra',0.)), service=svc,
                    total=float(leg.get('time_total',0.))+svc, distance=float(leg.get('distance',0.)),
                    start_before=None, end_after=None)
            event_no += 1; step_events.append(ev)
        # Determine region/field and new coverage.
        if method == 'cc':
            res=info.get('result') or {}; field_id=info.get('selected_field')
            region=[p.id for p in (res.get('region') or [])]
            entry=info.get('entry_boundary'); exit_=info.get('exit_boundary')
            cov_only=info.get('coverage_only_distance'); transit=info.get('transit_distance')
        else:
            field_id=action['plan']['field_id']; region=list(action['plan']['point_ids'])
            entry=action['plan'].get('entry_boundary_id'); exit_=action['plan'].get('exit_boundary_id')
            cov_only=info.get('coverage_only_distance'); transit=info.get('transit_distance')
        if field_id is None: raise ValueError('Replay missing field identity')
        new_ids=set(after_cov[field_id])-set(before_cov[field_id])
        coverage_area=_area(new_ids,areas,field_id)
        assigned_area=_area(region,areas,field_id)
        # Events physically precede coverage for explicit-region refill; otherwise follow it.
        pre_events = step_events if (method != 'cc' and action is not None and action.get('resupply_before')) else []
        post_events = [e for e in step_events if e not in pre_events]
        visit_start=step_start
        if pre_events:
            t=step_start
            for ev in pre_events:
                ev['start_before']=t; ev['end_after']=t+ev['total']
                t=ev['end_after']
                _add_event(current_trip,ev)
                current_trip['resupply_count'] += 1
                close_trip(current_trip, ev['end_after'], 'resupply')
                current_trip=new_trip(ev['end_after'],'resupply')
            visit_start=t
        visit_no += 1
        ev_all=step_events
        event_tot={k:sum(float(e.get(k,0.) or 0.) for e in ev_all) for k in ('work','nonwork','maneuver','service','total','distance')}
        _terminal_leg = info.get('return_to_depot')
        if _terminal_leg is not None:
            event_tot['nonwork'] += float(_terminal_leg.get('time_nonwork_base',0.))
            event_tot['maneuver'] += float(_terminal_leg.get('time_maneuver_extra',0.))
            event_tot['total'] += float(_terminal_leg.get('time_total',0.))
            event_tot['distance'] += float(_terminal_leg.get('distance',0.))
        visit=dict(instance_id=identity['id'], scale=identity['fields'], method=method,
                   visit_index=visit_no, trip_id=current_trip['trip_id'], field_id=field_id,
                   region_point_ids=region, region_area=assigned_area,
                   new_point_ids=sorted(new_ids), new_area=coverage_area,
                   entry_boundary=entry, exit_boundary=exit_,
                   start_position=before['position'], end_position=after['position'],
                   before_energy=before['energy'], after_energy=after['energy'],
                   before_payload=before['payload'], after_payload=after['payload'],
                   before_energy_ratio=before['energy_ratio'], after_energy_ratio=after['energy_ratio'],
                   before_payload_ratio=before['payload_ratio'], after_payload_ratio=after['payload_ratio'],
                   before_heading_valid=before['heading_valid'], after_heading_valid=after['heading_valid'],
                   work=float(info.get('time_work_base',0.))-event_tot['work'],
                   nonwork=float(info.get('time_nonwork_base',0.))-event_tot['nonwork'],
                   maneuver=float(info.get('time_maneuver_extra',0.))-event_tot['maneuver'],
                   service=float(info.get('time_service',0.))-event_tot['service'],
                   total=float(info.get('time_total',0.))-event_tot['total'],
                   distance=float(info.get('path_length',0.))-event_tot['distance'],
                   coverage_distance=cov_only, transfer_distance=transit,
                   coverage_gain=float(info.get('coverage_gain',0.)),
                   coverage_ratio=coverage_area/assigned_area if assigned_area else None,
                   region_components=_components_for(region,env,field_id),
                   assigned_area=assigned_area,
                   supplies_before_visit=bool(pre_events),
                   start_time=visit_start, end_time=visit_start+ (float(info.get('time_total',0.))-event_tot['total']),
                   step_index=step_index)
        visits.append(visit)
        cur=current_trip
        cur['visits'] += 1
        cur['coverage'] += coverage_area
        cur['work'] += visit['work']; cur['nonwork'] += visit['nonwork']
        cur['maneuver'] += visit['maneuver']; cur['service'] += visit['service']
        cur['total'] += visit['total']; cur['distance'] += visit['distance']
        if field_id not in cur['fields']: cur['fields'].append(field_id)
        covered=after_cov
        # Add event components to current trip and close on post events.
        if post_events:
            t=visit['end_time']
            for ev in post_events:
                ev['start_before']=t; ev['end_after']=t+ev['total']; t=ev['end_after']
                _add_event(current_trip,ev); current_trip['resupply_count'] += 1
            close_trip(current_trip, t, 'resupply')
            current_trip=new_trip(t,'resupply')
        # Terminal return is a separate event and closes the trip.
        terminal=None
        if info.get('return_to_depot') is not None:
            leg=info['return_to_depot']; svc=0.0
            terminal=dict(instance_id=identity['id'],scale=identity['fields'],method=method,
                          event_index=event_no,type='terminal_return',trigger='mission_complete',
                          trip_id=current_trip['trip_id'] if current_trip else None,
                          before_energy=after['energy']+float(leg.get('energy',0.)),
                          before_payload=after['payload'],
                          before_energy_ratio=(after['energy']+float(leg.get('energy',0.)))/env.uavs[0].max_energy,
                          before_payload_ratio=after['payload_ratio'],
                          arrival_energy=after['energy'], arrival_payload=after['payload'],
                          after_energy=after['energy'], after_payload=after['payload'],
                          after_energy_ratio=after['energy_ratio'], after_payload_ratio=after['payload_ratio'],
                          return_distance=float(leg.get('distance',0.)), return_energy=float(leg.get('energy',0.)),
                          return_nominal_time=float(leg.get('time_total',0.)), service_time=0.,
                          work=0., nonwork=float(leg.get('time_nonwork_base',0.)),
                          maneuver=float(leg.get('time_maneuver_extra',0.)), service=0.,
                          total=float(leg.get('time_total',0.)), distance=float(leg.get('distance',0.)),
                          start_before=None, end_after=None)
            terminal['start_before']=visit['end_time']; terminal['end_after']=visit['end_time']+terminal['total']
            event_no += 1; events.append(terminal)
            _add_event(current_trip,terminal)
            close_trip(current_trip,terminal['end_after'],'terminal')
            current_trip=None
        events.extend(step_events)
        if terminal is not None and current_trip is None: pass
        if done and terminal is None and not step_events:
            # A completed step without a terminal leg is a ledger inconsistency.
            raise ValueError('Mission completed without terminal return event')
    # Any open trip (should be none) is recorded.
    if current_trip is not None:
        close_trip(current_trip, env.get_summary()['total_time'], 'unclosed')
    summary=env.get_summary(); archived=best.get('summary') or {}
    diffs={k:abs(float(summary[k])-float(archived[k])) for k in LEDGER_KEYS if k in archived}
    max_diff=max(diffs.values()) if diffs else None
    replay=dict(instance_id=identity['id'],scale=identity['fields'],method=method,
                completed=bool(best.get('completed')),done=bool(done),
                returned_to_depot=bool(summary['returned_to_depot']),
                termination_reason=best.get('reason'), archived_total=archived.get('total_time'),
                replay_total=summary['total_time'], max_ledger_diff=max_diff,
                ledger_diffs=diffs, time_closure_error=summary['time_closure_error'],
                resupply_count=summary['resupply_count'], execution_count=summary['execution_count'],
                coverage_ratio=summary['coverage_ratio'], visits=len(visits), trips=len(trips), events=len(events))
    if max_diff is None or max_diff>TOL:
        raise ValueError(f'Replay archived ledger mismatch {identity} {diffs}')
    if not replay['completed'] or not replay['returned_to_depot']:
        raise ValueError(f'Archived trace not complete {identity}')
    # field-level aggregate
    field_rows=[]
    for f in env.fields:
        fvis=[v for v in visits if v['field_id']==f.field_id]
        uncovered={p.id for p in f.grid_points if not p.covered}
        field_rows.append(dict(instance_id=identity['id'],scale=identity['fields'],method=method,field_id=f.field_id,
            visits=len(fvis),new_area=sum(v['new_area'] for v in fvis),assigned_area=sum(v['assigned_area'] for v in fvis),
            trip_ids=sorted({v['trip_id'] for v in fvis}),uncovered_components=_components_for(tuple(uncovered),env,f.field_id),
            uncovered_points=len(uncovered),final_covered=not bool(uncovered)))
    return dict(replay=replay, visits=visits, events=events, trips=trips, fields=field_rows,
                summary=summary, archived_summary=archived)

def initial_features(env):
    fs=env.fields; depot=env.depot.position
    rows=[]
    for f in fs:
        rect=f.xy_polygon.minimum_rotated_rectangle
        coords=list(rect.exterior.coords); sides=[]
        for a,b in zip(coords,coords[1:]):
            sides.append(math.hypot(b[0]-a[0],b[1]-a[1]))
        sides=[s for s in sides if s>0]
        aspect=max(sides)/min(sides) if sides else None
        dist=float(f.xy_polygon.distance(Point(depot)))
        rows.append(dict(field_id=f.field_id,area=float(f.area),aspect_ratio=aspect,depot_distance=dist,
                         normalized_depot_distance=dist/math.sqrt(float(f.area)) if f.area>0 else None))
    areas=[r['area'] for r in rows]
    nearest=[]
    for i,f in enumerate(fs):
        ds=[f.xy_polygon.distance(g.xy_polygon) for j,g in enumerate(fs) if j!=i]
        nearest.append(min(ds) if ds else None)
    u=env.uavs[0]; qmax=float(u.max_payload)/float(u.payload_per_area) if u.payload_per_area else None
    for r,n in zip(rows,nearest):
        r['nearest_field_distance']=float(n) if n is not None else None
        r['nearest_normalized']=(float(n)/math.sqrt(r['area'])) if n is not None and r['area']>0 else None
        r['qmax_area']=qmax; r['area_over_qmax']=(r['area']/qmax) if qmax else None
    return dict(fields=rows,total_area=sum(areas),field_area_mean=mean(areas),field_area_std=pstdev(areas),
                field_area_cv=pstdev(areas)/mean(areas) if mean(areas) else None,
                max_area=max(areas),min_area=min(areas),qmax_area=qmax,
                payload_per_area=float(u.payload_per_area),max_payload=float(u.max_payload),
                field_area_over_qmax=[r['area_over_qmax'] for r in rows])

def load_frozen(manifest_path):
    from coverage_env.rl.evaluation.framework_comparison import load_panel
    manifest,panel=load_panel(manifest_path)
    return manifest,panel

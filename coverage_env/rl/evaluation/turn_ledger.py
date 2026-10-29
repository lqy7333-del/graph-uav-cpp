"""Three-level turn/distance ledger built from real execution records.

Design rules (from review):
* Read mappings with item access. A required field that is missing raises; it must
  never degrade into a zero metric (that is how a getattr bug produced a false
  "turns never change" result).
* A successful coverage with a positive area MUST carry a non-empty path and
  non-empty strokes, otherwise raise.
* Whole-task distance and heading MUST come from the full flown trajectory, not
  from concatenated per-visit coverage paths (that would drop resupply/return legs
  and invent links that were never flown).
* Turn events are identified on the full trajectory and then classified; segments
  never count turns on their own, so interfaces cannot be double counted.
* Resupply stops are flagged separately: a heading change across a stop is not a
  turn in continuous flight.
"""
from __future__ import annotations

import math

REQUIRED_KEYS = ("coverage_path", "coverage_strokes", "total_path",
                 "entry_transit_path", "exit_transit_path", "distance")


def require_record(record, *, covered_area: float, index: int | None = None) -> dict:
    """Validate one execution record and return it.

    Raises instead of returning empty geometry: an empty path for a successful
    positive-area coverage is a data error, not a zero-turn measurement.
    """
    if not isinstance(record, dict):
        raise TypeError(f"execution record must be a mapping, got {type(record).__name__}")
    missing = [k for k in REQUIRED_KEYS if k not in record]
    if missing:
        raise KeyError(f"execution record {index} is missing required keys: {missing}")
    if covered_area > 0:
        for key in ("coverage_path", "coverage_strokes"):
            value = record[key]
            if not value:
                raise ValueError(
                    f"execution record {index} covered {covered_area:.1f} m2 but "
                    f"{key!r} is empty; refusing to report zero turns"
                )
    return record


def _segments(path):
    return [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-9]


def _angle(seg_a, seg_b):
    na, nb = math.hypot(*seg_a), math.hypot(*seg_b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    dot = (seg_a[0] * seg_b[0] + seg_a[1] * seg_b[1]) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def heading_profile(path, stop_points=(), tol=1e-6):
    """Heading changes along one flown path, with turns at stops flagged.

    Returns (total_heading_deg, turn_events, stop_events) where ``turn_events`` are
    (index, degrees) for changes above tolerance and ``stop_events`` are the ones
    that occur where the aircraft was serviced (not a continuous-flight turn).
    """
    segs = _segments(path)
    total, turns, stops = 0.0, [], []
    for index, (a, b) in enumerate(zip(segs, segs[1:]), start=1):
        theta = _angle(a, b)
        if theta <= tol:
            continue
        total += theta
        at_stop = any(math.hypot(path[index][0] - p[0], path[index][1] - p[1]) <= 1e-6
                      for p in stop_points)
        (stops if at_stop else turns).append((index, theta))
    return total, turns, stops


def build_ledger(environment):
    """Return {visits, fields, mission} ledgers plus closure errors."""
    records = environment.execution_records
    planned = []
    for index, rec in enumerate(records):
        area = float(rec.get("coverage") or 0.0) if isinstance(rec, dict) else 0.0
        require_record(rec, covered_area=area, index=index)
        cov = [tuple(p) for p in rec["coverage_path"]]
        entry = [tuple(p) for p in rec["entry_transit_path"]]
        exit_ = [tuple(p) for p in rec["exit_transit_path"]]
        cov_head, cov_turns, _ = heading_profile(cov)
        planned.append({
            "index": index,
            "field_id": rec.get("field_id"),
            "entry_boundary_id": rec.get("entry_boundary_id"),
            "exit_boundary_id": rec.get("exit_boundary_id"),
            "covered_area": area,
            "n_strokes": len(rec["coverage_strokes"]),
            "coverage_reversals": _reversals(rec["coverage_strokes"]),
            "coverage_distance": _length(cov),
            "entry_distance": _length(entry),
            "exit_distance": _length(exit_),
            "coverage_heading_deg": cov_head,
            "coverage_turn_events": len(cov_turns),
        })

    summary = environment.get_summary()
    traj = [tuple(p) for p in (summary.get("uav_trajectory", {}).get("UAV-1") or [])]
    if len(traj) < 2:
        raise ValueError("full flown trajectory is empty; refusing to report a mission ledger")
    depot = tuple(environment.depot.get_position())
    stops = [depot] if summary.get("resupply_count", 0) else []
    full_heading, turns, stop_turns = heading_profile(traj, stops)
    full_distance = _length(traj)

    fields = {}
    for visit in planned:
        entry = fields.setdefault(visit["field_id"], {"visits": [], "covered_area": 0.0})
        entry["visits"].append(visit["index"])
        entry["covered_area"] += visit["covered_area"]
    mission = {
        "full_distance": full_distance,
        "sum_visit_distance": sum(v["coverage_distance"] for v in planned),
        "sum_entry_distance": sum(v["entry_distance"] for v in planned),
        "sum_exit_distance": sum(v["exit_distance"] for v in planned),
        "reported_total_distance": float(summary.get("total_distance", 0.0)),
        "full_heading_deg": full_heading,
        "sum_coverage_heading_deg": sum(v["coverage_heading_deg"] for v in planned),
        "n_turn_events": len(turns),
        "n_stop_events": len(stop_turns),
        "n_visits": len(planned),
        "n_fields": len(fields),
        "resupplies": int(summary.get("resupply_count", 0)),
    }
    mission["distance_closure_error"] = full_distance - mission["reported_total_distance"]
    mission["heading_coverage_gap_deg"] = full_heading - mission["sum_coverage_heading_deg"]
    return {"visits": planned, "fields": fields, "mission": mission}


def _length(path):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def _reversals(strokes):
    vectors = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        seg = (stroke[-1][0] - stroke[0][0], stroke[-1][1] - stroke[0][1])
        if math.hypot(*seg) > 1e-8:
            vectors.append(seg)
    return sum((a[0] * b[0] + a[1] * b[1]) / (math.hypot(*a) * math.hypot(*b)) < -0.99
               for a, b in zip(vectors, vectors[1:]))

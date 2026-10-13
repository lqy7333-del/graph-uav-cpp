"""Feasibility filtering for workload-free spatial policy actions."""

from __future__ import annotations

from math import hypot, sqrt

from coverage_env.rl.action_space import CoverageAction


def get_action_mask(actions, state_or_environment, *, return_diagnostics=False):
    """Return a 1/0 mask; workload feasibility is estimated by the environment."""
    environment = state_or_environment
    if hasattr(environment, "estimate_workload"):
        masks, diagnostics = [], []
        for action in actions:
            try:
                ratio = environment.estimate_workload(action)
                ok = ratio > 0.0
                reason = "feasible" if ok else "no positive workload is energy/reachability feasible"
                data = {"estimated_workload": ratio}
            except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                ok, reason, data = False, str(exc), {}
            masks.append(int(ok))
            diagnostics.append({"feasible": ok, "reason": reason, **data})
    elif isinstance(environment, dict):
        masks, diagnostics = _mask_from_state(actions, environment)
    else:
        raise TypeError("Mask evaluation expects an Environment or state dictionary.")
    return (masks, diagnostics) if return_diagnostics else masks


def _mask_from_state(actions, state):
    """Best-effort feasibility check using only serialized state fields."""
    fields = state.get("fields", {})
    boundaries = {}
    for boundary in state.get("boundaries", []):
        boundaries.setdefault(boundary["field_id"], {})[boundary["boundary_id"]] = boundary
    uavs = list(state.get("uavs", {}).values())
    uav = uavs[0] if uavs else {}
    position = uav.get("position") or state.get("depot", {}).get("position") or (0.0, 0.0)
    depot = state.get("depot", {}).get("position") or (0.0, 0.0)
    alpha = max(0.0, float(uav.get("alpha", 0.2)))
    beta = max(0.0, float(uav.get("beta", 0.02)))
    width = max(1e-9, float(uav.get("coverage_width", 10.0)))
    energy = float(uav.get("remaining_energy", 0.0))
    masks, diagnostics = [], []
    for action in actions:
        reason, details = "feasible", {}
        if not isinstance(action, CoverageAction):
            reason = "action type is invalid"
        else:
            field = fields.get(action.field_id)
            pair = boundaries.get(action.field_id, {})
            entry, exit_ = pair.get(action.entry_boundary_id), pair.get(action.exit_boundary_id)
            if field is None:
                reason = "unknown field"
            elif float(field.get("remaining_area", 0.0)) <= 0.0:
                reason = "field is complete"
            elif entry is None or exit_ is None or entry.get("length", 0) <= 0 or exit_.get("length", 0) <= 0:
                reason = "invalid boundary combination"
            else:
                reachable = float(field.get("reachable_area_by_boundary", {}).get(action.entry_boundary_id, 0.0))
                if reachable <= 0.0:
                    reason = "entry boundary has no reachable uncovered component"
                else:
                    entry_point = _closest_point(position, entry["start_point"], entry["end_point"])
                    distance = hypot(entry_point[0]-position[0], entry_point[1]-position[1])
                    return_distance = hypot(exit_["center"][0]-depot[0], exit_["center"][1]-depot[1])
                    reserve = alpha * return_distance
                    budget = energy - reserve - alpha * distance
                    min_area = min(reachable, float(field.get("remaining_area", 0.0)))
                    # State observations do not expose point areas; approximate
                    # the smallest executable region by one remaining cell.
                    point_count = max(1, int(field.get("uncovered_points", 1)))
                    probe = min_area / point_count
                    estimated = alpha * 1.25 * (probe / width + 2.0 * sqrt(probe)) + beta * probe
                    details = {"reachable_area": reachable, "return_energy_reserve": reserve,
                               "estimated_probe_energy": estimated}
                    if budget <= estimated:
                        reason = "no positive workload fits remaining energy after return reserve"
        ok = reason == "feasible"
        masks.append(int(ok))
        diagnostics.append({"feasible": ok, "reason": reason, **details})
    return masks, diagnostics


def _closest_point(point, start, end):
    dx, dy = end[0]-start[0], end[1]-start[1]
    denominator = dx*dx + dy*dy
    if denominator <= 0.0:
        return tuple(start)
    t = min(1.0, max(0.0, ((point[0]-start[0])*dx + (point[1]-start[1])*dy)/denominator))
    return start[0]+t*dx, start[1]+t*dy

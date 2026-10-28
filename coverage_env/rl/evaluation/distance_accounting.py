"""Disjoint distance buckets taken from simulator feedback, never inferred from centers."""
import math


def step_distances(info):
    """Return coverage, transit, resupply, final return; enforce per-step closure."""
    coverage = float(info.get("coverage_only_distance", 0.0))
    transit = float(info.get("transit_distance", 0.0))
    legs = float(info.get("resupply_or_return_distance", 0.0))
    final_return = float(info.get("final_return_distance", 0.0))
    resupply = legs - final_return
    total = float(info.get("path_length", 0.0))
    if resupply < -1e-6 or not math.isclose(coverage+transit+legs, total, abs_tol=1e-5, rel_tol=1e-8):
        raise ValueError("Distance metadata does not close; do not mix legacy path proxies and simulator buckets.")
    return coverage, transit, max(0.0, resupply), final_return

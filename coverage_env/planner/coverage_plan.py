"""Versioned, immutable geometry; no mutable GridPoint references or billed costs."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(',', ':'), default=str).encode()).hexdigest()


def fingerprints(env, field):
    """Include geometry, capabilities, cost configuration and dynamic mission state.

    Works with the lightweight preview sandbox used by the existing CC refiner.
    No identity-based hashing, so plans also work on an identical deep copy.
    """
    u = env.uavs[0]
    instance = _digest([
        [(f.field_id, f.xy_polygon.wkb_hex,
          [(p.id, p.x, p.y, p.point_area, p.neighbors) for p in f.grid_points],
          [(b.id, b.start_point, b.end_point) for b in f.boundaries]) for f in env.fields],
        env.depot.get_position(),
    ])
    config = _digest([u.uav_id, u.max_energy, u.max_payload, u.alpha, u.beta,
                      u.payload_per_area, u.coverage_width, u.speed,
                      getattr(env, 'rl_config', {})])
    state = _digest([
        instance, config, field.field_id,
        [(f.field_id, [(p.id, p.covered) for p in f.grid_points]) for f in env.fields],
        u.get_state(), u.total_energy_consumed, u.total_payload_consumed,
        getattr(env, '_heading_valid', {}), getattr(env, '_mission_trajectories', {}),
        getattr(env, '_time_totals', {}), getattr(env, '_resupply_count', 0),
    ])
    return instance, config, state


@dataclass(frozen=True)
class CoveragePlan:
    schema_version: int
    source: str
    task_id: str
    field_id: str | int
    entry_boundary_id: str
    exit_boundary_id: str
    point_ids: tuple
    entry_boundary_point: tuple
    entry_transit_path: tuple
    coverage_path: tuple
    exit_transit_path: tuple
    coverage_strokes: tuple
    scan_transit_paths: tuple
    orientation: float
    scan_order_ids: tuple
    growth_edge_id: str
    max_reachable_area: float
    instance_fingerprint: str
    config_fingerprint: str
    state_fingerprint: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        def immutable(value):
            return tuple(immutable(x) for x in value) if isinstance(value, (list, tuple)) else value
        for key in ('point_ids', 'entry_boundary_point', 'entry_transit_path', 'coverage_path', 'exit_transit_path',
                    'coverage_strokes', 'scan_transit_paths', 'scan_order_ids'):
            data[key] = immutable(data[key])
        if data['schema_version'] != 1:
            raise ValueError('Unsupported coverage plan version')
        return cls(**data)

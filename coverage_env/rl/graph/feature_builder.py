"""Node and global feature construction for field-boundary graphs."""

from __future__ import annotations

from math import atan2, ceil, degrees, hypot, pi

import torch


def _angle_to_unit(angle_degrees: float) -> float:
    """Map an unoriented axis angle in degrees to a stable [-1, 1] value."""
    return max(-1.0, min(1.0, float(angle_degrees) / 90.0))


class FieldFeatureBuilder:
    """Build seven-dimensional Field features, including its static center."""

    def __init__(self, max_area: float, max_perimeter: float,
                 position_scale_m: float = 10_000.0) -> None:
        if max_area <= 0.0 or max_perimeter <= 0.0:
            raise ValueError("max_area and max_perimeter must be positive.")
        if position_scale_m <= 0.0:
            raise ValueError("position_scale_m must be positive.")
        self.max_area = float(max_area)
        self.max_perimeter = float(max_perimeter)
        self.position_scale_m = float(position_scale_m)

    def build(self, environment) -> torch.Tensor:
        """Return area/geometry, remaining work, and per-parcel look-ahead.

        The look-ahead columns exist because the audit showed the valuable
        improvements are not orderings within a fixed trip structure but *changes to
        the structure itself*: the order decides how much of each parcel gets covered
        before the payload runs out, which decides the total coverage distance. A
        policy that only sees the current payload cannot anticipate that, so each
        parcel now reports whether one visit would finish it, what would be left
        afterwards, and whether the UAV could still reach any other parcel.
        """
        rows = []
        uav = environment.uavs[0] if environment.uavs else None
        remaining_payload = float(getattr(uav, "remaining_payload", 0.0)) if uav else 0.0
        max_payload = float(getattr(uav, "max_payload", 1.0)) if uav else 1.0
        density = float(getattr(uav, "payload_per_area", 0.004)) if uav else 0.004
        needs = []
        for candidate_field in environment.fields:
            uncovered = sum(
                point.point_area for point in candidate_field.grid_points if not point.covered
            )
            needs.append(uncovered * density)
        for index, field in enumerate(environment.fields):
            total_grid_area = sum(point.point_area for point in field.grid_points)
            remaining = sum(point.point_area for point in field.grid_points if not point.covered)
            remaining_ratio = (
                remaining / total_grid_area if total_grid_area > 0.0 else 1.0
            )
            compactness = (
                4.0 * pi * field.area / (field.perimeter ** 2)
                if field.perimeter > 0.0 else 0.0
            )
            rows.append([
                min(1.0, field.area / self.max_area),
                min(1.0, field.perimeter / self.max_perimeter),
                max(0.0, min(1.0, compactness)),
                _angle_to_unit(field.orientation),
                max(0.0, min(1.0, remaining_ratio)),
                max(-1.0, min(1.0, float(field.centroid[0]) / self.position_scale_m)),
                max(-1.0, min(1.0, float(field.centroid[1]) / self.position_scale_m)),
                # ---- look-ahead ----
                1.0 if remaining_payload + 1e-9 >= needs[index] else 0.0,
                max(0.0, min(1.0, (remaining_payload - needs[index]) / max_payload))
                if max_payload > 0.0 else 0.0,
                1.0 if any(
                    other != index
                    and remaining_payload - needs[index] + 1e-9 >= needs[other]
                    and needs[other] > 1e-9
                    for other in range(len(needs))
                ) else 0.0,
            ])
        return torch.tensor(rows, dtype=torch.float32).reshape(-1, 10)


class BoundaryFeatureBuilder:
    """Build four-dimensional Boundary features, including its static center."""

    def __init__(self, max_boundary_length: float,
                 position_scale_m: float = 10_000.0) -> None:
        if max_boundary_length <= 0.0:
            raise ValueError("max_boundary_length must be positive.")
        if position_scale_m <= 0.0:
            raise ValueError("position_scale_m must be positive.")
        self.max_boundary_length = float(max_boundary_length)
        self.position_scale_m = float(position_scale_m)

    def build(self, environment, boundaries=None) -> torch.Tensor:
        """Return rows [length_norm, orientation, center_x, center_y]."""
        boundaries = boundaries or [
            (field, boundary)
            for field in environment.fields
            for boundary in field.boundaries
        ]
        rows = []
        for _field, boundary in boundaries:
            angle = degrees(atan2(
                boundary.tangent_vector[1], boundary.tangent_vector[0]
            ))
            # Boundary segments are unoriented axes: normalize modulo 180 degrees.
            angle = (angle + 90.0) % 180.0 - 90.0
            rows.append([
                min(1.0, boundary.length / self.max_boundary_length),
                _angle_to_unit(angle),
                max(-1.0, min(1.0, float(boundary.center[0]) / self.position_scale_m)),
                max(-1.0, min(1.0, float(boundary.center[1]) / self.position_scale_m)),
                # Distance from this gate to the depot: leaving through a gate close
                # to the depot is what makes a resupply leg cheap, which the beam
                # search exploited for 462-698 m per episode.
                max(0.0, min(1.0, hypot(
                    boundary.center[0] - environment.depot.get_position()[0],
                    boundary.center[1] - environment.depot.get_position()[1],
                ) / self.position_scale_m)),
            ])
        return torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)


class UavNodeFeatureBuilder:
    """Build the six-dimensional UAV node feature vector.

    The vector carries the UAV's *state* only. Its position is deliberately left
    out: every other relation in the graph is relative, so keeping absolute
    coordinates out preserves translation invariance, and the geometry the policy
    actually needs travels on the ``UAV -> Field`` / ``UAV -> Boundary`` edge
    features instead (relative offset plus normalized distance).
    """

    def __init__(
        self,
        max_energy_scale: float = 1_800.0,
        max_payload_scale: float = 60.0,
        speed_scale: float = 10.0,
        coverage_width_scale: float = 12.0,
        alpha_scale: float = 0.10,
        beta_scale: float = 0.024,
        payload_per_area_scale: float = 0.004,
    ) -> None:
        if min(
            max_energy_scale, max_payload_scale, speed_scale, coverage_width_scale
        ) <= 0.0:
            raise ValueError("UAV feature scales must be positive.")
        self.max_energy_scale = float(max_energy_scale)
        self.max_payload_scale = float(max_payload_scale)
        self.speed_scale = float(speed_scale)
        self.coverage_width_scale = float(coverage_width_scale)
        self.alpha_scale = float(alpha_scale)
        self.beta_scale = float(beta_scale)
        self.payload_per_area_scale = float(payload_per_area_scale)

    def build(self, environment) -> torch.Tensor:
        """Return ``[1, 9]``: state, capability, and consumption-rate channels."""
        if not environment.uavs:
            raise ValueError("Cannot build a UAV node without a UAV in the environment.")
        uav = environment.uavs[0]
        max_energy = float(getattr(uav, "max_energy", 0.0))
        max_payload = float(getattr(uav, "max_payload", 0.0))
        remaining_energy = float(getattr(uav, "remaining_energy", 0.0))
        remaining_payload = float(
            getattr(uav, "remaining_payload", max_payload)
        )

        def ratio(remaining, maximum):
            return max(0.0, min(1.0, remaining / maximum)) if maximum > 0.0 else 0.0

        def norm(value, scale):
            return max(0.0, min(1.0, float(value) / scale))

        return torch.tensor([[
            ratio(remaining_energy, max_energy),
            ratio(remaining_payload, max_payload),
            norm(getattr(uav, "speed", 0.0), self.speed_scale),
            norm(getattr(uav, "coverage_width", 0.0), self.coverage_width_scale),
            norm(max_energy, self.max_energy_scale),
            norm(max_payload, self.max_payload_scale),
            norm(getattr(uav, "alpha", 0.0), self.alpha_scale),
            norm(getattr(uav, "beta", 0.0), self.beta_scale),
            norm(getattr(uav, "payload_per_area", 0.0),
                 self.payload_per_area_scale),
            # ---- look-ahead ----
            # How many more payload-loads the remaining work needs: the single
            # clearest summary of the trip structure the policy has to plan.
            min(1.0, ceil(
                (sum(
                    sum(point.point_area for point in f.grid_points if not point.covered)
                    for f in environment.fields
                ) * float(getattr(uav, "payload_per_area", 0.004)))
                / max(remaining_payload, 1e-9)
            ) / 8.0) if remaining_payload > 1e-9 else 1.0,
            min(1.0, (
                sum(sum(point.point_area for point in f.grid_points if not point.covered)
                    for f in environment.fields)
                / max(sum(sum(point.point_area for point in f.grid_points)
                          for f in environment.fields), 1e-9)
            )),
        ]], dtype=torch.float32)


class GlobalFeatureBuilder:
    """Build the five-value UAV/depot global feature vector."""

    def __init__(self, position_scale_m: float = 10_000.0) -> None:
        if position_scale_m <= 0.0:
            raise ValueError("position_scale_m must be positive.")
        self.position_scale_m = float(position_scale_m)

    def build(self, environment) -> torch.Tensor:
        """Return ``[uav_x, uav_y, energy_ratio, depot_x, depot_y]``."""
        if environment.uavs:
            uav = environment.uavs[0]
            position = uav.current_position or environment.depot.get_position()
            energy_ratio = uav.remaining_energy / uav.max_energy
        else:
            position = environment.depot.get_position()
            energy_ratio = 0.0
        depot = environment.depot.get_position()

        def normalize(value):
            return max(-1.0, min(1.0, float(value) / self.position_scale_m))

        return torch.tensor([
            normalize(position[0]), normalize(position[1]),
            max(0.0, min(1.0, energy_ratio)),
            normalize(depot[0]), normalize(depot[1]),
        ], dtype=torch.float32)

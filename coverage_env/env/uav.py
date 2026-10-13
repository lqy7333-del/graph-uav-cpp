"""UAV state and a simple distance/coverage energy model."""

from __future__ import annotations

from math import hypot, isfinite
from typing import Mapping, Optional, Sequence, Tuple, Union

Position2D = Tuple[float, float]
Path = Sequence[Position2D]


class UAV:
    """An energy-limited UAV with path execution and coverage state.

    Energy is measured in Wh. The default model parameters are ``alpha=0.2``
    Wh/m and ``beta=0.02`` Wh/m². Supply explicit values for application data.
    Methods do not modify static flight and energy parameters.
    """

    def __init__(
        self,
        uav_id: Union[str, int],
        max_energy: float,
        speed: float,
        coverage_width: float,
        sensor_width: float,
        energy_consumption_parameters: Optional[Mapping[str, float]] = None,
        *,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        max_payload: Optional[float] = None,
        payload_per_area: float = 0.0,
    ) -> None:
        """Initialize static vehicle parameters and an empty dynamic state."""
        self.uav_id = uav_id
        self.max_energy = self._positive_value("max_energy", max_energy)
        self.speed = self._positive_value("speed", speed)
        self.coverage_width = self._positive_value("coverage_width", coverage_width)
        self.sensor_width = self._positive_value("sensor_width", sensor_width)
        self.max_payload = self._positive_value(
            "max_payload", max_energy if max_payload is None else max_payload
        )
        self.payload_per_area = self._nonnegative_value(
            "payload_per_area", payload_per_area
        )

        parameters = dict(energy_consumption_parameters or {})
        self.alpha = self._nonnegative_value(
            "alpha", parameters.get("alpha", 0.2) if alpha is None else alpha
        )
        self.beta = self._nonnegative_value(
            "beta", parameters.get("beta", 0.02) if beta is None else beta
        )
        parameters.update({"alpha": self.alpha, "beta": self.beta})
        self.energy_consumption_parameters = parameters

        self.current_position: Optional[Position2D] = None
        self.remaining_energy = self.max_energy
        self.remaining_payload = self.max_payload
        self.total_distance = 0.0
        self.covered_area = 0.0
        self.visited_fields: list[Union[str, int]] = []
        self.current_path: list[Position2D] = []
        self.last_energy_consumption = 0.0
        self.total_energy_consumed = 0.0

    @staticmethod
    def _positive_value(name: str, value: float) -> float:
        """Validate a finite positive scalar parameter."""
        converted = float(value)
        if not isfinite(converted) or converted <= 0.0:
            raise ValueError(f"{name} must be a finite positive number.")
        return converted

    @staticmethod
    def _nonnegative_value(name: str, value: float) -> float:
        """Validate a finite non-negative energy coefficient."""
        converted = float(value)
        if not isfinite(converted) or converted < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number.")
        return converted

    @staticmethod
    def _validate_position(position: Position2D) -> Position2D:
        """Convert a local XY coordinate to a finite float pair."""
        if len(position) != 2:
            raise ValueError("position must contain exactly (x, y).")
        x, y = float(position[0]), float(position[1])
        if not isfinite(x) or not isfinite(y):
            raise ValueError("position coordinates must be finite.")
        return x, y

    def calculate_energy(self, distance: float, area: float) -> float:
        """Return ``alpha * distance + beta * area`` in Wh."""
        distance = self._nonnegative_value("distance", distance)
        area = self._nonnegative_value("area", area)
        return self.alpha * distance + self.beta * area

    def execute_path(self, path: Path, area: float = 0.0) -> dict:
        """Execute a waypoint sequence and update energy and position.

        If the first waypoint differs from the current position, the direct
        connector to that waypoint is included in the executed distance. The
        optional ``area`` is the area covered while traversing the path.
        Insufficient energy raises ``ValueError`` without partially updating
        the UAV state. Returns a compact execution-feedback dictionary.
        """
        if not path:
            raise ValueError("path must contain at least one waypoint.")
        waypoints = [self._validate_position(point) for point in path]
        if self.current_position is not None and waypoints[0] != self.current_position:
            executed_path = [self.current_position, *waypoints]
        else:
            executed_path = waypoints

        distance = sum(
            hypot(end[0] - start[0], end[1] - start[1])
            for start, end in zip(executed_path[:-1], executed_path[1:])
        )
        area = self._nonnegative_value("area", area)
        energy = self.calculate_energy(distance, area)
        payload = self.payload_per_area * area
        if energy > self.remaining_energy:
            raise ValueError(
                f"UAV {self.uav_id!r} needs {energy:.3f} Wh but has "
                f"{self.remaining_energy:.3f} Wh remaining."
            )
        if payload > self.remaining_payload:
            raise ValueError(
                f"UAV {self.uav_id!r} needs {payload:.3f} payload but has "
                f"{self.remaining_payload:.3f} remaining."
            )

        self.current_position = waypoints[-1]
        self.current_path = list(executed_path)
        self.total_distance += distance
        self.remaining_energy -= energy
        self.remaining_payload -= payload
        self.covered_area += area
        self.last_energy_consumption = energy
        self.total_energy_consumed += energy
        return {
            "path_length": distance,
            "energy_consumption": energy,
            "remaining_energy": self.remaining_energy,
            "final_position": self.current_position,
        }

    def reset(self, position: Position2D) -> None:
        """Restore the UAV and its cumulative task state at ``position``."""
        self.current_position = self._validate_position(position)
        self.remaining_energy = self.max_energy
        self.remaining_payload = self.max_payload
        self.total_distance = 0.0
        self.covered_area = 0.0
        self.visited_fields.clear()
        self.current_path.clear()
        self.last_energy_consumption = 0.0
        self.total_energy_consumed = 0.0

    def get_state(self) -> dict:
        """Return dynamic state plus the static capability and payload fields."""
        energy_ratio = self.remaining_energy / self.max_energy if self.max_energy > 0.0 else 0.0
        payload_ratio = self.remaining_payload / self.max_payload if self.max_payload > 0.0 else 0.0
        return {
            "position": self.current_position,
            "remaining_energy": self.remaining_energy,
            "max_energy": self.max_energy,
            "speed": self.speed,
            "coverage_width": self.coverage_width,
            "max_payload": self.max_payload,
            "remaining_payload": self.remaining_payload,
            "energy_ratio": energy_ratio,
            "payload_ratio": payload_ratio,
            "total_distance": self.total_distance,
            "covered_area": self.covered_area,
        }

    def __repr__(self) -> str:
        return (
            f"UAV(uav_id={self.uav_id!r}, position={self.current_position!r}, "
            f"remaining_energy={self.remaining_energy:.2f} Wh)"
        )

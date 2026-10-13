"""High-level field, boundary, and workload action representation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Union


@dataclass(frozen=True)
class CoverageConfiguration:
    """A coverage action without preselected entry or exit coordinates.

    The environment resolves an entry point from the UAV state and entry
    boundary. The exit point is chosen after the coverage path is generated.
    ``workload_ratio`` is the fraction of the field's *remaining discrete area*
    requested by this action.
    """

    field_id: Union[str, int]
    entry_boundary_id: str
    exit_boundary_id: str
    workload_ratio: float

    def __post_init__(self) -> None:
        """Validate boundary identifiers and the requested area fraction."""
        if not self.entry_boundary_id:
            raise ValueError("entry_boundary_id must not be empty.")
        if not self.exit_boundary_id:
            raise ValueError("exit_boundary_id must not be empty.")
        ratio = float(self.workload_ratio)
        if not isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ValueError("workload_ratio must be finite and in (0, 1].")
        object.__setattr__(self, "workload_ratio", ratio)

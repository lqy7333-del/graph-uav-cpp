"""Generate workload-free spatial decisions for the policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class CoverageAction:
    """Policy action selecting a field and its entry/exit boundaries."""

    field_id: Union[str, int]
    entry_boundary_id: str
    exit_boundary_id: str

    def __post_init__(self) -> None:
        if not self.entry_boundary_id or not self.exit_boundary_id:
            raise ValueError("Entry and exit boundary ids must not be empty.")


class ActionGenerator:
    """Enumerate ``(field, entry boundary, exit boundary)`` policy actions."""

    def generate(self, environment_or_state) -> list[CoverageAction]:
        """Generate ordered boundary pairs for each field with work remaining."""
        if hasattr(environment_or_state, "fields"):
            records = [
                (field.field_id, bool(field.get_uncovered_points()),
                 [boundary.id for boundary in field.boundaries])
                for field in environment_or_state.fields
            ]
        elif isinstance(environment_or_state, dict):
            by_field = {}
            for boundary in environment_or_state.get("boundaries", []):
                by_field.setdefault(boundary["field_id"], []).append(
                    boundary["boundary_id"]
                )
            records = [
                (field_id, float(field.get("remaining_area", 0.0)) > 0.0,
                 by_field.get(field_id, []))
                for field_id, field in environment_or_state.get("fields", {}).items()
            ]
        else:
            raise TypeError("generate expects an Environment or state dictionary.")
        return [
            CoverageAction(field_id, entry, exit_)
            for field_id, available, boundaries in records if available
            for entry in boundaries for exit_ in boundaries
        ]

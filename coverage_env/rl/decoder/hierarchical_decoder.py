"""Factorised decode over the ``(field, entry, exit)`` product space.

Why
---
A flat softmax over every feasible ``(field, entry, exit)`` candidate dilutes the
one decision that decides the tour: *which parcel to visit next*. Measured on this
task, a step offers ~41 feasible candidates for ~3 unfinished parcels, i.e. ~14
``(entry, exit)`` variants per parcel -- and those variants move the **immediate**
transit cost that dominates the per-step reward, while the payoff of a good parcel
order is deferred. The field decision was therefore a 1/14 slice of the output and
most of the gradient went into picking entry/exit points, which two-opt already
showed are good enough (reordering alone closes 100% of the remaining gap).

How
---
Decode as a chain: ``p(field) * p(entry | field) * p(exit | field, entry)``. By the
chain rule this is **exact**, not an approximation:

* the joint over candidates is ``p_F(f) * p_E(e|f) * p_X(x|f,e)``, which still sums
  to one because every level is a proper conditional over its feasible children,
  so ``sample_action``/``greedy_action``/``top_k_actions`` and PPO's log-prob
  lookup keep working on the flat vector unchanged;
* the joint entropy equals ``H(F) + H(E|F) + H(X|F,E)``, so PPO's entropy term --
  and ``entropy_coef`` -- needs no re-tuning.

The benefit is inductive, not expressive: separate heads give the field decision
its own gradient instead of a 1/14 share of a product-space softmax.

Note that each level is degenerate when it has a single feasible child: with one
unfinished parcel ``p(field)`` is exactly 1 and the field head receives no
gradient, which is correct rather than buggy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

REMAINING_FEATURE_DIM = 4


def segment_softmax(
    scores: torch.Tensor,
    segments: torch.Tensor,
    mask: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Softmax of ``scores`` within each segment, ignoring masked entries.

    Segments with no feasible entry yield all zeros rather than NaN, which keeps a
    degenerate level harmless instead of poisoning the whole batch.
    """
    filled = scores.masked_fill(~mask, float("-inf"))
    maxima = torch.full(
        (num_segments,), float("-inf"), device=scores.device, dtype=scores.dtype
    )
    maxima = maxima.scatter_reduce(0, segments, filled, reduce="amax")
    maxima = torch.where(torch.isinf(maxima), torch.zeros_like(maxima), maxima)
    weights = torch.exp(filled - maxima[segments]) * mask.to(scores.dtype)
    totals = torch.zeros(num_segments, device=scores.device, dtype=scores.dtype)
    totals = totals.index_add(0, segments, weights)
    return weights / totals[segments].clamp_min(1e-12)


def segment_max(values, segments, mask, num_segments):
    """Max-pool ``values`` within each segment, ignoring masked entries.

    Used to summarise a parent's children: a parcel is as attractive as its *best*
    entry/exit pair, not its average one.
    """
    filled = values.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    width = values.shape[1]
    pooled = torch.full(
        (num_segments, width), float("-inf"), device=values.device, dtype=values.dtype
    )
    pooled = pooled.scatter_reduce(
        0, segments.unsqueeze(-1).expand(-1, width), filled, reduce="amax"
    )
    return torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)


def remaining_set_features(
    field_indices: torch.Tensor,
    graph,
    mask: torch.Tensor | None = None,
):
    """Return ``[C, 4]`` describing where the unfinished work sits.

    ``[distance to the remaining centroid, distance to the farthest remaining
    parcel, remaining fraction, remaining spread]`` in units of ``position_scale_m``
    (the field coordinates are already normalised by it).

    The pooled global context only carries aggregate pair-distance statistics, so
    without these the field head cannot tell whether a candidate sits with the rest
    of the work or off on its own -- exactly the information a tour decision needs.
    """
    positions = graph.x_field[:, 5:7]
    remaining = graph.x_field[:, 4] > 1e-6
    if not bool(remaining.any()):
        return positions.new_zeros((field_indices.shape[0], REMAINING_FEATURE_DIM))
    remaining_positions = positions[remaining]
    centroid = remaining_positions.mean(dim=0)
    spread = torch.cdist(remaining_positions, remaining_positions).max()
    candidate_positions = positions[field_indices]
    to_centroid = (candidate_positions - centroid).norm(dim=-1)
    to_farthest = torch.cdist(candidate_positions, remaining_positions).max(dim=-1).values
    fraction = remaining_positions.shape[0] / max(1, positions.shape[0])
    ones = torch.ones_like(to_centroid)
    return torch.stack(
        [to_centroid, to_farthest, ones * fraction, ones * spread], dim=-1
    )


class HierarchicalActionDecoder(nn.Module):
    """Score ``p(field)``, ``p(entry | field)`` and ``p(exit | field, entry)``.

    Each head reads only the slots it needs from the shared action embedding
    ``[z_F, z_B(entry)+r_entry, z_B(exit)+r_exit]``, so the entry head cannot be
    influenced by the exit choice and the field head sees neither.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        remaining_dim: int = REMAINING_FEATURE_DIM,
        head_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        head_hidden = int(head_hidden or hidden_dim)
        # Parcel head sees a pooled summary of that parcel's entry/exit options, so
        # "which parcel next" knows how good its gate is. Scoring parcels from z_F
        # alone made the order decision blind to the entry/exit cost -- exactly the
        # dependency that makes "parcel first, then edge" work.
        self.child_projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        # z_F + z_G + remaining-set geometry + pooled child summary
        self.field_head = self._mlp(self.hidden_dim * 3 + remaining_dim, head_hidden)
        # z_F + z_entry + z_G + pooled exits available behind this entry
        self.entry_head = self._mlp(self.hidden_dim * 4, head_hidden)
        # z_F + z_entry + z_exit + z_G
        self.exit_head = self._mlp(self.hidden_dim * 4, head_hidden)

    @staticmethod
    def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def grouping(actions):
        """Map every candidate onto its field and ``(field, entry)`` groups.

        Returns ``(field_segments, field_of_candidate_fields,
        entry_segments, field_representatives, entry_representatives)``: per
        candidate segment ids plus one representative candidate per group, whose
        embedding row supplies that group's ``z_F``/``z_entry``.
        """
        field_lookup: dict = {}
        field_segments: list[int] = []
        field_representatives: list[int] = []
        entry_lookup: dict = {}
        entry_segments: list[int] = []
        entry_representatives: list[int] = []
        for index, action in enumerate(actions):
            key = action.field_id
            if key not in field_lookup:
                field_lookup[key] = len(field_representatives)
                field_representatives.append(index)
            field_segments.append(field_lookup[key])
            entry_key = (action.field_id, action.entry_boundary_id)
            if entry_key not in entry_lookup:
                entry_lookup[entry_key] = len(entry_representatives)
                entry_representatives.append(index)
            entry_segments.append(entry_lookup[entry_key])
        field_of_entry_group = [field_segments[index] for index in entry_representatives]
        return (
            field_segments,
            entry_segments,
            field_representatives,
            entry_representatives,
            field_of_entry_group,
        )

    def forward(self, actions, action_vectors, global_embedding, mask, graph):
        """Return ``(joint probabilities, extras)`` over the candidate list."""
        device = action_vectors.device
        hidden = self.hidden_dim
        (
            field_segments,
            entry_segments,
            field_representatives,
            entry_representatives,
            field_of_entry_group,
        ) = self.grouping(actions)
        field_segment_tensor = torch.tensor(field_segments, dtype=torch.long, device=device)
        entry_segment_tensor = torch.tensor(entry_segments, dtype=torch.long, device=device)
        num_fields = len(field_representatives)
        num_entries = len(entry_representatives)

        z_field = action_vectors[:, :hidden]
        z_entry = action_vectors[:, hidden:2 * hidden]
        z_exit = action_vectors[:, 2 * hidden:]
        context = global_embedding.unsqueeze(0).expand(action_vectors.shape[0], -1)

        # Level 1: parcels. A parcel is feasible when any of its variants is.
        field_indices = torch.tensor(
            [graph.field_id_to_index[actions[index].field_id]
             for index in field_representatives],
            dtype=torch.long, device=device,
        )
        remaining = remaining_set_features(field_indices, graph)
        field_vectors = z_field[torch.tensor(field_representatives, dtype=torch.long, device=device)]
        # Parent sees its best child: max-pool the entry/exit slots over this
        # parcel's feasible candidates. This is the reverse half of the dependency.
        pooled_children = self.child_projection(torch.cat([
            segment_max(z_entry, field_segment_tensor, mask, num_fields),
            segment_max(z_exit, field_segment_tensor, mask, num_fields),
        ], dim=-1))
        field_logits = self.field_head(
            torch.cat([
                field_vectors,
                global_embedding.unsqueeze(0).expand(field_vectors.shape[0], -1),
                remaining,
                pooled_children,
            ], dim=-1)
        ).squeeze(-1)
        field_feasible = torch.zeros(num_fields, device=device)
        field_feasible = field_feasible.index_add(
            0, field_segment_tensor, mask.to(field_feasible.dtype)
        ) > 0
        # Level 1 is a single softmax across the parcels -- not a per-parcel
        # segment softmax, which would trivially return 1 for every parcel.
        field_probabilities = torch.softmax(
            field_logits.masked_fill(~field_feasible, float("-inf")), dim=0
        )
        field_probabilities = torch.where(
            field_feasible, field_probabilities, torch.zeros_like(field_probabilities)
        )

        # Level 2: entry boundary, conditioned on the parcel. Scored once per
        # (parcel, entry) group and normalised across the groups of that parcel.
        entry_representative_tensor = torch.tensor(
            entry_representatives, dtype=torch.long, device=device
        )
        # An entry is as good as the exits it leads to, so it also sees its own
        # pooled exit summary.
        pooled_exits = segment_max(z_exit, entry_segment_tensor, mask, num_entries)
        entry_logits = self.entry_head(torch.cat([
            z_field[entry_representative_tensor],
            z_entry[entry_representative_tensor],
            global_embedding.unsqueeze(0).expand(num_entries, -1),
            pooled_exits,
        ], dim=-1)).squeeze(-1)
        entry_group_feasible = torch.zeros(num_entries, device=device)
        entry_group_feasible = entry_group_feasible.index_add(
            0, entry_segment_tensor, mask.to(entry_group_feasible.dtype)
        ) > 0
        entry_group_probabilities = segment_softmax(
            entry_logits,
            torch.tensor(field_of_entry_group, dtype=torch.long, device=device),
            entry_group_feasible,
            num_fields,
        )
        entry_probabilities = entry_group_probabilities[entry_segment_tensor]

        # Level 3: exit boundary, conditioned on (parcel, entry).
        exit_logits = self.exit_head(
            torch.cat([z_field, z_entry, z_exit, context], dim=-1)
        ).squeeze(-1)
        exit_probabilities = segment_softmax(
            exit_logits, entry_segment_tensor, mask, num_entries
        )

        joint = (
            field_probabilities[field_segment_tensor]
            * entry_probabilities
            * exit_probabilities
        )
        return joint, {
            "field_probabilities": field_probabilities,
            "field_logits": field_logits,
            "field_segments": field_segment_tensor,
            "entry_probabilities": entry_probabilities,
            "exit_probabilities": exit_probabilities,
            "field_representatives": field_representatives,
            "entry_representatives": entry_representatives,
        }


def checkpoint_uses_hierarchical(state_dict) -> bool:
    """Detect which decoder a saved actor state dict belongs to.

    Checkpoints record only their parameter names, so loading a flat-decoder
    checkpoint into the factored decoder fails with a wall of missing keys. Tools
    that load a checkpoint for evaluation use this to pick the right decoder
    instead of relying on a flag somebody has to remember.
    """
    return any(str(key).startswith("hierarchical_decoder.") for key in state_dict)

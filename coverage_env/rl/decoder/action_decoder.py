"""Coverage Configuration score decoder and categorical actor interface."""

from __future__ import annotations

import torch
import torch.nn as nn

from coverage_env.rl.action_mask import get_action_mask
from coverage_env.rl.action_space import ActionGenerator, CoverageAction
from coverage_env.rl.decoder.action_embedding import ActionEmbedding
from coverage_env.rl.decoder.global_state_encoder import (
    GlobalStateEncoder,
    build_depot_feature,
    build_graph_context,
    build_uav_feature,
)
from coverage_env.rl.decoder.value_encoder import ValueEncoder
from coverage_env.rl.encoder.hetero_encoder import CoverageGraphEncoder
from coverage_env.rl.graph.graph_builder import GraphBuilder


class ActionDecoder(nn.Module):
    """Map a candidate action plus global context to one scalar score."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, action_input):
        """Return scores with shape ``[N_candidate, 1]``."""
        return self.network(action_input)


class CoveragePolicy(nn.Module):
    """Produce a masked probability distribution over spatial CoverageActions.

    The policy contains a graph encoder and score decoder, but no sampling
    objective, optimizer, PPO logic, or training loop. Bind an Environment at
    construction or supply it in ``forward`` so the graph can be rebuilt from
    current geometry and state.
    """

    def __init__(
        self,
        environment=None,
        *,
        config: dict | None = None,
        encoder: CoverageGraphEncoder | None = None,
    ) -> None:
        super().__init__()
        config = config or {}
        self.rl_config = config.get("rl", config)
        self.environment = environment
        self.graph_builder = GraphBuilder(config)
        encoder_config = self.rl_config.get("encoder", {})
        self.encoder = encoder or CoverageGraphEncoder.from_config(encoder_config)
        self.action_generator = ActionGenerator()
        hidden_dim = self.encoder.hidden_dim
        self.action_embedding = ActionEmbedding(hidden_dim=hidden_dim)
        self.global_state_encoder = GlobalStateEncoder(
            graph_dim=2 * hidden_dim, uav_dim=10, depot_dim=2, hidden_dim=hidden_dim
        )
        decoder_input_dim = self.action_embedding.output_dim + hidden_dim
        self.action_decoder = ActionDecoder(decoder_input_dim, hidden_dim)
        self.value_encoder = ValueEncoder(hidden_dim)

    def forward(
        self, state: dict, environment=None, graph=None,
        candidate_actions=None, feasible_mask=None,
    ):
        """Encode state and return candidate actions, mask, scores, and probs."""
        environment = environment or self.environment
        if graph is None:
            if environment is None:
                raise ValueError(
                    "CoveragePolicy needs an Environment or prebuilt graph."
                )
            graph = self.graph_builder.build_graph(environment)

        device = next(self.parameters()).device
        if hasattr(graph, "to"):
            graph = graph.to(device)
        embeddings = self.encoder(graph)
        actions = candidate_actions or self.action_generator.generate(
            environment if environment is not None else state
        )
        mask_values = feasible_mask
        if mask_values is None:
            mask_values = get_action_mask(
                actions, environment if environment is not None else state
            )
        mask = torch.tensor(mask_values, dtype=torch.bool, device=device)

        global_embedding = self._build_global_embedding(embeddings, environment, state)
        action_vectors = self.action_embedding(
            actions,
            embeddings["field_embedding"],
            embeddings["boundary_embedding"],
            graph,
        )
        action_context = torch.cat([
            action_vectors,
            global_embedding.unsqueeze(0).expand(action_vectors.shape[0], -1),
        ], dim=-1)
        scores = self.action_decoder(action_context)
        masked_scores = scores.squeeze(-1).masked_fill(~mask, float("-inf"))
        if mask.numel() and bool(mask.any()):
            distribution = torch.softmax(masked_scores, dim=0)
        else:
            distribution = scores.new_zeros((scores.shape[0],))

        state_embedding = self.value_encoder(
            embeddings["field_embedding"],
            embeddings["boundary_embedding"],
            global_embedding,
        )
        return {
            "actions": actions,
            "mask": mask,
            "scores": scores,
            "masked_scores": masked_scores,
            "action_distribution": distribution,
            "global_embedding": global_embedding,
            "action_input": action_context,
            "state_embedding": state_embedding,
            "encoder_output": embeddings,
            "graph": graph,
        }

    @staticmethod
    def _global_sources(environment, state):
        """Resolve ``(uav_source, depot_source)`` from environment or state dict."""
        if environment is not None and getattr(environment, "uavs", None):
            return environment.uavs[0], environment.depot
        if isinstance(state, dict):
            uavs = state.get("uavs", {})
            if uavs:
                return next(iter(uavs.values())), state.get("depot", {})
        return None, None

    def _build_global_embedding(self, embeddings, environment=None, state=None):
        """Build the global context ``z_G`` from graph pooling, UAV, and depot."""
        device = embeddings["field_embedding"].device
        uav_source, depot_source = self._global_sources(environment, state)
        g_graph = build_graph_context(
            embeddings["field_embedding"], embeddings["boundary_embedding"]
        )
        uav_dim = self.global_state_encoder.uav_dim
        depot_dim = self.global_state_encoder.depot_dim
        x_uav = (
            build_uav_feature(uav_source).to(device)
            if uav_source is not None else torch.zeros(uav_dim, device=device)
        )
        x_depot = (
            build_depot_feature(depot_source).to(device)
            if depot_source is not None else torch.zeros(depot_dim, device=device)
        )
        return self.global_state_encoder(g_graph, x_uav, x_depot)

    def encode_graph(self, graph, environment=None, state=None):
        """Encode a saved graph without generating candidates or action masks."""
        device = next(self.parameters()).device
        if hasattr(graph, "to"):
            graph = graph.to(device)
        embeddings = self.encoder(graph)
        global_embedding = self._build_global_embedding(embeddings, environment, state)
        return embeddings, global_embedding

    @staticmethod
    def sample_action(output, *, generator=None) -> CoverageAction:
        """Sample one feasible action from a forward result."""
        distribution = output["action_distribution"]
        if distribution.numel() == 0 or float(distribution.sum()) <= 0.0:
            raise RuntimeError("No feasible spatial action is available.")
        index = int(torch.multinomial(distribution, 1, generator=generator).item())
        return output["actions"][index]

    @staticmethod
    def greedy_action(output) -> CoverageAction:
        """Select the highest-probability feasible action."""
        distribution = output["action_distribution"]
        if distribution.numel() == 0 or float(distribution.sum()) <= 0.0:
            raise RuntimeError("No feasible spatial action is available.")
        index = int(distribution.argmax().item())
        return output["actions"][index]

    @staticmethod
    def top_k_actions(output, k: int = 5) -> list[dict]:
        """Return human-readable top-k action/probability rows."""
        if k <= 0:
            raise ValueError("k must be positive.")
        distribution = output["action_distribution"]
        if distribution.numel() == 0:
            return []
        valid_indices = torch.where(output["mask"])[0]
        if valid_indices.numel() == 0:
            return []
        order = valid_indices[
            torch.argsort(distribution[valid_indices], descending=True)[:k]
        ]
        rows = []
        for index_tensor in order:
            index = int(index_tensor.item())
            action = output["actions"][index]
            rows.append({
                "field": action.field_id,
                "entry": action.entry_boundary_id,
                "exit": action.exit_boundary_id,
                "probability": float(distribution[index].item()),
            })
        return rows

    @classmethod
    def print_top_k(cls, output, k: int = 5) -> None:
        """Print a compact Top-K action table for manual policy inspection."""
        rows = cls.top_k_actions(output, k)
        print("Top-K actions:")
        for rank, row in enumerate(rows, start=1):
            print(
                f"{rank}. Field={row['field']} Entry={row['entry']} "
                f"Exit={row['exit']} "
                f"Probability={row['probability']:.6f}"
            )

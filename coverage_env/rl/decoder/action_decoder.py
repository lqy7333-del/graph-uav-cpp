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
    build_site_feature,
    build_uav_feature,
)
from coverage_env.rl.encoder.batched_encoder import (
    batched_graph_context,
    collate_graphs,
    encode_graph_batch,
)
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
        # ``site_dim`` carries an explicit site-layout summary: pooled graph
        # embeddings are position- and count-invariant, so without it neither the
        # critic nor the actor can see where the fields are relative to the depot.
        self.global_state_encoder = GlobalStateEncoder(
            graph_dim=2 * hidden_dim, uav_dim=13, depot_dim=2, site_dim=12,
            hidden_dim=hidden_dim,
        )
        # Candidate Field/Boundary embeddings now carry static metric coordinates;
        # explicit per-action distance features are disabled for this ablation.
        decoder_input_dim = self.action_embedding.output_dim + hidden_dim
        self.action_decoder = ActionDecoder(decoder_input_dim, hidden_dim)

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

        global_embedding, critic_input = self._build_global_embedding(
            embeddings, environment, state
        )
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

        return {
            "actions": actions,
            "mask": mask,
            "scores": scores,
            "masked_scores": masked_scores,
            "action_distribution": distribution,
            "global_embedding": global_embedding,
            "critic_input": critic_input,
            "action_input": action_context,
            "encoder_output": embeddings,
            "graph": graph,
        }

    def forward_batched(self, contexts, environment=None):
        """Encode several transitions with one batched GNN pass, then decode each.

        ``contexts`` is a list of mappings shaped like the arguments of
        :meth:`forward` — ``observation``, ``graph``, ``actions``, ``mask`` — one
        per transition. The graph encoder runs **once** over the disjoint union
        of every graph in the list; the (much cheaper) action embedding and
        decoder still run per transition, reading their rows out of the batched
        embeddings via the batch's per-graph offsets.

        Returns one output mapping per context, with the same keys :meth:`forward`
        returns, so the two paths are interchangeable. This is the "batch the
        encoder only" variant: it changes no attention indexing, only how many
        times the GNN is launched.
        """
        context_list = list(contexts)
        if not context_list:
            return []
        environment = environment or self.environment
        device = next(self.parameters()).device
        batch = collate_graphs(
            [context["graph"] for context in context_list]
        ).to(device)
        embeddings = encode_graph_batch(self.encoder, batch)

        # Per-transition global features; pooled graph context is per graph.
        uav_dim = self.global_state_encoder.uav_dim
        depot_dim = self.global_state_encoder.depot_dim
        site_dim = self.global_state_encoder.site_dim
        uav_features, depot_features, site_features = [], [], []
        for context in context_list:
            observation = context["observation"]
            uav_source, depot_source = self._global_sources(environment, observation)
            site_source = self._site_source(environment, observation)
            uav_features.append(
                build_uav_feature(uav_source).to(device)
                if uav_source is not None else torch.zeros(uav_dim, device=device)
            )
            depot_features.append(
                build_depot_feature(depot_source).to(device)
                if depot_source is not None else torch.zeros(depot_dim, device=device)
            )
            site_features.append(
                build_site_feature(site_source).to(device)
                if site_source is not None else torch.zeros(site_dim, device=device)
            )
        uav_stack = torch.stack(uav_features)
        depot_stack = torch.stack(depot_features)
        site_stack = torch.stack(site_features)
        global_embeddings = self.global_state_encoder.forward_batched(
            batched_graph_context(
                embeddings["field_embedding"], embeddings["boundary_embedding"], batch
            ),
            uav_stack,
            depot_stack,
            site_stack,
        )
        critic_inputs = torch.cat(
            [global_embeddings, uav_stack, depot_stack, site_stack], dim=-1
        )

        field_offsets = batch.field_offsets.tolist()
        boundary_offsets = batch.boundary_offsets.tolist()
        outputs = []
        for index, context in enumerate(context_list):
            actions = context["actions"]
            graph = context["graph"]
            mask = torch.tensor(
                context["mask"], dtype=torch.bool, device=device
            )
            action_vectors = self.action_embedding(
                actions,
                embeddings["field_embedding"],
                embeddings["boundary_embedding"],
                graph,
                field_offset=field_offsets[index],
                boundary_offset=boundary_offsets[index],
            )
            global_embedding = global_embeddings[index]
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
            outputs.append({
                "actions": actions,
                "mask": mask,
                "scores": scores,
                "masked_scores": masked_scores,
                "action_distribution": distribution,
                "global_embedding": global_embedding,
                "critic_input": critic_inputs[index],
                "action_input": action_context,
                "graph": graph,
            })
        return outputs

    @staticmethod
    def _global_sources(environment, state):
        """Resolve ``(uav_source, depot_source)``, preferring the stored state.

        PPO replay must condition on the observation saved with the transition:
        by the time ``update()`` runs, the live environment has advanced past the
        rollout, so its UAV/depot no longer describe the transition being
        replayed. Using them makes the replayed ``z_G`` — and therefore the
        log-probs and values — inconsistent with the ones recorded during
        collection, so the on-policy ratio is not 1 at the first epoch.

        The environment stays available as a fallback for callers that have no
        state dict, such as ``value_for_graph`` during rollout (where the
        environment is current anyway).
        """
        state = state if isinstance(state, dict) else {}
        uavs = state.get("uavs") or {}
        uav_source = next(iter(uavs.values()), None)
        depot_source = state.get("depot") or None
        if environment is not None:
            if uav_source is None and getattr(environment, "uavs", None):
                uav_source = environment.uavs[0]
            if depot_source is None:
                depot_source = environment.depot
        return uav_source, depot_source

    @staticmethod
    def _site_source(environment, state):
        """Resolve the layout source (fields + depot), preferring the stored state.

        Same reasoning as ``_global_sources``: PPO replay must see the layout of
        the transition's own instance, and by then the live environment has moved
        on (or resampled a different instance entirely).
        """
        if isinstance(state, dict) and state.get("fields"):
            return state
        if environment is not None and getattr(environment, "fields", None):
            return environment
        return None

    def _build_global_embedding(self, embeddings, environment=None, state=None):
        """Build ``(z_G, critic_input)`` from graph pooling, UAV, depot, and layout.

        ``critic_input`` is ``z_G`` concatenated with the **raw** UAV, depot, and
        site features. ``GlobalStateEncoder`` is optimized only by the actor loss
        and is detached before the critic, so routing the layout information
        solely through it leaves the value function unable to shape -- or easily
        read -- the representation it needs. The raw features give the critic a
        direct, linear path to them.
        """
        device = embeddings["field_embedding"].device
        uav_source, depot_source = self._global_sources(environment, state)
        site_source = self._site_source(environment, state)
        g_graph = build_graph_context(
            embeddings["field_embedding"], embeddings["boundary_embedding"]
        )
        uav_dim = self.global_state_encoder.uav_dim
        depot_dim = self.global_state_encoder.depot_dim
        site_dim = self.global_state_encoder.site_dim
        x_uav = (
            build_uav_feature(uav_source).to(device)
            if uav_source is not None else torch.zeros(uav_dim, device=device)
        )
        x_depot = (
            build_depot_feature(depot_source).to(device)
            if depot_source is not None else torch.zeros(depot_dim, device=device)
        )
        x_site = (
            build_site_feature(site_source).to(device)
            if site_source is not None else torch.zeros(site_dim, device=device)
        )
        global_embedding = self.global_state_encoder(g_graph, x_uav, x_depot, x_site)
        critic_input = torch.cat(
            [global_embedding, x_uav, x_depot, x_site], dim=-1
        )
        return global_embedding, critic_input

    def encode_graph(self, graph, environment=None, state=None):
        """Encode a saved graph; returns ``(embeddings, z_G, critic_input)``."""
        device = next(self.parameters()).device
        if hasattr(graph, "to"):
            graph = graph.to(device)
        embeddings = self.encoder(graph)
        global_embedding, critic_input = self._build_global_embedding(
            embeddings, environment, state
        )
        return embeddings, global_embedding, critic_input

    @staticmethod
    def sample_action(output, *, generator=None) -> CoverageAction:
        """Sample one feasible action from a forward result."""
        distribution = output["action_distribution"]
        if distribution.numel() == 0 or float(distribution.detach().sum()) <= 0.0:
            raise RuntimeError("No feasible spatial action is available.")
        index = int(torch.multinomial(distribution, 1, generator=generator).item())
        return output["actions"][index]

    @staticmethod
    def greedy_action(output) -> CoverageAction:
        """Select the highest-probability feasible action."""
        distribution = output["action_distribution"]
        if distribution.numel() == 0 or float(distribution.detach().sum()) <= 0.0:
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

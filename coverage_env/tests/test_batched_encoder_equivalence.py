"""Numerical equivalence of disjoint-union batching vs the per-graph encoder.

Prototype verification for P1 in ``docs/training_performance_notes.md``: the
batched encoder must reproduce, per graph, exactly what the existing
single-graph ``CoverageGraphEncoder`` produces for the same weights.
"""

from __future__ import annotations

import copy
import time
import unittest

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.global_state_encoder import build_graph_context
from coverage_env.rl.encoder.batched_encoder import (
    BatchedCoverageGraphEncoder,
    collate_graphs,
)
from coverage_env.rl.encoder.hetero_encoder import CoverageGraphEncoder
from coverage_env.rl.graph.graph_builder import GraphBuilder
from coverage_env.rl.graph.relation_builder import (
    EDGE_BB_INTER,
    EDGE_BB_INTRA,
    EDGE_FB,
)


def build_graphs(config: dict, count: int = 6, seed: int = 20261015) -> list:
    """Return ``count`` graphs drawn from independently resampled instances."""
    environment = InstanceGenerator(config).generate(seed=seed)
    builder = GraphBuilder(config["rl"])
    graphs = []
    for _ in range(count):
        environment.reset()
        graphs.append(builder.build_graph(environment))
    return graphs


def strip_inter_edges(graph):
    """Return a copy of ``graph`` with every cross-field relation removed."""
    keep = graph.edge_type != EDGE_BB_INTER
    stripped = copy.deepcopy(graph)
    stripped.edge_index = graph.edge_index[:, keep]
    stripped.edge_type = graph.edge_type[keep]
    stripped.edge_attr = graph.edge_attr[keep]
    return stripped


def max_abs_diff(first: torch.Tensor, second: torch.Tensor) -> float:
    """Return the largest absolute elementwise difference (0.0 when empty)."""
    if first.numel() == 0 or second.numel() == 0:
        return 0.0
    if first.shape != second.shape:
        raise AssertionError(f"shape mismatch: {tuple(first.shape)} vs {tuple(second.shape)}")
    return float((first.double() - second.double()).abs().max().item())


def _edge_counts(graphs, edge_type: int) -> list[int]:
    return [int((graph.edge_type == edge_type).sum().item()) for graph in graphs]


def equivalence_rows(encoder: BatchedCoverageGraphEncoder, graphs, device) -> list[dict]:
    """Run both paths and return one diff report per graph.

    The reference path calls the inherited single-graph ``forward`` once per
    graph; the batched path calls ``forward_batched`` once for the whole list.
    """
    encoder = encoder.to(device)
    encoder.eval()

    reference = []
    with torch.no_grad():
        for graph in graphs:
            moved = graph.to(device) if hasattr(graph, "to") else graph
            output = encoder(moved)
            reference.append({
                "field": output["field_embedding"],
                "boundary": output["boundary_embedding"],
                "context": build_graph_context(
                    output["field_embedding"], output["boundary_embedding"]
                ),
                "global": output["global_feature"].reshape(-1),
                "attention": {
                    "boundary_inter": encoder.attention_debug["boundary_inter"]["attention"],
                    "boundary_intra": encoder.attention_debug["boundary_intra"]["attention"],
                    "field_boundary": encoder.attention_debug["field_boundary"]["attention"],
                },
            })

    batch = collate_graphs(graphs).to(device)
    with torch.no_grad():
        batched = encoder.forward_batched(batch)

    field_offsets = batch.field_offsets.tolist()
    boundary_offsets = batch.boundary_offsets.tolist()
    inter_counts = _edge_counts(graphs, EDGE_BB_INTER)
    intra_counts = _edge_counts(graphs, EDGE_BB_INTRA)
    fb_counts = _edge_counts(graphs, EDGE_FB)
    batched_inter = torch.split(batched["attention"]["boundary_inter"], inter_counts)
    batched_intra = torch.split(batched["attention"]["boundary_intra"], intra_counts)
    # FieldBoundaryEncoder emits [all forward FB edges, all reversed FB edges],
    # so a graph's two halves are not adjacent and must be reassembled.
    fb_total = sum(fb_counts)
    fb_forward = torch.split(batched["attention"]["field_boundary"][:fb_total], fb_counts)
    fb_reverse = torch.split(batched["attention"]["field_boundary"][fb_total:], fb_counts)
    batched_fb = [
        torch.cat([forward, reverse], dim=0)
        for forward, reverse in zip(fb_forward, fb_reverse)
    ]

    rows = []
    for index, graph in enumerate(graphs):
        start, stop = field_offsets[index], field_offsets[index + 1]
        b_start, b_stop = boundary_offsets[index], boundary_offsets[index + 1]
        rows.append({
            "index": index,
            "fields": stop - start,
            "boundaries": b_stop - b_start,
            "edges": int(graph.edge_index.shape[1]),
            "inter_edges": inter_counts[index],
            "field": max_abs_diff(
                reference[index]["field"], batched["field_embedding"][start:stop]
            ),
            "boundary": max_abs_diff(
                reference[index]["boundary"],
                batched["boundary_embedding"][b_start:b_stop],
            ),
            "context": max_abs_diff(
                reference[index]["context"], batched["graph_context"][index]
            ),
            "global": max_abs_diff(
                reference[index]["global"], batched["global_feature"][index]
            ),
            "inter_attention": max_abs_diff(
                reference[index]["attention"]["boundary_inter"], batched_inter[index]
            ),
            "intra_attention": max_abs_diff(
                reference[index]["attention"]["boundary_intra"], batched_intra[index]
            ),
            "fb_attention": max_abs_diff(
                reference[index]["attention"]["field_boundary"], batched_fb[index]
            ),
        })
    return rows


def print_report(rows: list[dict], device, label: str) -> None:
    """Print a compact per-graph equivalence table."""
    print(f"\n=== {label} (device={device}) ===")
    header = (f"{'graph':>5} {'fields':>6} {'bnd':>5} {'edges':>6} {'E_int':>6} "
              f"{'field':>9} {'bnd':>9} {'ctx':>9} {'attn':>9}")
    print(header)
    for row in rows:
        attention = max(
            row["inter_attention"], row["intra_attention"], row["fb_attention"]
        )
        print(f"{row['index']:>5} {row['fields']:>6} {row['boundaries']:>5} "
              f"{row['edges']:>6} {row['inter_edges']:>6} "
              f"{row['field']:>9.2e} {row['boundary']:>9.2e} "
              f"{row['context']:>9.2e} {attention:>9.2e}")
    worst = max(
        max(r["field"], r["boundary"], r["context"], r["global"],
            r["inter_attention"], r["intra_attention"], r["fb_attention"])
        for r in rows
    )
    print(f"worst absolute difference across all tensors: {worst:.3e} "
          f"({'BIT-EXACT' if worst == 0.0 else 'round-off only'})")


def make_encoder(config: dict) -> BatchedCoverageGraphEncoder:
    """Return a batched encoder whose weights match a fresh single-graph one."""
    torch.manual_seed(0)
    reference = CoverageGraphEncoder.from_config(config["rl"])
    batched = BatchedCoverageGraphEncoder.from_config(config["rl"])
    batched.load_state_dict(reference.state_dict())
    return batched


class BatchedEncoderEquivalenceTests(unittest.TestCase):
    """Batching must not change per-graph embeddings, context, or attention."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG_PATH)
        cls.encoder = make_encoder(cls.config)
        cls.graphs = build_graphs(cls.config, count=6)

    def _assert_node_outputs_bit_exact(self, rows):
        """Node embeddings, attention, and raw global features must be identical."""
        for row in rows:
            for key in ("field", "boundary", "global",
                        "inter_attention", "intra_attention", "fb_attention"):
                self.assertEqual(row[key], 0.0, f"{key} is not bit-exact: {row}")

    def _assert_context_close(self, rows, tolerance):
        """Pooled context may differ by float32 round-off (index_add vs mean)."""
        for row in rows:
            self.assertLessEqual(row["context"], tolerance, row)

    def test_matches_per_graph_on_real_instances(self):
        rows = equivalence_rows(self.encoder, self.graphs, "cpu")
        self._assert_node_outputs_bit_exact(rows)
        self._assert_context_close(rows, 1e-6)
        self.assertGreater(len({row["fields"] for row in rows}), 1,
                           "expected resampled instances with different field counts")

    def test_matches_with_mixed_empty_inter_edges(self):
        graphs = [strip_inter_edges(graph) for graph in self.graphs]
        rows = equivalence_rows(self.encoder, graphs, "cpu")
        # Every graph now has zero cross-field edges; the batched layer still
        # takes its non-empty path and must reproduce the residual-only result.
        self.assertTrue(all(row["inter_edges"] == 0 for row in rows))
        self._assert_node_outputs_bit_exact(rows)
        self._assert_context_close(rows, 1e-6)

    def test_single_graph_batch_is_equivalent(self):
        rows = equivalence_rows(self.encoder, self.graphs[:1], "cpu")
        self._assert_node_outputs_bit_exact(rows)
        self._assert_context_close(rows, 1e-6)

    def test_boundary_indices_stay_inside_their_owning_graph(self):
        """Regression guard: per-graph boundary rows must never alias.

        Collapsing each graph onto its own ``[0, n_boundary_k)`` range makes
        graph 1's boundaries index graph 0's feature rows; the concatenated
        layout must instead keep ``b_off[k] + j``.
        """
        batch = collate_graphs(self.graphs)
        offsets = batch.boundary_offsets.tolist()
        for edge_type in (EDGE_BB_INTRA, EDGE_BB_INTER):
            mask = batch.edge_type == edge_type
            local = batch.edge_index[:, mask] - batch.num_fields
            owners = batch.edge_batch[mask]
            for owner, target in zip(owners.tolist(), local[1].tolist()):
                low, high = offsets[owner], offsets[owner + 1]
                self.assertTrue(
                    low <= target < high,
                    f"graph {owner} targets boundary row {target}, "
                    f"outside its own range [{low}, {high})",
                )

    def test_global_feature_is_stacked_per_graph(self):
        batch = collate_graphs(self.graphs)
        self.assertEqual(tuple(batch.global_feature.shape),
                         (len(self.graphs), 5))
        for index, graph in enumerate(self.graphs):
            self.assertTrue(torch.equal(
                batch.global_feature[index], graph.global_feature.reshape(-1)
            ))

    def test_edges_never_cross_components(self):
        batch = collate_graphs(self.graphs)
        field_offsets = batch.field_offsets.tolist()
        boundary_offsets = batch.boundary_offsets.tolist()
        for edge in range(batch.edge_index.shape[1]):
            source, target = batch.edge_index[:, edge].tolist()
            self.assertEqual(
                self._owner(source, batch, field_offsets, boundary_offsets),
                self._owner(target, batch, field_offsets, boundary_offsets),
            )

    @staticmethod
    def _owner(node_index, batch, field_offsets, boundary_offsets):
        if node_index < batch.num_fields:
            for graph in range(batch.num_graphs):
                if field_offsets[graph] <= node_index < field_offsets[graph + 1]:
                    return graph
        else:
            local = node_index - batch.num_fields
            for graph in range(batch.num_graphs):
                if boundary_offsets[graph] <= local < boundary_offsets[graph + 1]:
                    return graph
        raise AssertionError(f"node {node_index} belongs to no graph")

    def test_empty_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            collate_graphs([])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_matches_per_graph_on_cuda(self):
        rows = equivalence_rows(self.encoder, self.graphs, "cuda")
        for row in rows:
            for key in ("field", "boundary", "context", "global",
                        "inter_attention", "intra_attention", "fb_attention"):
                self.assertLessEqual(row[key], 1e-5, f"{key}: {row}")


def benchmark_batched_vs_per_graph(
    config: dict, count: int = 64, device: str = "cuda", repeats: int = 3
) -> dict:
    """Time one batched forward against ``count`` per-graph forwards."""
    graphs = build_graphs(config, count=count)
    single = CoverageGraphEncoder.from_config(config["rl"]).to(device).eval()
    batched = make_encoder(config).to(device).eval()
    moved = [graph.to(device) if hasattr(graph, "to") else graph for graph in graphs]
    batch = collate_graphs(graphs).to(device)

    def per_graph():
        with torch.no_grad():
            return [single(graph)["field_embedding"] for graph in moved]

    def one_batch():
        with torch.no_grad():
            return batched.forward_batched(batch)

    per_graph()
    one_batch()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    results = {}
    for label, function in (("per_graph", per_graph), ("batched", one_batch)):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        results[label] = (time.perf_counter() - start) / repeats
    results["speedup"] = results["per_graph"] / results["batched"]
    results["nodes"] = sum(
        int(graph.x_field.shape[0] + graph.x_boundary.shape[0]) for graph in graphs
    )
    return results


if __name__ == "__main__":
    import sys

    configuration = load_config(DEFAULT_CONFIG_PATH)
    model = make_encoder(configuration)
    cases = {
        "real-resampled-instances": build_graphs(configuration, count=6),
        "all-inter-edges-stripped": [
            strip_inter_edges(graph) for graph in build_graphs(configuration, count=4)
        ],
        "single-graph-batch": build_graphs(configuration, count=1),
    }
    for label, case_graphs in cases.items():
        print_report(equivalence_rows(model, case_graphs, "cpu"), "cpu", label)
    if torch.cuda.is_available():
        print_report(
            equivalence_rows(model, build_graphs(configuration, count=6), "cuda"),
            "cuda", "real-resampled-instances",
        )

    print("\n=== encoder throughput ===")
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        result = benchmark_batched_vs_per_graph(configuration, device=device)
        print(f"{device:4s}: {result['nodes']} nodes | "
              f"per-graph {result['per_graph']*1000:8.1f} ms | "
              f"batched {result['batched']*1000:8.1f} ms | "
              f"speedup {result['speedup']:5.2f}x")
    sys.exit(0)

"""Immutable instance bank for training-set overfitting diagnostics.

Instances are serialized once, not regenerated from seeds at each reset.  The
pool index is logging metadata only and is never added to the observation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import torch

from coverage_env.rl.trainer.episode import instance_fingerprint


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bank(directory, config, environments, seeds, *, initialization=None):
    """Persist pristine snapshots and their identities; never replace a bank."""
    directory = Path(directory).resolve()
    if (directory / "manifest.json").exists() or (directory / "instances.pt").exists():
        raise FileExistsError("Instance bank already exists; reuse it or choose a new directory.")
    if not environments or len(environments) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Bank needs nonempty snapshots and one distinct seed per snapshot.")
    entries = []
    snapshots = []
    for index, (env, seed) in enumerate(zip(environments, seeds)):
        if env._randomize_each_reset or env._randomize_uav:
            raise ValueError("Overfit snapshots must disable geometry and UAV resampling.")
        snapshot = deepcopy(env)
        snapshot.reset()
        entries.append({"index": index, "seed": seed, "field_count": len(env.fields),
                        "fingerprint": instance_fingerprint(snapshot)})
        snapshots.append(snapshot)
    if len({entry["fingerprint"] for entry in entries}) != len(entries):
        raise ValueError("Fixed bank contains duplicate instances; do not count copies as different tasks.")
    directory.mkdir(parents=True, exist_ok=True)
    bank = directory / "instances.pt"
    torch.save({"config": deepcopy(config), "environments": snapshots}, bank)
    manifest = {"format_version": 1, "purpose": "training_set_overfit_not_generalization",
                "config": deepcopy(config), "initialization": initialization,
                "bank_sha256": file_sha256(bank), "instances": entries}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_bank(directory, *, expected_manifest_sha=None):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    if expected_manifest_sha and file_sha256(manifest_path) != expected_manifest_sha:
        raise ValueError("Fixed-instance manifest changed since training/checkpoint creation.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_sha256(directory / "instances.pt") != manifest["bank_sha256"]:
        raise ValueError("Fixed-instance snapshot file checksum mismatch.")
    saved = torch.load(directory / "instances.pt", map_location="cpu", weights_only=False)
    environments = saved["environments"]
    if saved["config"] != manifest["config"] or len(environments) != len(manifest["instances"]):
        raise ValueError("Snapshot/config count differs from the frozen manifest.")
    for env, entry in zip(environments, manifest["instances"]):
        if env._randomize_each_reset or env._randomize_uav:
            raise ValueError("Resampling is enabled in a supposedly fixed snapshot.")
        if instance_fingerprint(env) != entry["fingerprint"]:
            raise ValueError(f"Pristine snapshot identity mismatch: {entry['index']}")
    return manifest, environments


class FixedInstancePool:
    """Delegate execution to fresh copies of a shuffled, finite instance bank.

    Every cycle contains each instance once. Private RNG/order/cursor and the
    active episode survive checkpointing. Immutable templates stay on disk so
    each DDP checkpoint need not duplicate the entire bank for every rank.
    """
    def __init__(self, directory, *, seed=42):
        self.directory = str(Path(directory).resolve())
        self.manifest_sha = file_sha256(Path(self.directory) / "manifest.json")
        self.manifest, self._templates = load_bank(self.directory)
        self._rng = random.Random(seed)
        self._order = []
        self._cursor = 0
        self.current_index = None
        self.starts = [0] * len(self._templates)
        self.completions = [0] * len(self._templates)
        self.transitions = [0] * len(self._templates)
        self._episode_done = False
        # Construction exposes fields for the actor without consuming a draw.
        self._active = deepcopy(self._templates[0])

    def __getattr__(self, name):
        active = self.__dict__.get("_active")
        if active is None:
            raise AttributeError(name)
        return getattr(active, name)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_templates"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def reset(self):
        if self._templates is None:
            self.manifest, self._templates = load_bank(
                self.directory, expected_manifest_sha=self.manifest_sha)
        if self._cursor >= len(self._order):
            self._order = list(range(len(self._templates)))
            self._rng.shuffle(self._order)
            self._cursor = 0
        index = self._order[self._cursor]
        self._cursor += 1
        self._active = deepcopy(self._templates[index])
        state = self._active.reset()
        if instance_fingerprint(self._active) != self.manifest["instances"][index]["fingerprint"]:
            raise RuntimeError("Reset changed fixed geometry/resources or failed to clear coverage.")
        self.current_index = index
        self.starts[index] += 1
        self._episode_done = False
        return state

    def step(self, action):
        if self.current_index is None or self._episode_done:
            raise RuntimeError("Reset is required before stepping a fixed instance.")
        result = self._active.step(action)
        self.transitions[self.current_index] += 1
        if result[2]:
            self.completions[self.current_index] += 1
            self._episode_done = True
        return result

    def usage(self):
        return {"starts": list(self.starts), "completed_episodes": list(self.completions),
                "transitions": list(self.transitions), "manifest_sha": self.manifest_sha}

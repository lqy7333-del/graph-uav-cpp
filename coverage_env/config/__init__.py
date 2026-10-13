"""Experiment configuration loading helpers."""

from __future__ import annotations

from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load a YAML experiment configuration into a plain dictionary."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a YAML mapping.")
    return config


def energy_parameters(config: dict | None = None) -> dict:
    """Return the UAV energy coefficients ``{"alpha", "beta"}`` from config.

    This is the single source of truth for the linear energy model's
    coefficients, so tests and other consumers follow ``default.yaml`` changes
    instead of hard-coding values that later drift.
    """
    config = config or load_config()
    energy = config["uav"]["energy"]
    return {"alpha": float(energy["alpha"]), "beta": float(energy["beta"])}


__all__ = ["DEFAULT_CONFIG_PATH", "load_config", "energy_parameters"]

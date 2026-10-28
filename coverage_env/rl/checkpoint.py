"""Checkpoint provenance, random streams, and evaluation configuration."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
import subprocess
import warnings

import numpy as np
import torch


def capture_rng_state():
    """Capture process-local random streams; environment RNG lives in its snapshot."""
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": ({"device": torch.cuda.current_device(), "state": torch.cuda.get_rng_state()}
                 if torch.cuda.is_initialized() else None),
    }


def restore_rng_state(state):
    """Restore streams after model/environment construction, never before it."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        cuda = state["cuda"]
        torch.cuda.set_rng_state(cuda["state"].cpu(), device=cuda["device"])


def assert_rank_parameters_equal(*modules):
    """Check full actor and critic tensors, not a checksum that can cancel errors."""
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return
    for module in modules:
        flat = torch.cat([p.detach().reshape(-1) for p in module.parameters()])
        reference = flat.clone()
        dist.broadcast(reference, src=0)
        mismatch = torch.tensor(int(not torch.equal(flat, reference)), device=flat.device)
        dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
        if mismatch.item():
            raise RuntimeError("DDP model parameters differ across ranks; refusing to save inconsistent weights.")


def provenance():
    """Record the source revision and whether uncommitted code was present."""
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"revision": revision, "dirty": dirty, "torch": str(torch.__version__)}


def evaluation_config(checkpoint, fallback=None):
    """Use saved experiment semantics, with an explicit warning for legacy weights."""
    from coverage_env.config import load_config
    from coverage_env.rl.decoder.hierarchical_decoder import checkpoint_decoder_flags

    if checkpoint.get("resolved_config"):
        config = deepcopy(checkpoint["resolved_config"])
    else:
        warnings.warn("Legacy checkpoint has no resolved_config; evaluation uses the supplied/default config.",
                      RuntimeWarning, stacklevel=2)
        config = deepcopy(fallback if fallback is not None else load_config())
    config.setdefault("rl", {}).setdefault("decoder", {}).update(
        checkpoint_decoder_flags(checkpoint["actor"]))
    return config

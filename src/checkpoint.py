"""Saving and loading training checkpoints.

A checkpoint is one .pt file holding everything needed to continue a run:

    {
        "format_version":       2,
        "model_state_dict":     network weights (CPU tensors, no "_orig_mod." prefix),
        "optimizer_state_dict": optimizer state (Adam/AdamW moments, SGD momentum, ...),
        "iteration":            number of COMPLETED iterations, counted across all jobs,
        "games_played":         self-play games played so far, counted across all jobs,
        "config":               the training config that produced it (plain types only),
        "saved_at":             local time the file was written,
    }

Files written before this format existed contain only the network weights (a bare
state_dict). load_checkpoint accepts both, so older checkpoints still load for evaluation.

Everything stored is a tensor or a plain Python value. torch.load defaults to
weights_only=True from torch 2.6, which refuses numpy arrays/scalars and arbitrary objects;
keeping to plain types means the defaults just work.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import torch

FORMAT_VERSION = 2


def plain_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of the config with only JSON-style values (dict, list, str, int, float,
    bool, None). Tuples become lists; anything else (e.g. a numpy number) becomes its
    string form. It is a record of the run, not something the code reads back."""
    return json.loads(json.dumps(config, default=str))


def atomic_torch_save(obj: Any, path: str) -> None:
    """Write to a temporary file in the same directory, then rename over the target.
    A crash mid-write leaves the previous file intact instead of a truncated one."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    path: str,
    model_state_dict: Dict[str, torch.Tensor],
    optimizer_state_dict: Optional[Dict[str, Any]],
    iteration: int,
    games_played: int,
    config: Dict[str, Any],
) -> None:
    atomic_torch_save(
        {
            "format_version": FORMAT_VERSION,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "iteration": int(iteration),
            "games_played": int(games_played),
            "config": plain_config(config),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        path,
    )


def load_checkpoint(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a checkpoint in either format and return it in the new layout.

    An old weights-only file comes back with optimizer_state_dict, iteration,
    games_played and config set to None.
    """
    data = torch.load(path, map_location=map_location)
    if isinstance(data, dict) and "model_state_dict" in data:
        return data
    return {
        "format_version": 1,
        "model_state_dict": data,
        "optimizer_state_dict": None,
        "iteration": None,
        "games_played": None,
        "config": None,
        "saved_at": None,
    }


def load_model_state_dict(path: str, map_location: Any = "cpu") -> Dict[str, torch.Tensor]:
    """Just the network weights, from either format. What evaluation code wants."""
    return load_checkpoint(path, map_location)["model_state_dict"]

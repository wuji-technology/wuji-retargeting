"""Path-resolution helpers shared across example scripts.

resolve_mujoco_model_dir(config_path):
    If the retarget config YAML sets optimizer.mjcf_path, return the
    grandparent directory (the dir that contains wuji_hand_description/).
    Return None when mjcf_path is absent — caller falls back to its default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_mujoco_model_dir(config_path: Path) -> Optional[str]:
    import yaml
    config_path = Path(config_path).resolve()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    mjcf_rel = (cfg.get("optimizer") or {}).get("mjcf_path")
    if not mjcf_rel:
        return None
    mjcf_abs = (config_path.parent / mjcf_rel).resolve()
    # mjcf_abs = .../wuji_hand_description/mjcf/<hand>.xml
    # parents[2] = dir that contains wuji_hand_description/
    if len(mjcf_abs.parents) <= 2:
        raise ValueError(
            "optimizer.mjcf_path must resolve to "
            ".../wuji_hand_description/mjcf/<hand>.xml, "
            f"got {mjcf_abs}"
        )
    return str(mjcf_abs.parents[2])

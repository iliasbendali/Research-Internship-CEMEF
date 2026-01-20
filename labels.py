# labels.py

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from domain import LABEL_TO_IDX

def extract_annotations_from_match_json(match_json: Dict[str, Any], half: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    actions = match_json.get("actions", {})
    for _, action_obj in actions.items():
        meta = action_obj.get("imageMetadata", {})
        if int(meta.get("half", 0)) != int(half):
            continue
        label = meta.get("label", None)
        position = meta.get("position", None)
        if label is None or position is None:
            continue
        if meta.get("visibility") == "not shown":
            continue
        if label not in LABEL_TO_IDX:
            continue
        out.append({"label_idx": LABEL_TO_IDX[label], "position": int(position)})
    return out

def build_targets_for_half(
    T: int,
    annotations: List[Dict[str, Any]],
    step_seconds: float,
    sigma_by_idx: List[float],
    radius_sigmas: float = 4.0,
) -> np.ndarray:
    Y = np.zeros((T, 17), dtype=np.float32)
    for ann in annotations:
        j = int(ann["label_idx"])
        pos_ms = int(ann["position"])
        time_sec = pos_ms / 1000.0
        t0 = int(round(time_sec / step_seconds))
        if t0 < 0 or t0 >= T:
            continue

        sigma_sec = float(sigma_by_idx[j])
        sigma_frames = max(1e-6, sigma_sec / step_seconds)

        radius = int(round(radius_sigmas * sigma_frames))
        lo = max(0, t0 - radius)
        hi = min(T - 1, t0 + radius)

        for t in range(lo, hi + 1):
            z = (t - t0) / sigma_frames
            g = np.exp(-0.5 * z * z)
            if g > Y[t, j]:
                Y[t, j] = g
    return Y

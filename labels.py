# labels.py

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from domain import LABEL_TO_IDX

def extract_annotations_from_match_json(match_json: Dict[str, Any], half: int) -> List[Dict[str, Any]]:
    """
    Retourne une liste d'annotations au format:
    [{"half": 1, "position_ms": 123456, "label": "Foul"}, ...]
    Supporte le format SoccerNet où les events sont dans match_json["actions"][*.png]["imageMetadata"].
    """
    anns: List[Dict[str, Any]] = []

    # Format "actions" (dict de png -> {imageMetadata,...})
    actions = match_json.get("actions", None)
    if isinstance(actions, dict):
        for _, item in actions.items():
            meta = item.get("imageMetadata", {})
            try:
                h = int(meta.get("half"))
            except Exception:
                continue
            if h != int(half):
                continue

            label = meta.get("label", None)
            pos = meta.get("position", None)  # ms normalement
            if label is None or pos is None:
                continue

            try:
                position_ms = int(pos)
            except Exception:
                continue

            anns.append({"half": h, "position_ms": position_ms, "label": str(label)})
        return anns

    # (Optionnel) fallback si tu as un autre format ailleurs
    annotations = match_json.get("annotations", None)
    if isinstance(annotations, list):
        for a in annotations:
            try:
                h = int(a.get("half"))
            except Exception:
                continue
            if h != int(half):
                continue
            label = a.get("label", None)
            pos = a.get("position", None)
            if label is None or pos is None:
                continue
            anns.append({"half": h, "position_ms": int(pos), "label": str(label)})
        return anns

    return anns


def build_targets_for_half(
    T: int,
    annotations: List[Dict[str, Any]],
    step_seconds: float,
    sigma_by_idx: List[float],
    radius_sigmas: float = 4.0,
) -> np.ndarray:
    Y = np.zeros((T, 17), dtype=np.float32)
    for ann in annotations:
        label = ann.get("label", None)
        if "label_idx" in ann:
            j = int(ann["label_idx"])
        else:
            if label is None:
                continue
            # normalisation minimale (pour éviter des mismatches)
            lab = str(label).strip()

            # mapping string -> idx (doit correspondre à tes 17 classes)
            LABEL_TO_IDX = {
                "Goal": 0,
                "Kick-off": 1,
                "Penalty": 2,
                "Yellow card": 3,
                "Red card": 4,
                "Yellow->red card": 5,
                "Foul": 6,
                "Substitution": 7,
                "Offside": 8,
                "Ball out of play": 9,
                "Throw-in": 10,
                "Clearance": 11,
                "Corner": 12,
                "Direct free-kick": 13,
                "Indirect free-kick": 14,
                "Shots on target": 15,
                "Shots off target": 16,
            }

            # SoccerNet peut avoir "Shots on target/off target" (pluriel)
            if lab == "Shot on target":
                lab = "Shots on target"
            if lab == "Shot off target":
                lab = "Shots off target"

            if lab not in LABEL_TO_IDX:
                # label inconnu -> on ignore
                continue
            j = LABEL_TO_IDX[lab]
        pos_ms = ann.get("position", ann.get("position_ms", None))
        if pos_ms is None:
            continue
        pos_ms = int(pos_ms)
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

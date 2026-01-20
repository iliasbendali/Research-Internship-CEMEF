# windowing.py : créer les fenêtres une fois le modèle déjà entraîné

# inference.py
from __future__ import annotations

import numpy as np
import torch
from typing import Any, Dict, Optional

from domain import HalfEmbeddings


class SlidingWindowInferencer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        window_seconds: float = 180.0,
        stride_seconds: float = 90.0,
        apply_sigmoid: bool = True,  # je code ça avant de savoir quelle sera la sortie du PatchTST : logits ou proba
    ):
        self.model = model
        self.device = device
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.apply_sigmoid = bool(apply_sigmoid)

    @staticmethod
    def _to_frames(seconds: float, step_seconds: float) -> int:
        return max(1, int(round(float(seconds) / float(step_seconds))))

    def predict_half_scores(self, half: HalfEmbeddings) -> np.ndarray:
        """
        Retourne scores_full: (T,17) float32 (probas si apply_sigmoid=True, sinon logits).
        """
        X = half.embeddings
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        X = np.asarray(X, dtype=np.float32)

        T = int(X.shape[0])
        if T == 0:
            return np.zeros((0, 17), dtype=np.float32)

        step = float(getattr(half, "step_seconds", 0.5) or 0.5)
        L = self._to_frames(self.window_seconds, step)  # ex 360
        S = self._to_frames(self.stride_seconds, step)  # ex 180

        # Cas rare: T < L => padding + truncate
        # ce cas n'existe que si le dataset a un bug sur un match car L = 3min et T <= 45min
        if T < L:
            X_pad = np.zeros((L, X.shape[1]), dtype=np.float32)
            X_pad[:T] = X
            y_pad = self._run_model_logits(X_pad)  # (L,17) logits
            y = y_pad[:T]
            return self._to_output(y)

        sum_scores = np.zeros((T, 17), dtype=np.float32)
        count = np.zeros((T, 1), dtype=np.float32)

        starts = list(range(0, T - L + 1, S))
        if not starts:
            starts = [0]
        if starts[-1] != (T - L):
            starts.append(T - L)

        for start in starts:
            end = start + L
            X_win = X[start:end]  # (L,512)
            y_win = self._run_model_logits(X_win)  # (L,17) logits
            y_win = self._to_output(y_win)         # probs ou logits

            sum_scores[start:end] += y_win
            count[start:end] += 1.0

        scores_full = sum_scores / np.maximum(count, 1.0)
        return scores_full.astype(np.float32)

    def _run_model_logits(self, X_win: np.ndarray) -> np.ndarray:
        """
        Exécute le modèle sur une fenêtre (L,512) et renvoie des logits (L,17).
        """
        x = torch.from_numpy(X_win).to(self.device)
        x = x.unsqueeze(0)  # (1,L,512)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x)

        # support dict {"logits": ...} ou sortie directe
        if isinstance(out, dict):
            if "logits" in out:
                y = out["logits"]
            elif "probs" in out:
                # si le modèle renvoie déjà des probas, on les traite comme "logits" au sens output,
                # et apply_sigmoid devra être False
                y = out["probs"]
            else:
                raise RuntimeError(f"Sortie dict inconnue: keys={list(out.keys())}")
        else:
            y = out

        if y.dim() == 3:
            y = y.squeeze(0)  # (L,17)

        y = y.detach().float().cpu().numpy().astype(np.float32)
        return y

    def _to_output(self, y: np.ndarray) -> np.ndarray:
        if not self.apply_sigmoid:
            return y.astype(np.float32)
        # sigmoid stable
        return (1.0 / (1.0 + np.exp(-y))).astype(np.float32)


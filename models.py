# ActionDetector ; TemporalPostProcessor
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from domain import HalfEmbeddings, ActionEvent, EventLabel, LABEL_TO_IDX

import torch

class ActionDetector:
    """
    doit retourner des scores (probas) (T, 17) pour une mi-temps.
    """

    def __init__(self, model=None, device: str = "cpu"):
        """
        model: typiquement un torch.nn.Module déjà pré-entraîné (transformer + tête 17 classes)
        device: "cpu" ou "cuda"
        """
        self.model = model
        self.device = device

    def predict_scores(self, half: HalfEmbeddings) -> np.ndarray:
        """
        Retourne un ndarray float32 shape (T, 17).
        - Si self.model est None: erreur explicite (tu dois plug ton modèle)
        - Si modèle torch: on fait l'inférence proprement
        """
        if self.model is None:
            raise RuntimeError(
                "ActionDetector.model est None. "
                "Plug ton transformer + head (torch.nn.Module) ou remplace predict_scores()."
            )

        x = half.embeddings
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        x = x.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, T, D)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x)  # attendu: (1, T, 17) ou dict/logits

        if isinstance(out, dict):
            if "probs" in out:
                y = out["probs"]
            elif "logits" in out:
                y = out["logits"]
            else:
                raise RuntimeError(f"Sortie dict inconnue: keys={list(out.keys())}")
        else:
            y = out

        # ici je veux changer le format de y: (1, T, 17) -> (T, 17)
        if y.dim() == 3:
            y = y.squeeze(0)

        y = y.detach().float().cpu().numpy()

        # bon pour l'instant je garde la ligne du dessous mais si mon transformer 
        # me donne direct des probas faudra la supprimer
        y = 1.0 / (1.0 + np.exp(-y))

        return y.astype(np.float32)



@dataclass
class TemporalPostProcessor:
    threshold: float = 0.5
    min_separation_seconds: float = 20.0   # éviter les pics quasi identiques, peut-être que j'utiliserai 
                                           # un dico pour la suite
    top_k: Optional[int] = None            # None = pas de limite par label

    def _min_sep_frames(self, half: HalfEmbeddings) -> int:
        step = float(getattr(half, "step_seconds", 0.5) or 0.5)
        return max(1, int(round(self.min_separation_seconds / step)))

    def extract_events_for_label(
        self,
        probs_1d: np.ndarray,      # shape (T,)
        half: HalfEmbeddings,
        label: EventLabel,
        source: str = "transformer",
    ) -> List[ActionEvent]:
        """
        ici je transforme une série de probas (T,) en événements (pics) pour un label.
        comment ?
        - on garde les indices au-dessus du threshold (peut-être que j'implémenterai un dico 
                                                threshold avec une valeur par classe, à voir)
        - on groupe les indices proches (<= min_sep_frames)
        - par groupe, on garde le max (argmax local)
        - optionnel: top_k (par label)
        """
        T = int(probs_1d.shape[0])
        if T == 0:
            return []

        thr = float(self.threshold)
        candidates = np.where(probs_1d >= thr)[0]
        if candidates.size == 0:
            return []

        min_sep = self._min_sep_frames(half)

        # Grouper en "blocs" d'indices proches
        peaks: List[int] = []
        start = int(candidates[0])
        prev = start

        for idx in candidates[1:]:
            idx = int(idx)
            if idx - prev <= min_sep:
                prev = idx
            else:
                # bloc [start..prev] -> peak = argmax
                seg = probs_1d[start : prev + 1]
                peak = start + int(np.argmax(seg))
                peaks.append(peak)
                start = idx
                prev = idx

        # dernier bloc
        seg = probs_1d[start : prev + 1]
        peak = start + int(np.argmax(seg))
        peaks.append(peak)

        # Construire ActionEvent
        events = [
            ActionEvent(
                match_id=half.match_id,
                half=half.half,
                time_sec=half.index_to_time_sec(t),
                label=label,
                confidence=float(probs_1d[t]),
                source=source,
                extra={"t_idx": int(t)},
            )
            for t in peaks
        ]

        # top_k par label (si demandé)
        if self.top_k is not None:
            k = int(self.top_k)
            events.sort(key=lambda e: e.confidence, reverse=True)
            events = events[:k]
            events.sort(key=lambda e: e.time_sec)

        return events

    def extract_events(
        self,
        scores: np.ndarray,                # shape (T, 17)
        half: HalfEmbeddings,
        labels: Optional[Sequence[EventLabel]] = None,
        source: str = "transformer",
    ) -> List[ActionEvent]:
        """
        Multi-label: extrait les events pour plusieurs labels et fusionne.
        """
        if scores.ndim != 2 or scores.shape[1] != 17:
            raise ValueError(f"scores doit être (T,17). Reçu: {scores.shape}")

        # labels à traiter
        if labels is None:
            labels = list(LABEL_TO_IDX.keys())  # type: ignore[assignment]

        events: List[ActionEvent] = []

        for label in labels:
            if label not in LABEL_TO_IDX:
                raise ValueError(f"Label inconnu: {label}")
            j = LABEL_TO_IDX[label]
            probs_1d = scores[:, j]
            events.extend(
                self.extract_events_for_label(
                    probs_1d=probs_1d,
                    half=half,
                    label=label,  # type: ignore[arg-type]
                    source=source,
                )
            )

        # tri global
        events.sort(key=lambda e: e.time_sec)
        return events

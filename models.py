# ActionDetector ; TemporalPostProcessor
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Tuple, Any

import numpy as np

from domain import HalfEmbeddings, ActionEvent, EventLabel, LABEL_TO_IDX, label_int

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
    """
    Convertit des scores frame-level (T,17) en événements discrets.

    Stratégie:
    1) Pour chaque label: candidats = indices où prob >= threshold
    2) Optionnel: ne garder que des maxima locaux (recommandé)
    3) NMS temporel greedy (fenêtre symétrique): on garde le meilleur pic,
       puis on supprime tous les autres pics dans [t-min_sep, t+min_sep].
    4) Optionnel: top_k par label

    Notes:
    - threshold et min_separation_seconds peuvent être globaux ou overridés par label via per_label.
    """

    threshold: float = 0.5
    min_separation_seconds: float = 20.0
    top_k: Optional[int] = None

    # (optionnel) override par label: {"Goal": {"threshold":0.4, "min_sep":30}, ...}
    per_label: Optional[Dict[str, Dict[str, float]]] = None

    # si True: ne garde que les maxima locaux parmi les points au-dessus du seuil
    local_max_only: bool = True

    def _step_seconds(self, half: HalfEmbeddings) -> float:
        step = float(getattr(half, "step_seconds", 0.5) or 0.5)
        return step if step > 0 else 0.5

    def _min_sep_frames(self, half: HalfEmbeddings, min_sep_seconds: float) -> int:
        step = self._step_seconds(half)
        return max(1, int(round(float(min_sep_seconds) / step)))

    def _get_params_for_label(self, label: str) -> Tuple[float, float]:
        thr = float(self.threshold)
        min_sep = float(self.min_separation_seconds)
        if self.per_label and label in self.per_label:
            cfg = self.per_label[label]
            if "threshold" in cfg:
                thr = float(cfg["threshold"])
            if "min_sep" in cfg:
                min_sep = float(cfg["min_sep"])
        return thr, min_sep

    @staticmethod
    def _filter_local_maxima(indices: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """
        Garde seulement les indices qui sont des maxima locaux (strict ou plateau géré).
        - On accepte un plateau si l'indice est au sommet du plateau (proba == max local).
        """
        if indices.size == 0:
            return indices

        # Pour décider "max local", on regarde voisins immédiats dans la série complète.
        # On gère les bords.
        kept = []
        T = probs.shape[0]

        for t in indices.tolist():
            p = probs[t]
            left = probs[t - 1] if t - 1 >= 0 else -np.inf
            right = probs[t + 1] if t + 1 < T else -np.inf

            # cas simple: strictement >= voisins
            if p >= left and p >= right:
                kept.append(t)

        return np.array(kept, dtype=np.int64)

    def _nms_greedy(self, candidates: np.ndarray, probs: np.ndarray, min_sep_frames: int) -> np.ndarray:
        """
        NMS greedy symétrique: garde les meilleurs scores, supprime dans +/- min_sep_frames.
        """
        if candidates.size == 0:
            return candidates

        # Trier par score décroissant
        order = np.argsort(probs[candidates])[::-1]
        cand_sorted = candidates[order]

        selected: List[int] = []
        suppressed = np.zeros(probs.shape[0], dtype=bool)  # suppression par frame

        for t in cand_sorted.tolist():
            if suppressed[t]:
                continue
            selected.append(t)
            lo = max(0, t - min_sep_frames)
            hi = min(probs.shape[0] - 1, t + min_sep_frames)
            suppressed[lo : hi + 1] = True

        selected = np.array(selected, dtype=np.int64)
        selected.sort()  # tri chronologique
        return selected

    def extract_events_for_label(
        self,
        probs_1d: np.ndarray,      # shape (T,)
        half: HalfEmbeddings,
        label: EventLabel,
        source: str = "transformer",
    ) -> List[ActionEvent]:
        T = int(probs_1d.shape[0])
        if T == 0:
            return []

        thr, min_sep_seconds = self._get_params_for_label(str(label))
        candidates = np.where(probs_1d >= thr)[0].astype(np.int64)
        if candidates.size == 0:
            return []

        # Option: ne garder que maxima locaux
        if self.local_max_only:
            candidates = self._filter_local_maxima(candidates, probs_1d)
            if candidates.size == 0:
                return []

        # NMS greedy
        min_sep_frames = self._min_sep_frames(half, min_sep_seconds)
        peaks = self._nms_greedy(candidates, probs_1d, min_sep_frames)

        # top_k par label (après NMS)
        if self.top_k is not None:
            k = int(self.top_k)
            # ici peaks est trié par temps; pour top_k on trie par confiance puis on reprend le temps
            if peaks.size > 0:
                order = np.argsort(probs_1d[peaks])[::-1]
                peaks = peaks[order[:k]]
                peaks.sort()

        events = [
            ActionEvent(
                match_id=half.match_id,
                half=half.half,
                time_sec=half.index_to_time_sec(int(t)),
                label=label,
                confidence=float(probs_1d[int(t)]),
                source=source,
                extra={"t_idx": int(t)},
            )
            for t in peaks.tolist()
        ]

        return events

    def extract_events(
        self,
        scores: np.ndarray,                # shape (T, 17)
        half: HalfEmbeddings,
        labels: Optional[Sequence[EventLabel]] = None,
        source: str = "transformer",
    ) -> List[ActionEvent]:
        if scores.ndim != 2 or scores.shape[1] != 17:
            raise ValueError(f"scores doit être (T,17). Reçu: {scores.shape}")

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

        events.sort(key=lambda e: e.time_sec)
        return events


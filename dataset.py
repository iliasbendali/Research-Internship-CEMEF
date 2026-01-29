# dataset.py 
# ici on va créer les window juste pour l'entraînement

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from data import SoccerNetDataClient
from domain import Half
from labels import extract_annotations_from_match_json, build_targets_for_half

from collections import Counter
import logging

@dataclass(frozen=True)
class HalfKey:
    match_id: str
    half: Half


class LRUHalfCache:
    """
    Cache LRU sur des mi-temps.
    Stocke (X_full, Y_full, step_seconds).
    """
    def __init__(self, max_items: int = 4):
        self.max_items = int(max_items)
        self._cache: "OrderedDict[HalfKey, Tuple[np.ndarray, np.ndarray, float]]" = OrderedDict()

    def get(self, key: HalfKey) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: HalfKey, value: Tuple[np.ndarray, np.ndarray, float]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


class SoccerNetWindowDataset(Dataset):
    """
    Dataset fenêtré pour action spotting.
    Chaque item = fenêtre (L,512) + cibles (L,17) + mask (L,)
    """
    def __init__(
        self,
        root_dir: str,
        match_ids: List[str],
        sigma_by_idx: List[float],
        window_seconds: float = 180.0,
        stride_seconds: float = 90.0,
        cache_size_halves: int = 4,
        default_step_seconds: float = 0.5,  # 2 Hz
    ):
        self.client = SoccerNetDataClient(root_dir)
        self.match_ids = match_ids
        self.sigma_by_idx = sigma_by_idx

        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.default_step_seconds = float(default_step_seconds)

        self.cache = LRUHalfCache(max_items=cache_size_halves)

        # Index léger: (match_id, half, start_frame)
        self.index: List[Tuple[str, Half, int]] = []
        self._build_index()

    @staticmethod
    def _sec_to_frames(seconds: float, step_seconds: float) -> int:
        return max(1, int(round(seconds / step_seconds)))

    def _get_half_full(self, match_id: str, half: Half) -> Tuple[np.ndarray, np.ndarray, float]:
        key = HalfKey(match_id, half)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        # 1) Embeddings
        half_emb = self.client.load_half(match_id, half)
        X_full = np.asarray(half_emb.embeddings, dtype=np.float32)  # (T,512)
        step = float(getattr(half_emb, "step_seconds", self.default_step_seconds) or self.default_step_seconds)

        # 2) Labels JSON -> annotations -> Y_full (robuste)
        try:
            match_json = self.client.load_labels(match_id)
            annotations = extract_annotations_from_match_json(match_json, half=int(half))
            if not hasattr(self, "_label_counter"):
                self._label_counter = Counter()

            for ann in annotations:
                lab = str(ann.get("label", "")).strip()
                if lab:
                    self._label_counter[lab] += 1
        except Exception as e:
            # JSON invalide / tronqué / etc -> on skip ce match pour les labels
            # (on garde les embeddings, mais Y_full = 0)
            if not hasattr(self, "_bad_labels"):
                self._bad_labels = set()
            if match_id not in self._bad_labels:
                logging.warning(...)

                self._bad_labels.add(match_id)
            annotations = []

        Y_full = build_targets_for_half(
            T=int(X_full.shape[0]),
            annotations=annotations,
            step_seconds=step,
            sigma_by_idx=self.sigma_by_idx,
            radius_sigmas=4.0,
        ).astype(np.float32)  # (T,17)

        self.cache.put(key, (X_full, Y_full, step))
        return X_full, Y_full, step


    def _build_index(self) -> None:
        # Crée toutes les fenêtres sliding pour chaque mi-temps
        for match_id in self.match_ids:
            for half in (1, 2):
                half_emb = self.client.load_half(match_id, half)
                T = int(half_emb.num_steps())
                step = float(getattr(half_emb, "step_seconds", self.default_step_seconds) or self.default_step_seconds)

                if T <= 0:
                    continue

                L = self._sec_to_frames(self.window_seconds, step)
                S = self._sec_to_frames(self.stride_seconds, step)

                if T < L:
                    # une seule fenêtre (padding géré au __getitem__)
                    self.index.append((match_id, half, 0))
                    continue

                starts = list(range(0, T - L + 1, S))
                if not starts:
                    starts = [0]
                if starts[-1] != (T - L):
                    starts.append(T - L)

                for start in starts:
                    self.index.append((match_id, half, int(start)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        match_id, half, start = self.index[idx]
        X_full, Y_full, step = self._get_half_full(match_id, half)

        T = int(X_full.shape[0])
        L = self._sec_to_frames(self.window_seconds, step)

        if T < L:
            X = np.zeros((L, X_full.shape[1]), dtype=np.float32)
            Y = np.zeros((L, 17), dtype=np.float32)
            X[:T] = X_full
            Y[:T] = Y_full
            mask = np.zeros((L,), dtype=np.float32)
            mask[:T] = 1.0
        else:
            end = start + L
            X = X_full[start:end]
            Y = Y_full[start:end]
            mask = np.ones((L,), dtype=np.float32)

        return {
            "x": torch.from_numpy(X),        # (L,512)
            "y": torch.from_numpy(Y),        # (L,17)
            "mask": torch.from_numpy(mask),  # (L,)
        }

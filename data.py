from pathlib import Path
import numpy as np
import re
from domain import HalfEmbeddings, Half
import json
from typing import Union, List


class SoccerNetDataClient:
    HALF_FILES = {
        1: "1_ResNET_TF2_PCA512.npy",
        2: "2_ResNET_TF2_PCA512.npy",
    }           # passage des fichiers embeddings aux mi-temps

    LABEL_FILES = ["Labels-v3.json", "Labels-v3"]  # windows peut masquer .json

    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(self.root_dir)

        self.match_index = self._build_match_index()

    # INDEX : puisqu'on ne peut pas obtenir le nom du dossier associé au match uniquement
    # grâce à la requette LLM (car il y a le score dans la requette), on utilise un index

    def _build_match_index(self) -> dict:
        """
        Construit un index :
        (date, team1, team2, competition, season) -> match_id
        """
        index = {}

        # ✅ parsing robuste du nom de dossier match
        # Ex: "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
        pattern = re.compile(
            r"^(?P<date>\d{4}-\d{2}-\d{2}) - (?P<time>\d{2}-\d{2}) (?P<team1>.+?) \d+ - \d+ (?P<team2>.+)$"
        )

        for competition_dir in self.root_dir.iterdir():
            if not competition_dir.is_dir():
                continue

            competition = competition_dir.name

            for season_dir in competition_dir.iterdir():
                if not season_dir.is_dir():
                    continue

                season = season_dir.name

                for match_dir in season_dir.iterdir():
                    if not match_dir.is_dir():
                        continue

                    name = match_dir.name

                    # ✅ on remplace l'ancien parsing fragile par la regex
                    m = pattern.match(name)
                    if not m:
                        continue

                    date_part = m.group("date")
                    team1 = m.group("team1").strip()
                    team2 = m.group("team2").strip()

                    key = (
                        date_part,
                        *sorted([team1.lower(), team2.lower()]),
                        competition,
                        season,
                    )

                    index[key] = f"{competition}/{season}/{match_dir.name}"

        return index

    # RÉSOLUTION

    def resolve_match_id(
        self,
        team1: str,
        team2: str,
        match_date: str,
        competition: str,
        season: str,
    ) -> str:
        key = (
            match_date,
            *sorted([team1.lower(), team2.lower()]),  # même résultat quel que soit l'ordre
            competition,
            season,
        )

        if key not in self.match_index:
            raise KeyError(f"Match introuvable pour {key}")

        return self.match_index[key]

    # CHARGEMENT

    def load_half(self, match_id: str, half: Half) -> HalfEmbeddings:
        match_dir = self.root_dir / match_id
        file = match_dir / self.HALF_FILES[half]

        embeddings = np.load(file)

        return HalfEmbeddings(
            match_id=match_id,
            half=half,
            embeddings=embeddings,
            step_seconds=0.5,
            metadata={"path": str(file)},
        )
    
    def load_match(self, match_id: str) -> List[HalfEmbeddings]:
        """
        Charge les embeddings des deux mi-temps d'un match.
        """
        return [
            self.load_half(match_id, 1),
            self.load_half(match_id, 2),
        ]

    def load_labels(self, match_id: str) -> dict:
        """
        Charge le JSON de labels du match (Labels-v3).
        """
        match_dir = self.root_dir / match_id

        label_path = None
        for name in self.LABEL_FILES:
            p = match_dir / name
            if p.exists():
                label_path = p
                break

        if label_path is None:
            raise FileNotFoundError(f"Labels introuvables dans {match_dir} (attendu: {self.LABEL_FILES})")

        with open(label_path, "r", encoding="utf-8") as f:
            return json.load(f)
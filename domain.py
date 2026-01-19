# HalfEmbeddings ; ActionEvent ; Intent

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Literal, Set



Half = Literal[1, 2]

# Commandes "utilisateur" (LLM -> orchestration).
# trucs que le LLM doit comprendre et donner en sortie
IntentType = Literal[
    "get_events",
    "get_goals",
    "get_highlights",
]

LABELS = [
    "Goal",
    "Kick-off",
    "Penalty",
    "Yellow card",
    "Red card",
    "Yellow->red card",
    "Foul",
    "Substitution",
    "Offside",
    "Ball out of play",
    "Throw-in",
    "Clearance",
    "Corner",
    "Direct free-kick",
    "Indirect free-kick",
    "Shot on target",
    "Shot off target",
]

LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}

# 17 classes SoccerNet version 3 (labels d'événements).
EventLabel = Literal[
    "Goal",
    "Kick-off",
    "Penalty",
    "Yellow card",
    "Red card",
    "Yellow->red card",
    "Foul",
    "Substitution",
    "Offside",
    "Ball out of play",
    "Throw-in",
    "Clearance",
    "Corner",
    "Direct free-kick",
    "Indirect free-kick",
    "Shot on target",
    "Shot off target",
]

ALL_EVENT_LABELS: Set[EventLabel] = set(LABELS)  # ou Set[str] si tu préfères, mais c'est moins safe


# conversion d'un temps en secondes vers 'MM:SS' (arrondi à la seconde).

def format_mmss(time_sec: float) -> str:
    if time_sec < 0:
        time_sec = 0.0
    total = int(round(time_sec))
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"


@dataclass(frozen=True)
class HalfEmbeddings:
    """
    Représente une mi-temps : embeddings temporels + info de conversion index -> secondes.

    embeddings: shape (T, D) (numpy ou torch) 
        T est 2x le nombre de secondes
        D est le nombre de features issues du ResNet (réduit après une PCA)
    step_seconds: durée entre deux pas temporels.
      - Avec 2 embeddings/sec => step_seconds = 0.5
    """
    match_id: str # ex: "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"
    half: Half
    team: Optional[str] = None # car l'équipe qui m'intéresse peut être renseignée ou pas (à enlever à priori)

    embeddings: Any = None  # np.ndarray ou torch.Tensor
    step_seconds: float = 0.5  # pas de temps
    start_time_seconds: float = 0.0  # début de mi-temps

    metadata: Dict[str, Any] = field(default_factory=dict)

    def num_steps(self) -> int:
        """Retourne T (nombre de pas temporels) si possible."""
        try:
            return int(self.embeddings.shape[0])
        except Exception:
            return 0

    def index_to_time_sec(self, t: int) -> float:
        """Index temporel -> secondes depuis le début de la mi-temps."""
        return self.start_time_seconds + float(t) * float(self.step_seconds)

    def index_to_mmss(self, t: int) -> str:
        """Index temporel -> 'MM:SS'."""
        return format_mmss(self.index_to_time_sec(t))
    

@dataclass(frozen=True)
class ActionEvent:
    """
    Événement détecté par le modèle à un instant précis.
    time_sec est la représentation canonique (très pratique pour tout le reste).
    """
    match_id: str
    half: Half
    time_sec: float

    label: EventLabel
    confidence: float

    source: str = "transformer"
    extra: Dict[str, Any] = field(default_factory=dict)

    def mmss(self) -> str:
        return format_mmss(self.time_sec)

    def window(self, around_seconds: float = 40.0) -> Tuple[float, float]:
        """Fenêtre autour de l'événement : [t-around, t+2xaround]."""
        a = max(0.0, self.time_sec - around_seconds)
        b = self.time_sec + 2*around_seconds # en général, les ralentis durent plus longtemps après le but
        return (a, b)


@dataclass(frozen=True)
class Intent:
    """
    Intent = commande structurée issue du LLM.
    C'est ce que l'orchestrator exécute.
    Ici on ne distingue pas encore "équipe focus" vs "adversaire".
    On stocke team1/team2 pour identifier le match, et ensuite on détecte
    les événements sans attribuer une équipe.
    """
    type: IntentType

    competition: Optional[str] = None     # ex: "england_epl"
    season: Optional[str] = None          # ex: "2014-2015"
    match_date: Optional[str] = None      # ex: "2015-02-21" 
    team1: Optional[str] = None           # ex: "Chelsea"
    team2: Optional[str] = None           # ex: "Burnley"
    match_id: Optional[str] = None        # si déjà connu, on bypass la résolution

    # ce que je veux extraire 
    label_filter: Optional[EventLabel] = None  # ex: "Goal" parmi les 17 classes que j'ai

    around_seconds: float = 40.0
    max_results: int = 50

    params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validation minimale pour s'assurer que l'intent est exploitable.
        """

        # 1) Vérifier qu'on peut identifier un match
        if not self.match_id:
            # Cas sans match_id : il faut suffisamment d'infos pour le résoudre
            has_teams = self.team1 is not None and self.team2 is not None
            has_date = self.match_date is not None

            if not (has_teams or has_date):
                raise ValueError(
                    "Intent invalide: fournir match_id OU (team1 + team2) OU match_date."
                )

        # 2) Vérifier max_results
        if self.max_results <= 0:
            raise ValueError("Intent invalide: max_results doit être > 0.")

        # 3) Vérifier around_seconds
        if self.around_seconds < 0:
            raise ValueError("Intent invalide: around_seconds doit être >= 0.")

        # 4) Vérifier label_filter
        if self.label_filter is not None and self.label_filter not in ALL_EVENT_LABELS:
            raise ValueError(
                f"Intent invalide: label_filter='{self.label_filter}' n'est pas un label SoccerNet valide."
            )




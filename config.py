# Config (dataclass)
# je peux piloter tous les params choisis arbitrairement ici

POSTPROC = {
    "threshold": 0.5,
    "min_separation_seconds": 20.0,
    "top_k": None,
    "local_max_only": True,
    "per_label": {
        "Goal": {"threshold": 0.35, "min_sep": 30.0},
        "Yellow card": {"threshold": 0.45, "min_sep": 15.0},
        "Throw-in": {"threshold": 0.55, "min_sep": 6.0},
    },
}

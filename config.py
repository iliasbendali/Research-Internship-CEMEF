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

label_int = {
    "Goal" : 0,
    "Kick-off" : 1,
    "Penalty" : 2,
    "Yellow card" : 3,
    "Red card" : 4,
    "Yellow->red card" : 5,
    "Foul" : 6,
    "Substitution" : 7,
    "Offside" : 8,
    "Ball out of play" : 9,
    "Throw-in" : 10,
    "Clearance" : 11,
    "Corner" : 12,
    "Direct free-kick" : 13,
    "Indirect free-kick" : 14,
    "Shot on target" : 15,
    "Shot off target" : 16,
}

sigma_by_idx = [
    3.0,  # 0 Goal
    4.0,  # 1 Kick-off
    2.5,  # 2 Penalty
    2.0,  # 3 Yellow card
    2.0,  # 4 Red card
    2.0,  # 5 Yellow->red card
    3.0,  # 6 Foul
    3.0,  # 7 Substitution
    2.0,  # 8 Offside
    4.0,  # 9 Ball out of play
    3.0,  # 10 Throw-in
    3.0,  # 11 Clearance
    3.0,  # 12 Corner
    2.5,  # 13 Direct free-kick
    2.5,  # 14 Indirect free-kick
    2.0,  # 15 Shot on target
    2.0,  # 16 Shot off target
]

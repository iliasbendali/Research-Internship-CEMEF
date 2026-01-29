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
    "Shots on target" : 15,
    "Shots off target" : 16,
}

sigma_by_idx = [
    0.75,  # 0 Goal
    1.0,  # 1 Kick-off
    0.5,  # 2 Penalty
    0.5,  # 3 Yellow card
    0.5,  # 4 Red card
    0.5,  # 5 Yellow->red card
    0.75,  # 6 Foul
    0.75,  # 7 Substitution
    0.5,  # 8 Offside
    1.0,  # 9 Ball out of play
    0.75,  # 10 Throw-in
    0.75,  # 11 Clearance
    0.75,  # 12 Corner
    0.7,  # 13 Direct free-kick
    0.7,  # 14 Indirect free-kick
    0.5,  # 15 Shot on target
    0.5,  # 16 Shot off target
]

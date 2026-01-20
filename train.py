import torch
from torch.utils.data import DataLoader

from dataset import SoccerNetWindowDataset
from models_patchtst import PatchTSTSpotter
from losses import (
    compute_pos_weight_from_loader,
    masked_bce_with_logits_loss,
)

# ------------------
# CONFIG
# ------------------
ROOT_DIR = "C:/Users/Ilias/Documents/cours_MINES/TR/projet/data/data"
BATCH_SIZE = 4             # Zenbook safe
NUM_WORKERS = 2            # CPU loading
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_SECONDS = 180.0
STRIDE_SECONDS = 90.0

# sigma_by_idx (ce qu'on a défini)
sigma_by_idx = [
    3.0, 4.0, 2.5, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0,
    4.0, 3.0, 3.0, 3.0, 2.5, 2.5, 2.0, 2.0
]

# ------------------
# MATCH SPLIT (EXEMPLE)
# ------------------
# IMPORTANT: split par match, pas par fenêtre
train_match_ids = []
for competition_dir in ROOT_DIR.iterdir():
    for season_dir in competition_dir.iterdir():
        for match_dir in season_dir.iterdir():
            train_match_ids.append(
                f"{competition_dir.name}/{season_dir.name}/{match_dir.name}"
            )

# ------------------
# DATASET + DATALOADER
# ------------------
train_dataset = SoccerNetWindowDataset(
    root_dir=ROOT_DIR,
    match_ids=train_match_ids,
    sigma_by_idx=sigma_by_idx,
    window_seconds=WINDOW_SECONDS,
    stride_seconds=STRIDE_SECONDS,
    cache_size_halves=4,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# ------------------
# MODEL
# ------------------
model = PatchTSTSpotter(
    num_input_channels=512,
    d_model=256,
    num_hidden_layers=6,
    num_attention_heads=8,
    ffn_dim=1024,
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)

# ------------------
# POS_WEIGHT (approx sur N batches)
# ------------------
pos_weight = compute_pos_weight_from_loader(
    train_loader,
    device=DEVICE,
    max_batches=200,
)

print("pos_weight:", pos_weight.tolist())

# ------------------
# TRAIN 1 EPOCH
# ------------------
model.train()
total_loss = 0.0

for batch in train_loader:
    x = batch["x"].to(DEVICE)      # (B,L,512)
    y = batch["y"].to(DEVICE)      # (B,L,17)
    mask = batch["mask"].to(DEVICE)# (B,L)

    optimizer.zero_grad()

    out = model(x)
    logits = out["logits"]         # (B,L,17)

    loss = masked_bce_with_logits_loss(
        logits=logits,
        targets=y,
        mask=mask,
        pos_weight=pos_weight,
    )

    loss.backward()
    optimizer.step()

    total_loss += loss.item()

print("Train loss (1 epoch):", total_loss / len(train_loader))

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import SoccerNetWindowDataset
from models_patchtst import PatchTSTSpotter
from losses import (
    compute_pos_weight_from_loader,
    masked_bce_with_logits_loss,
)

# ------------------
# CONFIG
# ------------------
ROOT_DIR = Path("/home/ibendali/soccernet_data/data/")
BATCH_SIZE = 2       
NUM_WORKERS = 0            # CPU loading
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

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

train_match_ids = train_match_ids[:7]  # <-- limite pour tester
"""
train_match_ids = ["england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"]
"""
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
    pin_memory = (DEVICE == "cuda" and NUM_WORKERS > 0),
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
# ---- sanity check label density ----

with torch.no_grad():
    batch = next(iter(train_loader))
    y0 = batch["y"]          # (B,L,17)
    mask0 = batch["mask"]    # (B,L)
    m = mask0.unsqueeze(-1)
    pos_per_class = (y0 * m).sum(dim=(0,1))          # somme des valeurs gaussiennes
    max_per_class = (y0 * m).amax(dim=(0,1))         # max (devrait être proche de 1 si un event est dans la fenêtre)
    print("sanity pos_sum (1 batch):", pos_per_class.tolist())
    print("sanity pos_max (1 batch):", max_per_class.tolist())


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

    import torch.nn.functional as F

    out = model(x)
    logits = out["logits"]  # (B,Lp,17)

    # --- align targets/mask to logits length ---
    B, Lp, C = logits.shape
    L = y.shape[1]

    if Lp != L:
        y_aligned = F.interpolate(
            y.permute(0, 2, 1),      # (B,17,L)
            size=Lp,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)           # (B,Lp,17)

        mask_aligned = F.interpolate(
            mask.unsqueeze(1),       # (B,1,L)
            size=Lp,
            mode="nearest",
        ).squeeze(1)                 # (B,Lp)
    else:
        y_aligned = y
        mask_aligned = mask

    loss = masked_bce_with_logits_loss(
        logits=logits,
        targets=y_aligned,
        mask=mask_aligned,
        pos_weight=pos_weight,
    )


    loss.backward()
    optimizer.step()

    total_loss += loss.item()

print("Train loss (1 epoch):", total_loss / len(train_loader))

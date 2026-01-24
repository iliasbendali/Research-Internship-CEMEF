# train.py
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SoccerNetWindowDataset
from models_patchtst import PatchTSTSpotter
from losses import compute_pos_weight_from_loader, masked_bce_with_logits_loss, masked_asymmetric_focal_loss_with_logits

# ------------------
# CONFIG
# ------------------
ROOT_DIR = Path("/home/ibendali/soccernet_data/data/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

BATCH_SIZE = 1
NUM_WORKERS = 0
PIN_MEMORY = (DEVICE == "cuda" and NUM_WORKERS > 0)

WINDOW_SECONDS = 180.0
STRIDE_SECONDS = 90.0
EPOCHS = 8               # ✅ change ici ton nombre d'epochs
ACCUM_STEPS = 1           # ✅ mets 2/4 si tu veux batch effectif >1
LR = 1e-4
WEIGHT_DECAY = 1e-4

# sigma_by_idx
sigma_by_idx = [
    3.0, 4.0, 2.5, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0,
    4.0, 3.0, 3.0, 3.0, 2.5, 2.5, 2.0, 2.0
]

# ------------------
# MATCH LIST + FILTER LABELS
# ------------------
match_ids = []
glo = 0
no_labels = 0

for competition_dir in ROOT_DIR.iterdir():
    if not competition_dir.is_dir():
        continue
    for season_dir in competition_dir.iterdir():
        if not season_dir.is_dir():
            continue
        for match_dir in season_dir.iterdir():
            if not match_dir.is_dir():
                continue
            glo += 1
            labels = list(match_dir.glob("Labels*"))
            if len(labels) > 0:
                match_ids.append(f"{competition_dir.name}/{season_dir.name}/{match_dir.name}")
            else:
                no_labels += 1

print(f"matches total scanned: {glo} | with labels: {len(match_ids)} | without labels: {no_labels} | ratio_without={no_labels/max(glo,1):.3f}")

# ✅ pour debug rapide, limite le nombre de matchs
random.shuffle(match_ids)
all_matches = match_ids

n = len(all_matches)
train_ids = all_matches[:int(0.8 * n)]
val_ids   = all_matches[int(0.8 * n):int(0.9 * n)]
test_ids  = all_matches[int(0.9 * n):]

print(f"split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

# ------------------
# DATASETS + LOADERS
# ------------------
train_dataset = SoccerNetWindowDataset(
    root_dir=ROOT_DIR,
    match_ids=train_ids,
    sigma_by_idx=sigma_by_idx,
    window_seconds=WINDOW_SECONDS,
    stride_seconds=STRIDE_SECONDS,
    cache_size_halves=4,
)

val_dataset = SoccerNetWindowDataset(
    root_dir=ROOT_DIR,
    match_ids=val_ids,
    sigma_by_idx=sigma_by_idx,
    window_seconds=WINDOW_SECONDS,
    stride_seconds=STRIDE_SECONDS,
    cache_size_halves=2,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

def quick_target_sanity(loader, n=50):
    pos_sum = torch.zeros(17)
    maxv = torch.zeros(17)
    pos_ratio = torch.zeros(17)
    tot = 0

    for i, batch in enumerate(loader):
        y = batch["y"].float()          # (B,L,17)
        m = batch["mask"].float()       # (B,L)

        # ne garde que les frames valides
        m3 = m.unsqueeze(-1)
        yv = y * m3

        pos_sum += yv.sum(dim=(0,1))
        maxv = torch.maximum(maxv, yv.max(dim=1).values.max(dim=0).values)
        pos_ratio += (yv > 0.1).float().sum(dim=(0,1))  # seuil soft (gauss)
        tot += int(m.sum().item())  # nb frames valides

        if i+1 >= n:
            break

    pos_ratio = pos_ratio / max(tot, 1)
    print("Frames valides total:", tot)
    print("Y max par classe:", maxv.tolist())
    print("Y sum par classe:", pos_sum.tolist())
    print("Ratio (Y>0.1) par classe:", pos_ratio.tolist())

quick_target_sanity(train_loader, n=50)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

# ------------------
# MODEL + OPTIM
# ------------------
model = PatchTSTSpotter(
    num_input_channels=512,
    d_model=256,
    num_hidden_layers=6,
    num_attention_heads=8,
    ffn_dim=1024,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ------------------
# POS_WEIGHT (sur TRAIN uniquement)
# ------------------
pos_weight = compute_pos_weight_from_loader(
    train_loader,
    device=DEVICE,
    max_batches=400,   # ✅ augmente si tu veux une meilleure estimation
)
print("pos_weight:", pos_weight.tolist())

# ------------------
# HELPERS
# ------------------
def align_targets_to_logits(y, mask, logits):
    """Downsample y/mask de (B,L,*) vers (B,Lp,*) pour matcher logits."""
    B, Lp, C = logits.shape
    L = y.shape[1]
    if Lp == L:
        return y, mask
    y_aligned = F.interpolate(
        y.permute(0, 2, 1), size=Lp, mode="linear", align_corners=False
    ).permute(0, 2, 1)
    mask_aligned = F.interpolate(
        mask.unsqueeze(1), size=Lp, mode="nearest"
    ).squeeze(1)
    return y_aligned, mask_aligned

def pick_peaks_1d(probs_1d: torch.Tensor, threshold: float, min_sep: int):
    """
    probs_1d: (L,) tensor CPU/torch
    retourne liste d'indices pics.
    """
    p = probs_1d
    idx = (p >= threshold).nonzero(as_tuple=False).flatten().tolist()
    if not idx:
        return []
    # pics = points au-dessus seuil, puis on garde les maxima locaux avec séparation min_sep
    # stratégie simple: trier par proba décroissante, puis greedy suppression
    idx_sorted = sorted(idx, key=lambda i: float(p[i]), reverse=True)
    kept = []
    for i in idx_sorted:
        if all(abs(i - j) >= min_sep for j in kept):
            kept.append(i)
    kept.sort()
    return kept

@torch.no_grad()
def eval_val_events(model, loader, threshold=0.3, y_pos_thr=0.5, tol_sec=5.0, min_sep_sec=10.0):
    """
    Évaluation événementielle simple:
    - GT events: indices où y_aligned >= y_pos_thr, puis pics par NMS aussi
    - Pred events: pics sur probs
    Match TP si un pred est à <= tol_sec d'un GT (par classe).
    """
    model.eval()
    tp = torch.zeros(17, dtype=torch.long)
    fp = torch.zeros(17, dtype=torch.long)
    fn = torch.zeros(17, dtype=torch.long)

    for batch in loader:
        x = batch["x"].to(DEVICE)
        y = batch["y"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        logits = model(x)["logits"]           # (B,Lp,17)
        y_aligned, mask_aligned = align_targets_to_logits(y, mask, logits)
        probs = torch.sigmoid(logits)

        # conversion token -> secondes (approx)
        # fenêtre = WINDOW_SECONDS, Lp tokens => dt_token = WINDOW_SECONDS / Lp
        Lp = probs.shape[1]
        dt = WINDOW_SECONDS / float(Lp)
        min_sep = max(1, int(round(min_sep_sec / dt)))
        tol = max(1, int(round(tol_sec / dt)))

        for b in range(probs.shape[0]):
            m = mask_aligned[b].bool()  # (Lp,)
            for c in range(17):
                p1 = probs[b, :, c].detach().cpu()
                y1 = y_aligned[b, :, c].detach().cpu()

                # ignore padding
                p1 = p1[m.cpu()]
                y1 = y1[m.cpu()]

                pred_peaks = pick_peaks_1d(p1, threshold=threshold, min_sep=min_sep)
                gt_peaks   = pick_peaks_1d(y1, threshold=y_pos_thr,  min_sep=min_sep)

                matched = set()
                for pi in pred_peaks:
                    # trouve un gt à distance <= tol
                    ok = None
                    for gi, g in enumerate(gt_peaks):
                        if gi in matched:
                            continue
                        if abs(pi - g) <= tol:
                            ok = gi
                            break
                    if ok is not None:
                        tp[c] += 1
                        matched.add(ok)
                    else:
                        fp[c] += 1

                fn[c] += (len(gt_peaks) - len(matched))

    support = (tp + fn)  # nb GT events (approximé) par classe
    active = support > 0
    if active.any():
        precision = (tp[active].float() / (tp[active] + fp[active]).float().clamp(min=1)).mean().item()
        recall    = (tp[active].float() / (tp[active] + fn[active]).float().clamp(min=1)).mean().item()
    else:
        precision, recall = 0.0, 0.0
    f1 = float(2 * precision * recall / (precision + recall + 1e-9))

    return precision, recall, f1


@torch.no_grad()
def eval_val_proxy_multi(model, loader, thresholds=(0.1, 0.2, 0.3, 0.5), y_pos_thr=0.3):
    model.eval()
    results = {}
    for th in thresholds:
        tp = torch.zeros(17, dtype=torch.long)
        fp = torch.zeros(17, dtype=torch.long)
        fn = torch.zeros(17, dtype=torch.long)

        for batch in loader:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            logits = model(x)["logits"]
            y_aligned, mask_aligned = align_targets_to_logits(y, mask, logits)

            probs = torch.sigmoid(logits)
            pred = (probs >= th)

            y_pos = (y_aligned >= y_pos_thr)

            m = mask_aligned.unsqueeze(-1).bool()
            pred = pred & m
            y_pos = y_pos & m

            tp += (pred & y_pos).sum(dim=(0,1)).cpu()
            fp += (pred & (~y_pos)).sum(dim=(0,1)).cpu()
            fn += ((~pred) & y_pos).sum(dim=(0,1)).cpu()

        support = (tp + fn)  # nb GT events (approximé) par classe
        active = support > 0
        if active.any():
            precision = (tp[active].float() / (tp[active] + fp[active]).float().clamp(min=1)).mean().item()
            recall    = (tp[active].float() / (tp[active] + fn[active]).float().clamp(min=1)).mean().item()
        else:
            precision, recall = 0.0, 0.0
        f1 = float(2 * precision * recall / (precision + recall + 1e-9))

        results[th] = (precision, recall, float(f1))
    return results


# ------------------
# TRAIN + VAL LOOP
# ------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    steps = 0

    for step, batch in enumerate(train_loader):
        x = batch["x"].to(DEVICE)
        y = batch["y"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        out = model(x)
        logits = out["logits"]  # (B,Lp,17)

        y_aligned, mask_aligned = align_targets_to_logits(y, mask, logits)


        loss = masked_bce_with_logits_loss(
            logits=logits,
            targets=y_aligned,
            mask=mask_aligned,
            pos_weight=pos_weight,   # IMPORTANT
        )



        (loss / ACCUM_STEPS).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        steps += 1

    # flush restant
    if steps % ACCUM_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss / max(steps, 1)

    # ✅ validation proxy
    val_p, val_r, val_f1 = eval_val_events(model, val_loader, threshold=0.3, y_pos_thr=0.5, tol_sec=5.0, min_sep_sec=10.0)
    print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} | val_event_P={val_p:.4f} val_event_R={val_r:.4f} val_event_F1={val_f1:.4f}")



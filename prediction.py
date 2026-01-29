# prediction.py
from pathlib import Path
import random
import json
import numpy as np
import torch

from data import SoccerNetDataClient
from models_patchtst import PatchTSTSpotter

# ====== CONFIG ======
ROOT_DIR = Path("/home/ibendali/soccernet_data/data/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ou chemin absolu si tu préfères:
CKPT_PATH = Path("/home/ibendali/checkpoints/best_f1.pt")

WINDOW_SECONDS = 180.0
STRIDE_SECONDS = 90.0
STEP_SECONDS_DEFAULT = 0.5  # 2Hz


# mêmes sigmas que train (ou ceux que tu utilises en ce moment)
sigma_by_idx = [
    3.0,4.0,2.5,2.0,2.0,2.0,3.0,3.0,2.0,4.0,3.0,3.0,3.0,2.5,2.5,2.0,2.0
]

# Postprocess simple (sans dépendre de ton domain.py)
LABELS = [
    "Goal","Kick-off","Penalty","Yellow card","Red card","Yellow->red card",
    "Foul","Substitution","Offside","Ball out of play","Throw-in","Clearance",
    "Corner","Direct free-kick","Indirect free-kick","Shots on target","Shots off target"
]

def sec_to_frames(seconds: float, step_seconds: float) -> int:
    return max(1, int(round(seconds / step_seconds)))

def pick_peaks_1d(p, threshold, min_sep):
    idx = np.where(p >= threshold)[0]
    if idx.size == 0:
        return []
    idx_sorted = sorted(idx.tolist(), key=lambda i: float(p[i]), reverse=True)
    kept = []
    for i in idx_sorted:
        if all(abs(i - j) >= min_sep for j in kept):
            kept.append(i)
    kept.sort()
    return kept

def postprocess(scores_T17, step_seconds, per_class_percentile=99.9, min_sep_sec=15.0, top_k=None):
    T = scores_T17.shape[0]
    min_sep = max(1, int(round(min_sep_sec / step_seconds)))
    events = []
    for c in range(17):
        p = scores_T17[:, c]
        pmax = float(p.max())
        if pmax < 0.05:
            continue

        threshold = float(np.percentile(p, per_class_percentile))
        # threshold = max(0.05, min(threshold, pmax * 0.9))
        threshold = 0.5 
        peaks = pick_peaks_1d(p, threshold=threshold, min_sep=min_sep)
        if top_k is not None and len(peaks) > top_k:
            peaks = sorted(peaks, key=lambda t: p[t], reverse=True)[:top_k]
            peaks = sorted(peaks)
        for t in peaks:
            ms = int(round(1000.0 * (t * step_seconds)))
            events.append((ms, LABELS[c], float(p[t]), threshold))
    events.sort(key=lambda x: x[0])
    return events

@torch.no_grad()
def infer_half_scores_framelevel(model, half_embeddings_T512, step_seconds):
    """
    Inference fenêtrée: on passe des fenêtres (L,512),
    modèle -> logits (Lp,17),
    puis on upsample vers L frames (nearest),
    et on agrège sur toute la mi-temps.
    """
    model.eval()
    X_full = half_embeddings_T512.astype(np.float32)
    T_full = X_full.shape[0]

    L = sec_to_frames(WINDOW_SECONDS, step_seconds)
    S = sec_to_frames(STRIDE_SECONDS, step_seconds)

    # sorties agrégées
    agg = np.zeros((T_full, 17), dtype=np.float32)
    cnt = np.zeros((T_full, 1), dtype=np.float32)

    # fenêtres couvrant la mi-temps
    if T_full <= L:
        starts = [0]
    else:
        starts = list(range(0, T_full - L + 1, S))
        if starts[-1] != (T_full - L):
            starts.append(T_full - L)

    for start in starts:
        end = start + L
        x_win = X_full[start:end]
        if x_win.shape[0] < L:
            pad = np.zeros((L - x_win.shape[0], 512), dtype=np.float32)
            x_win = np.concatenate([x_win, pad], axis=0)

        x = torch.from_numpy(x_win).unsqueeze(0).to(DEVICE)  # (1,L,512)
        logits = model(x)["logits"]  # (1,Lp,17)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()  # (Lp,17)

        # upsample token->frames de la fenêtre (nearest pour ne pas lisser)
        Lp = probs.shape[0]
        # indices frames -> token
        # map frame i in [0,L) to token floor(i * Lp / L)
        idx_tok = (np.floor(np.arange(L) * (Lp / float(L)))).astype(np.int64)
        idx_tok = np.clip(idx_tok, 0, Lp - 1)
        probs_frames = probs[idx_tok]  # (L,17)

        # agrégation sur la mi-temps (moyenne simple)
        real_len = min(L, T_full - start)
        agg[start:start+real_len] += probs_frames[:real_len]
        cnt[start:start+real_len] += 1.0

    scores = agg / np.clip(cnt, 1.0, None)
    return scores  # (T_full,17) frame-level (2Hz)

def main():
    with open("splits.json", "r") as f:
        splits = json.load(f)

    val_ids = splits["val"]
    match_id = val_ids[0]   # ou random.choice(val_ids)
    print("Using VAL match_id:", match_id)


    # 2) Charger modèle
    model = PatchTSTSpotter(
        num_input_channels=512,
        d_model=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        ffn_dim=1024,
    ).to(DEVICE)

    assert CKPT_PATH.exists(), f"Checkpoint introuvable: {CKPT_PATH.resolve()}"

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    print(f"✅ Loaded checkpoint: {CKPT_PATH} | epoch={ckpt.get('epoch')} | val_f1={ckpt.get('val_f1')}")



    client = SoccerNetDataClient(ROOT_DIR)

    for half in [1, 2]:
        half_obj = client.load_half(match_id, half)
        X = np.asarray(half_obj.embeddings, dtype=np.float32)  # (T,512)
        step = float(getattr(half_obj, "step_seconds", STEP_SECONDS_DEFAULT) or STEP_SECONDS_DEFAULT)

        scores = infer_half_scores_framelevel(model, X, step_seconds=step)
        # stats rapides
        m = scores.mean(axis=0)
        s = scores.std(axis=0)
        mx = scores.max(axis=0)

        print("mean per class:", np.round(m, 3))
        print("std  per class:", np.round(s, 3))
        print("max  per class:", np.round(mx, 3))

        events = postprocess(
            scores, step_seconds=step,
            per_class_percentile=99.9,      # ajuste
            min_sep_sec=15.0,   # ajuste
            top_k=None
        )

        print(f"\n--- HALF {half} --- step={step}s | T={scores.shape[0]}")
        for ms, lab, sc, thr in events[:200]:
            print(f"{ms:>8d} ms | {lab:<18s} | score={sc:.3f} | thr={thr:.3f}")
    match_json = client.load_labels(match_id)
    print("keys:", list(match_json.keys())[:30])

    for k in ["annotations", "Annotations", "labels", "Labels", "events", "Events"]:
        v = match_json.get(k, None)
        if isinstance(v, list):
            print(f"Found list key '{k}' with len={len(v)}")
            if len(v) > 0:
                print("sample item:", v[0])
            break
    else:
        # aucune liste trouvée
        print("No obvious list of annotations found.")
        # affiche un extrait du json pour comprendre
        s = json.dumps(match_json)[:800]
        print("json head:", s)


if __name__ == "__main__":
    main()



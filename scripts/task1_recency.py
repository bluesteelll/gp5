"""recency-взвешивание обучения для задачи 1.

тест целиком приходится на ковид и заметно поляризован (53% пятёрок, 22% единиц против 43%/14%
в train) -- это сдвиг распределения во времени. взвешиваем обучающие примеры по свежести,
w = exp(-(t_end_train - date)/tau), чтобы сеть сильнее опиралась на недавние паттерны, ближе к
val/test. tau подбираем по val. голова и архитектура -- как у baseline (ёмкость+ классиф. голова).
утечки нет: веса зависят только от даты строки, не от таргета и не от будущего.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from _constants import PROCESSED, ARTIFACTS
import task1_lib as L

BIG = dict(d_cat=32, d_branch=48, hidden=128, top=(256, 128))
PRED_DIR = ARTIFACTS / "exp_preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def recency_weights(splits, tau_days):
    """вес train-строки по свежести относительно конца train-окна; val/test не используются."""
    dates = pd.read_parquet(PROCESSED / "task1_dataset.parquet", columns=["review_id", "date"])
    dates["date"] = pd.to_datetime(dates["date"])
    tr = splits["train"].merge(dates, on="review_id", how="left")
    t_end = tr["date"].max()
    age = (t_end - tr["date"]).dt.total_seconds().to_numpy() / 86400.0
    return np.exp(-age / tau_days).astype(np.float32)


def run_weighted(name, factory, splits, y_val, n_cat_vocab, w_train, seed=42):
    tens = {s: L.make_tensors(splits[s], L.U_FULL, L.B_FULL, L.C_FULL, n_cat_vocab)
            for s in ["train", "val", "test"]}
    tens["train"]["w"] = torch.tensor(w_train, device=L.device)
    torch.manual_seed(seed)
    model = factory(len(L.U_FULL), len(L.B_FULL), len(L.C_FULL)).to(L.device)

    def decode(out):
        return (torch.softmax(out, dim=1) * L.STAR_VALUES).sum(dim=1)

    @torch.no_grad()
    def predict_t(t):
        model.eval()
        out = []
        for b in L.batches(t, 8192, shuffle=False):
            o = decode(model({k: v for k, v in b.items() if k not in ("y", "w")}))
            out.append(o.float().cpu().numpy())
        return np.concatenate(out)

    opt = torch.optim.AdamW(model.parameters(), lr=L.LR, weight_decay=L.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best = {"rmse": np.inf, "state": None, "epoch": -1}
    t0 = time.time()
    for epoch in range(L.MAX_EPOCHS):
        model.train()
        for b in L.batches(tens["train"], L.BATCH, shuffle=True):
            if len(b["y"]) < 2:
                continue
            opt.zero_grad()
            out = model({k: v for k, v in b.items() if k not in ("y", "w")})
            target = (b["y"] - 1).long()
            loss = (F.cross_entropy(out, target, reduction="none") * b["w"]).sum() / b["w"].sum()
            loss.backward()
            opt.step()
        val_rmse = L.rmse(y_val, predict_t(tens["val"]))
        sched.step(val_rmse)
        if val_rmse < best["rmse"]:
            best = {"rmse": val_rmse, "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch}
        if epoch - best["epoch"] >= L.PATIENCE:
            break
    model.load_state_dict(best["state"])
    print(f"{name} val {best['rmse']:.4f} {time.time() - t0:.0f}c")
    return {"val_rmse": best["rmse"], "val_pred": predict_t(tens["val"]),
            "test_pred": predict_t(tens["test"]), "state": best["state"]}


def main():
    D = L.prepare()
    Inter = D["models"]["InteractionMLP"]
    factory = lambda du, db, dc: Inter(du, db, dc, **BIG, out=5)
    base_val = L.rmse(D["y_val"], np.load(PRED_DIR / "cls_big_seed42_val.npy")) if (PRED_DIR / "cls_big_seed42_val.npy").exists() else None
    print("baseline сид42 val:", round(base_val, 4) if base_val else "n/a")
    best = None
    for tau in [180, 365, 730, 1460]:
        w = recency_weights(D["splits"], tau)
        r = run_weighted(f"recency tau={tau}", factory, D["splits"], D["y_val"], D["n_cat_vocab"], w, seed=42)
        print(f"  tau={tau} val {r['val_rmse']:.4f} test {L.rmse(D['y_te'], r['test_pred']):.4f}")
        if best is None or r["val_rmse"] < best["val_rmse"]:
            best, best_tau = r, tau
    print(f"лучший tau={best_tau} val {best['val_rmse']:.4f} test {L.rmse(D['y_te'], best['test_pred']):.4f}")
    # сохраняем как ещё один диверсный член бленда (один сид -- индикативно)
    np.save(PRED_DIR / "recency_val.npy", best["val_pred"])
    np.save(PRED_DIR / "recency_test.npy", best["test_pred"])


if __name__ == "__main__":
    main()

"""CORAL — порядковая голова для задачи 1 (предложена в разделе «дальше» отчёта).

вместо softmax-матожидания обучаем K-1 связанных бинарных порогов «оценка > k» с общими весами
и монотонными биасами (Cao et al. 2019, arxiv 1901.07884). это уважает порядок оценок 1<2<3<4<5.
декодирование в RMSE: ожидаемый ранг = 1 + сумма сигмоид порогов. протокол обучения тот же, что в
ноутбуке: AdamW, ранняя остановка по val RMSE, лучшие веса. предсказания кэшируются для бленда.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from _constants import ARTIFACTS
import task1_lib as L

BIG = dict(d_cat=32, d_branch=48, hidden=128, top=(256, 128))
PRED_DIR = ARTIFACTS / "exp_preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)


class CoralHead(nn.Module):
    """общий скаляр-проекция базовой сети + K-1 монотонных биасов -> логиты порогов."""
    def __init__(self, base, k_thresh=4):
        super().__init__()
        self.base = base
        self.biases = nn.Parameter(torch.zeros(k_thresh))

    def forward(self, b):
        s = self.base(b)
        return s.unsqueeze(1) + self.biases


def coral_loss(logits, y):
    r = (y - 1).long().unsqueeze(1)
    levels = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    targets = (r > levels).float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def decode_coral(logits):
    return 1.0 + torch.sigmoid(logits).sum(dim=1)


def run_coral(name, make_base, splits, y_val, n_cat_vocab, seed=42):
    tens = {s: L.make_tensors(splits[s], L.U_FULL, L.B_FULL, L.C_FULL, n_cat_vocab)
            for s in ["train", "val", "test"]}
    torch.manual_seed(seed)
    model = CoralHead(make_base(len(L.U_FULL), len(L.B_FULL), len(L.C_FULL))).to(L.device)

    @torch.no_grad()
    def predict_t(t):
        model.eval()
        out = []
        for b in L.batches(t, 8192, shuffle=False):
            o = decode_coral(model({k: v for k, v in b.items() if k != "y"}))
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
            logits = model({k: v for k, v in b.items() if k != "y"})
            loss = coral_loss(logits, b["y"])
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
    make_base = lambda du, db, dc: Inter(du, db, dc, **BIG, out=1)
    vals, tests = [], []
    for seed in L.SEEDS_MULTI:
        r = run_coral(f"CORAL big seed{seed}", make_base, D["splits"], D["y_val"], D["n_cat_vocab"], seed=seed)
        vals.append(r["val_pred"])
        tests.append(r["test_pred"])
    ev = np.mean(vals, axis=0)
    et = np.mean(tests, axis=0)
    np.save(PRED_DIR / "coral_val.npy", ev)
    np.save(PRED_DIR / "coral_test.npy", et)
    print(f"CORAL ансамбль val {L.rmse(D['y_val'], ev):.4f} test {L.rmse(D['y_te'], et):.4f}")


if __name__ == "__main__":
    main()

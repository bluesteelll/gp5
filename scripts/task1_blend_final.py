"""финальный бленд задачи 1: собирает все базовые модели из artifacts/exp_preds/ и ищет
лучшую комбинацию по val (жадный отбор + оптимизация весов). test печатается один раз.

базовые модели (если есть в кэше): ens (baseline-ансамбль), ens_ext (обогащённые priors),
reg (регрессия), gbdt, coral. веса настраиваются только на val, отбор моделей -- по val.
итог пишется в reports/task1_experiments.json (ключ final_blend).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from _constants import REPORTS, ARTIFACTS
import task1_lib as L

PRED_DIR = ARTIFACTS / "exp_preds"


def load_base():
    """собирает базовые модели из кэша; ens/ens_ext усредняются по сидам на лету."""
    base = {}

    def mean_seeds(prefix):
        vs = sorted(PRED_DIR.glob(f"{prefix}_seed*_val.npy"))
        ts = sorted(PRED_DIR.glob(f"{prefix}_seed*_test.npy"))
        if not vs:
            return None
        return np.mean([np.load(p) for p in vs], 0), np.mean([np.load(p) for p in ts], 0)

    ens = mean_seeds("cls_big")
    if ens:
        base["ens"] = ens
    ens_ext = mean_seeds("cls_big_ext")
    if ens_ext:
        base["ens_ext"] = ens_ext
    for name, key in [("reg_big_seed42", "reg"), ("gbdt", "gbdt"), ("coral", "coral"), ("recency", "recency")]:
        v, t = PRED_DIR / f"{name}_val.npy", PRED_DIR / f"{name}_test.npy"
        if v.exists() and t.exists():
            base[key] = (np.load(v), np.load(t))
    return base


def opt_weights(preds_val, y_val):
    P = np.stack(preds_val, 1)
    k = P.shape[1]
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    res = minimize(lambda w: L.rmse(y_val, P @ w), np.full(k, 1 / k),
                   method="SLSQP", bounds=[(0, 1)] * k, constraints=cons)
    return res.x


def greedy(base, y_val):
    """жадный форвард-отбор по val: добавляем модель, пока бленд улучшается."""
    keys = list(base)
    chosen, best_rmse = [], np.inf
    while True:
        cand = None
        for k in keys:
            if k in chosen:
                continue
            trial = chosen + [k]
            w = opt_weights([base[m][0] for m in trial], y_val)
            r = L.rmse(y_val, np.stack([base[m][0] for m in trial], 1) @ w)
            if r < best_rmse - 1e-5:
                best_rmse, cand = r, k
        if cand is None:
            break
        chosen.append(cand)
    return chosen


def main():
    D = L.prepare()
    y_val, y_te = D["y_val"], D["y_te"]
    warm_user, warm_biz = D["warm_user"], D["warm_biz"]
    base = load_base()
    print("базовые модели:", {k: round(L.rmse(y_val, v[0]), 4) for k, v in base.items()})
    print("           test:", {k: round(L.rmse(y_te, v[1]), 4) for k, v in base.items()})

    chosen = greedy(base, y_val)
    w = opt_weights([base[m][0] for m in chosen], y_val)
    bv = np.stack([base[m][0] for m in chosen], 1) @ w
    bt = np.stack([base[m][1] for m in chosen], 1) @ w
    print(f"\nжадный бленд: {chosen} w={np.round(w,3)}")
    print(f"val {L.rmse(y_val, bv):.4f}  test {L.rmse(y_te, bt):.4f}")

    allk = list(base)
    wa = opt_weights([base[m][0] for m in allk], y_val)
    av = np.stack([base[m][0] for m in allk], 1) @ wa
    at = np.stack([base[m][1] for m in allk], 1) @ wa
    print(f"полный бленд {allk} w={np.round(wa,3)} val {L.rmse(y_val, av):.4f} test {L.rmse(y_te, at):.4f}")

    out = {}
    if (REPORTS / "task1_experiments.json").exists():
        out = json.load(open(REPORTS / "task1_experiments.json", encoding="utf-8"))
    out["final_blend"] = {
        "members": chosen, "weights": [round(float(x), 4) for x in w],
        "val_rmse": L.rmse(y_val, bv), "test_rmse": L.rmse(y_te, bt),
        "segments": L.segment_report(bt, y_te, warm_user, warm_biz),
        "base_val_rmse": {k: L.rmse(y_val, v[0]) for k, v in base.items()},
        "base_test_rmse": {k: L.rmse(y_te, v[1]) for k, v in base.items()},
    }
    json.dump(out, open(REPORTS / "task1_experiments.json", "w"), ensure_ascii=False, indent=2)
    print("сохранено в reports/task1_experiments.json (final_blend)")


if __name__ == "__main__":
    main()

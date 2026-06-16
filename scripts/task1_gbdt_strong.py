"""более ёмкий поиск GBDT: больше листьев и меньше регуляризации, чтобы GBDT приблизился к сети
и сильнее вкладывался в бленд. лучший по val сохраняется в кэш artifacts/exp_preds/gbdt_*.npy.
"""
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from _constants import ARTIFACTS
import task1_lib as L
from task1_gbdt import fit_gbdt

PRED_DIR = ARTIFACTS / "exp_preds"
GRID = [
    {"learning_rate": 0.05, "max_leaf_nodes": 127, "min_samples_leaf": 50},
    {"learning_rate": 0.05, "max_leaf_nodes": 255, "min_samples_leaf": 30},
    {"learning_rate": 0.03, "max_leaf_nodes": 127, "min_samples_leaf": 50, "max_iter": 1000},
    {"learning_rate": 0.02, "max_leaf_nodes": 63, "min_samples_leaf": 100, "max_iter": 1200},
]


def main():
    D = L.prepare()
    splits, y_val, y_te = D["splits"], D["y_val"], D["y_te"]
    cur = float("inf")
    cv = PRED_DIR / "gbdt_val.npy"
    if cv.exists():
        cur = L.rmse(y_val, np.load(cv))
    print(f"текущий лучший GBDT по val: {cur:.4f}")
    best = None
    for g in GRID:
        r = fit_gbdt(D["df"], splits, y_val, y_te, params=g)
        print(f"GBDT {g} val {r['val_rmse']:.4f} test {r['test_rmse']:.4f} iters {r['n_iter']}")
        if best is None or r["val_rmse"] < best["val_rmse"]:
            best = r
    if best["val_rmse"] < cur:
        np.save(PRED_DIR / "gbdt_val.npy", best["val_pred"])
        np.save(PRED_DIR / "gbdt_test.npy", best["test_pred"])
        print(f"обновлён кэш GBDT: val {best['val_rmse']:.4f} test {best['test_rmse']:.4f}")
    else:
        print(f"улучшения нет, кэш не трогаем (лучший новый val {best['val_rmse']:.4f})")


if __name__ == "__main__":
    main()

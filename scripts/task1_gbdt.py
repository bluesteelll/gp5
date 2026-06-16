"""GBDT-ветка для задачи 1: HistGradientBoostingRegressor на плоской матрице признаков.

GBDT в проекте не пробовали, а на табличных данных это сильный базовый метод и хороший
кандидат в гетерогенный бленд с сетью (ошибки декоррелируют). признаки те же, что у сети
(числовые + prior + время + категории multi-hot + город), утечки нет: GBDT учится только на train,
ранняя остановка на внутреннем train-холдауте, выбор гиперпараметров по нашему val.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from _constants import PROCESSED
from task1_lib import USER_NUM, BIZ_NUM, CROSS, TIME_COLS, USER_PRIOR, BIZ_PRIOR, CROSS_PRIOR, rmse

NUM_ALL = USER_NUM + BIZ_NUM + CROSS + USER_PRIOR + BIZ_PRIOR + CROSS_PRIOR


def build_matrix(df):
    """плоская матрица признаков для GBDT: числовые сети + prior + время + категории + город."""
    cat_cols_src = pd.read_parquet(PROCESSED / "task1_features.parquet")
    cat_cols = [c for c in cat_cols_src.columns if c.startswith("cat_") or c.startswith("city_")]
    extra = cat_cols_src[["review_id"] + cat_cols]
    d = df.merge(extra, on="review_id", validate="one_to_one")
    feat_cols = NUM_ALL + TIME_COLS + cat_cols
    return d, feat_cols


def fit_gbdt(df, splits, y_val, y_te, seed=42, params=None):
    """обучает HistGBR на train, возвращает предсказания на val/test (клип в [1,5] делает rmse)."""
    d, feat_cols = build_matrix(df)
    dsplit = {s: d[d["split"] == s] for s in ["train", "val", "test"]}
    Xtr, ytr = dsplit["train"][feat_cols].to_numpy(np.float32), dsplit["train"]["stars"].to_numpy()
    p = dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=200,
             l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
             n_iter_no_change=30, random_state=seed)
    if params:
        p.update(params)
    model = HistGradientBoostingRegressor(**p)
    model.fit(Xtr, ytr)
    val_pred = model.predict(dsplit["val"][feat_cols].to_numpy(np.float32))
    test_pred = model.predict(dsplit["test"][feat_cols].to_numpy(np.float32))
    return {"name": f"GBDT seed{seed}", "model": model, "val_pred": val_pred, "test_pred": test_pred,
            "val_rmse": rmse(y_val, val_pred), "test_rmse": rmse(y_te, test_pred), "n_iter": model.n_iter_}


def tune_gbdt(df, splits, y_val, y_te):
    """малый перебор гиперпараметров по нашему val, печатает таблицу, возвращает лучший прогон."""
    grid = [
        {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 200},
        {"learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 100},
        {"learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 300},
        {"learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 200},
    ]
    best, rows = None, []
    for g in grid:
        r = fit_gbdt(df, splits, y_val, y_te, params=g)
        rows.append({**g, "val_rmse": r["val_rmse"], "test_rmse": r["test_rmse"], "n_iter": r["n_iter"]})
        print(f"GBDT {g} val {r['val_rmse']:.4f} test {r['test_rmse']:.4f} iters {r['n_iter']}")
        if best is None or r["val_rmse"] < best["val_rmse"]:
            best = r
    return best, pd.DataFrame(rows)

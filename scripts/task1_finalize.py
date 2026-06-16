"""финализация задачи 1: обучает все члены гетерогенного бленда, подбирает веса по val и
сохраняет восстановимый артефакт logs/task1/final/blend.pt (+ gbdt.joblib), затем обновляет
reports/task1_model_metrics.json. test печатается один раз.

члены бленда (каждый -- честный прогон того же протокола: отбор по val, без новых утечек):
  ens      -- baseline-ансамбль InteractionMLP (ёмкость+ классиф. голова, CE), 3 сида
  ens_ext  -- то же + обогащённые prior-признаки (разброс и доли крайних оценок), 3 сида
  coral    -- та же сеть с порядковой головой CORAL, 3 сида
  recency  -- та же сеть с recency-взвешиванием обучения (tau=1460), 1 сид
  gbdt     -- HistGradientBoostingRegressor на тех же признаках
веса членов подбираются жадно на val (стэкинг), test -- один раз.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from _constants import REPORTS
import task1_lib as L
from task1_priors_ext import attach_ext_priors
from task1_gbdt import fit_gbdt, build_matrix
from task1_coral import run_coral, BIG
from task1_recency import recency_weights, run_weighted
from task1_blend_final import greedy, opt_weights

FINAL_DIR = root / "logs" / "task1" / "final"
GBDT_PARAMS = {"learning_rate": 0.03, "max_leaf_nodes": 127, "min_samples_leaf": 50, "max_iter": 1000}
RECENCY_TAU = 1460


def main():
    D = L.prepare()
    splits, y_val, y_te = D["splits"], D["y_val"], D["y_te"]
    warm_user, warm_biz = D["warm_user"], D["warm_biz"]
    Inter = D["models"]["InteractionMLP"]
    ncv = D["n_cat_vocab"]
    cls_factory = lambda du, db, dc: Inter(du, db, dc, **BIG, out=5)
    base, states = {}, {}

    # ens -- baseline-ансамбль
    vs, ts, st = [], [], []
    for seed in L.SEEDS_MULTI:
        r = L.run_experiment(f"ens s{seed}", cls_factory, "CE", L.U_FULL, L.B_FULL, L.C_FULL,
                             splits, y_val, ncv, seed=seed)
        vs.append(r["val_pred"]); ts.append(r["test_pred"]); st.append(r["state"])
    base["ens"] = (np.mean(vs, 0), np.mean(ts, 0)); states["ens"] = st

    # ens_ext -- обогащённые priors
    df_ext, U_EXT, B_EXT = attach_ext_priors(D["df"])
    splits_ext = {s: df_ext[df_ext["split"] == s] for s in ["train", "val", "test"]}
    vs, ts, st = [], [], []
    for seed in L.SEEDS_MULTI:
        r = L.run_experiment(f"ens_ext s{seed}", cls_factory, "CE", U_EXT, B_EXT, L.C_FULL,
                             splits_ext, y_val, ncv, seed=seed)
        vs.append(r["val_pred"]); ts.append(r["test_pred"]); st.append(r["state"])
    base["ens_ext"] = (np.mean(vs, 0), np.mean(ts, 0)); states["ens_ext"] = st

    # coral -- порядковая голова
    make_base = lambda du, db, dc: Inter(du, db, dc, **BIG, out=1)
    vs, ts, st = [], [], []
    for seed in L.SEEDS_MULTI:
        r = run_coral(f"coral s{seed}", make_base, splits, y_val, ncv, seed=seed)
        vs.append(r["val_pred"]); ts.append(r["test_pred"]); st.append(r["state"])
    base["coral"] = (np.mean(vs, 0), np.mean(ts, 0)); states["coral"] = st

    # recency -- взвешивание свежести
    w = recency_weights(splits, RECENCY_TAU)
    r = run_weighted(f"recency tau={RECENCY_TAU}", cls_factory, splits, y_val, ncv, w, seed=42)
    base["recency"] = (r["val_pred"], r["test_pred"]); states["recency"] = [r["state"]]

    # gbdt
    g = fit_gbdt(D["df"], splits, y_val, y_te, params=GBDT_PARAMS)
    base["gbdt"] = (g["val_pred"], g["test_pred"])

    print("\nбазовые члены val:", {k: round(L.rmse(y_val, v[0]), 4) for k, v in base.items()})

    # ── веса бленда: жадный отбор по val ──
    chosen = greedy(base, y_val)
    w_blend = opt_weights([base[m][0] for m in chosen], y_val)
    bv = np.stack([base[m][0] for m in chosen], 1) @ w_blend
    bt = np.stack([base[m][1] for m in chosen], 1) @ w_blend
    weights = {m: float(round(x, 4)) for m, x in zip(chosen, w_blend)}
    blend_val, blend_test = L.rmse(y_val, bv), L.rmse(y_te, bt)
    base_ens_test = L.rmse(y_te, base["ens"][1])
    print(f"\nбленд {chosen} w={list(weights.values())}")
    print(f"val {blend_val:.4f}  test {blend_test:.4f}  (baseline-ансамбль test {base_ens_test:.4f})")
    seg = L.segment_report(bt, y_te, warm_user, warm_biz)
    print("сегменты:", {k: round(v, 4) for k, v in seg.items()})

    # ── сохранение восстановимого артефакта ──
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    _, gbdt_feats = build_matrix(D["df"])
    joblib.dump(g["model"], FINAL_DIR / "gbdt.joblib")
    artifact = {
        "weights": weights,
        "members": {
            "ens": {"kind": "cls_big", "seeds": L.SEEDS_MULTI, "states": states["ens"],
                    "ucols": L.U_FULL, "bcols": L.B_FULL, "ccols": L.C_FULL},
            "ens_ext": {"kind": "cls_big", "seeds": L.SEEDS_MULTI, "states": states["ens_ext"],
                        "ucols": U_EXT, "bcols": B_EXT, "ccols": L.C_FULL},
            "coral": {"kind": "coral", "seeds": L.SEEDS_MULTI, "states": states["coral"],
                      "ucols": L.U_FULL, "bcols": L.B_FULL, "ccols": L.C_FULL},
            "recency": {"kind": "cls_big", "seeds": [42], "tau": RECENCY_TAU, "states": states["recency"],
                        "ucols": L.U_FULL, "bcols": L.B_FULL, "ccols": L.C_FULL},
            "gbdt": {"kind": "gbdt", "path": "gbdt.joblib", "feat_cols": gbdt_feats, "params": GBDT_PARAMS},
        },
        "big": BIG,
        "n_cat_vocab": ncv,
        "decode": {"cls_big": "softmax_expectation_1to5", "coral": "1+sum_sigmoid", "gbdt": "clip_1_5"},
    }
    torch.save(artifact, FINAL_DIR / "blend.pt")
    print(f"\nсохранено: {FINAL_DIR / 'blend.pt'} и gbdt.joblib")

    # ── обновление метрик (существующие данные сохраняем) ──
    mp = REPORTS / "task1_model_metrics.json"
    m = json.load(open(mp, encoding="utf-8")) if mp.exists() else {}
    m["final_model"] = "гетерогенный бленд (ens + обогащённые priors + CORAL + recency + GBDT, веса по val)"
    m["final_test_rmse"] = blend_test
    m["blend"] = {
        "members": chosen,
        "weights": weights,
        "val_rmse": blend_val,
        "test_rmse": blend_test,
        "segments": seg,
        "base_val_rmse": {k: L.rmse(y_val, v[0]) for k, v in base.items()},
        "base_test_rmse": {k: L.rmse(y_te, v[1]) for k, v in base.items()},
        "baseline_ensemble_test_rmse_same_env": base_ens_test,
        "note": ("бленд и его базовые члены пересчитаны в едином окружении (cuda, torch 2.12); "
                 "коммит-baseline 1.0447 воспроизведён как " + f"{base_ens_test:.4f}, "
                 "выигрыш бленда измеряется относительно него"),
    }
    json.dump(m, open(mp, "w"), ensure_ascii=False, indent=2)
    print(f"обновлено: {mp}")


if __name__ == "__main__":
    main()

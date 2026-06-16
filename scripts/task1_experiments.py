"""драйвер экспериментов задачи 1: воспроизводит baseline в этом окружении и проверяет
улучшения. отбор по val, test печатается для сводки. ничего из коммита не перезаписывает,
результаты кладёт в reports/task1_experiments.json и предсказания в artifacts/exp_preds/.

стадии:
  0. baseline: ансамбль 3 сидов InteractionMLP (ёмкость+ классиф. голова, CE) -- как в ноутбуке 04
  1. диверсификация головы: ёмкость+ регрессия (MSE) -- другой индуктивный сдвиг для бленда
  2. GBDT: HistGradientBoostingRegressor на тех же признаках (в проекте не пробовали)
  3. бленды: среднее и веса, настроенные на val (стэкинг), test -- один раз
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
from task1_gbdt import tune_gbdt
from task1_priors_ext import attach_ext_priors

PRED_DIR = ARTIFACTS / "exp_preds"
PRED_DIR.mkdir(parents=True, exist_ok=True)

BIG = dict(d_cat=32, d_branch=48, hidden=128, top=(256, 128))


def cache(name, fn):
    """кэширует пару (val_pred, test_pred) в npy, чтобы повторные запуски не переобучали."""
    vp, tp = PRED_DIR / f"{name}_val.npy", PRED_DIR / f"{name}_test.npy"
    if vp.exists() and tp.exists():
        return np.load(vp), np.load(tp)
    val_pred, test_pred = fn()
    np.save(vp, val_pred)
    np.save(tp, test_pred)
    return val_pred, test_pred


def blend_weights(preds_val, y_val):
    """неотрицательные веса с суммой 1, минимизируют val RMSE блендированного предсказания."""
    P = np.stack(preds_val, axis=1)
    k = P.shape[1]

    def obj(w):
        return L.rmse(y_val, P @ w)

    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bnds = [(0, 1)] * k
    res = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=bnds, constraints=cons)
    return res.x


def main():
    D = L.prepare()
    splits, y_val, y_te = D["splits"], D["y_val"], D["y_te"]
    warm_user, warm_biz = D["warm_user"], D["warm_biz"]
    Inter = D["models"]["InteractionMLP"]
    ncv = D["n_cat_vocab"]
    print(f"device {L.device} | строк {len(D['df']):,} | cat_vocab {ncv}")

    # bias-бейзлайн для справки
    te = splits["test"]
    mu = splits["train"]["stars"].mean()
    p_bias = np.clip(te["average_stars"] + te["biz_avg_stars"] - mu, 1, 5).to_numpy()
    bias_rmse = L.rmse(y_te, p_bias)

    results = {"device": L.device, "bias_rmse": bias_rmse, "models": {}, "blends": {}}

    def run(name, factory, loss, seed):
        return cache(name, lambda: (
            lambda r: (r["val_pred"], r["test_pred"]))(
            L.run_experiment(name, factory, loss, L.U_FULL, L.B_FULL, L.C_FULL,
                             splits, y_val, ncv, seed=seed)))

    # ── стадия 0: baseline-ансамбль (ёмкость+ классиф. голова, CE), 3 сида ──
    cls_factory = lambda du, db, dc: Inter(du, db, dc, **BIG, out=5)
    ens_val_members, ens_test_members = [], []
    for seed in L.SEEDS_MULTI:
        vp, tp = run(f"cls_big_seed{seed}", cls_factory, "CE", seed)
        ens_val_members.append(vp)
        ens_test_members.append(tp)
    ens_val = np.mean(ens_val_members, axis=0)
    ens_test = np.mean(ens_test_members, axis=0)
    results["models"]["baseline_ensemble (cls big x3)"] = {
        "val_rmse": L.rmse(y_val, ens_val), "test_rmse": L.rmse(y_te, ens_test)}
    print(f"\n[baseline] ансамбль val {L.rmse(y_val, ens_val):.4f} test {L.rmse(y_te, ens_test):.4f}")

    # ── стадия 1: диверсификация головы -- ёмкость+ регрессия MSE ──
    reg_factory = lambda du, db, dc: Inter(du, db, dc, **BIG, out=1)
    reg_val, reg_test = run("reg_big_seed42", reg_factory, "MSE", 42)
    results["models"]["reg big (MSE)"] = {
        "val_rmse": L.rmse(y_val, reg_val), "test_rmse": L.rmse(y_te, reg_test)}
    print(f"[diverse] ёмкость+ регрессия val {L.rmse(y_val, reg_val):.4f} test {L.rmse(y_te, reg_test):.4f}")

    # ── стадия 1.5: обогащённые prior-признаки (разброс и доли крайних оценок) ──
    df_ext, U_EXT, B_EXT = attach_ext_priors(D["df"])
    splits_ext = {s: df_ext[df_ext["split"] == s] for s in ["train", "val", "test"]}
    ens_ext_val, ens_ext_test = [], []
    for seed in L.SEEDS_MULTI:
        name = f"cls_big_ext_seed{seed}"
        vp, tp = cache(name, (lambda sd, nm: lambda: (lambda r: (r["val_pred"], r["test_pred"]))(
            L.run_experiment(nm, cls_factory, "CE", U_EXT, B_EXT, L.C_FULL,
                             splits_ext, y_val, ncv, seed=sd)))(seed, name))
        ens_ext_val.append(vp)
        ens_ext_test.append(tp)
    ens_ext_v = np.mean(ens_ext_val, axis=0)
    ens_ext_t = np.mean(ens_ext_test, axis=0)
    results["models"]["ensemble + обогащённые priors (cls big x3)"] = {
        "val_rmse": L.rmse(y_val, ens_ext_v), "test_rmse": L.rmse(y_te, ens_ext_t)}
    print(f"[priors+] обогащённые priors ансамбль val {L.rmse(y_val, ens_ext_v):.4f} test {L.rmse(y_te, ens_ext_t):.4f}")

    # ── стадия 2: GBDT ──
    gbdt_best, gbdt_table = tune_gbdt(D["df"], splits, y_val, y_te)
    np.save(PRED_DIR / "gbdt_val.npy", gbdt_best["val_pred"])
    np.save(PRED_DIR / "gbdt_test.npy", gbdt_best["test_pred"])
    results["models"]["GBDT (best)"] = {
        "val_rmse": gbdt_best["val_rmse"], "test_rmse": gbdt_best["test_rmse"]}
    print(f"[gbdt] лучший val {gbdt_best['val_rmse']:.4f} test {gbdt_best['test_rmse']:.4f}")

    # ── стадия 3: бленды (веса по val, test один раз) ──
    base = {
        "ens": (ens_val, ens_test),
        "reg": (reg_val, reg_test),
        "gbdt": (gbdt_best["val_pred"], gbdt_best["test_pred"]),
        "ens_ext": (ens_ext_v, ens_ext_t),
    }

    def add_blend(name, keys):
        pv = [base[k][0] for k in keys]
        pt = [base[k][1] for k in keys]
        w = blend_weights(pv, y_val)
        bv = np.stack(pv, 1) @ w
        bt = np.stack(pt, 1) @ w
        results["blends"][name] = {
            "members": keys, "weights": [round(float(x), 3) for x in w],
            "val_rmse": L.rmse(y_val, bv), "test_rmse": L.rmse(y_te, bt),
            "segments": L.segment_report(bt, y_te, warm_user, warm_biz)}
        print(f"[blend] {name} w={np.round(w,3)} val {L.rmse(y_val,bv):.4f} test {L.rmse(y_te,bt):.4f}")
        return bv, bt

    add_blend("ens+gbdt", ["ens", "gbdt"])
    add_blend("ens+reg", ["ens", "reg"])
    add_blend("ens+reg+gbdt", ["ens", "reg", "gbdt"])
    add_blend("ens_ext+gbdt", ["ens_ext", "gbdt"])
    add_blend("ens_ext+reg+gbdt", ["ens_ext", "reg", "gbdt"])
    add_blend("все (ens+ens_ext+reg+gbdt)", ["ens", "ens_ext", "reg", "gbdt"])

    # ── сводка: всё, отсортировано по val ──
    rows = []
    for k, v in {**results["models"], **results["blends"]}.items():
        rows.append((v["val_rmse"], v["test_rmse"], k))
    rows.sort()
    print("\n=== сводка (сорт по val) ===")
    print(f"{'val':>8} {'test':>8}  модель")
    print(f"{'-':>8} {bias_rmse:>8.4f}  bias-бейзлайн (test only)")
    for vr, tr, k in rows:
        print(f"{vr:>8.4f} {tr:>8.4f}  {k}")

    results["baseline_test_rmse_committed"] = 1.04468003719117
    json.dump(results, open(REPORTS / "task1_experiments.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nсохранено в {REPORTS / 'task1_experiments.json'}")


if __name__ == "__main__":
    main()

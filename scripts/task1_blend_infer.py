"""инференс финального бленда задачи 1 из logs/task1/final/blend.pt.

восстанавливает все члены бленда по сохранённым весам и конфигу и выдаёт предсказание без
переобучения и без чтения ноутбука. так финальная модель воспроизводима как единый артефакт.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
import task1_lib as L
from task1_priors_ext import attach_ext_priors
from task1_gbdt import build_matrix
from task1_coral import CoralHead, decode_coral

FINAL_DIR = root / "logs" / "task1" / "final"


@torch.no_grad()
def _predict_nn(model, d, ucols, bcols, ccols, ncv, decode):
    model.eval()
    t = L.make_tensors(d, ucols, bcols, ccols, ncv)
    out = []
    for b in L.batches(t, 8192, shuffle=False):
        o = decode(model({k: v for k, v in b.items() if k != "y"}))
        out.append(o.float().cpu().numpy())
    return np.concatenate(out)


def predict_member(key, spec, art, D, df_ext, splits_ext, which):
    """предсказание одного члена бленда на наборе which in {val,test}."""
    Inter = D["models"]["InteractionMLP"]
    ncv = art["n_cat_vocab"]
    big = art["big"]
    if spec["kind"] == "gbdt":
        d, feats = build_matrix(D["df"])
        dd = d[d["split"] == which]
        return joblib.load(FINAL_DIR / spec["path"]).predict(dd[spec["feat_cols"]].to_numpy(np.float32))
    use_ext = key == "ens_ext"
    splits = splits_ext if use_ext else D["splits"]
    d = splits[which]
    preds = []
    for state in spec["states"]:
        if spec["kind"] == "coral":
            model = CoralHead(Inter(len(spec["ucols"]), len(spec["bcols"]), len(spec["ccols"]), **big, out=1)).to(L.device)
            model.load_state_dict(state)
            preds.append(_predict_nn(model, d, spec["ucols"], spec["bcols"], spec["ccols"], ncv, decode_coral))
        else:  # cls_big
            model = Inter(len(spec["ucols"]), len(spec["bcols"]), len(spec["ccols"]), **big, out=5).to(L.device)
            model.load_state_dict(state)
            dec = lambda o: (torch.softmax(o, 1) * L.STAR_VALUES).sum(1)
            preds.append(_predict_nn(model, d, spec["ucols"], spec["bcols"], spec["ccols"], ncv, dec))
    return np.mean(preds, 0)


def predict_blend(which="test"):
    """возвращает (предсказание бленда, таргет) на наборе which."""
    art = torch.load(FINAL_DIR / "blend.pt", map_location=L.device, weights_only=False)
    D = L.prepare()
    df_ext, _, _ = attach_ext_priors(D["df"])
    splits_ext = {s: df_ext[df_ext["split"] == s] for s in ["train", "val", "test"]}
    y = D["y_te"] if which == "test" else D["y_val"]
    # порядок как в task1_finalize: взвешиваем сырые предсказания членов и клипаем сумму один раз
    # (gbdt иногда выходит за [1,5]), так инференс точно совпадает с метрикой финализатора
    blend = np.zeros(len(y))
    for key, w in art["weights"].items():
        p = predict_member(key, art["members"][key], art, D, df_ext, splits_ext, which)
        blend += w * p
    return np.clip(blend, 1, 5), y, D


def interpretable_metrics(pred, y):
    """понятные метрики качества финальной модели: попадание в звезду, ошибка ≤1★, бинарно."""
    p = np.clip(pred, 1, 5)
    err = np.abs(p - y)
    like_t, like_p = y >= 4, p >= 4
    return {
        "exact_hit": float((np.rint(p) == y).mean()),
        "within_1_star": float((err <= 1.0).mean()),
        "binary_acc_thr4": float((like_t == like_p).mean()),
        "binary_naive": float(like_t.mean()),
    }


if __name__ == "__main__":
    import json
    from _constants import REPORTS
    pred, y, D = predict_blend("test")
    print(f"бленд из артефакта: test RMSE {L.rmse(y, pred):.4f}")
    print("сегменты:", {k: round(v, 4) for k, v in
                        L.segment_report(pred, y, D["warm_user"], D["warm_biz"]).items()})
    im = interpretable_metrics(pred, y)
    print("интерпретируемые:", {k: round(v, 4) for k, v in im.items()})
    mp = REPORTS / "task1_model_metrics.json"
    m = json.load(open(mp, encoding="utf-8"))
    m.setdefault("blend", {})["interpretable"] = im
    json.dump(m, open(mp, "w"), ensure_ascii=False, indent=2)
    print("интерпретируемые метрики записаны в blend.interpretable")

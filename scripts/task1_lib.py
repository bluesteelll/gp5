"""общая библиотека задачи 1, воспроизводит пайплайн 04_task1_rating_mlp.ipynb как импортируемый код.

собирает prior-признаки строго по прошлому (cells 13-19), единый фрейм (cell 22), тензоры и
сегменты, модели FlatMLP/InteractionMLP/DCNv2 и цикл обучения run_experiment (cells 25-43).
эксперименты импортируют это и переиспользуют, не дублируя пайплайн.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from _constants import PROCESSED, ARTIFACTS, REPORTS  # noqa: F401

# ── константы, как в ноутбуке 04 ────────────────────────────────────────────
SEED = 42
SEEDS_MULTI = [42, 7, 2024]
MAXLEN = 10
D_CAT = 16
D_BRANCH = 32
D_BIZ_EMB = 32
N_CITY = 4
BATCH = 1024
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

USER_NUM = ["num_user_review_count", "num_average_stars", "num_user_useful", "num_user_funny",
            "num_user_cool", "num_fans", "num_n_friends", "num_n_elite_years", "num_account_age_days"]
BIZ_NUM = ["num_biz_avg_stars", "num_biz_review_count", "num_is_open", "num_price_range", "num_price_known"]
CROSS = ["num_user_avg_minus_biz"]
TIME_COLS = ["month_sin", "month_cos", "dow_sin", "dow_cos", "hour_sin", "hour_cos", "is_weekend"]
USER_PRIOR = ["u_prior_mean", "u_cnt", "u_days_prev", "u_has_hist"]
BIZ_PRIOR = ["b_prior_mean", "b_cnt", "b_days_prev", "b_has_hist"]
CROSS_PRIOR = ["u_geo_dist", "price_aff"]
CATID_COLS = [f"catid_{i}" for i in range(MAXLEN)]

U_FULL = USER_NUM + USER_PRIOR
B_FULL = BIZ_NUM + BIZ_PRIOR
C_FULL = CROSS + TIME_COLS

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
STAR_VALUES = torch.arange(1.0, 6.0, device=device)


# ── метрики ─────────────────────────────────────────────────────────────────
def rmse(y, p):
    return float(np.sqrt(np.mean((y - np.clip(p, 1, 5)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - np.clip(p, 1, 5))))


# ── prior-признаки строго по прошлому (cells 13-19) ──────────────────────────
def build_prior_feats(force=False):
    """expanding-статистики юзера и заведения без текущего отзыва, скейлер с fit на train.
    сохраняет task1_prior_feats.parquet и task1_prior_scaler.joblib, как в ноутбуке."""
    out_path = PROCESSED / "task1_prior_feats.parquet"
    if out_path.exists() and not force:
        return
    import joblib
    from sklearn.preprocessing import StandardScaler
    cols = ["review_id", "user_id", "business_id", "stars", "date", "latitude", "longitude", "price_range", "split"]
    ds = pd.read_parquet(PROCESSED / "task1_dataset.parquet", columns=cols)
    ds["date"] = pd.to_datetime(ds["date"])
    ds = ds.sort_values(["date", "review_id"], kind="mergesort").reset_index(drop=True)

    gu = ds.groupby("user_id", sort=False)
    ds["u_cnt"] = gu.cumcount()
    ds["u_prior_mean"] = np.where(ds.u_cnt > 0, (gu["stars"].cumsum() - ds["stars"]) / ds.u_cnt, np.nan)
    ds["u_days_prev"] = gu["date"].diff().dt.total_seconds() / 86400
    for c in ["latitude", "longitude"]:
        ds[f"u_pc_{c}"] = np.where(ds.u_cnt > 0, (gu[c].cumsum() - ds[c]) / ds.u_cnt, np.nan)
    lat1, lon1 = np.radians(ds.latitude), np.radians(ds.longitude)
    lat2, lon2 = np.radians(ds.u_pc_latitude), np.radians(ds.u_pc_longitude)
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    ds["u_geo_dist"] = 2 * 6371 * np.arcsin(np.sqrt(h))
    ds["u_prior_price"] = np.where(ds.u_cnt > 0, (gu["price_range"].cumsum() - ds["price_range"]) / ds.u_cnt, np.nan)
    ds["price_aff"] = ds["u_prior_price"] - ds["price_range"]

    gb = ds.groupby("business_id", sort=False)
    ds["b_cnt"] = gb.cumcount()
    ds["b_prior_mean"] = np.where(ds.b_cnt > 0, (gb["stars"].cumsum() - ds["stars"]) / ds.b_cnt, np.nan)
    ds["b_days_prev"] = gb["date"].diff().dt.total_seconds() / 86400

    mu_train = ds.loc[ds.split == "train", "stars"].mean()
    ds["u_has_hist"] = (ds.u_cnt > 0).astype(float)
    ds["b_has_hist"] = (ds.b_cnt > 0).astype(float)
    fill = {"u_prior_mean": mu_train, "b_prior_mean": mu_train, "u_days_prev": 3650.0,
            "b_days_prev": 3650.0, "u_geo_dist": 0.0, "price_aff": 0.0}
    ds = ds.fillna(fill)
    for c in ["u_cnt", "b_cnt", "u_days_prev", "b_days_prev", "u_geo_dist"]:
        ds[c] = np.log1p(ds[c])
    PRIOR_ALL = USER_PRIOR + BIZ_PRIOR + CROSS_PRIOR
    prior_scaler = StandardScaler().fit(ds.loc[ds.split == "train", PRIOR_ALL])
    ds[PRIOR_ALL] = prior_scaler.transform(ds[PRIOR_ALL])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ds[["review_id"] + PRIOR_ALL].to_parquet(out_path, index=False)
    joblib.dump(prior_scaler, ARTIFACTS / "task1_prior_scaler.joblib")


# ── единый фрейм (cell 22) ───────────────────────────────────────────────────
def load_frame():
    """склеивает embed_inputs + features + ключи + prior по review_id в один DataFrame."""
    build_prior_feats()
    PRIOR_ALL = USER_PRIOR + BIZ_PRIOR + CROSS_PRIOR
    emb = pd.read_parquet(PROCESSED / "task1_embed_inputs.parquet")
    feats = pd.read_parquet(PROCESSED / "task1_features.parquet", columns=["review_id"] + TIME_COLS)
    keycols = ["review_id", "user_id", "business_id", "biz_avg_stars", "average_stars"]
    keys = pd.read_parquet(PROCESSED / "task1_dataset.parquet", columns=keycols)
    prior = pd.read_parquet(PROCESSED / "task1_prior_feats.parquet")
    df = emb.merge(feats, on="review_id", validate="one_to_one")
    df = df.merge(keys, on="review_id", validate="one_to_one")
    df = df.merge(prior, on="review_id", validate="one_to_one")
    cat_vocab = json.load(open(ARTIFACTS / "task1_vocab_categories.json"))
    biz_vocab = json.load(open(ARTIFACTS / "task1_vocab_business.json"))
    n_cat_vocab = len(cat_vocab) + 1
    n_biz_vocab = len(biz_vocab) + 1
    assert df["city_idx"].max() < N_CITY and df["biz_idx"].max() < n_biz_vocab
    assert not df[USER_NUM + BIZ_NUM + CROSS + TIME_COLS + PRIOR_ALL].isna().any().any()
    return df, n_cat_vocab, n_biz_vocab


# ── тензоры и сегменты (cells 25-27) ─────────────────────────────────────────
def make_tensors(d, ucols, bcols, ccols, n_cat_vocab):
    t = lambda a, dt: torch.tensor(np.asarray(a), dtype=dt, device=device)
    out = {}
    out["xu"] = t(d[ucols].to_numpy(np.float32), torch.float32)
    out["xb"] = t(d[bcols].to_numpy(np.float32), torch.float32)
    out["ctx"] = t(d[ccols].to_numpy(np.float32), torch.float32)
    out["cats"] = t(d[CATID_COLS].to_numpy(np.int64), torch.long)
    out["city"] = torch.nn.functional.one_hot(t(d["city_idx"].to_numpy(np.int64), torch.long), N_CITY).float()
    out["biz"] = t(d["biz_idx"].to_numpy(np.int64), torch.long)
    out["y"] = t(d["stars"].to_numpy(np.float32), torch.float32)
    return out


def make_splits(df):
    splits = {s: df[df["split"] == s] for s in ["train", "val", "test"]}
    y_val = splits["val"]["stars"].to_numpy()
    y_te = splits["test"]["stars"].to_numpy()
    train_users = set(splits["train"]["user_id"])
    train_biz = set(splits["train"]["business_id"])
    warm_user = splits["test"]["user_id"].isin(train_users).to_numpy()
    warm_biz = splits["test"]["business_id"].isin(train_biz).to_numpy()
    return splits, y_val, y_te, warm_user, warm_biz


def segment_report(pred, y_te, warm_user, warm_biz):
    p = np.clip(pred, 1, 5)
    seg = {"все": np.ones(len(y_te), bool), "warm юзер": warm_user, "cold юзер": ~warm_user,
           "warm заведение": warm_biz, "cold заведение": ~warm_biz}
    return {k: rmse(y_te[m], p[m]) for k, m in seg.items()}


# ── модели (cells 34-37) ─────────────────────────────────────────────────────
def mlp(dims, p_drop=DROPOUT):
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), nn.ReLU(), nn.Dropout(p_drop)]
    return nn.Sequential(*layers)


def make_models(n_cat_vocab, n_biz_vocab):
    """фабрики моделей замыкаются на размеры словарей текущего среза."""
    class FlatMLP(nn.Module):
        def __init__(self, du, db, dc):
            super().__init__()
            self.cat_emb = nn.EmbeddingBag(n_cat_vocab, D_CAT, mode="mean", padding_idx=0)
            self.body = nn.Sequential(mlp([du + db + dc + D_CAT + N_CITY, 256, 128, 64]), nn.Linear(64, 1))

        def forward(self, b):
            x = torch.cat([b["xu"], b["xb"], b["ctx"], self.cat_emb(b["cats"]), b["city"]], dim=1)
            return self.body(x).squeeze(1)

    class InteractionMLP(nn.Module):
        def __init__(self, du, db, dc, use_biz_emb=False, d_cat=D_CAT, d_branch=D_BRANCH, hidden=64, top=(128, 64), out=1):
            super().__init__()
            self.cat_emb = nn.EmbeddingBag(n_cat_vocab, d_cat, mode="mean", padding_idx=0)
            self.use_biz_emb = use_biz_emb
            d_biz_in = db + d_cat + N_CITY
            if use_biz_emb:
                self.biz_emb = nn.Embedding(n_biz_vocab, D_BIZ_EMB, padding_idx=0)
                d_biz_in += D_BIZ_EMB
            self.user_mlp = mlp([du, hidden, d_branch])
            self.biz_mlp = mlp([d_biz_in, hidden, d_branch])
            self.top = nn.Sequential(mlp([3 * d_branch + 1 + dc, *top]), nn.Linear(top[-1], out))

        def forward(self, b):
            u = self.user_mlp(b["xu"])
            biz_in = [b["xb"], self.cat_emb(b["cats"]), b["city"]]
            if self.use_biz_emb:
                biz_in.append(self.biz_emb(b["biz"]))
            v = self.biz_mlp(torch.cat(biz_in, dim=1))
            had = u * v
            dot = had.sum(dim=1, keepdim=True)
            o = self.top(torch.cat([u, v, had, dot, b["ctx"]], dim=1))
            return o.squeeze(1) if o.shape[1] == 1 else o

    class CrossNet(nn.Module):
        def __init__(self, d, n_layers=3):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])

        def forward(self, x0):
            x = x0
            for lin in self.layers:
                x = x0 * lin(x) + x
            return x

    class DCNv2(nn.Module):
        def __init__(self, du, db, dc, d_cat=D_CAT, n_cross=3):
            super().__init__()
            self.cat_emb = nn.EmbeddingBag(n_cat_vocab, d_cat, mode="mean", padding_idx=0)
            d_in = du + db + dc + d_cat + N_CITY
            self.cross = CrossNet(d_in, n_cross)
            self.deep = mlp([d_in, 256, 128])
            self.head = nn.Linear(d_in + 128, 1)

        def forward(self, b):
            x = torch.cat([b["xu"], b["xb"], b["ctx"], self.cat_emb(b["cats"]), b["city"]], dim=1)
            return self.head(torch.cat([self.cross(x), self.deep(x)], dim=1)).squeeze(1)

    return {"FlatMLP": FlatMLP, "InteractionMLP": InteractionMLP, "DCNv2": DCNv2}


# ── цикл обучения (cell 43) ──────────────────────────────────────────────────
LOSS_FNS = {"MSE": nn.MSELoss(), "Huber": nn.SmoothL1Loss(beta=1.0), "CE": nn.CrossEntropyLoss()}


def batches(tensors, batch_size, shuffle):
    n = len(tensors["y"])
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield {k: v[j] for k, v in tensors.items()}


def run_experiment(name, factory, loss_kind, ucols, bcols, ccols, splits, y_val, n_cat_vocab,
                   seed=SEED, verbose=False, decode_fn=None):
    """один цикл обучения для всех моделей: AdamW, early stopping по val RMSE, лучшие веса.
    decode_fn(out)->pred переопределяет декодирование (для CORAL и пр.); по умолчанию как в ноутбуке."""
    tens = {s: make_tensors(splits[s], ucols, bcols, ccols, n_cat_vocab) for s in ["train", "val", "test"]}
    is_cls = loss_kind == "CE"
    loss_fn = LOSS_FNS[loss_kind]
    torch.manual_seed(seed)
    model = factory(len(ucols), len(bcols), len(ccols)).to(device)

    def decode(out):
        if decode_fn is not None:
            return decode_fn(out)
        return (torch.softmax(out, dim=1) * STAR_VALUES).sum(dim=1) if is_cls else out

    @torch.no_grad()
    def predict_t(tensors):
        model.eval()
        preds = []
        for b in batches(tensors, 8192, shuffle=False):
            o = decode(model({k: v for k, v in b.items() if k != "y"}))
            preds.append(o.float().cpu().numpy())
        return np.concatenate(preds)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best = {"rmse": np.inf, "state": None, "epoch": -1}
    history = []
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        tot, nb = 0.0, 0
        for b in batches(tens["train"], BATCH, shuffle=True):
            if len(b["y"]) < 2:
                continue
            opt.zero_grad()
            out = model({k: v for k, v in b.items() if k != "y"})
            target = (b["y"] - 1).long() if is_cls else b["y"]
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        val_rmse = rmse(y_val, predict_t(tens["val"]))
        assert np.isfinite(val_rmse)
        sched.step(val_rmse)
        history.append({"epoch": epoch, "train_loss": tot / nb, "val_rmse": val_rmse})
        if val_rmse < best["rmse"]:
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best = {"rmse": val_rmse, "state": state, "epoch": epoch}
        if verbose:
            print(f"{name} эпоха {epoch:2d} train {tot / nb:.4f} val {val_rmse:.4f}")
        if epoch - best["epoch"] >= PATIENCE:
            break
    model.load_state_dict(best["state"])
    res = {"name": name, "val_rmse": best["rmse"], "history": pd.DataFrame(history), "state": best["state"]}
    res["val_pred"] = predict_t(tens["val"])
    res["test_pred"] = predict_t(tens["test"])
    print(f"{name} val {best['rmse']:.4f} {time.time() - t0:.0f}c")
    return res


def prepare():
    """всё, что нужно экспериментам: фрейм, сплиты, словари, фабрики моделей."""
    df, n_cat_vocab, n_biz_vocab = load_frame()
    splits, y_val, y_te, warm_user, warm_biz = make_splits(df)
    models = make_models(n_cat_vocab, n_biz_vocab)
    return dict(df=df, n_cat_vocab=n_cat_vocab, n_biz_vocab=n_biz_vocab, splits=splits,
                y_val=y_val, y_te=y_te, warm_user=warm_user, warm_biz=warm_biz, models=models)

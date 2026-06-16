"""собирает данные для задачи 1 из среза, воспроизводит data-ячейки 03_task1_dataset.ipynb
(без EDA и графиков), чтобы пайплайн запускался headless и был воспроизводим.

на вход data/processed/{business,reviews,users}.parquet (их делает preprocess.py),
на выход task1_dataset.parquet, task1_embed_inputs.parquet, task1_features.parquet,
словари task1_vocab_*.json и скейлеры в artifacts/. логика один в один с ноутбуком 03.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from _constants import PROCESSED, ARTIFACTS, REVIEWS_PARQUET, BUSINESS_PARQUET, USERS_PARQUET

TOPK = 20
MIN_CAT_FREQ = 20
MAXLEN = 10
OUT = PROCESSED / "task1_dataset.parquet"


def build_dataset():
    """ячейки 3-18: джойн reviews+business+users, фичи, временной сплит -> task1_dataset.parquet"""
    cols = ["review_id", "user_id", "business_id", "stars", "date"]
    reviews = pd.read_parquet(REVIEWS_PARQUET, columns=cols)
    business = pd.read_parquet(BUSINESS_PARQUET)
    business = business.drop(columns=["name", "postal_code", "city", "state"])
    business = business.rename(columns={"stars": "biz_avg_stars", "review_count": "biz_review_count"})
    users = pd.read_parquet(USERS_PARQUET)
    users = users.rename(columns={"review_count": "user_review_count", "useful": "user_useful",
                                  "funny": "user_funny", "cool": "user_cool"})

    df = reviews.merge(business, on="business_id", how="left")
    df = df.merge(users, on="user_id", how="left")
    assert len(df) == len(reviews)
    before = len(df)
    df = df.dropna(subset=["average_stars"]).reset_index(drop=True)
    print("удалено строк без профиля юзера", before - len(df))

    df["date"] = pd.to_datetime(df["date"])
    df["yelping_since"] = pd.to_datetime(df["yelping_since"])
    df["account_age_days"] = (df["date"] - df["yelping_since"]).dt.days.clip(lower=0)
    df["price_known"] = df["price_range"].notna().astype(int)
    df["price_range"] = df["price_range"].fillna(df["price_range"].median())
    df["user_avg_minus_biz"] = df["average_stars"] - df["biz_avg_stars"]
    df = df.drop(columns=["yelping_since"])

    cat_sets = df["categories"].fillna("").apply(lambda s: {x.strip() for x in s.split(",") if x.strip()})
    top_cats = df["categories"].dropna().str.split(", ").explode().value_counts().head(TOPK).index.tolist()

    def slug(name):
        return "cat_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    for cat in top_cats:
        df[slug(cat)] = cat_sets.apply(lambda st: int(cat in st))
    city_oh = pd.get_dummies(df["city_state"], prefix="city").astype(int)
    df = pd.concat([df, city_oh], axis=1)
    df = df.drop(columns=["categories", "city_state"])

    df = df.sort_values("date").reset_index(drop=True)
    q70, q85 = df["date"].quantile([0.70, 0.85])
    df["split"] = np.where(df["date"] <= q70, "train", np.where(df["date"] <= q85, "val", "test"))
    print(df["split"].value_counts(normalize=True).round(3).to_dict(), q70.date(), q85.date())

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print("task1_dataset.parquet", df.shape)
    return df


def build_embed_inputs():
    """ячейки 37-48: индексы категорий/города/заведения + числовые log1p+scaler -> task1_embed_inputs.parquet"""
    cols = ["review_id", "business_id", "split", "stars",
            "biz_avg_stars", "biz_review_count", "is_open", "price_range", "price_known",
            "user_review_count", "average_stars", "user_useful", "user_funny", "user_cool",
            "fans", "n_friends", "n_elite_years", "account_age_days", "user_avg_minus_biz"]
    emb = pd.read_parquet(OUT)[cols].copy()
    biz = pd.read_parquet(BUSINESS_PARQUET, columns=["business_id", "categories", "city_state"])
    biz["cat_list"] = biz["categories"].fillna("").apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    emb = emb.merge(biz[["business_id", "cat_list", "city_state"]], on="business_id", how="left")
    train_mask = emb["split"] == "train"

    train = emb[train_mask]
    train_biz_ids = set(train["business_id"])
    train_biz = biz[biz["business_id"].isin(train_biz_ids)]
    freq = Counter(c for lst in train_biz["cat_list"] for c in lst)
    kept = sorted([c for c, v in freq.items() if v >= MIN_CAT_FREQ])
    cat_vocab = {c: i + 1 for i, c in enumerate(kept)}
    city_vocab = {c: i + 1 for i, c in enumerate(sorted(train["city_state"].dropna().unique()))}
    biz_vocab = {b: i + 1 for i, b in enumerate(sorted(train["business_id"].unique()))}
    print("словари: категории", len(cat_vocab), "города", len(city_vocab), "заведения", len(biz_vocab))

    def encode_cats(lst):
        ids = [cat_vocab[c] for c in (lst or []) if c in cat_vocab][:MAXLEN]
        return ids + [0] * (MAXLEN - len(ids))

    cat_mat = np.array(emb["cat_list"].apply(encode_cats).to_list(), dtype="int64")
    emb["city_idx"] = emb["city_state"].map(city_vocab).fillna(0).astype("int64")
    emb["biz_idx"] = emb["business_id"].map(biz_vocab).fillna(0).astype("int64")

    NUM = ["biz_avg_stars", "biz_review_count", "is_open", "price_range", "price_known",
           "user_review_count", "average_stars", "user_useful", "user_funny", "user_cool",
           "fans", "n_friends", "n_elite_years", "account_age_days", "user_avg_minus_biz"]
    LOG = ["biz_review_count", "user_review_count", "user_useful", "user_funny",
           "user_cool", "fans", "n_friends"]
    Xnum = emb[NUM].copy()
    Xnum[LOG] = np.log1p(Xnum[LOG])
    scaler = StandardScaler().fit(Xnum[train_mask.to_numpy()])
    Xnum_scaled = pd.DataFrame(scaler.transform(Xnum), columns=["num_" + c for c in NUM])

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    json.dump(cat_vocab, open(ARTIFACTS / "task1_vocab_categories.json", "w"), ensure_ascii=False)
    json.dump(city_vocab, open(ARTIFACTS / "task1_vocab_city.json", "w"), ensure_ascii=False)
    json.dump(biz_vocab, open(ARTIFACTS / "task1_vocab_business.json", "w"), ensure_ascii=False)
    joblib.dump(scaler, ARTIFACTS / "task1_scaler.joblib")
    out = pd.concat([
        emb[["review_id", "split", "stars", "city_idx", "biz_idx"]].reset_index(drop=True),
        pd.DataFrame(cat_mat, columns=[f"catid_{i}" for i in range(MAXLEN)]),
        Xnum_scaled.reset_index(drop=True),
    ], axis=1)
    out.to_parquet(PROCESSED / "task1_embed_inputs.parquet", index=False)
    print("task1_embed_inputs.parquet", out.shape)


def build_features():
    """ячейки 58-69: город one-hot, циклическое время, винзоризация+log1p+scaler -> task1_features.parquet"""
    ff = pd.read_parquet(OUT)
    ff["date"] = pd.to_datetime(ff["date"])
    tr = ff["split"] == "train"

    d = ff["date"].dt
    ff["year"] = d.year
    ff["is_weekend"] = (d.dayofweek >= 5).astype(int)

    def cyclic(values, period):
        ang = 2 * np.pi * values / period
        return np.sin(ang), np.cos(ang)

    ff["month_sin"], ff["month_cos"] = cyclic(d.month, 12)
    ff["dow_sin"], ff["dow_cos"] = cyclic(d.dayofweek, 7)
    ff["hour_sin"], ff["hour_cos"] = cyclic(d.hour, 24)

    HEAVY = ["biz_review_count", "user_review_count", "user_useful", "user_funny",
             "user_cool", "fans", "n_friends"]
    winsor = {}
    for c in HEAVY:
        lo, hi = ff.loc[tr, c].quantile([0.01, 0.99])
        winsor[c] = {"lo": float(lo), "hi": float(hi)}
        ff[c] = ff[c].clip(lo, hi)
    LOG = HEAVY
    NUM_SCALE = ["biz_avg_stars", "biz_review_count", "price_range", "user_review_count",
                 "average_stars", "user_useful", "user_funny", "user_cool", "fans",
                 "n_friends", "n_elite_years", "account_age_days", "user_avg_minus_biz", "year"]
    ff[LOG] = np.log1p(ff[LOG])
    scaler = StandardScaler().fit(ff.loc[tr, NUM_SCALE])
    ff[NUM_SCALE] = scaler.transform(ff[NUM_SCALE])

    final = ff.drop(columns=["date"])
    final.to_parquet(PROCESSED / "task1_features.parquet", index=False)
    json.dump(winsor, open(ARTIFACTS / "task1_winsor_bounds.json", "w"), indent=0)
    joblib.dump(scaler, ARTIFACTS / "task1_scaler_final.joblib")
    print("task1_features.parquet", final.shape)


if __name__ == "__main__":
    build_dataset()
    build_embed_inputs()
    build_features()
    print("данные задачи 1 собраны")

"""обогащённые time-aware prior-признаки для задачи 1.

prior-признаки дали основной прирост в ноутбуке 04 (~0.047 RMSE). расширяем тот же рычаг:
кроме средней по прошлому добавляем разброс оценок и доли крайних оценок (5 и 1) у юзера и
заведения. это напрямую кодирует бимодальность, ради которой в ноутбуке вводили классиф. голову,
но как признак. всё строго по прошлому относительно даты отзыва (cumsum минус текущее значение),
утечки нет по построению, ровно как у существующих prior-признаков.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from _constants import PROCESSED

EXT_PATH = PROCESSED / "task1_prior_ext.parquet"
U_PRIOR_EXT = ["u_prior_std", "u_prior_p5", "u_prior_p1"]
B_PRIOR_EXT = ["b_prior_std", "b_prior_p5", "b_prior_p1"]


def build_ext_priors(force=False):
    if EXT_PATH.exists() and not force:
        return
    cols = ["review_id", "user_id", "business_id", "stars", "date", "split"]
    ds = pd.read_parquet(PROCESSED / "task1_dataset.parquet", columns=cols)
    ds["date"] = pd.to_datetime(ds["date"])
    ds = ds.sort_values(["date", "review_id"], kind="mergesort").reset_index(drop=True)
    ds["is5"] = (ds["stars"] == 5).astype(float)
    ds["is1"] = (ds["stars"] == 1).astype(float)
    ds["sq"] = ds["stars"].astype(float) ** 2

    def add(prefix, key):
        g = ds.groupby(key, sort=False)
        cnt = g.cumcount()
        mean = np.where(cnt > 0, (g["stars"].cumsum() - ds["stars"]) / cnt, np.nan)
        sqmean = np.where(cnt > 0, (g["sq"].cumsum() - ds["sq"]) / cnt, np.nan)
        var = np.clip(sqmean - mean ** 2, 0, None)
        ds[f"{prefix}_prior_std"] = np.sqrt(var)
        ds[f"{prefix}_prior_p5"] = np.where(cnt > 0, (g["is5"].cumsum() - ds["is5"]) / cnt, np.nan)
        ds[f"{prefix}_prior_p1"] = np.where(cnt > 0, (g["is1"].cumsum() - ds["is1"]) / cnt, np.nan)

    add("u", "user_id")
    add("b", "business_id")

    ALL = U_PRIOR_EXT + B_PRIOR_EXT
    # пропуски у новых юзеров/заведений -> 0 (нет истории), has_hist уже есть в базовых prior
    ds[ALL] = ds[ALL].fillna(0.0)
    scaler = StandardScaler().fit(ds.loc[ds.split == "train", ALL])
    ds[ALL] = scaler.transform(ds[ALL])
    ds[["review_id"] + ALL].to_parquet(EXT_PATH, index=False)
    print("task1_prior_ext.parquet", ds[["review_id"] + ALL].shape)


def attach_ext_priors(df):
    """мерджит обогащённые priors в df и возвращает (df_ext, U_FULL_ext, B_FULL_ext)."""
    import task1_lib as L
    build_ext_priors()
    ext = pd.read_parquet(EXT_PATH)
    df2 = df.merge(ext, on="review_id", validate="one_to_one")
    return df2, L.U_FULL + U_PRIOR_EXT, L.B_FULL + B_PRIOR_EXT


if __name__ == "__main__":
    build_ext_priors()

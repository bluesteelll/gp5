import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _constants import REVIEWS_PARQUET, MISMATCH, MISMATCH_PARQUET

STARS = [1, 2, 3, 4, 5]
ALLOWED = {s: [t for t in STARS if abs(t - s) > 1] for s in STARS}
CELLS = [(s, t) for s in STARS for t in ALLOWED[s]]


def corrupt_random(orig, seed):
    rng = np.random.default_rng(seed)
    new = orig.copy()
    for s in STARS:
        m = orig == s
        if m.any():
            new[m] = rng.choice(ALLOWED[s], size=int(m.sum()))
    return new


def solve_transport(n_src, n_dst, q):
    from scipy.optimize import linprog
    idx = {c: i for i, c in enumerate(CELLS)}
    nv = len(CELLS) + 1
    M = nv - 1
    cobj = np.zeros(nv)
    cobj[M] = -1.0
    A_eq, b_eq = [], []
    for t in STARS:
        row = np.zeros(nv)
        for s in STARS:
            if (s, t) in idx:
                row[idx[(s, t)]] = 1
        row[M] = -q[t]
        A_eq.append(row)
        b_eq.append(0.0)
    A_ub, b_ub = [], []
    for s in STARS:
        row = np.zeros(nv)
        for t in ALLOWED[s]:
            row[idx[(s, t)]] = 1
        A_ub.append(row)
        b_ub.append(n_src[s])
    for t in STARS:
        row = np.zeros(nv)
        row[M] = q[t]
        A_ub.append(row)
        b_ub.append(n_dst[t])
    res = linprog(cobj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=[(0, None)] * nv, method="highs")

    x = np.floor(res.x[:len(CELLS)] + 1e-6).astype(int)
    return {c: int(x[idx[c]]) for c in CELLS}


def build_balanced(clean, corrupt, seed):
    n_src = corrupt["stars"].value_counts().reindex(STARS, fill_value=0).to_dict()
    n_dst = clean["stars"].value_counts().reindex(STARS, fill_value=0).to_dict()
    q = clean["stars"].value_counts(normalize=True).reindex(STARS, fill_value=0).to_dict()
    X = solve_transport(n_src, n_dst, q)
    c_t = {t: sum(X[(s, t)] for s in STARS if (s, t) in X) for t in STARS}

    corrupt = corrupt.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    new = np.full(len(corrupt), -1, dtype=int)
    sa = corrupt["stars"].to_numpy()
    for s in STARS:
        pos = np.where(sa == s)[0]
        cur = 0
        for t in ALLOWED[s]:
            k = X[(s, t)]
            new[pos[cur:cur + k]] = t
            cur += k
    corrupt = corrupt.assign(orig_stars=corrupt["stars"], stars=new, label=1)
    corrupt = corrupt[corrupt["stars"] > 0]

    parts = []
    for t in STARS:
        g = clean[clean["stars"] == t]
        parts.append(g.sample(n=c_t[t], random_state=seed))
    clean = pd.concat(parts).assign(orig_stars=lambda d: d["stars"], label=0)
    return clean, corrupt


ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["balanced", "random"], default="balanced")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()


if not REVIEWS_PARQUET.exists():
    print("нет reviews.parquet - сначала preprocess.py")
    sys.exit(1)

df = pd.read_parquet(REVIEWS_PARQUET, columns=["review_id", "user_id", "business_id", "stars", "text", "date"]).copy()
df["stars"] = df["stars"].astype(int)
print("отзывов", len(df), "режим", args.mode)

half_clean, half_corrupt = train_test_split(df, test_size=0.5, stratify=df["stars"], random_state=args.seed)

if args.mode == "random":
    clean = half_clean.assign(orig_stars=lambda d: d["stars"], label=0)
    corrupt = half_corrupt.copy()
    corrupt["orig_stars"] = corrupt["stars"]
    corrupt["stars"] = corrupt_random(corrupt["stars"].to_numpy(), args.seed)
    corrupt["label"] = 1
else:
    clean, corrupt = build_balanced(half_clean, half_corrupt, args.seed)

out = pd.concat([clean, corrupt], ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
out = out[["review_id", "user_id", "business_id", "text", "date", "stars", "orig_stars", "label"]]

d1 = out.loc[out["label"] == 1]
assert np.all(np.abs(d1["stars"] - d1["orig_stars"]) > 1)

MISMATCH.mkdir(parents=True, exist_ok=True)
out.to_parquet(MISMATCH_PARQUET, index=False)
print("сохранено", len(out), "label0", int((out["label"] == 0).sum()), "label1", int((out["label"] == 1).sum()))

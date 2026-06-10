#!/usr/bin/env python3
"""
build_mismatch_dataset.py — датасет для детектора несогласованности «текст ↔ оценка».

Бизнес (выбрано): чистка данных/достоверность рейтинга + UX-подсказка автору при публикации
(«написали негативно, но 5★ — точно?»). Меток «фейк» нет → учим бинарный классификатор отличать
согласованную пару (text, stars) от рассогласованной через СИНТЕТИЧЕСКУЮ порчу.

Общая схема:
  1. Берём исходные отзывы (data/processed/reviews.parquet) — НЕ изменяя их.
  2. Стратифицированно по `stars` делим на две равные половины.
  3. Половина A → label = 0 (оценка как есть, согласовано).
  4. Половине B портим оценку: новый stars c |new - orig| > 1 → label = 1.
  5. Склеиваем A+B (перемешиваем) и сохраняем НОВЫЙ датасет в data/mismatch/.

Режим порчи (--mode):
  balanced (по умолчанию) — распределение `stars` делается ОДИНАКОВЫМ в обоих классах
      (транспортная задача + подвыборка), чтобы по одной только оценке нельзя было угадать label
      (убираем shortcut; stars-only ≈ 50%). Цена — часть строк отбрасывается.
  random — новый stars равновероятен среди допустимых (|Δ|>1); проще, но даёт сдвиг распределения
      stars между классами (модель может частично читать label по самой оценке).

Запуск:
  python scripts/build_mismatch_dataset.py                  # balanced, seed=42
  python scripts/build_mismatch_dataset.py --mode random
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # корень проекта — для _constants
from _constants import REVIEWS_PARQUET, MISMATCH, MISMATCH_PARQUET

STARS = [1, 2, 3, 4, 5]
# Допустимые «испорченные» значения для каждой исходной оценки (|new - orig| > 1).
ALLOWED = {s: [t for t in STARS if abs(t - s) > 1] for s in STARS}
CELLS = [(s, t) for s in STARS for t in ALLOWED[s]]


def corrupt_random(orig: np.ndarray, seed: int) -> np.ndarray:
    """Режим A: новый stars равновероятен среди допустимых (|Δ|>1)."""
    rng = np.random.default_rng(seed)
    new = orig.copy()
    for s in STARS:
        m = orig == s
        if m.any():
            new[m] = rng.choice(ALLOWED[s], size=int(m.sum()))
    return new


def solve_transport(n_src: dict[int, int], n_dst: dict[int, int],
                    q: dict[int, float]) -> dict[tuple[int, int], int]:
    """Максимизируем объём M так, чтобы НОВЫЕ оценки шли в пропорции q (натуральное
    распределение — сохраняем все звёзды и реалистичную форму), при этом:
      - расход по источнику:    sum_t x[s,t] <= n_src[s];
      - столбец = доля от M:     sum_s x[s,t] = q[t]*M   (одинаковая маргиналь у обоих классов);
      - чистых строк хватает:    q[t]*M <= n_dst[t].
    n_src[s] — испорченные строки с исходной оценкой s; n_dst[t] — доступные чистые строки с t."""
    from scipy.optimize import linprog
    idx = {c: i for i, c in enumerate(CELLS)}
    nvar = len(CELLS) + 1                          # переменные X + объём M (последняя)
    M = nvar - 1
    cobj = np.zeros(nvar); cobj[M] = -1.0          # максимизируем M

    A_eq, b_eq = [], []
    for t in STARS:                                # sum_s x[s,t] - q[t]*M = 0
        row = np.zeros(nvar)
        for s in STARS:
            if (s, t) in idx:
                row[idx[(s, t)]] = 1
        row[M] = -q[t]
        A_eq.append(row); b_eq.append(0.0)

    A_ub, b_ub = [], []
    for s in STARS:                                # sum_t x[s,t] <= n_src[s]
        row = np.zeros(nvar)
        for t in ALLOWED[s]:
            row[idx[(s, t)]] = 1
        A_ub.append(row); b_ub.append(n_src[s])
    for t in STARS:                                # q[t]*M <= n_dst[t]
        row = np.zeros(nvar); row[M] = q[t]
        A_ub.append(row); b_ub.append(n_dst[t])

    res = linprog(cobj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=[(0, None)] * nvar, method="highs")
    x = np.floor(res.x[:len(CELLS)] + 1e-6).astype(int)   # целые потоки в пределах лимитов
    return {c: int(x[idx[c]]) for c in CELLS}


def build_balanced(clean: pd.DataFrame, corrupt: pd.DataFrame, seed: int):
    """Режим B: одинаковое распределение stars в обоих классах (убираем shortcut)."""
    n_src = corrupt["stars"].value_counts().reindex(STARS, fill_value=0).to_dict()
    n_dst = clean["stars"].value_counts().reindex(STARS, fill_value=0).to_dict()
    q = clean["stars"].value_counts(normalize=True).reindex(STARS, fill_value=0).to_dict()
    X = solve_transport(n_src, n_dst, q)
    c_t = {t: sum(X[(s, t)] for s in STARS if (s, t) in X) for t in STARS}  # общая маргиналь

    # --- испорченная половина: назначаем новые оценки по X, лишние строки отбрасываем ---
    corrupt = corrupt.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    new = np.full(len(corrupt), -1, dtype=int)
    s_arr = corrupt["stars"].to_numpy()
    for s in STARS:
        pos = np.where(s_arr == s)[0]
        cur = 0
        for t in ALLOWED[s]:
            k = X[(s, t)]
            new[pos[cur:cur + k]] = t
            cur += k
    corrupt = corrupt.assign(orig_stars=corrupt["stars"], stars=new, label=1)
    corrupt = corrupt[corrupt["stars"] > 0]      # отбрасываем неназначенные

    # --- чистая половина: подвыборка до тех же c_t на каждую оценку ---
    parts = []
    for t in STARS:
        grp = clean[clean["stars"] == t]
        parts.append(grp.sample(n=c_t[t], random_state=seed))
    clean = pd.concat(parts).assign(orig_stars=lambda d: d["stars"], label=0)
    return clean, corrupt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["balanced", "random"], default="balanced")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not REVIEWS_PARQUET.exists():
        print(f"[error] нет {REVIEWS_PARQUET}. Сначала собери срез: scripts/preprocess.py",
              file=sys.stderr)
        return 1

    cols = ["review_id", "user_id", "business_id", "stars", "text", "date"]
    df = pd.read_parquet(REVIEWS_PARQUET, columns=cols).copy()
    df["stars"] = df["stars"].astype(int)
    print(f"[load] {REVIEWS_PARQUET.name}: {len(df):,} отзывов | режим: {args.mode}")
    print("  распределение stars:", df["stars"].value_counts().sort_index().to_dict())

    # Стратифицированный сплит 50/50 по stars.
    half_clean, half_corrupt = train_test_split(
        df, test_size=0.5, stratify=df["stars"], random_state=args.seed)

    if args.mode == "random":
        clean = half_clean.assign(orig_stars=lambda d: d["stars"], label=0)
        corrupt = half_corrupt.copy()
        corrupt["orig_stars"] = corrupt["stars"]
        corrupt["stars"] = corrupt_random(corrupt["stars"].to_numpy(), args.seed)
        corrupt["label"] = 1
    else:
        clean, corrupt = build_balanced(half_clean, half_corrupt, args.seed)

    out = (pd.concat([clean, corrupt], ignore_index=True)
             .sample(frac=1.0, random_state=args.seed).reset_index(drop=True))
    out = out[["review_id", "user_id", "business_id", "text", "date",
               "stars", "orig_stars", "label"]]

    # --- проверки целостности ---
    d1 = out.loc[out["label"] == 1]
    assert np.all(np.abs(d1["stars"] - d1["orig_stars"]) > 1), "нарушено |new - orig| > 1"

    MISMATCH.mkdir(parents=True, exist_ok=True)
    out.to_parquet(MISMATCH_PARQUET, index=False)

    dist = out.groupby("label")["stars"].value_counts(normalize=True).unstack().reindex(columns=STARS).round(3)
    print(f"\n[save] {MISMATCH_PARQUET} ({MISMATCH_PARQUET.stat().st_size/1e6:.1f} MB)")
    print(f"  всего: {len(out):,} | label=0: {(out['label']==0).sum():,} | "
          f"label=1: {(out['label']==1).sum():,} | сохранено от исходных: {len(out)/len(df):.0%}")
    print("  распределение stars по классам (хотим одинаковое в режиме balanced):")
    print(dist.to_string())
    print("  исходный reviews.parquet НЕ изменялся.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

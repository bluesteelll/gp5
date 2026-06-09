#!/usr/bin/env python3
"""
02_preprocess.py — Первичная обработка и подготовка данных (Критерий 2).

Из сырых JSON-lines файлов Yelp делает компактный срез ОДНОГО метро
(совет README: один город -> все данные реально влезают и логируются целиком,
Критерий 5) и сохраняет аккуратные parquet-таблицы в data/processed/.

Что делает:
  1. Читает business.json, печатает топ метро (city, state) по числу отзывов,
     выбирает крупнейший (или заданный через --city/--state / env YELP_CITY).
  2. Фильтрует бизнесы выбранного метро, парсит ценовой диапазон и категории.
  3. ПОТОКОВО (streaming, не загружая 5 ГБ в память) фильтрует review.json
     по business_id метро -> reviews.parquet.
  4. По появившимся в отзывах user_id потоково фильтрует user.json,
     сразу сворачивая тяжёлые поля friends/elite в счётчики -> users.parquet.
  5. Фильтрует tip.json (текст без оценки — таргет для inference) -> tips.parquet.

Запуск:
  python scripts/02_preprocess.py                 # авто-выбор крупнейшего метро
  python scripts/02_preprocess.py --city Philadelphia --state PA
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import polars as pl

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "data" / "processed"

BUSINESS = RAW / "yelp_academic_dataset_business.json"
REVIEW = RAW / "yelp_academic_dataset_review.json"
USER = RAW / "yelp_academic_dataset_user.json"
TIP = RAW / "yelp_academic_dataset_tip.json"


def choose_metro(args) -> tuple[str, str, pl.DataFrame]:
    """Грузит business.json (он небольшой), печатает топ метро, возвращает выбранное."""
    print(f"[load] {BUSINESS.name}")
    biz = pl.read_ndjson(BUSINESS, infer_schema_length=2000)
    # Нормализуем город (регистр/пробелы) для надёжной группировки.
    biz = biz.with_columns(
        pl.col("city").str.strip_chars().alias("city"),
    )
    metro = (
        biz.group_by(["city", "state"])
        .agg(
            pl.len().alias("n_business"),
            pl.col("review_count").sum().alias("n_reviews"),
        )
        .sort("n_reviews", descending=True)
    )
    print("\n[top-10 метро по числу отзывов]")
    print(metro.head(10))

    if args.city:
        city, state = args.city, args.state
        if state is None:
            # выберем штат с макс. отзывами для этого города
            row = metro.filter(pl.col("city") == city).head(1)
            state = row["state"][0]
    else:
        top = metro.head(1)
        city, state = top["city"][0], top["state"][0]
    print(f"\n[выбрано метро] {city}, {state}")
    return city, state, biz


def save_business(biz: pl.DataFrame, city: str, state: str) -> pl.DataFrame:
    sub = biz.filter((pl.col("city") == city) & (pl.col("state") == state))

    # Ценовой диапазон ($..$$$$) лежит в attributes.RestaurantsPriceRange2.
    price = None
    if "attributes" in sub.columns:
        try:
            price = (
                sub.select(
                    pl.col("attributes").struct.field("RestaurantsPriceRange2")
                )
                .to_series()
                .cast(pl.Utf8)
            )
        except Exception:
            price = None

    keep = [
        "business_id", "name", "city", "state", "postal_code",
        "latitude", "longitude", "stars", "review_count", "is_open", "categories",
    ]
    out = sub.select([c for c in keep if c in sub.columns])
    if price is not None:
        out = out.with_columns(
            price.alias("price_range_raw"),
            pl.col("review_count").alias("review_count"),
        )
        out = out.with_columns(
            pl.col("price_range_raw")
            .cast(pl.Utf8)
            .replace({"None": None})
            .cast(pl.Int64, strict=False)
            .alias("price_range")
        ).drop("price_range_raw")

    OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT / "business.parquet")
    print(f"[save] business.parquet: {out.height:,} строк")
    return out


def stream_filter_reviews(biz_ids: list[str]) -> list[str]:
    """Потоково фильтрует review.json по business_id метро, пишет parquet.
    Возвращает список user_id, встретившихся в этих отзывах."""
    keep = ["review_id", "user_id", "business_id", "stars",
            "useful", "funny", "cool", "text", "date"]
    print(f"[stream] {REVIEW.name} -> reviews.parquet (фильтр по {len(biz_ids):,} бизнесам)")
    lf = (
        pl.scan_ndjson(REVIEW, infer_schema_length=2000)
        .select(keep)
        .filter(pl.col("business_id").is_in(biz_ids))
    )
    lf.sink_parquet(OUT / "reviews.parquet")
    rv = pl.read_parquet(OUT / "reviews.parquet", columns=["user_id"])
    print(f"[save] reviews.parquet: {rv.height:,} отзывов")
    return rv["user_id"].unique().to_list()


def stream_filter_users(user_ids: list[str]) -> None:
    """Потоково фильтрует user.json; тяжёлые friends/elite сразу сворачиваем в счётчики."""
    print(f"[stream] {USER.name} -> users.parquet (фильтр по {len(user_ids):,} юзерам)")
    base = ["user_id", "review_count", "yelping_since", "useful", "funny",
            "cool", "fans", "average_stars"]
    lf = (
        pl.scan_ndjson(USER, infer_schema_length=2000)
        .filter(pl.col("user_id").is_in(user_ids))
        .with_columns(
            # friends / elite — длинные строки; считаем их длину, сами строки не храним
            pl.when(pl.col("friends").is_null() | (pl.col("friends") == "None"))
            .then(0)
            .otherwise(pl.col("friends").str.split(", ").list.len())
            .alias("n_friends"),
            pl.when(pl.col("elite").is_null() | (pl.col("elite") == ""))
            .then(0)
            .otherwise(pl.col("elite").cast(pl.Utf8).str.split(",").list.len())
            .alias("n_elite_years"),
        )
        .select(base + ["n_friends", "n_elite_years"])
    )
    lf.sink_parquet(OUT / "users.parquet")
    n = pl.read_parquet(OUT / "users.parquet", columns=["user_id"]).height
    print(f"[save] users.parquet: {n:,} юзеров")


def stream_filter_tips(biz_ids: list[str]) -> None:
    if not TIP.exists():
        print("[skip] tip.json отсутствует")
        return
    print(f"[stream] {TIP.name} -> tips.parquet")
    lf = (
        pl.scan_ndjson(TIP, infer_schema_length=2000)
        .filter(pl.col("business_id").is_in(biz_ids))
    )
    lf.sink_parquet(OUT / "tips.parquet")
    n = pl.read_parquet(OUT / "tips.parquet", columns=["business_id"]).height
    print(f"[save] tips.parquet: {n:,} типов (текст без оценки)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=os.environ.get("YELP_CITY"))
    ap.add_argument("--state", default=os.environ.get("YELP_STATE"))
    args = ap.parse_args()

    for f in (BUSINESS, REVIEW, USER):
        if not f.exists():
            print(f"[error] нет файла {f}. Сначала запусти scripts/01_download.py")
            return 1

    city, state, biz = choose_metro(args)
    biz_sub = save_business(biz, city, state)
    biz_ids = biz_sub["business_id"].to_list()

    user_ids = stream_filter_reviews(biz_ids)
    stream_filter_users(user_ids)
    stream_filter_tips(biz_ids)

    # Метаданные среза — пригодятся в EDA и для воспроизводимости.
    meta = pl.DataFrame({"city": [city], "state": [state],
                         "n_business": [len(biz_ids)], "n_users": [len(user_ids)]})
    meta.write_parquet(OUT / "_meta.parquet")
    print("\n[ok] Готово. Срез метро в data/processed/:")
    for p in sorted(OUT.glob("*.parquet")):
        print(f"   {p.name}: {p.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

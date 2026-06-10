#!/usr/bin/env python3
"""
preprocess.py — Первичная обработка и подготовка данных (Критерий 2).

Из сырых JSON-lines файлов Yelp делает компактный срез выбранных городов
(по умолчанию — набор _constants.DEFAULT_CITIES, обоснованный в notebooks/EDA_1.ipynb)
и сохраняет аккуратные parquet-таблицы в data/processed/.

Что делает:
  1. Читает business.json, нормализует «город, штат» и отбирает заведения выбранных городов.
  2. Парсит ценовой диапазон, сохраняет business.parquet.
  3. ПОТОКОВО (streaming, не загружая 5 ГБ в память) фильтрует review.json
     по business_id среза -> reviews.parquet.
  4. По появившимся в отзывах user_id потоково фильтрует user.json,
     сразу сворачивая тяжёлые поля friends/elite в счётчики -> users.parquet.
  5. Фильтрует tip.json (текст без оценки — таргет для inference) -> tips.parquet.

Запуск:
  python scripts/preprocess.py                                  # срез DEFAULT_CITIES
  python scripts/preprocess.py --cities "Tucson, AZ;Boise, ID"  # свои города (ключи "City, ST")
  python scripts/preprocess.py --city Philadelphia --state PA   # один город
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import polars as pl

# _constants.py лежит в корне проекта — добавляем корень в путь импорта
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _constants import (
    PROCESSED,
    BUSINESS,
    REVIEW,
    USER,
    TIP,
    BUSINESS_PARQUET,
    REVIEWS_PARQUET,
    USERS_PARQUET,
    TIPS_PARQUET,
    META_PARQUET,
    DEFAULT_CITIES,
)


def city_state_expr() -> pl.Expr:
    """Канонический ключ 'City, ST' (склеивает Saint/St.) — как в EDA_1."""
    c = (
        pl.col("city").fill_null("").str.strip_chars().str.to_lowercase()
        .str.replace_all(".", "", literal=True)
        .str.replace(r"^saint\s+", "st ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars().str.to_titlecase()
    )
    return (c + pl.lit(", ") + pl.col("state").fill_null("")).alias("city_state")


def select_cities(args) -> list[str]:
    """Список ключей городов 'City, ST' для среза."""
    if args.cities:
        return [c.strip() for c in args.cities.split(";") if c.strip()]
    if args.city:
        key = (
            pl.DataFrame({"city": [args.city], "state": [args.state or ""]})
            .with_columns(city_state_expr())["city_state"][0]
        )
        return [key]
    return list(DEFAULT_CITIES)


def save_business(biz: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    sub = biz.filter(pl.col("city_state").is_in(keys))

    # Ценовой диапазон ($..$$$$) лежит в attributes.RestaurantsPriceRange2.
    price = None
    if "attributes" in sub.columns:
        try:
            price = (
                sub.select(pl.col("attributes").struct.field("RestaurantsPriceRange2"))
                .to_series()
                .cast(pl.Utf8)
            )
        except Exception:
            price = None

    keep = [
        "business_id", "name", "city", "state", "city_state", "postal_code",
        "latitude", "longitude", "stars", "review_count", "is_open", "categories",
    ]
    out = sub.select([c for c in keep if c in sub.columns])
    if price is not None:
        out = out.with_columns(price.alias("price_range_raw"))
        out = out.with_columns(
            pl.col("price_range_raw")
            .cast(pl.Utf8)
            .replace({"None": None})
            .cast(pl.Int64, strict=False)
            .alias("price_range")
        ).drop("price_range_raw")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.write_parquet(BUSINESS_PARQUET)
    print(f"[save] business.parquet: {out.height:,} заведений")
    return out


def stream_filter_reviews(biz_ids: list[str]) -> list[str]:
    """Потоково фильтрует review.json по business_id среза, пишет parquet.
    Возвращает список user_id, встретившихся в этих отзывах."""
    keep = ["review_id", "user_id", "business_id", "stars",
            "useful", "funny", "cool", "text", "date"]
    print(f"[stream] {REVIEW.name} -> reviews.parquet (фильтр по {len(biz_ids):,} бизнесам)")
    lf = (
        pl.scan_ndjson(REVIEW, infer_schema_length=2000)
        .select(keep)
        .filter(pl.col("business_id").is_in(biz_ids))
    )
    lf.sink_parquet(REVIEWS_PARQUET)
    rv = pl.read_parquet(REVIEWS_PARQUET, columns=["user_id"])
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
    lf.sink_parquet(USERS_PARQUET)
    n = pl.read_parquet(USERS_PARQUET, columns=["user_id"]).height
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
    lf.sink_parquet(TIPS_PARQUET)
    n = pl.read_parquet(TIPS_PARQUET, columns=["business_id"]).height
    print(f"[save] tips.parquet: {n:,} типсов (текст без оценки)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default=None,
                    help='ключи городов через ";", напр. "Tucson, AZ;Boise, ID"')
    ap.add_argument("--city", default=None, help="один город (вместе с --state)")
    ap.add_argument("--state", default=None)
    args = ap.parse_args()

    for f in (BUSINESS, REVIEW, USER):
        if not f.exists():
            print(f"[error] нет файла {f}. Сначала запусти scripts/download.py")
            return 1

    keys = select_cities(args)
    print(f"[load] {BUSINESS.name}")
    biz = pl.read_ndjson(BUSINESS, infer_schema_length=2000).with_columns(city_state_expr())
    print(f"[срез] города: {keys}")

    biz_sub = save_business(biz, keys)
    if biz_sub.height == 0:
        print(f"[error] под выбранные города нет заведений: {keys}", file=sys.stderr)
        return 1
    # Сколько заведений попало по каждому городу
    by_city = biz_sub.group_by("city_state").len().sort("len", descending=True)
    for row in by_city.iter_rows(named=True):
        print(f"   {row['city_state']}: {row['len']:,} заведений")
    biz_ids = biz_sub["business_id"].to_list()

    user_ids = stream_filter_reviews(biz_ids)
    stream_filter_users(user_ids)
    stream_filter_tips(biz_ids)

    n_reviews = pl.read_parquet(REVIEWS_PARQUET, columns=["review_id"]).height
    # Метаданные среза — пригодятся в EDA и для воспроизводимости.
    meta = pl.DataFrame({"cities": [" + ".join(keys)],
                         "n_cities": [len(keys)],
                         "n_business": [biz_sub.height],
                         "n_users": [len(user_ids)],
                         "n_reviews": [n_reviews]})
    meta.write_parquet(META_PARQUET)
    print("\n[ok] Готово. Срез в data/processed/:")
    for p in sorted(PROCESSED.glob("*.parquet")):
        print(f"   {p.name}: {p.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

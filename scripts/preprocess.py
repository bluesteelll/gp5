import argparse, sys
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _constants import (PROCESSED, BUSINESS, REVIEW, USER, TIP, BUSINESS_PARQUET,
                        REVIEWS_PARQUET, USERS_PARQUET, TIPS_PARQUET, META_PARQUET, DEFAULT_CITIES)

def city_key():
    c = (pl.col("city").fill_null("").str.strip_chars().str.to_lowercase()
         .str.replace_all(".", "", literal=True)
         .str.replace(r"^saint\s+", "st ")
         .str.replace_all(r"\s+", " ")
         .str.strip_chars().str.to_titlecase())
    return (c + pl.lit(", ") + pl.col("state").fill_null("")).alias("city_state")

ap = argparse.ArgumentParser()
ap.add_argument("--cities", default=None)
ap.add_argument("--city", default=None)
ap.add_argument("--state", default=None)
args = ap.parse_args()

for f in (BUSINESS, REVIEW, USER):
    if not f.exists():
        print("нет файла", f, "- сначала download.py")
        sys.exit(1)

if args.cities:
    keys = [c.strip() for c in args.cities.split(";") if c.strip()]
elif args.city:
        tmp = pl.DataFrame({"city": [args.city], "state": [args.state or ""]})
        keys = [tmp.with_columns(city_key())["city_state"][0]]
else:
    keys = list(DEFAULT_CITIES)
print("города:", keys)

biz = pl.read_ndjson(BUSINESS, infer_schema_length=2000).with_columns(city_key())
sub = biz.filter(pl.col("city_state").is_in(keys))
if sub.height == 0:
    print("под выбранные города нет заведений:", keys)
    sys.exit(1)

price = None
if "attributes" in sub.columns:
    try:
        price = sub.select(pl.col("attributes").struct.field("RestaurantsPriceRange2")).to_series().cast(pl.Utf8)
    except Exception:
        price = None

keep = ["business_id", "name", "city", "state", "city_state", "postal_code","latitude", "longitude", "stars", "review_count", "is_open", "categories"]
out = sub.select([c for c in keep if c in sub.columns])
if price is not None:
    out = out.with_columns(price.alias("pr"))
    out = out.with_columns(pl.col("pr").cast(pl.Utf8).replace({"None": None})
                           .cast(pl.Int64, strict=False).alias("price_range")).drop("pr")

PROCESSED.mkdir(parents=True, exist_ok=True)
out.write_parquet(BUSINESS_PARQUET)
print("business.parquet:", out.height)
biz_ids = out["business_id"].to_list()

cols = ["review_id", "user_id", "business_id", "stars", "useful", "funny", "cool", "text", "date"]
pl.scan_ndjson(REVIEW, infer_schema_length=2000).select(cols).filter(pl.col("business_id").is_in(biz_ids)).sink_parquet(REVIEWS_PARQUET)
rv = pl.read_parquet(REVIEWS_PARQUET, columns=["user_id", "review_id"])
print("reviews.parquet:", rv.height)
user_ids = rv["user_id"].unique().to_list()

base = ["user_id", "review_count", "yelping_since", "useful", "funny", "cool", "fans", "average_stars"]
n_friends = (
    pl.when(pl.col("friends").is_null() | (pl.col("friends") == "None"))
    .then(0)
    .otherwise(pl.col("friends").str.split(", ").list.len())
    .alias("n_friends")
)
n_elite_years = (
    pl.when(pl.col("elite").is_null() | (pl.col("elite") == ""))
    .then(0)
    .otherwise(pl.col("elite").cast(pl.Utf8).str.split(",").list.len())
    .alias("n_elite_years")
)
users = pl.scan_ndjson(USER, infer_schema_length=2000)
users = users.filter(pl.col("user_id").is_in(user_ids))
users = users.with_columns(n_friends, n_elite_years)
users = users.select(base + ["n_friends", "n_elite_years"])
users.sink_parquet(USERS_PARQUET)
print("users.parquet:", pl.read_parquet(USERS_PARQUET, columns=["user_id"]).height)

if TIP.exists():
    tips = pl.scan_ndjson(TIP, infer_schema_length=2000)
    tips = tips.filter(pl.col("business_id").is_in(biz_ids))
    tips.sink_parquet(TIPS_PARQUET)
    print("tips.parquet:", pl.read_parquet(TIPS_PARQUET, columns=["business_id"]).height)

pl.DataFrame({"cities": [" + ".join(keys)], "n_cities": [len(keys)], "n_business": [out.height],
             "n_users": [len(user_ids)], "n_reviews": [rv.height]}).write_parquet(META_PARQUET)
print("готово")
import os
from pathlib import Path

DATASET = "yelp-dataset/yelp-dataset"

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
ARTIFACTS = PROJECT / "artifacts"

FILES = ["yelp_academic_dataset_business.json","yelp_academic_dataset_review.json","yelp_academic_dataset_user.json","yelp_academic_dataset_tip.json",]

BUSINESS = RAW / "yelp_academic_dataset_business.json"
REVIEW = RAW / "yelp_academic_dataset_review.json"
USER = RAW / "yelp_academic_dataset_user.json"
TIP = RAW / "yelp_academic_dataset_tip.json"

BUSINESS_PARQUET = PROCESSED / "business.parquet"
REVIEWS_PARQUET = PROCESSED / "reviews.parquet"
USERS_PARQUET = PROCESSED / "users.parquet"
TIPS_PARQUET = PROCESSED / "tips.parquet"
META_PARQUET = PROCESSED / "_meta.parquet"

MISMATCH = PROJECT / "data" / "mismatch"
MISMATCH_PARQUET = MISMATCH / "mismatch_dataset.parquet"

DEFAULT_CITIES = ["Tucson, AZ", "St Petersburg, FL", "Edmonton, AB"]


def load_dotenv(path=None):
    if path:
        p = Path(path)
    else:
        p = PROJECT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on", "y"}


load_dotenv()

ENABLE_LOGGING = env_bool("ENABLE_LOGGING", False)
ENABLE_ARTIFACTS = env_bool("ENABLE_ARTIFACTS", False)
ENABLE_FAST_DEV_RUN = env_bool("ENABLE_FAST_DEV_RUN", False)

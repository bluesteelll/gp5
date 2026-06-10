from pathlib import Path

DATASET = "yelp-dataset/yelp-dataset"

# Каталоги проекта
PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
FIGURES = PROJECT / "figures"
REPORTS = PROJECT / "reports"

# Сырые JSON, которые скачиваем с Kaggle
FILES = [
    "yelp_academic_dataset_business.json",
    "yelp_academic_dataset_review.json",
    "yelp_academic_dataset_user.json",
    "yelp_academic_dataset_tip.json",
]

# Сырые данные
BUSINESS = RAW / "yelp_academic_dataset_business.json"
REVIEW = RAW / "yelp_academic_dataset_review.json"
USER = RAW / "yelp_academic_dataset_user.json"
TIP = RAW / "yelp_academic_dataset_tip.json"

# Обработанные данные
BUSINESS_PARQUET = PROCESSED / "business.parquet"
REVIEWS_PARQUET = PROCESSED / "reviews.parquet"
USERS_PARQUET = PROCESSED / "users.parquet"
TIPS_PARQUET = PROCESSED / "tips.parquet"
META_PARQUET = PROCESSED / "_meta.parquet"

# Города среза по умолчанию (нормализованные ключи "City, ST").
# Набор выбран в notebooks/EDA_1.ipynb: 2-3 крупных города со сбалансированными
# классами оценок и максимумом типсов (сырьё для Задачи 2), суммарно ~600-700k отзывов.
DEFAULT_CITIES = [
    "Tucson, AZ",
    "St Petersburg, FL",
    "Edmonton, AB",
]

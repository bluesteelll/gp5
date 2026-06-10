import os
from pathlib import Path

DATASET = "yelp-dataset/yelp-dataset"

# Каталоги проекта
PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
ARTIFACTS = PROJECT / "artifacts"
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

# Датасет для детектора несогласованности «текст↔оценка» (синтетическая порча, отдельная папка).
MISMATCH = PROJECT / "data" / "mismatch"
MISMATCH_PARQUET = MISMATCH / "mismatch_dataset.parquet"

# Города среза по умолчанию (нормализованные ключи "City, ST").
# Набор выбран в notebooks/EDA_1.ipynb: 2-3 крупных города со сбалансированными
# классами оценок и максимумом типсов (сырьё для Задачи 2), суммарно ~600-700k отзывов.
DEFAULT_CITIES = [
    "Tucson, AZ",
    "St Petersburg, FL",
    "Edmonton, AB",
]


# ── Переменные окружения (.env) ───────────────────────────────────────────
def load_dotenv(path: str | Path | None = None) -> None:
    """Подхватывает переменные из .env в os.environ (реальное окружение в приоритете)."""
    p = Path(path) if path else PROJECT / ".env"
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on", "y"}


# Грузим .env при импорте — чтобы флаги работали и в скриптах, и в ноутбуках.
load_dotenv()

# Флаги (по умолчанию False)
ENABLE_LOGGING = env_bool("ENABLE_LOGGING", False)      # логирование экспериментов с моделями (используется позже)
ENABLE_ARTIFACTS = env_bool("ENABLE_ARTIFACTS", False)  # сохранять ли картинки EDA в artifacts/

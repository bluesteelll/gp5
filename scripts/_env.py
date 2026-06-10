from __future__ import annotations
import os
from pathlib import Path

from _constants import load_dotenv


def load_env(path: str | Path | None = None) -> None:
    """Загружает .env в окружение и настраивает Kaggle-аутентификацию."""
    load_dotenv(path)
    _materialize_kaggle_token()


def _materialize_kaggle_token() -> None:
    """Кладёт KAGGLE_API_TOKEN в ~/.kaggle/access_token, если файла ещё нет."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    kdir = Path.home() / ".kaggle"
    f = kdir / "access_token"
    if token and not f.exists():
        kdir.mkdir(exist_ok=True)
        f.write_text(token.strip(), encoding="utf-8")
        os.chmod(f, 0o600)


def have_kaggle_creds():
    return bool(os.environ.get("KAGGLE_API_TOKEN")) or (Path.home() / ".kaggle" / "access_token").exists()

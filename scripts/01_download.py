#!/usr/bin/env python3
"""
01_download.py — Автоматизированный сбор данных (Критерий 2).

Скачивает нужные файлы Yelp Open Dataset с Kaggle-зеркала
(yelp-dataset/yelp-dataset) через официальный Kaggle CLI и распаковывает
JSON в data/raw/. Качаем по одному файлу (resume-friendly) и пропускаем
checkin.json, который в наших задачах не нужен.

Авторизация: единственный способ — KAGGLE_API_TOKEN (OAuth access token "KGAT_...")
из файла .env. Получить: kaggle.com -> Settings -> API -> Create New Token.

Запуск:
  python scripts/01_download.py
"""
from __future__ import annotations
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import load_env, have_kaggle_creds

DATASET = "yelp-dataset/yelp-dataset"
PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"

# Файлы, которые реально нужны двум задачам (checkin.json пропускаем).
NEEDED = [
    "yelp_academic_dataset_business.json",
    "yelp_academic_dataset_tip.json",
    "yelp_academic_dataset_review.json",
    "yelp_academic_dataset_user.json",
]


def kaggle_bin() -> str:
    return shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))


def download_one(name: str) -> bool:
    target = RAW / name
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {name} уже есть ({target.stat().st_size/1e6:.1f} MB)")
        return True
    print(f"[download] {name} ...")
    cmd = [kaggle_bin(), "datasets", "download", "-d", DATASET,
           "-f", name, "-p", str(RAW)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[error] не удалось скачать {name}", file=sys.stderr)
        return False
    # Kaggle отдаёт <name>.zip — распакуем и удалим архив.
    z = RAW / f"{name}.zip"
    if z.exists():
        print(f"[unzip] {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW)
        z.unlink()
    return target.exists()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    load_env()  # подхватываем KAGGLE_API_TOKEN из .env
    if not have_kaggle_creds():
        print("[error] нет токена Kaggle. Заполни .env: KAGGLE_API_TOKEN=KGAT_... "
              "(шаблон в .env.example)",
              file=sys.stderr)
        return 1

    ok = True
    for name in NEEDED:
        ok = download_one(name) and ok

    print("\n[итог] data/raw:")
    for p in sorted(RAW.glob("*.json")):
        print(f"   {p.name}: {p.stat().st_size/1e6:.1f} MB")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

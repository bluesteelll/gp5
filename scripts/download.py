import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# _constants.py лежит в корне проекта — добавляем корень в путь импорта
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _env import load_env, have_kaggle_creds
from _constants import RAW, DATASET, FILES


def kaggle_bin() -> str:
    return shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))


def download_one(name: str) -> bool:
    target = RAW / name
    if target.exists() and target.stat().st_size > 0:
        print(f"[skip] {name} уже скачан ({target.stat().st_size/1e6:.1f} MB)")
        return True
    print(f"[download] {name} ...")
    cmd = [kaggle_bin(), "datasets", "download", "-d", DATASET,
           "-f", name, "-p", str(RAW)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[error] не удалось скачать {name}", file=sys.stderr)
        return False
    z = RAW / f"{name}.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW)
        z.unlink()
    return target.exists()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    load_env()
    if not have_kaggle_creds():
        print("[error] отсутствует OAuth-токен Kaggle",
              file=sys.stderr)
        return 1

    downloaded = True
    for name in FILES:
        downloaded = download_one(name) and downloaded

    print("\n[итог] data/raw:")
    for p in sorted(RAW.glob("*.json")):
        print(f"   {p.name}: {p.stat().st_size/1e6:.1f} MB")
    return 0 if downloaded else 1


if __name__ == "__main__":
    raise SystemExit(main())

import shutil, subprocess, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _env import load_env, have_kaggle_creds
from _constants import RAW, DATASET, FILES

load_env()
RAW.mkdir(parents=True, exist_ok=True)
if not have_kaggle_creds():
    print("нет токена Kaggle")
    sys.exit(1)

kaggle = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))

for f in FILES:
    p = RAW/f
    if p.exists() and p.stat().st_size > 0:
        print("skip", f)
        continue
    print("качаю", f)
    subprocess.run([kaggle, "datasets", "download", "-d", DATASET, "-f", f, "-p", str(RAW)])
    z = RAW / (f + ".zip")
    if z.exists():
        zipfile.ZipFile(z).extractall(RAW)
        z.unlink()

for p in sorted(RAW.glob("*.json")):
    print(p.name, round(p.stat().st_size / 1e6, 1), "MB")

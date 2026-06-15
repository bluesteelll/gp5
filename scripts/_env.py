import os
from pathlib import Path
from _constants import load_dotenv

def load_env(path=None):
    load_dotenv(path)
    t = os.environ.get("KAGGLE_API_TOKEN")
    f = Path.home() / ".kaggle" / "access_token"
    if t and not f.exists():
        f.parent.mkdir(exist_ok=True)
        f.write_text(t.strip())
        os.chmod(f, 0o600)

def have_kaggle_creds():
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    return (Path.home() / ".kaggle" / "access_token").exists()

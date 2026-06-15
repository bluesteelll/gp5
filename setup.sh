#!/usr/bin/env bash
set -euo pipefail


DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PY="${PYTHON:-python3}"
VENV="$DIR/.venv"

echo "==> [1/4] Проверка интерпретатора"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "!! Не найден интерпретатор '$PY'. Задай через PYTHON=... ./setup.sh" >&2
  exit 1
fi
"$PY" --version

echo "==> [2/4] Виртуальное окружение и зависимости"
if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
  echo "   создано $VENV"
else
  echo "   $VENV уже существует — переиспользую"
fi
source "$VENV/bin/activate"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r "$DIR/requirements.txt"
echo "   зависимости установлены"

echo "==> [3/4] Проверка токена Kaggle"
if [ -f "$DIR/.env" ]; then
  set -a
  source "$DIR/.env"
  set +a
  echo "   .env загружен"
else
  echo "   !! нет $DIR/.env — скопируй .env.example в .env и впиши токен" >&2
fi
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ ! -f "$HOME/.kaggle/access_token" ]; then
  echo "!! Не найден OAuth-токен для Kaggle. Впиши его в $DIR/.env" >&2
  exit 1
fi
echo "   токен найден"

echo "==> [4/4] Скачивание и первичная обработка"
python3 "$DIR/scripts/download.py"
python3 "$DIR/scripts/preprocess.py" "$@"

echo ""
echo "==> Done:) Окружение настроено."
ls -lh "$DIR/data/processed/" 2>/dev/null || true

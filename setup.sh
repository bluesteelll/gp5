#!/usr/bin/env bash
#
# setup.sh — bootstrap проекта Yelp DL (EDA).
#
# Делает «под ключ»:
#   1) создаёт виртуальное окружение .venv и ставит зависимости;
#   2) скачивает нужные файлы Yelp Open Dataset с Kaggle  (scripts/download.py);
#   3) делает срез одного метро и parquet-таблицы          (scripts/preprocess.py).
#
# Использование:
#   ./setup.sh                       # авто-выбор крупнейшего метро
#   ./setup.sh --city Philadelphia --state PA

set -euo pipefail

# --- пути относительно расположения скрипта (не зависит от cwd) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${python3}"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "==> [1/4] Проверка интерпретатора"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "!! Не найден интерпретатор '$PYTHON'. Задай через PYTHON=... ./setup.sh" >&2
  exit 1
fi
"$PYTHON" --version

echo "==> [2/4] Виртуальное окружение и зависимости"
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON" -m venv "$VENV_DIR"
  echo "   создано $VENV_DIR"
else
  echo "   $VENV_DIR уже существует — переиспользую"
fi

source "$VENV_DIR/bin/activate"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
echo "   зависимости установлены"

echo "==> [3/4] Проверка токена Kaggle"
# Подхватываем KAGGLE_API_TOKEN из .env.
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a; # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"; set +a
  echo "   .env загружен"
else
  echo "   !! нет $SCRIPT_DIR/.env — скопируй .env.example в .env и впиши токен" >&2
fi
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ ! -f "$HOME/.kaggle/access_token" ]; then
  echo "!! Не найден OAuth-токен для Kaggle. Впиши его в $SCRIPT_DIR/.env" >&2
  exit 1
fi
echo "   токен найден"

echo "==> [4/4] Скачивание и первичная обработка"
python3 "$SCRIPT_DIR/scripts/download.py"
python3 "$SCRIPT_DIR/scripts/preprocess.py" "$@"

echo ""
echo "==> Done:) Окружение настроено."
ls -lh "$SCRIPT_DIR/data/processed/" 2>/dev/null || true

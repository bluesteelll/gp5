#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
for f in notebooks/*.ipynb; do jupyter nbconvert --clear-output --inplace $f; done
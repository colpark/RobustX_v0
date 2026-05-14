#!/usr/bin/env bash
# Resumable downloader for the EventScape dataset (CARLA driving simulation).
#
# URLs verified from the rpg_ramnet README:
#   train (71 GB):  Town01-03_train.zip
#   val   (12 GB):  Town05_val.zip
#   test  (14 GB):  Town05_test.zip
#
# Usage:
#   ./get_eventscape.sh                  # train + test  (val you probably already have)
#   ./get_eventscape.sh val              # just val
#   ./get_eventscape.sh train val test   # everything (~ 100 GB)
#   OUT=/scratch/eventscape_raw ./get_eventscape.sh train
#
# Uses `wget --continue` so a dropped connection / Ctrl-C just resumes on re-run.
# After download each split is unzipped under $OUT/<split>/.

set -euo pipefail

OUT="${OUT:-./data/eventscape_raw}"
SPLITS="${@:-train test}"

BASE_URL="http://rpg.ifi.uzh.ch/data/RAM_Net/dataset"
declare -A FILES=(
  [train]="Town01-03_train.zip"
  [val]="Town05_val.zip"
  [test]="Town05_test.zip"
)

mkdir -p "$OUT" && cd "$OUT"

for split in $SPLITS; do
  fname="${FILES[$split]:-}"
  if [[ -z "$fname" ]]; then
    echo "unknown split: '$split' (use: train / val / test)"; continue
  fi
  url="$BASE_URL/$fname"
  echo
  echo "=== $split  →  $url ==="

  wget --continue --progress=dot:giga "$url" -O "$fname"

  echo "extracting $fname  →  $OUT/$split/"
  mkdir -p "$split"
  unzip -q -d "$split" "$fname"
  echo "  done. extracted to $OUT/$split/"
done

echo
echo "All requested splits in: $OUT/"
du -sh "$OUT"/* 2>/dev/null || true

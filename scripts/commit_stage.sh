#!/usr/bin/env bash
# Kullanım: bash scripts/commit_stage.sh 3 "pvlib pipeline birim testli"
STAGE=${1:-""}
MSG=${2:-""}
if [ -z "$STAGE" ] || [ -z "$MSG" ]; then
    echo "Kullanım: bash scripts/commit_stage.sh <stage_no> <açıklama>"
    exit 1
fi
git add .
git commit -m "STAGE-$STAGE: $MSG"
echo "✓ Commit atıldı: STAGE-$STAGE: $MSG"

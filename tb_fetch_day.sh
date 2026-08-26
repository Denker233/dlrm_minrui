#!/bin/bash
# Download one HF CriteoClickLogs day partition and convert it to the original
# Criteo Terabyte TSV file expected by the repo:  /home/cc/input/terabyte/day_<idx>
# Usage: tb_fetch_day.sh <hf_date> <day_idx>
set -uo pipefail
DATE=$1; IDX=$2
ROOT=/home/cc/input/terabyte
STAGE=$ROOT/_stage/day_$IDX
OUT=$ROOT/day_$IDX
REPO=https://huggingface.co/datasets/criteo/CriteoClickLogs/resolve/main
VENV=/home/cc/expr/dlrm_minrui/dlrm_env/bin/python
CONV=/home/cc/expr/dlrm_minrui/tb_convert_part.py

if [ -s "$OUT" ]; then echo "[day_$IDX] already built, skipping"; exit 0; fi
mkdir -p "$STAGE"

curl -s -m 120 "https://huggingface.co/api/datasets/criteo/CriteoClickLogs/tree/main/data/day=$DATE?recursive=true" \
 | "$VENV" -c "
import json,sys
d=json.load(sys.stdin)
for f in sorted(x['path'] for x in d if x.get('type')=='file'):
    print(f)
" > "$STAGE/files.list"
N=$(wc -l < "$STAGE/files.list")
echo "[day_$IDX] $DATE: $N parts"
[ "$N" -gt 0 ] || { echo "[day_$IDX] FAILED to list parts"; exit 1; }

cat > "$STAGE/worker.sh" <<WEOF
#!/bin/bash
set -uo pipefail
p="\$1"
base=\$(basename "\$p" .c000.snappy.parquet)
pq="$STAGE/\$base.parquet"
txt="$STAGE/\$base.txt"
[ -s "\$txt" ] && exit 0
for try in 1 2 3; do
  curl -sfL --retry 3 --retry-delay 2 -m 900 -o "\$pq" "$REPO/\$p" && break
  echo "[day_$IDX] retry \$try download \$base"; sleep 3
done
[ -s "\$pq" ] || { echo "[day_$IDX] DOWNLOAD_FAIL \$base"; exit 1; }
"$VENV" "$CONV" "\$pq" "\$txt" || { echo "[day_$IDX] CONVERT_FAIL \$base"; rm -f "\$pq" "\$txt"; exit 1; }
rm -f "\$pq"
WEOF
chmod +x "$STAGE/worker.sh"

xargs -a "$STAGE/files.list" -P 12 -n 1 "$STAGE/worker.sh"
rc=$?

DONE=$(ls -1 "$STAGE"/*.txt 2>/dev/null | wc -l)
echo "[day_$IDX] converted $DONE/$N parts (xargs rc=$rc)"
if [ "$DONE" -ne "$N" ]; then echo "[day_$IDX] INCOMPLETE - not assembling"; exit 1; fi

echo "[day_$IDX] assembling $OUT"
: > "$OUT.partial"
while read -r p; do
  base=$(basename "$p" .c000.snappy.parquet)
  cat "$STAGE/$base.txt" >> "$OUT.partial" && rm -f "$STAGE/$base.txt"
done < "$STAGE/files.list"
mv "$OUT.partial" "$OUT"
rm -rf "$STAGE"
echo "[day_$IDX] DONE $(du -h "$OUT" | cut -f1)  lines=$(wc -l < "$OUT")"

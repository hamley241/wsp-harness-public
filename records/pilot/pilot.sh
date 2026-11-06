#!/usr/bin/env bash
# rules/calibration-rule.md amendment 3, then the full run. Mechanical.
set -uo pipefail
cd "$(dirname "$0")/../../code"  # repository root of the code export
COMMON="--backend openrouter --model anthropic/claude-opus-5.5 --k 6 --seed 1 --hidden 16 --max-output-tokens 32000"
score() { uv run python -c "import json;a=json.load(open('$1/results.json'));r=a['arms']['A'][0];print(r['heldout'] if r['heldout'] is not None else 'none')" 2>/dev/null || echo none; }
N=32
for try in 1 2; do
  out=runs/calib-k6-n$N; rm -rf "$out"
  echo "=== calibration n_initial=$N start $(date -u +%H:%M:%SZ)"
  uv run python -m wsp.run $COMMON --rounds 0 --n-initial $N --max-attempts 5 --out "$out" > "$out.log" 2>&1
  echo "exit=$? end $(date -u +%H:%M:%SZ)"
  acc=$(score "$out"); echo "n_initial=$N D0_heldout=$acc"
  if [ "$acc" = "none" ]; then echo "NO SCORED D0: stop"; exit 2; fi
  if uv run python -c "import sys;sys.exit(0 if $acc >= 0.95 else 1)" && [ $try = 1 ]; then N=24; continue; fi
  break
done
echo "FROZEN: k=6 n_initial=$N D0_heldout=$acc" | tee runs/pilot-frozen.txt
out=runs/pilot-k6-n$N; rm -rf "$out"
echo "=== full run start $(date -u +%H:%M:%SZ)"
uv run python -m wsp.run $COMMON --rounds 3 --n-initial $N --max-attempts 3 --d0-from runs/calib-k6-n$N --out "$out" > "$out.log" 2>&1
echo "exit=$? end $(date -u +%H:%M:%SZ)"
ls "$out"

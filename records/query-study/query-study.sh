#!/bin/zsh
# Frozen launch for rules/query-study-rule.md (+ Amendment 1). Order: main, extension (only if needed), sensitivity.
set -euo pipefail
cd "$(dirname "$0")/../../code"  # repository root of the code export
OUT=runs/query-study
COMMON=(--n-main 20 --n-ext 10 --k 6 --n-initial 32 --rounds 3 --n-candidates 32
        --backend openrouter --model anthropic/claude-opus-5.5 --reasoning-tokens 16000 --max-output-tokens 32000
        --sat-timeout 30 --spend-cap-usd 10 --out $OUT)
ROSTER=${1:-main}
uv run python -m wsp.query_study run --roster $ROSTER "${COMMON[@]}"

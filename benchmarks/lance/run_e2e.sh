#!/usr/bin/env bash
# E2E training compute-sweep over the combined mixer: base vs lance, traces the
# data-bound -> compute-bound crossover by sweeping the per-step transformer size (--layers).
# Env (defaults match the dev box; override for another machine):
#   REPO, DATA, S, BUCKET, REGION  (as in run_matrix.sh)
#   REGIME   local | s3 | mixed     (default: mixed — the realistic regime)
#   LAYERS   space-separated layer counts to sweep (default: "1 2 4 8 16")
#   WORKERS  "a v s" for the per-loader DataLoaders (default: "18 4 18" — re-tune per core count)
#   RES      output file            (default: ./e2e_results.txt)
set +u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source .venv-gpu/bin/activate
export PYTHONPATH="$REPO" AWS_PROFILE="${AWS_PROFILE:-cosmosbench}" LANCE_IO_THREADS="${LANCE_IO_THREADS:-256}"

REGIME="${REGIME:-mixed}"
read -r AW VW SW <<< "${WORKERS:-18 4 18}"
RES="${RES:-./e2e_results.txt}"
: > "$RES"
for L in ${LAYERS:-1 2 4 8 16}; do
  for trio in base lance; do
    echo ">>> regime=$REGIME layers=$L trio=$trio workers=$AW/$VW/$SW" | tee -a "$RES"
    python benchmarks/lance/train_combined_e2e.py --trio "$trio" --regime "$REGIME" --layers "$L" \
      --action-workers "$AW" --vlm-workers "$VW" --vsft-workers "$SW" --batch-size 16 --steps 60 --warmup 18 2>&1 \
      | grep -iE "steps/s|compute:" | grep -v warn | sed "s/^/    [L=$L|$trio] /" | tee -a "$RES"
  done
done
echo "=== E2E DONE ($RES) ===" | tee -a "$RES"

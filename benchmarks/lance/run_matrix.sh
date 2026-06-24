#!/usr/bin/env bash
# Full combined-dataloader benchmark matrix: 3 storage regimes (LOCAL / full-S3 / MIXED)
# x 2 worker allocations x {base, lance}. Each cell is an isolated process. Results -> $RES.
#
# Env (defaults match the dev box; override for another machine):
#   REPO    repo root                         (default: this script's ../../..)
#   DATA    local dataset root                (default: /home/ubuntu/work/data)
#   FUSE    s3fs mountpoint of the bucket's cosmos/ prefix  (default: /home/ubuntu/s3mnt/cosmos)
#   S3      s3:// uri of the cosmos/ prefix    (default: s3://lancedb-datasets-dev-us-east-2-devrel/cosmos)
#   BUCKET  bucket name (for the boto3 vsft base)  (default: lancedb-datasets-dev-us-east-2-devrel)
#   REGION  AWS region                        (default: us-east-2)
#   ALLOCS  worker allocations to sweep, "a v s" per entry, ';'-separated
#           (default: "4 4 4;18 4 18" — RE-TUNE the 2nd for this machine's core count, see RUN_BENCHMARKS_H100.md)
#   RES     output file                       (default: ./matrix_results.txt)
# Requires: .venv-gpu active deps + an AWS profile "cosmosbench" + an s3fs mount for the S3 regime's base.
set +u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source .venv-gpu/bin/activate
[ -f benchmarks/lance/.creds.env ] && source benchmarks/lance/.creds.env
export PYTHONPATH="$REPO" AWS_PROFILE="${AWS_PROFILE:-cosmosbench}" LANCE_IO_THREADS="${LANCE_IO_THREADS:-256}"

DATA="${DATA:-/home/ubuntu/work/data}"
FUSE="${FUSE:-/home/ubuntu/s3mnt/cosmos}"
S="${S:-s3://lancedb-datasets-dev-us-east-2-devrel/cosmos}"
BUCKET="${BUCKET:-lancedb-datasets-dev-us-east-2-devrel}"
REGION="${REGION:-us-east-2}"
JSONL="$DATA/bridge_src/sft_dataset_bridge/train/video_dataset_file.jsonl"
RES="${RES:-./matrix_results.txt}"
: > "$RES"
R="--rounds 20 --warmup 6 --batch-size 16"

run() {  # label trio aw vw sw  <regime args...>
  local label="$1" trio="$2" aw="$3" vw="$4" sw="$5"; shift 5
  echo ">>> $label | $trio | $aw/$vw/$sw" | tee -a "$RES"
  python benchmarks/lance/bench_combined_faithful.py "$@" $R --trios "$trio" \
     --action-workers "$aw" --vlm-workers "$vw" --vsft-workers "$sw" 2>&1 \
     | grep -iE "standalone (action|vlm|vision)|combined mixer" | grep -v warn \
     | sed "s/^/    [$label|$trio|$aw\/$vw\/$sw] /" | tee -a "$RES"
}

LOCAL_ARGS=(--action-root $DATA/droid327/success --action-uri $DATA/lance/droid_composed327_plain
            --vlm-wds "$DATA/wds/llava_figureqa/shard-{00000..00019}.tar" --vlm-uri $DATA/lance/llava_figureqa
            --vsft-jsonl $JSONL --vsft-uri $DATA/lance/vision_sft_plain)
S3_ARGS=(--action-root $FUSE/droid327/base/success --action-uri $S/droid327/lance/droid_composed327_plain
         --vlm-wds "$FUSE/llava/wds/shard-{00000..00019}.tar" --vlm-uri $S/llava/lance/llava_figureqa
         --vsft-jsonl $JSONL --vsft-uri $S/vision_sft/lance/vision_sft_plain
         --vsft-s3-bucket $BUCKET --vsft-s3-prefix cosmos/vision_sft/base/sft_dataset_bridge/train --region $REGION)
MIXED_ARGS=(--action-root $DATA/droid327/success --action-uri $DATA/lance/droid_composed327_plain
            --vlm-wds "$DATA/wds/llava_figureqa/shard-{00000..00019}.tar" --vlm-uri $S/llava/lance/llava_figureqa
            --vsft-jsonl $JSONL --vsft-uri $S/vision_sft/lance/vision_sft_plain
            --vsft-s3-bucket $BUCKET --vsft-s3-prefix cosmos/vision_sft/base/sft_dataset_bridge/train
            --vlm-hf-subset "figureqa(cauldron,llava_format)" --region $REGION)

IFS=';' read -ra ALLOC_LIST <<< "${ALLOCS:-4 4 4;18 4 18}"
for alloc in "${ALLOC_LIST[@]}"; do
  set -- $alloc; A=$1 V=$2 Sw=$3
  for trio in base lance; do
    run LOCAL "$trio" $A $V $Sw "${LOCAL_ARGS[@]}"
    run S3    "$trio" $A $V $Sw "${S3_ARGS[@]}"
    run MIXED "$trio" $A $V $Sw "${MIXED_ARGS[@]}"
  done
done
echo "=== MATRIX DONE ($RES) ===" | tee -a "$RES"

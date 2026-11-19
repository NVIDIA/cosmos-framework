#!/usr/bin/env bash
# Original whole-node GB200 allocation: four GPUs, no separate GPU/GRES flags.
#SBATCH --job-name=failuregen-notebook
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --no-requeue

set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
: "${ROOT:?Set ROOT to a shared workspace mounted at the same path in the container}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COOKBOOK=${COOKBOOK:-$(cd "$SCRIPT_DIR/.." && pwd)}
FRAMEWORK=${FRAMEWORK:-$ROOT/repos/cosmos-framework}
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to a Cosmos-compatible container image}"
PYTHON_BIN=${PYTHON_BIN:-/opt/cosmos/venv/bin/python}
GENERATION_WORK_DIR=${GENERATION_WORK_DIR:-$ROOT/outputs/failuregen-work}
export FAILUREGEN_FRAMEWORK=${FAILUREGEN_FRAMEWORK:-$FRAMEWORK}
export FAILUREGEN_DATASET=${FAILUREGEN_DATASET:-$GENERATION_WORK_DIR/dataset}
export FAILUREGEN_PREPARED=${FAILUREGEN_PREPARED:-$GENERATION_WORK_DIR/prepared}
export FAILUREGEN_RUN_DIR=${FAILUREGEN_RUN_DIR:-$GENERATION_WORK_DIR/training/cosmos3_yam/fd_id/yam_eval_eef_gb200_v6}
export FAILUREGEN_CHECKPOINT=${FAILUREGEN_CHECKPOINT:-$FAILUREGEN_RUN_DIR/checkpoints/iter_000001243}
export FAILUREGEN_OUTPUT=${FAILUREGEN_OUTPUT:-$ROOT/outputs/failuregen-notebook/job$SLURM_JOB_ID}
JUPYTER_PORT=${JUPYTER_PORT:-8888}
JUPYTER_IP=${JUPYTER_IP:-127.0.0.1}
[[ $JUPYTER_PORT =~ ^[0-9]+$ ]] && (( JUPYTER_PORT >= 1024 && JUPYTER_PORT <= 65535 )) || {
    echo 'JUPYTER_PORT must be between 1024 and 65535' >&2; exit 1;
}
export ROOT COOKBOOK FRAMEWORK CONTAINER_IMAGE PYTHON_BIN GENERATION_WORK_DIR JUPYTER_PORT JUPYTER_IP

if [[ ${1:-} != --in-container ]]; then
    [[ ${SLURM_NNODES:-1} == 1 ]] || { echo 'Request exactly one node' >&2; exit 1; }
    # The notebook validates mode-specific data/model inputs. FD training can
    # start from base weights without an adapted run or prepared bundle yet.
    for path in "$COOKBOOK/src/failuregen/workflow.py" "$COOKBOOK/scripts/slurm_notebook.sbatch" \
        "$FAILUREGEN_FRAMEWORK/cosmos_framework/inference/args.py"; do
        [[ -r $path ]] || { echo "Missing input: $path" >&2; exit 1; }
        [[ $path == "$ROOT/"* ]] || { echo "Input must be under the mounted ROOT: $path" >&2; exit 1; }
    done
    [[ $FAILUREGEN_OUTPUT == "$ROOT/"* ]] || { echo 'Output must be under ROOT' >&2; exit 1; }
    if [[ $CONTAINER_IMAGE == /* ]]; then
        [[ -r $CONTAINER_IMAGE ]] || { echo "Missing container: $CONTAINER_IMAGE" >&2; exit 1; }
    fi
    mkdir -p "$ROOT/.cache/huggingface"
    payload=(env)
    for name in ROOT COOKBOOK FRAMEWORK CONTAINER_IMAGE PYTHON_BIN GENERATION_WORK_DIR JUPYTER_PORT JUPYTER_IP \
        FAILUREGEN_FRAMEWORK FAILUREGEN_DATASET FAILUREGEN_PREPARED FAILUREGEN_RUN_DIR \
        FAILUREGEN_CHECKPOINT FAILUREGEN_OUTPUT SLURM_JOB_ID; do
        payload+=("$name=${!name}")
    done
    echo "Notebook job $SLURM_JOB_ID on $(hostname); port $JUPYTER_PORT"
    exec srun --nodes=1 --ntasks=1 --cpu-bind=none --kill-on-bad-exit=1 \
        --container-image="$CONTAINER_IMAGE" --container-mounts="$ROOT:$ROOT" \
        --container-workdir="$COOKBOOK" --no-container-entrypoint \
        "${payload[@]}" bash "$COOKBOOK/scripts/slurm_notebook.sbatch" --in-container
fi

unset PYTHONHOME VIRTUAL_ENV COSMOS_DEVICE
if [[ -f /opt/cosmos/activate.sh ]]; then source /opt/cosmos/activate.sh; fi
export LD_LIBRARY_PATH=''
export PYTHONPATH="$COOKBOOK/src:$FAILUREGEN_FRAMEWORK"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
export HF_HOME="$ROOT/.cache/huggingface"
# Per-job kernel registration ensures the notebook uses this container's CUDA Python.
export JUPYTER_DATA_DIR="$ROOT/.cache/failuregen-jupyter/job$SLURM_JOB_ID/data"
export JUPYTER_RUNTIME_DIR="$ROOT/.cache/failuregen-jupyter/job$SLURM_JOB_ID/runtime"
mkdir -p "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR"
"$PYTHON_BIN" - <<'PY'
import importlib.util
import shutil
import torch
missing = [name for name in ('jupyterlab', 'ipykernel', 'av', 'PIL', 'numpy', 'pyarrow', 'yaml')
           if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Container is missing {missing}. See docs/slurm-notebook.md for notebook image setup.")
if torch.cuda.device_count() != 4:
    raise SystemExit('The original launch expects four visible GPUs in one Slurm task')
for index in range(4):
    props = torch.cuda.get_device_properties(index)
    if props.major < 10:
        raise SystemExit(f'Expected Blackwell GPU {index}; found {props.name}')
    print(f'GPU {index}: {props.name}, {props.total_memory / 2**30:.1f} GiB', flush=True)
if shutil.which('ffmpeg') is None:
    raise SystemExit('ffmpeg is required in the container')
PY
"$PYTHON_BIN" -m ipykernel install --user --name failuregen-cluster --display-name 'FailureGen Cluster'
echo "Jupyter compute node: $(hostname)"
echo "Listen address: $JUPYTER_IP:$JUPYTER_PORT; connect using the SSH tunnel in docs/slurm-notebook.md"
cd "$COOKBOOK"
exec "$PYTHON_BIN" -m jupyterlab --no-browser --ip="$JUPYTER_IP" --port="$JUPYTER_PORT" \
    --ServerApp.port_retries=0 --ServerApp.root_dir="$ROOT" --ServerApp.allow_root=True

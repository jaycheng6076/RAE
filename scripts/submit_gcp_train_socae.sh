#!/bin/bash
#SBATCH --job-name=ra_training
#SBATCH --partition=a4
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1          # Crucial: 1 task per node for torchrun
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64           # Increased: ~8 CPUs per GPU
#SBATCH --mem=0                      # Requests all memory on the node
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
export EXPERIMENT_NAME="socae"
# -----------------------------------------------------------------------------
# Configuration & Defaults
# -----------------------------------------------------------------------------
MODE=${MODE:-""}
CONFIG=${CONFIG:-""}
DATA_PATH=${DATA_PATH:-"/home/jianpengcheng_meta_com/imagenet/train"}
RESULTS_DIR=${RESULTS_DIR:-"results"}
SAMPLE_DIR=${SAMPLE_DIR:-"samples"}
IMAGE_SIZE=${IMAGE_SIZE:-256}
PRECISION=${PRECISION:-"fp32"}
CKPT=${CKPT:-""}
NUM_GPUS=8  # Matches --gpus-per-node
WANDB_ENABLED=${WANDB_ENABLED:-"true"}
LABEL_SAMPLING=${LABEL_SAMPLING:-"equal"}

# Validation
if [[ -z "$MODE" || -z "$CONFIG" ]]; then
    echo "ERROR: MODE and CONFIG are required exports."
    exit 1
fi

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
# Find Conda (more robust)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi

conda activate rae || { echo "ERROR: Conda environment 'rae' not found."; exit 1; }

# WandB Setup (Avoid hardcoding keys in scripts if possible; use 'wandb login')
export WANDB_API_KEY="758ed475ca16d6435193b0fd034dce7adceb85ab"
export WANDB_ENTITY="jianpengcheng"
export ENTITY="jianpengcheng"
export WANDB_PROJECT="socae"
export PROJECT="socae"

# -----------------------------------------------------------------------------
# Distributed Network Discovery
# -----------------------------------------------------------------------------
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export WORLD_SIZE=$((SLURM_JOB_NUM_NODES * NUM_GPUS))



echo "Master Node: $MASTER_ADDR : $MASTER_PORT"
echo "Total GPUs:  $WORLD_SIZE"

# -----------------------------------------------------------------------------
# Build Command
# -----------------------------------------------------------------------------
# We use --rdzv_backend=c10d for both single and multi-node for consistency
TORCHRUN_ARGS="--nnodes=$SLURM_JOB_NUM_NODES \
               --nproc_per_node=$NUM_GPUS \
               --rdzv_id=$SLURM_JOB_ID \
               --rdzv_backend=c10d \
               --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"

OPTIONAL_ARGS=""
[[ -n "$CKPT" ]] && OPTIONAL_ARGS="$OPTIONAL_ARGS --ckpt $CKPT"
[[ "$WANDB_ENABLED" == "true" ]] && OPTIONAL_ARGS="$OPTIONAL_ARGS --wandb"

case "$MODE" in
    train_stage1)
        SCRIPT="src/train_stage1.py --config $CONFIG --data-path $DATA_PATH --results-dir $RESULTS_DIR/stage1 --image-size $IMAGE_SIZE --precision $PRECISION $OPTIONAL_ARGS"
        ;;
    train_stage2)
        SCRIPT="src/train.py --config $CONFIG --data-path $DATA_PATH --results-dir $RESULTS_DIR/stage2 --precision $PRECISION $OPTIONAL_ARGS"
        ;;
    train_socae)
        SCRIPT="src/train_socae.py --config $CONFIG --data-path $DATA_PATH --results-dir $RESULTS_DIR/socae --image-size $IMAGE_SIZE --precision $PRECISION $OPTIONAL_ARGS"
        ;;
    sample)
        SCRIPT="src/sample_ddp.py --config $CONFIG --sample-dir $SAMPLE_DIR --precision $PRECISION --label-sampling $LABEL_SAMPLING"
        ;;
    stage1_sample)
        SCRIPT="src/stage1_sample_ddp.py --config $CONFIG --sample-dir $SAMPLE_DIR --precision $PRECISION --data-path $DATA_PATH"
        ;;
    *) echo "Unknown mode: $MODE"; exit 1 ;;
esac

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
echo "Launching on $SLURM_JOB_NUM_NODES nodes..."

# Prevent Python from looking at ~/.local/lib which causes version conflicts
export PYTHONNOUSERSITE=1

# Use srun to trigger the job.
# We call 'python -m torch.distributed.run' to ensure the conda-python is used.
srun bash -c "
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate rae
    export PYTHONNOUSERSITE=1
    echo '=== Python path ===' && which python
    echo '=== Conda env ===' && conda info --envs
    echo '=== PyTorch check ===' && python -c 'import torch; print(torch.__file__)'
    echo '=== LD_LIBRARY_PATH ===' && echo \$LD_LIBRARY_PATH
    python -m torch.distributed.run $TORCHRUN_ARGS $SCRIPT
"

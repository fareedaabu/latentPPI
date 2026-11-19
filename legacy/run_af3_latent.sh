#!/bin/bash
#SBATCH --job-name=stop
#SBATCH --partition=your_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1

## SBATCH --constraint="L40|A100"
#SBATCH --output=slurm_logs/stop_%A_%a.out
#SBATCH --error=slurm_logs/stop_%A_%a.err

source /etc/profile.d/modules.sh

module purge
module load singularity



# Slurm spools the script: export AF3_SCRIPT_DIR when submitting outside the repo.
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    SCRIPT_DIR=${AF3_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
else
    SCRIPT_DIR=${AF3_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
fi

AF3_IMAGE=${AF3_IMAGE:-/path/to/af3_latentppi_sf_v303.sif}
AF3_MODEL_PARAMETERS_DIR=${AF3_MODEL_PARAMETERS_DIR:-/path/to/af3/weights}
AF3_DATABASES_DIR=${AF3_DATABASES_DIR:-/path/to/af3/databases}
AF3_SCORER_DIR=${AF3_SCORER_DIR:-$SCRIPT_DIR/models}
AF3_INPUT_DIR=${AF3_INPUT_DIR:-$SCRIPT_DIR/dataset_build/ppi_dataset/inputs/jsons}
AF3_OUTPUT_DIR=${AF3_OUTPUT_DIR:-$SCRIPT_DIR/stopping/af3_predictions}
AF3_MSA_DIR=${AF3_MSA_DIR:-$SCRIPT_DIR/dataset_build/ppi_dataset/inputs/msa_cache}

# Extract input and output directories from the command file
CMD_FILE=${1:-${CMD_FILE:-stop_cmds.txt}}
if [[ ! -f "$CMD_FILE" ]]; then
  echo "Command file not found: $CMD_FILE"
  exit 1
fi
# Zero-based arrays use OFFSET=1; set OFFSET=0 for one-based arrays.
OFFSET=${OFFSET:-1}
RAW_CMD=$(sed -n "$((SLURM_ARRAY_TASK_ID + OFFSET))p" "$CMD_FILE")

# Process a single command from command file based on array task ID
if [[ -z "$RAW_CMD" ]]; then
    echo "No command found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $CMD_FILE"
    exit 1
fi

# Extract host paths from the command line before translating them to the
# paths exposed inside the Singularity container.
JSON_PATH=$(echo "$RAW_CMD" | sed -n 's/.*--json_path=\([^ ]*\).*/\1/p')
JSON_NAME=$(basename "$JSON_PATH")

# switch to read from af3_cmd name chosen through SLURM_ARRAY_TASK_ID
JSON_NAME=${JSON_NAME:-"test.json"}


mkdir -p "$AF3_OUTPUT_DIR"

# Optional Singularity argument: --env LATENTPPI_SAVE_PRE_PAIRFORMER=1
singularity exec --nv \
    --bind "$AF3_INPUT_DIR:/root/af_input" \
    --bind "$AF3_OUTPUT_DIR:/root/af_output" \
    --bind "$AF3_MODEL_PARAMETERS_DIR:/root/models" \
    --bind "$AF3_DATABASES_DIR:/root/public_databases" \
    --bind "$AF3_MSA_DIR:/root/msa_cache" \
    --bind "$AF3_SCORER_DIR:/root/scorer" \
    --env LATENTPPI_ENABLE=1 \
    --env LATENTPPI_MODEL_PATH=/root/scorer/latentppi_max_ipSAE_d0dom_model.pt \
    --env LATENTPPI_THRESHOLD=0.9 \
    --env XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter \
    "$AF3_IMAGE" \
    uv run python3 /app/alphafold/run_alphafold.py \
        --flash_attention_implementation=xla \
        --norun_data_pipeline \
        --json_path=/root/af_input/${JSON_NAME} \
        --model_dir=/root/models \
        --db_dir=/root/public_databases \
        --output_dir=/root/af_output \
        --num_diffusion_samples=1 \
        --force_output_dir \
        --num_recycles=1 
        # --embedding_recycle_index=0
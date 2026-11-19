#!/bin/bash
#SBATCH --job-name=screen
#SBATCH --partition=your_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --output=slurm_logs/af3_%A_%a.out
#SBATCH --error=slurm_logs/af3_%A_%a.err

# One command / protein pair per array task; see AF3_PIPELINE.md for usage.
# Create slurm_logs BEFORE sbatch. Array indices are 1-based (+ OFFSET).
set -Eeuo pipefail

echo "Running on $(hostname)"

RUN_DIR=${SLURM_SUBMIT_DIR:-$PWD}

# Slurm spools this script elsewhere: use the submit directory / explicit repo
# path to find Python helpers, rather than relying on BASH_SOURCE under Slurm.
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    SCRIPT_DIR=${AF3_SCRIPT_DIR:-$RUN_DIR}
else
    SCRIPT_DIR=${AF3_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
fi
AF3_SCRIPT_DIR=$SCRIPT_DIR


# ==================== EDIT USER SETTINGS HERE ====================
# Defaults can also be overridden with exports or sbatch --export.
# Empty AF3_OUTPUT_DIR uses the output directory in each AF3 command.
CMD_FILE=${1:-${CMD_FILE:-af3_cmds.txt}}
AF3_INPUT_DIR=${AF3_INPUT_DIR:-$AF3_SCRIPT_DIR/dataset_build/ppi_dataset/inputs/jsons}
AF3_OUTPUT_DIR=${AF3_OUTPUT_DIR:-}
SUMMARY_TSV=${SUMMARY_TSV:-$AF3_SCRIPT_DIR/dataset_build/ppi_dataset/dataset_summary.tsv}  # Empty: dataset_summary.tsv beside CMD_FILE.
SHARD_DIR=${SHARD_DIR:-$RUN_DIR/h5_shards}
IPSAE_OUTROOT=${IPSAE_OUTROOT:-$RUN_DIR/ipsae_scores_af3}
IPSAE_SUMMARY=${IPSAE_SUMMARY:-$RUN_DIR/ipsae_summary_af3.csv}

AF3_IMAGE=${AF3_IMAGE:-/path/to/af3_latentppi_sf_v303.sif}
AF3_MODEL_PARAMETERS_DIR=${AF3_MODEL_PARAMETERS_DIR:-/path/to/af3/weights}
AF3_DATABASES_DIR=${AF3_DATABASES_DIR:-/path/to/af3/databases}
PYTHON_BIN=${PYTHON_BIN:-python}

RUN_IPSAE_PARSE=${RUN_IPSAE_PARSE:-1}
SAVE_EMBEDDINGS=${SAVE_EMBEDDINGS:-0}
RUN_POOL=${RUN_POOL:-0}
RUN_COMPRESS=${RUN_COMPRESS:-1}
POOL_SEED=${POOL_SEED:-1}
KEEP_NPZ=${KEEP_NPZ:-0}
COMPRESS_EMBEDDINGS=${COMPRESS_EMBEDDINGS:-0}
ZSTD_LEVEL=${ZSTD_LEVEL:-19}
OFFSET=${OFFSET:-0}          # Added to the Slurm array task number.
DRY_RUN=${DRY_RUN:-0}        # Print the command without running the pipeline.
# ==================== END USER SETTINGS ==========================

source "$SCRIPT_DIR/af3_pipeline_helpers.sh"

# 1. Resolve this task, check dependencies, and lock its output directory.
prepare_task
printf 'Task %s, command line %s, system %s\n' "$TASK_ID" "$LINE_NUMBER" "$JOB_NAME"
printf 'Prediction folder: %s\nShards: %s/{prepairformer,recycle_%s}/\nSummary: %s\n' \
    "$AF3_JOB_SUBDIR" "$SHARD_DIR" "$RECYCLE_COUNT" "$SUMMARY_TSV"
if (( DRY_RUN == 0 )); then
    check_dependencies
    lock_task
fi
source "$SCRIPT_DIR/run_env_capture.sh"

# 2. Run AlphaFold. Scientific arguments come from the selected command line.
STAGE=fold
echo "[fold] Starting AF3 for $JOB_NAME"
start_time=$(date +%s)
EXIT_CODE=0
GPU_ARGS=(--nv)
if [[ ${AF3_CPU_ONLY:-0} == 1 ]]; then
    GPU_ARGS=()
fi
run_af3_command "$SINGULARITY_BIN" exec "${GPU_ARGS[@]}" "${BIND_ARGS[@]}" \
    --env LATENTPPI_ENABLE=0 --env "LATENTPPI_SAVE_PRE_PAIRFORMER=$SAVE_EMBEDDINGS" \
    --env XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter \
    "$AF3_IMAGE" uv run python3 /app/alphafold/run_alphafold.py \
    "${AF3_ARGS[@]}" --flash_attention_implementation=xla \
    "--json_path=$JSON_PATH" "--output_dir=$AF3_OUTPUT_DIR" \
    --model_dir=/root/models --db_dir=/root/public_databases \
    --run_inference "$SAVE_EMBEDDINGS_FLAG" --nocompress_large_output_files --force_output_dir
printf '%s: %ss (exit=%s)\n' "$AF3_JOB_SUBDIR" "$(( $(date +%s) - start_time ))" "$EXIT_CODE" >> "$RUN_DIR/af3_job_durations.txt"
(( EXIT_CODE == 0 )) || exit "$EXIT_CODE"
resolve_prediction_folder

# 3. Score the predicted structure with ipSAE.
STAGE=ipsae
if (( RUN_IPSAE_PARSE )); then
    echo "[ipsae] Scoring $AF3_JOB_SUBDIR"
    # The parser reports success on zero discoveries; require actual fold files.
    [[ -f "$AF3_JOB_SUBDIR/${PREDICTION_NAME}_model.cif" && -f "$AF3_JOB_SUBDIR/${PREDICTION_NAME}_confidences.json" ]] || fail 'Missing top-ranked structure/confidences for ipSAE'
    "$IPSAE_PYTHON" "$IPSAE_PARSER_PY" \
        --mode af3 --root "$AF3_JOB_SUBDIR" --outroot "$IPSAE_OUTROOT" \
        --summary "$IPSAE_SUMMARY" \
        --failed-log "$IPSAE_FAILED_LOG" \
        --done-ledger "$IPSAE_DONE_LEDGER" \
        --fail-ledger "$IPSAE_FAIL_LEDGER" \
        --ipsae-py "$IPSAE_PY" --python "$IPSAE_PYTHON" \
        --pae "$IPSAE_PAE" --dist "$IPSAE_DIST" \
        --verbose --input-json "$JSON_PATH" --prefer-seed-samples
else
    echo '[ipsae] Disabled (RUN_IPSAE_PARSE=0)'
fi

# 4. Pool prepairformer and final-recycle embeddings into separate shards.
STAGE=pool
if (( RUN_POOL )); then
    echo "[pool] Pooling seed $POOL_SEED from $AF3_JOB_SUBDIR"
    pool_and_publish_stages
else
    echo '[pool] Disabled (RUN_POOL=0)'
fi

# 5. Compress outputs, verifying each archive before deleting its source.
STAGE=compress
if (( RUN_COMPRESS )); then
    echo "[compress] Compressing files in $AF3_JOB_SUBDIR"
    compress_outputs
else
    echo '[compress] Disabled (RUN_COMPRESS=0)'
fi
STAGE=done
printf 'Completed %s (ipSAE=%s pool=%s compress=%s)\n' "$JOB_NAME" "$RUN_IPSAE_PARSE" "$RUN_POOL" "$RUN_COMPRESS"

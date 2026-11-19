#!/bin/bash
#SBATCH --job-name=screen
#SBATCH --partition=your_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4


#SBATCH --output=slurm_logs/stop_pipeline_%A_%a.out
#SBATCH --error=slurm_logs/stop_pipeline_%A_%a.err

# When submitting outside the repository root, export AF3_SCRIPT_DIR first.
# Example from stopping/: sbatch --array=1-61%4 ../run_af3_latent_compress_pool.sh stop_cmds.txt
# Array indices are one-based, matching run_af3_compress_pool.sh.
set -Eeuo pipefail

echo "Running on $(hostname)"

if [[ ${DRY_RUN:-0} == 1 || -z ${SLURM_JOB_ID:-} ]]; then
    RUN_DIR=$PWD
else
    RUN_DIR=${SLURM_SUBMIT_DIR:-$PWD}
fi
# Slurm spools the script: use AF3_SCRIPT_DIR or submit from the repository root.
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    SCRIPT_DIR=${AF3_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
else
    SCRIPT_DIR=${AF3_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
fi
AF3_SCRIPT_DIR=$SCRIPT_DIR

CMD_FILE=${1:-${CMD_FILE:-stop_cmds.txt}}
CMD_BASE_DIR=${CMD_BASE_DIR:-$RUN_DIR}
AF3_INPUT_DIR=${AF3_INPUT_DIR:-$SCRIPT_DIR/dataset_build/ppi_dataset/inputs/jsons}
AF3_MSA_DIR=${AF3_MSA_DIR:-$SCRIPT_DIR/dataset_build/ppi_dataset/inputs/msa_cache}
AF3_OUTPUT_DIR=${AF3_OUTPUT_DIR:-$SCRIPT_DIR/stopping/af3_predictions}
SUMMARY_TSV=${SUMMARY_TSV:-$SCRIPT_DIR/dataset_build/ppi_dataset/dataset_summary.tsv}
SHARD_DIR=${SHARD_DIR:-$RUN_DIR/h5_shards}
IPSAE_OUTROOT=${IPSAE_OUTROOT:-$RUN_DIR/ipsae_scores_af3}
IPSAE_SUMMARY=${IPSAE_SUMMARY:-$RUN_DIR/ipsae_summary_af3.csv}

AF3_IMAGE=${AF3_IMAGE:-/path/to/af3_latentppi_sf_v303.sif}
AF3_MODEL_PARAMETERS_DIR=${AF3_MODEL_PARAMETERS_DIR:-/path/to/af3/weights}
AF3_DATABASES_DIR=${AF3_DATABASES_DIR:-/path/to/af3/databases}
# YAML jobs export their scorer location from the run snapshot. Use the
# bundled checkpoint when a standalone invocation supplies neither override.
if [[ -z ${AF3_SCORER_DIR:-} && -z ${AF3_SCORER_FILENAME:-} ]]; then
    AF3_SCORER_DIR=$SCRIPT_DIR/models/checkpoint
    AF3_SCORER_FILENAME=latentppi_max_ipSAE_d0dom__emb-r1_score-r1_model.pt
fi
AF3_SCORER_DIR=${AF3_SCORER_DIR:-$SCRIPT_DIR/models}
AF3_SCORER_FILENAME=${AF3_SCORER_FILENAME:-latentppi_max_ipSAE_d0dom_model.pt}
LATENTPPI_THRESHOLD=${LATENTPPI_THRESHOLD:-0.9}
PYTHON_BIN=${PYTHON_BIN:-python}

RUN_IPSAE_PARSE=${RUN_IPSAE_PARSE:-1}
SAVE_EMBEDDINGS=${SAVE_EMBEDDINGS:-0}
RUN_POOL=${RUN_POOL:-0}
RUN_COMPRESS=${RUN_COMPRESS:-1}
POOL_SEED=${POOL_SEED:-1}
KEEP_NPZ=${KEEP_NPZ:-0}
COMPRESS_EMBEDDINGS=${COMPRESS_EMBEDDINGS:-0}
ZSTD_LEVEL=${ZSTD_LEVEL:-19}
OFFSET=${OFFSET:-0}
DEFAULT_NUM_RECYCLES=${DEFAULT_NUM_RECYCLES:-1}
DRY_RUN=${DRY_RUN:-0}

source "$SCRIPT_DIR/af3_pipeline_helpers.sh"

prepare_task
# The stop commands already specify this flag; put it in the final command
# exactly once while enforcing the working latent run's pipeline setting.
LATENT_AF3_ARGS=()
for arg in "${AF3_ARGS[@]}"; do
    [[ $arg == --norun_data_pipeline ]] || LATENT_AF3_ARGS+=("$arg")
done
printf 'Task %s, command line %s, system %s\n' "$TASK_ID" "$LINE_NUMBER" "$JOB_NAME"
printf 'Prediction folder: %s\nShards: %s/{prepairformer,recycle_%s}/\nSummary: %s\n' \
    "$AF3_JOB_SUBDIR" "$SHARD_DIR" "$RECYCLE_COUNT" "$SUMMARY_TSV"
if (( DRY_RUN == 0 )); then
    check_dependencies
    [[ -f "$AF3_SCORER_DIR/$AF3_SCORER_FILENAME" ]] || fail "Missing LatentPPI scorer: $AF3_SCORER_DIR/$AF3_SCORER_FILENAME"
    lock_task
fi
source "$SCRIPT_DIR/run_env_capture.sh"

# Keep the successful latent launcher's image, scorer, XLA, and inference flags.
# The helper resolves the JSON name and preserves scientific command-file flags.
STAGE=fold
echo "[fold] Starting latent AF3 for $JOB_NAME"
start_time=$(date +%s)
EXIT_CODE=0
GPU_ARGS=(--nv)
if [[ ${AF3_CPU_ONLY:-0} == 1 ]]; then
    GPU_ARGS=()
fi
run_af3_command "$SINGULARITY_BIN" exec "${GPU_ARGS[@]}" "${BIND_ARGS[@]}" \
    --bind "$AF3_SCORER_DIR:/root/scorer:ro" \
    --env LATENTPPI_ENABLE=1 \
    --env "LATENTPPI_SAVE_PRE_PAIRFORMER=$SAVE_EMBEDDINGS" \
    --env "LATENTPPI_MODEL_PATH=/root/scorer/$AF3_SCORER_FILENAME" \
    --env "LATENTPPI_THRESHOLD=$LATENTPPI_THRESHOLD" \
    --env XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter \
    "$AF3_IMAGE" uv run python3 /app/alphafold/run_alphafold.py \
    "${LATENT_AF3_ARGS[@]}" \
    --flash_attention_implementation=xla --norun_data_pipeline \
    "--json_path=$JSON_PATH" "--output_dir=$AF3_OUTPUT_DIR" \
    --model_dir=/root/models --db_dir=/root/public_databases \
    --run_inference "$SAVE_EMBEDDINGS_FLAG" --nocompress_large_output_files --force_output_dir
printf '%s: %ss (exit=%s)\n' "$AF3_JOB_SUBDIR" "$(( $(date +%s) - start_time ))" "$EXIT_CODE" >> "$RUN_DIR/af3_job_durations.txt"
(( EXIT_CODE == 0 )) || exit "$EXIT_CODE"
resolve_prediction_folder

# A successful scorer gate stops before diffusion: there is no structure to
# score with ipSAE, and only the pre-Pairformer embedding is exported.
GATED_EARLY_EXIT=0
TIMINGS=$AF3_JOB_SUBDIR/prediction_timings.jsonl
if [[ -f $TIMINGS ]]; then
    GATED_EARLY_EXIT=$("$PYTHON_BIN" - "$TIMINGS" "$PREDICTION_NAME" "$POOL_SEED" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    lines = [line for line in handle if line.strip()]
record = json.loads(lines[-1]) if lines else {}
if record and (record.get('input_name', '').lower() != sys.argv[2].lower()
               or str(record.get('seed')) != sys.argv[3]):
    raise SystemExit('Latest timing record belongs to another input or seed')
print(int(record.get('latentppi_scorer_gate_triggered') is True
          and record.get('latentppi_scorer_gate_status') == 'ok'))
PY
    )
fi

STAGE=ipsae
if (( GATED_EARLY_EXIT )); then
    echo '[ipsae] Scorer gate skipped diffusion; no structure exists to score'
elif (( RUN_IPSAE_PARSE )); then
    echo "[ipsae] Scoring $AF3_JOB_SUBDIR"
    [[ -f "$AF3_JOB_SUBDIR/${PREDICTION_NAME}_model.cif" && \
       -f "$AF3_JOB_SUBDIR/${PREDICTION_NAME}_confidences.json" ]] || \
        fail 'Missing top-ranked structure/confidences for ipSAE'
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

STAGE=pool
if (( RUN_POOL )); then
    if (( GATED_EARLY_EXIT )); then
        POOL_STAGES=(prepairformer)
        echo "[pool] Scorer gate: pooling pre-Pairformer embedding for seed $POOL_SEED"
    else
        echo "[pool] Pooling both embedding stages for seed $POOL_SEED"
    fi
    pool_and_publish_stages
else
    echo '[pool] Disabled (RUN_POOL=0)'
fi

STAGE=compress
if (( GATED_EARLY_EXIT )); then
    echo '[compress] Scorer gate produced no structure; skipping structure compression'
elif (( RUN_COMPRESS )); then
    echo "[compress] Compressing $AF3_JOB_SUBDIR"
    compress_outputs
else
    echo '[compress] Disabled (RUN_COMPRESS=0)'
fi
STAGE=done
printf 'Completed %s (gate=%s requested_ipSAE=%s requested_pool=%s requested_compress=%s)\n' \
    "$JOB_NAME" "$GATED_EARLY_EXIT" "$RUN_IPSAE_PARSE" "$RUN_POOL" "$RUN_COMPRESS"

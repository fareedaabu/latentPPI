#!/bin/bash
# Internal helpers for run_af3_compress_pool.sh. Source after its settings block.
# Functions share the runner's settings and task state. Call directly so Bash
# errexit remains active; do not invoke these functions as if/|| conditions.

# Compatibility overrides for alternate tools/environments; everyday settings
# belong in the runner. Keep these here so its settings block stays focused.
IPSAE_PYTHON=${IPSAE_PYTHON:-$PYTHON_BIN}
POOL_PYTHON=${POOL_PYTHON:-$PYTHON_BIN}
IPSAE_PARSER_PY=${IPSAE_PARSER_PY:-$SCRIPT_DIR/parse_ipsae.py}
IPSAE_PY=${IPSAE_PY:-$SCRIPT_DIR/ipsae.py}
POOL_PY=${POOL_PY:-$SCRIPT_DIR/pool_embeddings.py}
SINGULARITY_BIN=${SINGULARITY_BIN:-singularity}
ZSTD_BIN=${ZSTD_BIN:-zstd}
ZSTD_EMB_LEVEL=${ZSTD_EMB_LEVEL:-3}
ZSTD_THREADS=${ZSTD_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
IPSAE_FAILED_LOG=${IPSAE_FAILED_LOG:-$IPSAE_OUTROOT/failed.txt}
IPSAE_DONE_LEDGER=${IPSAE_DONE_LEDGER:-$IPSAE_OUTROOT/done_ledger.txt}
IPSAE_FAIL_LEDGER=${IPSAE_FAIL_LEDGER:-$IPSAE_OUTROOT/fail_ledger.txt}
IPSAE_PAE=${IPSAE_PAE:-15}
IPSAE_DIST=${IPSAE_DIST:-15}

fail() { echo "ERROR: $*" >&2; exit 1; }
STAGE=setup
RAW_CMD=
PARSED_FILE=
SHARD_TMP=
SHARD_TEMPS=()
COMPRESS_TMP=
FILE_LIST=
cleanup() {
    local status=$?
    trap - EXIT
    for temporary in "$PARSED_FILE" "$SHARD_TMP" "$COMPRESS_TMP" "$FILE_LIST" "${SHARD_TEMPS[@]}"; do
        [[ -z "$temporary" ]] || rm -f -- "$temporary"
    done
    if (( status != 0 )); then
        printf 'Failed stage=%s task=%s exit=%s\n' "$STAGE" "${SLURM_ARRAY_TASK_ID:-1}" "$status" >&2
        printf '%s\n' "$RAW_CMD" >> "$RUN_DIR/af3_failed_jobs.txt"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

prepare_task() {
    # Older Slurm-captured runners always saved embeddings and do not export
    # this setting. Preserve that behavior when they source this live helper.
    SAVE_EMBEDDINGS=${SAVE_EMBEDDINGS:-1}
    for switch in RUN_IPSAE_PARSE SAVE_EMBEDDINGS RUN_POOL RUN_COMPRESS KEEP_NPZ COMPRESS_EMBEDDINGS DRY_RUN; do
        [[ ${!switch} == 0 || ${!switch} == 1 ]] || fail "$switch must be 0 or 1"
    done
    # Both runners derive the AF3 CLI flag from the exported YAML setting.
    SAVE_EMBEDDINGS_FLAG=--nosave_embeddings
    if [[ $SAVE_EMBEDDINGS == 1 ]]; then
        SAVE_EMBEDDINGS_FLAG=--save_embeddings
    elif [[ $RUN_POOL == 1 ]]; then
        fail 'RUN_POOL=1 requires SAVE_EMBEDDINGS=1'
    fi
    [[ $POOL_SEED =~ ^[0-9]+$ ]] || fail 'POOL_SEED must be a nonnegative integer'
    [[ -f "$CMD_FILE" ]] || fail "Command file not found: $CMD_FILE"
    TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
    OFFSET=${OFFSET:-0}
    [[ $TASK_ID =~ ^[0-9]+$ && $OFFSET =~ ^[0-9]+$ ]] || fail 'Task ID and OFFSET must be nonnegative integers'
    LINE_NUMBER=$((10#$TASK_ID + 10#$OFFSET))
    (( LINE_NUMBER > 0 )) || fail 'Command line number must be >= 1 (use --array=1-N)'
    RAW_CMD=$(sed -n "${LINE_NUMBER}p" "$CMD_FILE")
    [[ -n "$RAW_CMD" ]] || fail "No command on line $LINE_NUMBER in $CMD_FILE"
    CMD_BASE_DIR=${CMD_BASE_DIR:-$(cd -- "$(dirname -- "$CMD_FILE")" && pwd)}
    SUMMARY_TSV=${SUMMARY_TSV:-$CMD_BASE_DIR/dataset_summary.tsv}

    # Tokenize shell quoting without eval; accept --flag=value and --flag value.
    # Preserve scientific flags from the command; replace paths and output controls.
    PARSED_FILE=$(mktemp)
    "$PYTHON_BIN" "$SCRIPT_DIR/parse_af3_command.py" \
        "$RAW_CMD" "$CMD_BASE_DIR" "$AF3_INPUT_DIR" "$AF3_OUTPUT_DIR" > "$PARSED_FILE"
    mapfile -d '' -t PARSED < "$PARSED_FILE"
    JSON_PATH=${PARSED[0]}
    AF3_OUTPUT_DIR=${PARSED[1]}
    JOB_NAME=${PARSED[2]}
    AF3_ARGS=("${PARSED[@]:3}")
    AF3_INPUT_DIR=$(dirname -- "$JSON_PATH")
    INPUT_PARENT=$(dirname -- "$AF3_INPUT_DIR")
    AF3_JOB_SUBDIR=$AF3_OUTPUT_DIR/$JOB_NAME
    # The standard NPZ contains the final recycle embedding. Derive its stage
    # from the inference arguments instead of labeling every run recycle_1.
    RECYCLE_COUNT=
    for ((i=0; i<${#AF3_ARGS[@]}; i++)); do
        case ${AF3_ARGS[i]} in
            --num_recycles=*) RECYCLE_COUNT=${AF3_ARGS[i]#*=} ;;
            --num_recycles) RECYCLE_COUNT=${AF3_ARGS[i+1]:-} ;;
        esac
    done
    if [[ -z $RECYCLE_COUNT && -n ${DEFAULT_NUM_RECYCLES:-} ]]; then
        RECYCLE_COUNT=$DEFAULT_NUM_RECYCLES
        AF3_ARGS+=("--num_recycles=$RECYCLE_COUNT")
    fi
    if (( RUN_POOL )); then
        [[ $RECYCLE_COUNT =~ ^[0-9]+$ ]] || fail 'Pooling requires explicit --num_recycles in the AF3 command'
    fi
    POOL_STAGES=(prepairformer "recycle_${RECYCLE_COUNT}")

    BIND_ARGS=(--bind "$INPUT_PARENT:$INPUT_PARENT:ro"
               --bind "$AF3_OUTPUT_DIR:$AF3_OUTPUT_DIR"
               --bind "$AF3_MODEL_PARAMETERS_DIR:/root/models:ro"
               --bind "$AF3_DATABASES_DIR:/root/public_databases:ro")
    # Optional additional cache outside the inputs tree. Also expose the alias
    # used by older JSONs with /root/msa_cache paths.
    AF3_MSA_DIR=${AF3_MSA_DIR:-$INPUT_PARENT/msa_cache}
    if [[ -d "$AF3_MSA_DIR" ]]; then
        AF3_MSA_DIR=$(cd -- "$AF3_MSA_DIR" && pwd)
        BIND_ARGS+=(--bind "$AF3_MSA_DIR:$AF3_MSA_DIR:ro" --bind "$AF3_MSA_DIR:/root/msa_cache:ro")
    fi

    # Screen JSONs can reference the original proteome A3Ms by absolute path.
    # Mount each external source directory at that same path inside the image.
    "$PYTHON_BIN" - "$JSON_PATH" "$INPUT_PARENT" > "$PARSED_FILE" <<'PY'
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
input_parent = Path(sys.argv[2])
data = json.loads(json_path.read_text())
directories = set()
for entry in data.get("sequences", []):
    protein = entry.get("protein", {})
    for key in ("unpairedMsaPath", "pairedMsaPath"):
        reference = protein.get(key)
        if not reference:
            continue
        path = Path(reference)
        if not path.is_absolute():
            path = json_path.parent / path
        if not path.is_file():
            raise SystemExit(f"MSA referenced by {json_path} is missing: {path}")
        path = path.absolute()
        if path.is_relative_to(input_parent):
            continue
        directories.add(str(path.parent))
for directory in sorted(directories):
    sys.stdout.buffer.write(directory.encode() + b"\0")
PY
    mapfile -d '' -t MSA_SOURCE_DIRS < "$PARSED_FILE"
    for msa_dir in "${MSA_SOURCE_DIRS[@]}"; do
        if [[ $msa_dir != "$AF3_MSA_DIR" ]]; then
            BIND_ARGS+=(--bind "$msa_dir:$msa_dir:ro")
        fi
    done
}

check_dependencies() {
    # Host Python must provide numpy/pandas/h5py; activate your environment before
    # submitting, or set the Python executable variables. No forced conda env.
    if ! command -v singularity >/dev/null 2>&1; then
  if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/lmod.sh \
                "${MODULESHOME:-/nonexistent}/init/bash" \
                /usr/share/modules/init/bash; do
      if [[ -r $init ]]; then
        set +u            # init scripts can reference unset vars; harmless if you don't use set -u
        source "$init"
        break
      fi
    done
  fi
  module load singularity
fi

    command -v "$SINGULARITY_BIN" >/dev/null || fail "Missing $SINGULARITY_BIN"
    [[ -f "$AF3_IMAGE" ]] || fail "Missing AF3 image: $AF3_IMAGE"
    [[ -d "$AF3_MODEL_PARAMETERS_DIR" && -d "$AF3_DATABASES_DIR" ]] || fail 'Missing AF3 weights/databases directory'
    if (( RUN_POOL )); then
        [[ -f "$POOL_PY" && -f "$SUMMARY_TSV" ]] || fail 'Missing pooler or summary TSV; set AF3_SCRIPT_DIR/SUMMARY_TSV'
        "$POOL_PYTHON" -c 'import numpy, pandas, h5py'
    fi
    if (( RUN_IPSAE_PARSE )); then
        [[ -f "$IPSAE_PARSER_PY" && -f "$IPSAE_PY" ]] || fail 'Missing ipSAE scripts; set AF3_SCRIPT_DIR'
        "$IPSAE_PYTHON" -c 'import numpy'
    fi
    if (( RUN_COMPRESS )); then
        command -v "$ZSTD_BIN" >/dev/null || fail "Compression requested but $ZSTD_BIN is unavailable"
    fi
}

lock_task() {
    mkdir -p -- "$AF3_OUTPUT_DIR" "$SHARD_DIR"
    # Serialize duplicate tasks/requeues for this pair, including inference, so
    # they cannot overwrite each other's prediction files or HDF5 shard.
    command -v flock >/dev/null || fail 'flock is required'
    exec 9> "$AF3_OUTPUT_DIR/.${JOB_NAME}.pipeline.lock"
    flock -n 9 || fail "Another task is processing $AF3_JOB_SUBDIR"
}

resolve_prediction_folder() {
    # Standard AF3 lowercases names; this patched image preserves input-name case.
    # Require one match so an old folder with different casing is not used silently.
    mapfile -d '' -t JOB_FOLDERS < <(find "$AF3_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -iname "$JOB_NAME" -print0)
    (( ${#JOB_FOLDERS[@]} == 1 )) || fail "Expected one output folder matching $JOB_NAME; found ${#JOB_FOLDERS[@]} in $AF3_OUTPUT_DIR"
    AF3_JOB_SUBDIR=${JOB_FOLDERS[0]}
    PREDICTION_NAME=$(basename -- "$AF3_JOB_SUBDIR")
    echo "[fold] Finished; using $AF3_JOB_SUBDIR"
}

pool_and_publish_stages() {
    EMBED_DIR=$AF3_JOB_SUBDIR/seed-${POOL_SEED}_embeddings
    NPZ_PATHS=("$EMBED_DIR/${PREDICTION_NAME}_seed-${POOL_SEED}_pre_pairformer_embeddings.npz")
    if (( ${GATED_EARLY_EXIT:-0} == 0 )); then
        NPZ_PATHS+=("$EMBED_DIR/${PREDICTION_NAME}_seed-${POOL_SEED}_embeddings.npz")
    fi
    for npz in "${NPZ_PATHS[@]}"; do
        [[ -f "$npz" ]] || fail "Missing embedding stage: $npz"
    done
    # Finish both temporary shards before publishing either or deleting inputs.
    for i in "${!POOL_STAGES[@]}"; do
        stage_dir=$SHARD_DIR/${POOL_STAGES[i]}
        mkdir -p -- "$stage_dir"
        SHARD_TMP=$(mktemp "$stage_dir/.embeddings_${JOB_NAME}.XXXXXX.partial")
        SHARD_TEMPS+=("$SHARD_TMP")
        rm -- "$SHARD_TMP"
        "$POOL_PYTHON" "$POOL_PY" --af3-out-dir "$AF3_JOB_SUBDIR" \
            --summary-tsv "$SUMMARY_TSV" --hdf5-path "$SHARD_TMP" \
            --seed "$POOL_SEED" --npz-path "${NPZ_PATHS[i]}" --keep-npz
        SHARD_TMP=
    done
    for i in "${!POOL_STAGES[@]}"; do
        SHARD_PATH=$SHARD_DIR/${POOL_STAGES[i]}/embeddings_$JOB_NAME.h5
        mv -f -- "${SHARD_TEMPS[i]}" "$SHARD_PATH"
        echo "[pool] Saved $SHARD_PATH"
    done
    SHARD_TEMPS=()
    # Compression explicitly requested for NPZ implies retaining raw embeddings
    # as archives. Otherwise retain only the pooled vectors unless KEEP_NPZ=1.
    if (( KEEP_NPZ == 0 && COMPRESS_EMBEDDINGS == 0 )); then
        rm -r -- "$AF3_JOB_SUBDIR/seed-${POOL_SEED}_embeddings"
    fi
}

compress_one() {
    local source=$1 level=$2
    # Always replace an old archive: rerunning AF3 can produce new contents.
    # Temp file + test + atomic rename prevents publishing partial archives.
    COMPRESS_TMP=$(mktemp "${source}.XXXXXX.partial")
    "$ZSTD_BIN" -q -f "-$level" "-T${ZSTD_THREADS}" --no-progress -o "$COMPRESS_TMP" -- "$source"
    "$ZSTD_BIN" -q -t -- "$COMPRESS_TMP"
    mv -f -- "$COMPRESS_TMP" "${source}.zst"
    COMPRESS_TMP=
    rm -- "$source"
}
compress_outputs() {
    # Include unprefixed seed/sample filenames. Keep small summary JSONs plain.
    FILE_LIST=$(mktemp)
    find "$AF3_JOB_SUBDIR" -type f \
        \( -name '*_confidences.json' -o -name 'confidences.json' -o -name '*_model.cif' -o -name 'model.cif' \) \
        ! -name '*summary_confidences.json' -print0 > "$FILE_LIST"
    while IFS= read -r -d '' file; do
        compress_one "$file" "$ZSTD_LEVEL"
    done < "$FILE_LIST"
    if (( COMPRESS_EMBEDDINGS )); then
        find "$AF3_JOB_SUBDIR" -type f -path '*/seed-*_embeddings/*.npz' -print0 > "$FILE_LIST"
        while IFS= read -r -d '' file; do
            compress_one "$file" "$ZSTD_EMB_LEVEL"
        done < "$FILE_LIST"
    fi
}

# Print the exact command in both normal and dry-run mode.
run_af3_command() {
    printf 'AF3 command: '; printf '%q ' "$@"; printf '\n'
    (( DRY_RUN == 0 )) || exit 0
    if [[ -n ${LATENTPPI_ATTEMPT_ID:-} ]]; then
        local error_log=$AF3_OUTPUT_DIR/.latentppi_attempts/$LATENTPPI_ATTEMPT_ID.stderr
        "$@" 2> "$error_log" || EXIT_CODE=$?
        cat -- "$error_log" >&2
    else
        "$@" || EXIT_CODE=$?
    fi
}

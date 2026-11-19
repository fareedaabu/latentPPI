#!/bin/bash
#SBATCH --job-name=inputs
#SBATCH --partition=your_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# send email about job start and end
# Set --mail-user on the sbatch command line when notifications are wanted.
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Configuration variables
CONDA_ENV=${CONDA_ENV:-colabfold}
OUTPUT_BASE=${OUTPUT_BASE:-/path/to/screen}
PROTEOME_MSA_DIR=${PROTEOME_MSA_DIR:-/path/to/proteome/msas}
FASTA_FILE=${FASTA_FILE:-your_fasta.fasta}
# Slurm spools the script: export AF3_SCRIPT_DIR when submitting outside the repo.
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    AF3_SCRIPT_DIR=${AF3_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
else
    AF3_SCRIPT_DIR=${AF3_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
fi
AF3_NUM_RECYCLES=${AF3_NUM_RECYCLES:-1}
AF3_NUM_DIFFUSION_SAMPLES=${AF3_NUM_DIFFUSION_SAMPLES:-1}
MAX_IDENTITY=${MAX_IDENTITY:-95.0}
FILTER_NON_STANDARD=${FILTER_NON_STANDARD:-1}
FORCE=${FORCE:-0}
RECREATE_MANIFEST=${RECREATE_MANIFEST:-0}
RECREATE_COMMANDS=${RECREATE_COMMANDS:-0}
SKIP_IF_MSA_EXISTS=${SKIP_IF_MSA_EXISTS:-1}


eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

##################################################
#                       MSA GENERATION
# Generate the query MSA if it is not already in the screen folder.
##################################################                    
echo "Checking if msa exists..."
echo " "
start_time=$(date +%s)
mkdir -p "$OUTPUT_BASE"
QUERY_A3M=$(python "$AF3_SCRIPT_DIR/query_msa.py" "$FASTA_FILE" "$OUTPUT_BASE" --expected)
TARGET_A3M=$(python "$AF3_SCRIPT_DIR/query_msa.py" "$FASTA_FILE" "$OUTPUT_BASE")
if [[ -z "$TARGET_A3M" || $SKIP_IF_MSA_EXISTS == 0 || $FORCE == 1 ]]; then
    echo "Generating query msa ..."
    # A fresh directory prevents ColabFold's cache from defeating FORCE.
    MSA_WORK_DIR=$(mktemp -d "$OUTPUT_BASE/.query-msa.XXXXXX")
    colabfold_batch "$FASTA_FILE" "$MSA_WORK_DIR" --msa-only
    GENERATED_A3M=$(python "$AF3_SCRIPT_DIR/query_msa.py" "$FASTA_FILE" "$MSA_WORK_DIR" --generated)
    [[ -n "$GENERATED_A3M" ]] || { echo "No query A3M generated in $MSA_WORK_DIR" >&2; exit 1; }
    TARGET_A3M=$QUERY_A3M
    cp -- "$GENERATED_A3M" "$TARGET_A3M"
    echo "Done generating msa"
else
    echo "MSA exists. skipping msa generation."
fi
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "MSA generation took $elapsed seconds"

# Get the query A3M written by ColabFold.
[[ -n "$TARGET_A3M" ]] || { echo "No query A3M found in $OUTPUT_BASE" >&2; exit 1; }
echo "target a3m: $TARGET_A3M"
if [[ $(awk '/^>/ {n++} END {print n+0}' "$TARGET_A3M") == 1 ]]; then
    echo "WARNING: SINGLE-SEQUENCE QUERY MSA; target has no coevolution signal: $TARGET_A3M" >&2
fi

##################################################
#               INPUT CREATION
################################################## 
start_time=$(date +%s)
INPUT_ARGS=(--fasta-single "$FASTA_FILE" --msa-dir "$PROTEOME_MSA_DIR"
            --msa-a "$TARGET_A3M" --output-dir "$OUTPUT_BASE"
            --af3-num-recycles "$AF3_NUM_RECYCLES"
            --af3-num-diffusion-samples "$AF3_NUM_DIFFUSION_SAMPLES"
            --max-identity "$MAX_IDENTITY")
[[ $FILTER_NON_STANDARD == 1 ]] || INPUT_ARGS+=(--no-filter-non-standard)
[[ $FORCE == 0 ]] || INPUT_ARGS+=(--force)
[[ $RECREATE_MANIFEST == 0 ]] || INPUT_ARGS+=(--recreate-manifest)
[[ $RECREATE_COMMANDS == 0 ]] || INPUT_ARGS+=(--recreate-commands)
python "$AF3_SCRIPT_DIR/run_pipeline.py" "${INPUT_ARGS[@]}"
printf 'Prepared pairs: %s; proteome A3Ms: %s; max_identity: %s\n' \
    "$(wc -l < "$OUTPUT_BASE/af3_cmds.txt")" \
    "$(find "$PROTEOME_MSA_DIR" -maxdepth 1 -name '*.a3m' -type f | wc -l)" "$MAX_IDENTITY"
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Creating inputs took $elapsed seconds"

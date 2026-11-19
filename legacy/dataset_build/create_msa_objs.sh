#!/bin/bash

#SBATCH -J msa
#SBATCH --partition=your_cpu

#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --error=logs/msa_%j.err
#SBATCH --output=logs/msa_%j.out

# send email about job start and end
# Set --mail-user on the sbatch command line when notifications are wanted.
#SBATCH --mail-type=END,FAIL


echo "Running on $(hostname)"
echo "Current directory: $(pwd)"
echo "Running on $(date)"
echo " "

eval "$(conda shell.bash hook)"
conda activate colabfold

# make directory in scratch and move to that directory
dir="$(pwd)"
# Set SCRATCH_ROOT if the scheduler does not provide TMPDIR.
SCRATCH_ROOT=${SCRATCH_ROOT:-${TMPDIR:-/path/to/scratch}}
COLABFOLD_DB_DIR=${COLABFOLD_DB_DIR:-/path/to/colabfold/databases}
scratch_dir="$SCRATCH_ROOT/f_msa_${SLURM_JOB_ID}"

mkdir -p "$scratch_dir"
cd "$scratch_dir"
mkdir msa_cache

# check cpu features to specify the right MMseqs2 options (e.g. --cpu-features=avx2)
# has only sse4.1 so wil point to that version of mmseqs2
# echo "CPU features: $(grep -o 'avx[^ ]*\|sse[^ ]*' /proc/cpuinfo | sort -u | head -20)"
# grep -o 'avx[^ ]*\|sse[^ ]*' /proc/cpuinfo | sort -u | head -20

# which mmseqs
# mmseqs --help 2>&1 | head -5   # often prints build info
# echo " "

# copy files from original directory to scratch
rsync -au "$dir/ppi_dataset/Q86VQ6.fasta" .

echo "Building MSAs for all sequences..."
CMD=(colabfold_search Q86VQ6.fasta "$COLABFOLD_DB_DIR" ./msa_cache --mmseqs "$dir/mmseqs/bin/mmseqs" --threads 32)
echo " "
printf 'Running: %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

if [ $? -ne 0 ]; then
    echo "Error: colabfold_search failed."
    exit 1
fi

echo "Done building MSAs."

# move the msa_cache contents back to the original directory
rsync -au msa_cache/* "$dir/ppi_dataset/inputs/msa_cache/."
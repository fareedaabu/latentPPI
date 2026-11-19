#!/bin/bash

#SBATCH --job-name=curate
#SBATCH --partition=your_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --error=logs/curate_%j.err
#SBATCH --output=logs/curate_%j.out

# send email about job start and end
# Set --mail-user on the sbatch command line when notifications are wanted.
#SBATCH --mail-type=END,FAIL


set -e

echo "Running on $(hostname)"
echo "Current directory: $(pwd)"
set -e

echo "Running on $(date)"
echo " "

eval "$(conda shell.bash hook)"
conda activate struct

python ppi_dataset_curate.py --outdir ./ppi_dataset --n_pairs 50000 --cluster_identity 0.9 --skip_cross_filter --auto_find_complexes --compartment_fraction 0.4 --max_complexes 5000 --max_len 2000


echo "Finished running ppi_dataset_curate.py pipeline"

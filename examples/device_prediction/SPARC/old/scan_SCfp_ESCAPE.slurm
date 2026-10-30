#!/bin/bash
#SBATCH --job-name=SPARC_SCfp_ESCAPE
#SBATCH --partition=LocalQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=14:00:00
#SBATCH --output=SPARC_param_scan%j.out
#SBATCH --error=SPARC_param_scan%j.err
#SBATCH --chdir=/mnt/homes_global/jal2351/software/saarelma-conner-ped/examples/device_prediction/SPARC

set -euo pipefail

# Firedrake venv (also sets PETSc env vars and MPLBACKEND=Agg)
source /mnt/codes/jal2351/venv/bin/activate

# Keep threading to a single CPU for this serial run
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Job started on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

python scan_SCfp_ESCAPE.py

echo "Job finished at $(date)"

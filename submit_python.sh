#!/bin/bash
#SBATCH --job-name=Py_Sim
#SBATCH --partition=qdisk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# 1. Activate the shared environment
source /clusterfs/opt/qupy/bin/activate

# 2. Align numerical libraries with the Slurm CPU allocation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 3. Execute the workflow
python pulse_dynamics_gaussian.py
#!/bin/bash
#SBATCH --exclusive
#SBATCH --output=time.out
#SBATCH --error=error.out
#SBATCH --job-name=arch_mpi
#SBATCH --partition=partition
#SBATCH --nodes=5

cd $SLURM_SUBMIT_DIR

## Copy conda init block here.
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
<<<< Conda init block >>>>>

srun --mpi=pmi2 flux start python workflow.py

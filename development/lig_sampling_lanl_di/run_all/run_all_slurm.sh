#!/bin/bash
#SBATCH --exclusive
#SBATCH --output=time.out
#SBATCH --error=error.out
#SBATCH --job-name=arch_mpi
#SBATCH --ntasks-per-node=48
#SBATCH --partition=<part>
#SBATCH --nodes=5

cd $SLURM_SUBMIT_DIR

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/path/to/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/path/to/conda/etc/profile.d/conda.sh" ]; then
        . "/path/to/conda/etc/profile.d/conda.sh"
    else
        export PATH="/path/to/conda/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/path/to/conda/etc/profile.d/mamba.sh" ]; then
    . "/path/to/conda/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<
conda activate conda_env


mpiexec -n 240 python -m mpi4py.futures mpirun.py

#!/bin/bash
#SBATCH --exclusive
#SBATCH --output=time.out
#SBATCH --error=error.out
#SBATCH --job-name=arch_mpi
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=2

cd $SLURM_SUBMIT_DIR

# Copy conda init block here: for example:
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/uname/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/uname/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/uname/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/uname/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Activate the correct environment
conda activate architector

mpiexec -n 48 python -m mpi4py.futures mpirun.py

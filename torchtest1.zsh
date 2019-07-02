#!/bin/zsh

#SBATCH -n 4
#SBATCH -A dlclim

module load gcc
module load openmpi/3.1.3
module load python/anaconda3
##module load cuda/10.1.105
module load torch/2017-10-04

date

echo "nodes: $SLURM_JOB_NODELIST"

mpiexec -n 4 ./torchtest1.py

date


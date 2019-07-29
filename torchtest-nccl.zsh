#!/bin/zsh

#SBATCH -n 4
#SBATCH -A dlclim
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=2

module load gcc
module load python/anaconda3
module load cuda/10.1.105
module load torch/2017-10-04

date

echo "nodes: $SLURM_JOB_NODELIST"

python -m torch.distributed.launch --nnodes $SLURM_NTASKS --nproc_per_node 2 ./torchtest-nccl.py

date


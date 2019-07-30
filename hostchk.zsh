#!/bin/zsh

#SBATCH -n 4
#SBATCH -A dlclim
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1

module load gcc
module load python/anaconda3
module load cuda/10.1.105
module load torch/2017-10-04

date

echo "nodes: $SLURM_JOB_NODELIST"

nodelist=`scontrol show hostnames $SLURM_JOB_NODELIST`

echo "nodelist:\n$nodelist"

bindir=`pwd`

NODERANK=0

while read -r node; do
     echo "Run hostchk on node $node  noderank=$NODERANK"
     srun -n 1 -N 1 ./hostchk.py &
     NODERANK=$(( NODERANK + 1 ))
done <<< $nodelist


wait

date


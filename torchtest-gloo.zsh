#!/bin/zsh

#SBATCH -n 2
#SBATCH -A dlclim
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH -J gloo-test

module load gcc
module load python/anaconda3
module load cuda/10.1.105
####module load torch/2017-10-04

date

echo "nodes: $SLURM_JOB_NODELIST"

env | grep SLURM

MASTER=`hostname -s`
PORT=24601

nodelist=`scontrol show hostnames $SLURM_JOB_NODELIST`

NODERANK=0
echo "starting master process on node $MASTER"
srun -n 1 -N 1 python -m torch.distributed.launch --nnodes $SLURM_JOB_NUM_NODES --nproc_per_node 2 --node_rank=$NODERANK --master_addr=$MASTER --master_port=$PORT ./torchtest-gloo.py &

while read -r node; do
    if [[ $node != $MASTER ]]; then
	NODERANK=$(( NODERANK + 1 ))
	echo "launching process on node $node  (noderank=$NODERANK)"
	srun -n 1 -N 1 python -m torch.distributed.launch --nnodes $SLURM_JOB_NUM_NODES --nproc_per_node 2 --node_rank=$NODERANK --master_addr=$MASTER --master_port=$PORT ./torchtest-gloo.py &
    fi
done <<< "$nodelist"


wait

date


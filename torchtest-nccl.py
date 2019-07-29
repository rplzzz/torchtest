#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import argparse

def main(length):
    """Set up an array of specified length and gather it back to the root process."""
    rank = dist.get_rank()
    comm_size = dist.get_world_size()

    print(f'Starting rank {rank} of {comm_size}')

    x = torch.ones(length) * rank # Default type is float, which is a good choice.

    buf = [torch.empty(length) for i in range(comm_size)]

    dist.all_gather(buf, x, buf)      # Synchronous collective: all processes block until complete.

    if rank==0:
        rslt = torch.stack(buf)
        print(f'rank: {rank}:\n{rslt}')
    else:
        print(f'rank: {rank}:  done.\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    args=parser.parse_args()

    rank = args.local_rank
    print(f'rank: {rank}   host: {os.uname()[1]}')
    
    dist.init_process_group('nccl', 'env://', rank=args.local_rank)
    main(1024)


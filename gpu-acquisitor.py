#!/usr/bin/env python3

# to be run on a machine with 2 GPUs (which have rights for, so -l gpu=2) as
# python3 scripts/gpu-acquisitor.py --id 1 & python3 scripts/gpu-acquisitor.py --id 2


import argparse
import time
import torch
from safe_gpu import safe_gpu
import logging


def main(args, logger):
    time.sleep(args.sleep)  # simulate other stuff
    results = []

    for gpu_no in range(args.nb_gpus):
        a = torch.zeros((2, 2), device=f'cuda:{gpu_no}')  # simulate CUDA computation
        results.append(a)

    time.sleep(args.sleep*2)  # simulate some more computation on all GPUs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sleep', default=3.0, type=float,
                        help='how long to sleep before trying to operate')
    parser.add_argument('--id', default=1, type=int,
                        help='just a number to identify processes')
    parser.add_argument('--nb-gpus', default=1, type=int,
                        help='how many gpus to take')
    args = parser.parse_args()

    logging.basicConfig(format='%(name)s %(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(f'GPUOwner{args.id}')
    logger.setLevel(logging.INFO)

    gpu_owner = safe_gpu.GPUOwner(
        nb_gpus=args.nb_gpus,
        logger=logger,
        debug_sleep=args.sleep,
    )
    main(args, logger)
    logger.info('Finished')

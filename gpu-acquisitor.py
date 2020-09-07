#!/usr/bin/env python3

# to be run on a machine with 2 GPUs (which have rights for, so -l gpu=2) as
# python3 scripts/gpu-acquisitor.py --id 1 & python3 scripts/gpu-acquisitor.py --id 2


import argparse
import time
import torch
from safe_gpu import safe_gpu
import logging


def main(args):
    time.sleep(args.sleep * 2)  # simulate other stuff
    a = torch.zeros((2, 2), device='cuda')  # simulate CUDA computation
    a.pow(2)  # get rid of complaints that a is unused


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sleep', default=3.0, type=float,
                        help='how long to sleep before trying to operate')
    parser.add_argument('--id', default=1, type=int,
                        help='just a number to identify processes')
    args = parser.parse_args()

    logging.basicConfig(format='%(name)s %(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(f'GPUOwner{args.id}')
    logger.setLevel(logging.INFO)

    gpu_owner = safe_gpu.GPUOwner(
        lambda: torch.zeros((1), device='cuda'),
        logger=logger,
        debug_sleep=args.sleep,
    )
    main(args)
    logger.info(f'Finished')

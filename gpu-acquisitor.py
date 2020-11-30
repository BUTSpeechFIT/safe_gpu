#!/usr/bin/env python3

# to be run on a machine with 2 GPUs (which have rights for, so -l gpu=2) as
# python3 scripts/gpu-acquisitor.py --id 1 & python3 scripts/gpu-acquisitor.py --id 2


import argparse
import time
from safe_gpu import safe_gpu
import logging


def simulate_computation_pytorch(nb_gpus):
    import torch
    results = []
    for gpu_no in range(args.nb_gpus):
        a = torch.zeros((2, 2), device=f'cuda:{gpu_no}')
        results.append(a)

    return results


def simulate_computation_tensorflow(nb_gpus):
    import tensorflow as tf
    results = []
    for gpu_no in range(args.nb_gpus):
        with tf.device(f'GPU:{gpu_no}'):
            a = tf.constant([1.0])
            results.append(a)

    return results
    

def main(args, logger):
    time.sleep(args.sleep)  # simulate other stuff, e.g. loading data etc.
    computations = {
        'pytorch': simulate_computation_pytorch,
        'tf': simulate_computation_tensorflow,
    }
    results = computations[args.backend](args.nb_gpus)

    time.sleep(args.sleep*2)  # simulate some more computation on all GPUs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sleep', default=3.0, type=float,
                        help='how long to sleep before trying to operate')
    parser.add_argument('--id', default=1, type=int,
                        help='just a number to identify processes')
    parser.add_argument('--nb-gpus', default=1, type=int,
                        help='how many gpus to take')
    parser.add_argument('--backend', default='pytorch', choices=['pytorch', 'tf'],
                        help='how many gpus to take')
    args = parser.parse_args()

    logging.basicConfig(format='%(name)s %(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(f'GPUOwner{args.id}')
    logger.setLevel(logging.INFO)

    placeholders = {
        'pytorch': safe_gpu.pytorch_placeholder,
        'tf': safe_gpu.tensorflow_placeholder,
    }

    gpu_owner = safe_gpu.GPUOwner(
        nb_gpus=args.nb_gpus,
        placeholder_fn=placeholders[args.backend],
        logger=logger,
        debug_sleep=args.sleep,
    )
    main(args, logger)
    logger.info('Finished')

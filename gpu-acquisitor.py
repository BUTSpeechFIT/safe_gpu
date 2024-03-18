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
    for gpu_no in range(nb_gpus):
        a = torch.zeros((2, 2), device=f'cuda:{gpu_no}')
        results.append(a)

    return results


def simulate_computation_tensorflow(nb_gpus):
    import tensorflow as tf
    results = []
    for gpu_no in range(nb_gpus):
        with tf.device(f'GPU:{gpu_no}'):
            a = tf.constant([1.0])
            results.append(a)

    return results


def simulate_computation_pycuda(nb_gpus):
    import pycuda.driver as cuda
    results = []

    class PycudaGpu:
        """
        Context manager for selecting 'active context'
        of a GPU for computation with pycuda.
        """

        def __init__(self, gpu_id: int):
            self.gpu_id = gpu_id

        def __enter__(self):
            safe_gpu.gpu_owner.placeholders[self.gpu_id].push()

        def __exit__(self, exc_type, exc_value, traceback):
            cuda.Context.pop()

    for gpu_no in range(nb_gpus):
        with PycudaGpu(gpu_no):
            # create byte array on a GPU,
            # - see: https://documen.tician.de/pycuda/tutorial.html
            array = bytes(b'HU7KCNWqOk4hma9VguEA5IK56lhfGnVC')
            array_gpu = cuda.mem_alloc(len(array))  # pointer to memory
            cuda.memcpy_htod(array_gpu, array)  # host -> device

            results.append({'ptr': array_gpu, 'gpu_no': gpu_no})

    return results


def simulate_computations(args, logger):
    time.sleep(args.sleep)  # simulate other stuff, e.g. loading data etc.

    computations = {
        'pytorch': simulate_computation_pytorch,
        'tf': simulate_computation_tensorflow,
        'pycuda': simulate_computation_pycuda,
    }
    results = computations[args.backend](args.nb_gpus)  # noqa: F841

    time.sleep(args.sleep*2)  # simulate some more computation on all GPUs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sleep', default=3.0, type=float,
                        help='how long to sleep before trying to operate')
    parser.add_argument('--id', default=1, type=int,
                        help='just a number to identify processes')
    parser.add_argument('--nb-gpus', default=1, type=int,
                        help='how many gpus to take')
    parser.add_argument('--backend', default='pytorch', choices=['pytorch', 'tf', 'pycuda'],
                        help='how many gpus to take')
    parser.add_argument('--explicit-owner-object', action='store_true',
                        help='construct actual GPUOwner object')
    return parser.parse_args()


def main():
    args = get_args()

    logging.basicConfig(format='%(name)s %(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(f'GPUOwner{args.id}')
    logger.setLevel(logging.INFO)

    placeholders = {
        'pytorch': safe_gpu.PytorchPlaceholder(),
        'tf': safe_gpu.tensorflow_placeholder,
        'pycuda': safe_gpu.PyCudaPlaceholder(),
    }

    if args.explicit_owner_object:
        gpu_owner = safe_gpu.GPUOwner(
            nb_gpus=args.nb_gpus,
            placeholder_fn=placeholders[args.backend],
            logger=logger,
            debug_sleep=args.sleep,
        )
        logger.info(f'Allocated devices: {gpu_owner.devices_taken}')
    else:
        safe_gpu.claim_gpus(
            nb_gpus=args.nb_gpus,
            placeholder_fn=placeholders[args.backend],
            logger=logger,
            debug_sleep=args.sleep,
        )
        logger.info(f'Allocated devices: {safe_gpu.gpu_owner.devices_taken}')

    print(args)
    simulate_computations(args, logger)
    logger.info('Finished')


if __name__ == '__main__':
    main()

import os
import time
import subprocess
import fcntl
import logging
import socket


LOCK_FILENAME = '/tmp/gpu-lock-magic-RaNdOM-This_Name-NEED_BE-THE_SAME-Across_Users'


def get_free_gpu():
    ''' Returns an integer. Note the possible race condition!
    '''
    # dirty hack for machines with non-exclusive GPUs
    if socket.gethostname().startswith('PCO'):
        return 0
    else:
        shell_line = 'nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"'
        return int(subprocess.check_output(shell_line, shell=True).decode())


def pytorch_placeholder():
    import torch
    return torch.zeros((1), device='cuda')


class SafeLocker:
    def __init__(self, fd):
        self._fd = fd

    def __enter__(self):
        fcntl.lockf(self._fd, fcntl.LOCK_EX)

    def __exit__(self, type, value, traceback):
        fcntl.lockf(self._fd, fcntl.LOCK_UN)


class SafeUmask:
    def __init__(self, working_mask):
        self.old_mask = None
        self.working_mask = working_mask

    def __enter__(self):
        self.old_mask = os.umask(self.working_mask)

    def __exit__(self, type, value, traceback):
        os.umask(self.old_mask)


class GPUOwner:
    def __init__(self, placeholder_fn=pytorch_placeholder, logger=None, debug_sleep=0.0):
        if logger is None:
            logger = logging

        with SafeUmask(0):  # do not mask any permission out by default
            with open(os.open(LOCK_FILENAME, os.O_CREAT | os.O_WRONLY, 0o666), 'w') as f:
                logger.info('acquiring lock')

                with SafeLocker(f):
                    logger.info('lock acquired')

                    free_gpu = get_free_gpu()
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(free_gpu)

                    if not os.environ['CUDA_VISIBLE_DEVICES']:
                        raise RuntimeError("No free GPUs found. Someone didn't properly declare their gpu resource?")

                    logger.info(f"Got CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
                    time.sleep(debug_sleep)

                    try:
                        self.placeholder = placeholder_fn()
                    except RuntimeError:
                        logger.error('Failed to acquire placeholder, truly marvellous')
                        raise
                logger.info('lock released')

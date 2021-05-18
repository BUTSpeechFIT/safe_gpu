import os
import time
import subprocess
import fcntl
import logging


LOCK_FILENAME = '/tmp/gpu-lock-magic-RaNdOM-This_Name-NEED_BE-THE_SAME-Across_Users'


class NvidiasmiError(Exception):
    pass


def get_free_gpus():
    ''' Returns a list of strings specifying GPUs not in use.

        Note that this information is volatile.
    '''
    raw_output = subprocess.check_output("nvidia-smi -q | grep 'Minor\|Processes' | grep -B1 'None' | tr -d ' ' | grep 'MinorNumber' | cut -d':' -f 2", shell=True)

    return raw_output.decode().split()


def pytorch_placeholder(device_no):
    import torch
    return torch.zeros((1), device=f'cuda:{device_no}')


def tensorflow_placeholder(device_no):
    import tensorflow as tf
    with tf.device(f'GPU:{device_no}'):
        return tf.constant([1.0])


class SafeLock:
    def __init__(self, fd, logger=None):
        self.logger = logger if logger else logging
        self._fd = fd

    def __enter__(self):
        self.logger.info('acquiring lock')
        fcntl.lockf(self._fd, fcntl.LOCK_EX)
        self.logger.info('lock acquired')

    def __exit__(self, type, value, traceback):
        fcntl.lockf(self._fd, fcntl.LOCK_UN)
        self.logger.info('lock released')


class SafeUmask:
    def __init__(self, working_mask):
        self.old_mask = None
        self.working_mask = working_mask

    def __enter__(self):
        self.old_mask = os.umask(self.working_mask)

    def __exit__(self, type, value, traceback):
        os.umask(self.old_mask)


def get_nvidia_smi_value(nvidia_smi_lines, key):
    for line in nvidia_smi_lines:
        if key in line:
            return line.split()[-1]
    else:
        raise KeyError(f'"{key}" not found in the provided nvidia-smi output')


def is_single_gpu_display_mode():
    res = subprocess.run(['nvidia-smi', '-q'], stdout=subprocess.PIPE)
    if res.returncode == 0:
        pass
    else:
        raise NvidiasmiError(res.stdout.decode('ascii'))

    stdout = res.stdout.decode('ascii')
    lines = stdout.split("\n")

    nb_gpus = int(get_nvidia_smi_value(lines, 'Attached GPUs'))
    if nb_gpus != 1:
        return False

    display_mode = get_nvidia_smi_value(lines, 'Display Mode')
    if display_mode == "Enabled":
        return True
    else:
        return False


class GPUOwner:
    def __init__(self, nb_gpus=1, placeholder_fn=pytorch_placeholder, logger=None, debug_sleep=0.0):
        if logger is None:
            logger = logging
        self.logger = logger
        self.debug_sleep = debug_sleep
        self.placeholder_fn = placeholder_fn

        # a workaround for machines where GPU is used also for actual display
        if is_single_gpu_display_mode():
            logger.info(f"Running on a machine with single GPU used for actual display")
            if nb_gpus == 1:
                self.allocate_gpus(['0'])
            else:
                raise ValueError(f'Requested {nb_gpus} GPUs on a machine with single one.')
        else:
            with SafeUmask(0):  # do not mask any permission out by default, allowing others to work with the lock
                with open(os.open(LOCK_FILENAME, os.O_CREAT | os.O_WRONLY, 0o666), 'w') as f:
                    with SafeLock(f, logger):
                        free_gpus = get_free_gpus()
                        if len(free_gpus) < nb_gpus:
                            raise RuntimeError(f"Required {nb_gpus} GPUs, only found these free: {free_gpus}. Somebody didn't properly declare their resources?")
                        gpus_to_allocate = free_gpus[:nb_gpus]
                        self.allocate_gpus(gpus_to_allocate)

    def allocate_gpus(self, gpu_device_numbers):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_device_numbers)
        self.logger.info(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

        time.sleep(self.debug_sleep)

        try:
            self.placeholders = [self.placeholder_fn(device_no) for device_no in range(len(gpu_device_numbers))]
        except RuntimeError:
            self.logger.error('Failed to acquire placeholder, truly marvellous. Race condition with someone not using `GPUOwner?`')
            raise

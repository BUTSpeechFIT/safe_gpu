import os
import time
import subprocess
import fcntl
import logging


LOCK_FILENAME = '/tmp/gpu-lock-magic-RaNdOM-This_Name-NEED_BE-THE_SAME-Across_Users'


def get_free_gpus():
    ''' Returns a list of integers specifying GPUs not in use.

        Note that this information is volatile.
    '''
    res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE)
    stdout = res.stdout.decode('ascii')
    lines = stdout.split("\n")

    i = 0

    # consume the headers
    while lines[i] != '|===============================+======================+======================|':
        i += 1
    i += 1

    # learn about devices
    devices = []
    while lines[i].split() != []:
        device = lines[i].split()[1]
        devices.append(device)
        i += 3

    # consume additional headers
    while lines[i] != '|=============================================================================|':
        i += 1
    i += 1

    processes = []
    # learn about processes p
    while lines[i] != '+-----------------------------------------------------------------------------+':
        process = lines[i].split()[1]
        processes.append(process)
        i += 1

    return [d for d in devices if d not in processes]


def pytorch_placeholder(device_no):
    import torch
    return torch.zeros((1), device=f'cuda:{device_no}')


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

        # a workaround for machines where GPU is used also for actual display
        if is_single_gpu_display_mode():
            if nb_gpus == 1:
                return
            else:
                raise ValueError(f'Requested {nb_gpus} GPUs on a machine with single one.')

        with SafeUmask(0):  # do not mask any permission out by default, allowing others to work with the lock
            with open(os.open(LOCK_FILENAME, os.O_CREAT | os.O_WRONLY, 0o666), 'w') as f:
                with SafeLock(f, logger):
                    free_gpus = get_free_gpus()
                    if len(free_gpus) < nb_gpus:
                        raise RuntimeError(f"Required {nb_gpus} GPUs, only found these free: {free_gpus}. Somebody didn't properly declare their resources?")
                    gpus_to_allocate = free_gpus[:nb_gpus]

                    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus_to_allocate)
                    logger.info(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

                    time.sleep(debug_sleep)

                    try:
                        self.placeholders = [placeholder_fn(device_no) for device_no in range(nb_gpus)]
                    except RuntimeError:
                        logger.error('Failed to acquire placeholder, truly marvellous. Race condition with someone not using `GPUOwner?`')
                        raise

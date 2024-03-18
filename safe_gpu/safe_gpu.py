import os
import time
import subprocess
import fcntl
import logging

LOCK_FILENAME = '/tmp/gpu-lock-magic-RaNdOM-This_Name-NEED_BE-THE_SAME-Across_Users'

gpu_owner = None


class NvidiasmiError(Exception):
    pass


def get_free_gpus():
    """
    Returns a list of strings specifying GPUs not in use.
    Note that this information is volatile.
    """
    gpus_smi = subprocess.check_output(["nvidia-smi", "--format=csv,noheader", "--query-gpu=index,gpu_bus_id"]).decode().strip()
    processes_smi = subprocess.check_output(["nvidia-smi", "--format=csv,noheader", "--query-compute-apps=pid,gpu_bus_id"]).decode().strip()

    gpus = {}
    for line in gpus_smi.split('\n'):
        idx, uuid = line.split(',')
        gpus[uuid.strip()] = idx.strip()

    # in case of no running processes
    if processes_smi == '':
        return list(gpus.values())

    used_gpus = set()
    for line in processes_smi.split('\n'):
        _, uuid = line.split(',')
        used_gpus.add(uuid.strip())

    free_gpus = list(set(gpus.keys()) - used_gpus)
    return sorted([gpus[free] for free in free_gpus])


def pytorch_placeholder(device_no):
    import torch
    return torch.zeros((1), device=f'cuda:{device_no}')

def tensorflow_placeholder(device_no):
    import tensorflow as tf
    with tf.device(f'GPU:{device_no}'):
        return tf.constant([1.0])


class PyCudaPlaceholder:
    def __call__(self, device_no):
        """
        Docs of pycuda `make_context()`:
        https://documen.tician.de/pycuda/driver.html#pycuda.driver.Device.make_context
        """
        import pycuda.driver as cuda
        cuda.init()
        device = cuda.Device(device_no)
        cuda_context = device.make_context()

        return cuda_context

    def release(self, placeholder):
        placeholder.detach()


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


def get_nvidia_smi_value(nvidia_smi_lines, key) -> str:
    for line in nvidia_smi_lines:
        if key in line:
            return line.split()[-1]
    else:
        raise KeyError(f'"{key}" not found in the provided nvidia-smi output')


def is_single_gpu_display_mode() -> bool:
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
    def __init__(
        self,
        nb_gpus=1,
        placeholder_fn=pytorch_placeholder,
        logger=None,
        debug_sleep=0.0,
    ):
        if logger is None:
            logger = logging
        self.logger = logger
        self.debug_sleep = debug_sleep
        self.placeholder_fn = placeholder_fn
        self.devices_taken = []
        self.placeholders = []

        # a workaround for machines where GPU is used also for actual display
        if is_single_gpu_display_mode():
            logger.info("Running on a machine with single GPU used for actual display")
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

    def allocate_gpus(self, gpu_device_numbers) -> None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_device_numbers)
        self.logger.info(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

        time.sleep(self.debug_sleep)

        try:
            self.placeholders.extend(self.placeholder_fn(device_no) for device_no in range(len(gpu_device_numbers)))
        except RuntimeError:
            self.logger.error(
                """
                Failed to acquire placeholder.
                Race condition with someone not using `GPUOwner` ?
                Or CUDA was already initialized when calling `claim_gpus()`?
                """
            )
            raise

        self.devices_taken.extend(int(gpu) for gpu in gpu_device_numbers)

    def __del__(self):
        # this destructor gets called when program ends
        # TODO: does it work ?
        self.release_gpus()

    def release_gpus(self) -> None:
        self.logger.info(f"RELEASING PLACEHOLDERS ON CUDA DEVICES : {self.devices_taken}")

        if isinstance(self.placeholder_fn, PyCudaPlaceholder):
            for ii, cuda_context in enumerate(self.placeholders):
                self.placeholder_fn.release(cuda_context)
                self.logger.info(f"released cuda_context on device {self.devices_taken[ii]}")

        elif self.placeholder_fn is pytorch_placeholder:
            while len(self.placeholders):
                # this does not really release the GPU...
                pytorch_tensor = self.placeholders.pop()
                del pytorch_tensor

        elif self.placeholder_fn is tensorflow_placeholder:
            while len(self.placeholders):
                # this does not really release the GPU...
                tensorflow_tensor = self.placeholders.pop()
                del tensorflow_tensor


def claim_gpus(
    nb_gpus=1,
    placeholder_fn=pytorch_placeholder,
    logger=None,
    debug_sleep=0.0,
) -> None:
    """
    Allocate the GPUs.

    :param nb_gpus: number of GPUs to be allocated.
    :param placeholder_fn: Placeholder function, it creates CUDA context at particular GPU.
                           Typically by creating a ``dummy'' Tensor that gets stored till the end
                           of the program life-time.
    :param logger: argument for passing in the logging module or object (e.g. `logger=logging`)
    :param debug_sleep: delay between setting `os.environ['CUDA_VISIBLE_DEVICES']` and executing `placeholder_fn` calls.

    IMPORTANT NOTICE:

    This function must be called before initializing CUDA:
    - The `import torch` can be done before calling `claim_gpus()`.
    - But `torch.cuda.is_available()` should not be called before `claim_gpus()`,
      otherwise setting `CUDA_VISIBLE_DEVICES`in `GPUOwner::allocate_gpus()` will **NOT** work well.

    """

    global gpu_owner
    gpu_owner = GPUOwner(nb_gpus, placeholder_fn, logger, debug_sleep)


def release_gpus() -> None:
    global gpu_owner
    gpu_owner.release_gpus()

# safe_gpu

A module for safe acquisition of GPUs in exclusive mode.
Relevant mainly in clusters with a purely declarative gpu resource, such as many versions of SGE.

Features:
* toolkit independence (PyTorch/TensorFlow/pycuda/...)
* multiple GPUs acquisition
* workaround for machines with a single GPU used for display and computation alike

Downsides:
* in order to really prevent the race condition, everyone on your cluster has to use this

## Instalation

`safe_gpu` is on PyPi, so you can simply:

```
pip install safe-gpu
```

## Usage
Prior to initializing CUDA (typically happens when you place something on GPU), instantiate `GPUOwner` and bind it to a variable, that's all.

```
from safe_gpu import safe_gpu

gpu_owner = safe_gpu.GPUOwner()
```

### Acquiring multiple GPUs
Pass the desired number to `GPUOwner`.

```
gpu_owner = safe_gpu.GPUOwner(nb_gpus)
```

### Avoiding PyTorch
The default implementation uses a PyTorch tensor to claim a GPU.
If you don't want / can't use that, provide your own GPU memory allocating function to `GPUOwner`.
It has to accept one parameter `device_no`, occupy a (preferrably negligible) piece of memory on that device, and return a pointer to it.

### Checking that it works
Together with this package, a small testing script is provided.
It exagerrates the time needed to acquire the GPU after polling nvidia-smi, make the race condition technically sure to happen.
To run it, get to a machine with 2 free GPUs and run two instances of the script in parallel.
You should see in the output that one of them really waited for the faster one to fully acquire the GPU.

```
~/safe_gpu $ python3 gpu-acquisitor.py --id 1 & python3 gpu-acquisitor.py --id 2
GPUOwner1 2020-09-11 13:11:05,476 [INFO] acquiring lock
GPUOwner2 2020-09-11 13:11:05,476 [INFO] acquiring lock
GPUOwner1 2020-09-11 13:11:05,476 [INFO] lock acquired
GPUOwner1 2020-09-11 13:11:05,872 [INFO] Got CUDA_VISIBLE_DEVICES=2
GPUOwner2 2020-09-11 13:11:11,035 [INFO] lock acquired
GPUOwner1 2020-09-11 13:11:11,035 [INFO] lock released
GPUOwner2 2020-09-11 13:11:11,326 [INFO] Got CUDA_VISIBLE_DEVICES=3
GPUOwner2 2020-09-11 13:11:16,507 [INFO] lock released
GPUOwner1 2020-09-11 13:11:17,066 [INFO] Finished
GPUOwner2 2020-09-11 13:11:22,538 [INFO] Finished
```

# safe_gpu

A module for safe acquisition of GPUs in exclusive mode.
Relevant mainly in clusters with a purely declarative gpu resource, such as many versions of SGE.

Features:
* toolkit independence (PyTorch/TensorFlow/pycuda/...)
* multiple GPUs acquisition
* workaround for machines with a single GPU used for display and computation alike

Downsides:
* in order to really prevent the race condition, everyone on your cluster has to use this

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

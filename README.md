# safe_gpu

A module for safe acquisition of GPUs in exclusive mode.
Relevant especially in clusters with purely declarative gpu resource, such as many versions of SGE.

## Usage
Prior to placing anything on GPU, instantiate a `GPUOwner` and bind it to a variable, that's all.

```
from safe_gpu import safe_gpu

gpu_owner = safe_gpu.GPUOWner()
```

### Avoiding PyTorch
The default implementation uses a PyTorch tensor to claim a GPU.
If you don't want / can't use that, provide your own GPU memory allocating function to `GPUOwner`, e.g.:

```
gpu_owner = safe_gpu.GPUOwner(placeholder_fn=lambda: cuda.mem_alloc(4))
```

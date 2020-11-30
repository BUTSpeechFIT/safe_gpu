# safe-gpu

A module for safe acquisition of GPUs in exclusive mode.
Relevant mainly in clusters with a purely declarative gpu resource, such as many versions of SGE.

Features:
* toolkit independence (PyTorch/TensorFlow/pycuda/...), this just sets `CUDA_VISIBLE_DEVICES` properly
* included support for PyTorch and TensorFlow2 backends, open to others
* multiple GPUs acquisition
* workaround for machines with a single GPU used for display and computation alike
* open to implementation in different languages

Downsides:
* in order to really prevent the race condition, everyone on your cluster has to use this

## Instalation

In addition to manual installation, `safe-gpu` is on PyPi, so you can simply:

```
pip install safe-gpu
```

Note that `safe-gpu` does not formally depend on any backend, giving you, the user, the freedom to pick one of your liking.

## Usage
Prior to initializing CUDA (typically happens in a lazy fashion when you place something on GPU), instantiate `GPUOwner` and bind it to a variable, that's all.

```
from safe_gpu import safe_gpu

gpu_owner = safe_gpu.GPUOwner()
```

If you want multiple GPUs, pass the desired number to `GPUOwner`:

```
gpu_owner = safe_gpu.GPUOwner(nb_gpus)
```

### Other backends
The default implementation uses a PyTorch tensor to claim a GPU.
Additionally, a TensorFlow2 placeholder is provided as `safe_gpu.tensorflow_placeholder`.

If you don't want to / can't use that, provide your own GPU memory allocating function as `GPUOwner`'s parameter `placeholder_fn`.
It has to accept one parameter `device_no`, occupy a (preferrably negligible) piece of memory on that device, and return a pointer to it.

Pull requests for other backends are welcome.

### Checking that it works
Together with this package, a small testing script is provided.
It exagerrates the time needed to acquire the GPU after polling nvidia-smi, making the race condition technically sure to happen.

To run the following example, get to a machine with 3 free GPUs and run two instances of the script in parallel as shown.
You should see in the output that one of them really waited for the faster one to fully acquire the GPU.

This script is not distributed along in the pip package, so please download it separately.

```
$ python3 gpu-acquisitor.py --backend pytorch --id 1 --nb-gpus 1 & python3 gpu-acquisitor.py --backend pytorch --id 2 --nb-gpus 2
GPUOwner1 2020-11-30 14:29:33,315 [INFO] acquiring lock
GPUOwner1 2020-11-30 14:29:33,315 [INFO] lock acquired
GPUOwner2 2020-11-30 14:29:33,361 [INFO] acquiring lock
GPUOwner1 2020-11-30 14:29:34,855 [INFO] Set CUDA_VISIBLE_DEVICES=2
GPUOwner2 2020-11-30 14:29:45,447 [INFO] lock acquired
GPUOwner1 2020-11-30 14:29:45,447 [INFO] lock released
GPUOwner2 2020-11-30 14:29:48,926 [INFO] Set CUDA_VISIBLE_DEVICES=4,5
GPUOwner1 2020-11-30 14:29:54,492 [INFO] Finished
GPUOwner2 2020-11-30 14:30:00,525 [INFO] lock released
GPUOwner2 2020-11-30 14:30:09,571 [INFO] Finished

```

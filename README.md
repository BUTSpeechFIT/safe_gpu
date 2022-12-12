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
Prior to initializing CUDA (typically happens in lazy fashion when you place something on GPU), call `claim_gpus`.

```
from safe_gpu import safe_gpu

safe_gpu.claim_gpus()
```

If you want multiple GPUs, pass the desired number to `claim_gpus`:

```
safe_gpu.claim_gpus(nb_gpus)
```

Internally, `claim_gpus()` constructs a `GPUOwner` and stores it in `safe_gpu.gpu_owner`.
If preferred, user code can construct `GPUOwner` itself, but care should be taken to keep it alive until actual data is placed on said GPUs.

### Usage with Horovod
Typical Horovod usage includes starting your script in several processes, one per GPU.
Therefore, only ask for one GPU in each process:

```
safe_gpu.claim_gpus()  # 1 GPU is the default, can be ommited
hvd.init()
```


### Common errors
In order to properly setup GPUs for your process, `claim_gpus` really needs be called before CUDA is initialized.
When CUDA does get initialized, it fixes your logical devices (e.g. PyTorch `cuda:1` etc.) to actual GPUs in your system.
If `CUDA_VISIBLE_DEVICES` are not set at that moment, CUDA will happily offer your process all of the visible GPUs, including those already occupied.

Most commonly, this issue occurs for users who try to play it safe and check CUDA availability beforehand:
```
if torch.cuda.is_available():  # This already initializes CUDA
  safe_gpu.claim_gpus(nb_gpus)  # So this can fail easily
```

If your workflow mandates on-the-fly checking of GPU availability, instead use:
```
try:
  safe_gpu.claim_gpus(nb_gpus)
except safe_gpu.NvidiasmiError:
  ...
```

Also, horovod users can be at risk:
```
hvd.init()
torch.cuda.set_device(hvd.local_rank())  # This initializes CUDA, too
safe_gpu.claim_gpus()  # Thus this is likely to fail
```

See above for proper solution.


### Other backends
The default implementation uses a PyTorch tensor to claim a GPU.
Additionally, a TensorFlow2 placeholder is provided as `safe_gpu.tensorflow_placeholder`.

If you don't want to / can't use that, provide your own GPU memory allocating function as `claim_gpus`'s parameter `placeholder_fn`.
It has to accept one parameter `device_no`, occupy a (preferably negligible) piece of memory on that device, and return a pointer to it.

Pull requests for other backends are welcome.

### Checking that it works
Together with this package, a small testing script is provided.
It exaggerates the time needed to acquire the GPU after polling nvidia-smi, making the race condition technically sure to happen.

To run the following example, get to a machine with 3 free GPUs and run two instances of the script in parallel as shown.
You should see in the output that one of them really waited for the faster one to fully acquire the GPU.

This script is not distributed along in the pip package, so please download it separately if needed.

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

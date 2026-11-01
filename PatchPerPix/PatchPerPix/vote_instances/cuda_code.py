import logging

import numpy as np
import pycuda
import pycuda.compiler

_logger = logging.getLogger(__name__)

_kernel_cache = {}


def make_kernel(code, options=None):
    # modules belong to the current context; blocks of the same shape reuse them
    key = (code, tuple(options) if options is not None else None)
    module = _kernel_cache.get(key)
    if module is None:
        module = pycuda.compiler.DynamicSourceModule(code, options=options)
        _kernel_cache[key] = module
    return module


def alloc_zero_array(shape, dtype):
    return pycuda.driver.managed_zeros(
        shape, dtype=dtype, mem_flags=pycuda.driver.mem_attach_flags.GLOBAL)


def alloc_empty_array(shape, dtype):
    return pycuda.driver.managed_empty(
        shape, dtype=dtype, mem_flags=pycuda.driver.mem_attach_flags.GLOBAL)


def alloc_zero_device_array(shape, dtype):
    """Zeroed array in device memory (falls back to managed memory if the
    device is full). Only touch it from kernels; use to_host() to read it."""
    import pycuda.gpuarray
    try:
        return pycuda.gpuarray.zeros(shape, dtype=dtype)
    except pycuda.driver.MemoryError:
        _logger.warning("device alloc of %s %s failed, using managed memory",
                       shape, np.dtype(dtype).name)
        return alloc_zero_array(shape, dtype)


def to_device_array(array):
    import pycuda.gpuarray
    try:
        return pycuda.gpuarray.to_gpu(np.ascontiguousarray(array))
    except pycuda.driver.MemoryError:
        tmp = alloc_empty_array(array.shape, array.dtype)
        tmp[:] = array
        return tmp


def cuda_arg(array):
    """Kernel argument for a GPUArray or a managed numpy array."""
    return array.gpudata if hasattr(array, 'gpudata') else array


def to_host(array):
    return array.get() if hasattr(array, 'get') else array


def free_array(array):
    if hasattr(array, 'gpudata'):
        array.gpudata.free()
    else:
        array.base.free()


def sync(context):
    context.synchronize()

def init_cuda():
    from pycuda.autoinit import context
    # import pycuda.driver as cuda

    # # Initialize CUDA
    # cuda.init()

    # # from pycuda.tools import make_default_context
    # global context
    # # context = make_default_context()
    # ndevices = cuda.Device.count()
    # for devn in range(ndevices):
    #     dev = cuda.Device(devn)
    #     try:
    #         context = dev.make_context(cuda.ctx_flags.SCHED_SPIN)
    #         break
    #     except cuda.Error:
    #         pass

    # device = context.get_device()

    # def _finish_up():
    #     global context
    #     context.pop()
    #     context = None

    #     from pycuda.tools import clear_context_caches
    #     clear_context_caches()

    # import atexit
    # atexit.register(_finish_up)
    return context

def delete_cuda(context):
    context.pop()
    context = None

    from pycuda.tools import clear_context_caches
    clear_context_caches()
    _kernel_cache.clear()

def get_cuda_stream():
    import pycuda.driver as cuda
    return cuda.Stream()

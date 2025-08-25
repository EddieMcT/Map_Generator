### Experimental code
# to quickly test gpu acceleration, uses cupy instead of numpy
# not expected to work on windows systems, can be enabled for WSL systems but is untested

gpu_enabled = False
debug = True
# detect if cuda is available
try:
    import cupy
    # Check if CUDA is available and functional
    if debug:
        print("cupy imported. Checking for CUDA availability...")
    if hasattr(cupy, 'cuda') and cupy.cuda.is_available():
        if debug:
            print("CUDA is available. Testing functionality.")
        try:
            # Attempt to allocate a small array on the GPU
            _ = cupy.random.rand(2,2,3)
            if debug:
                print("CUDA is functional. Enabling GPU acceleration.")
            gpu_enabled = True
        except Exception as e:
            if debug:
                print(f"CUDA test failed: {e}. Disabling GPU acceleration.")
            gpu_enabled = False
    else:
        if debug:
            print("CUDA is not available. Disabling GPU acceleration.")
        gpu_enabled = False
except:
    if debug:
        print("cupy import failed. Disabling GPU acceleration.")
    gpu_enabled = False

if gpu_enabled:
    # from cupy import *
    if debug:
        print("Using cupy for GPU acceleration.")
    from cupy import newaxis, array_split, asarray, min, power, max, ndarray, add, multiply, sin, linspace, floor, concatenate, arange, zeros_like, zeros, cos, clip, sum, argmin, minimum, sqrt, maximum, broadcast_to, random, pi
    from cupy import subtract, ones_like, meshgrid, log2, array, abs, arctan2, where, square, uint16, divide, sort, absolute, linalg, ogrid, lib, take_along_axis, indices, tile, isscalar, stack 
    
else:
    # from numpy import *
    if debug:
        print("Using numpy for CPU processing.")
    from numpy import newaxis, array_split, asarray, min, power, max, ndarray, add, multiply, sin, linspace, floor, concatenate, arange, zeros_like, zeros, cos, clip, sum, argmin, minimum, sqrt, maximum, broadcast_to, random, pi 
    from numpy import subtract, ones_like, meshgrid, log2, array, abs, arctan2, where, square, uint16, divide, sort, absolute, linalg, ogrid, lib, take_along_axis, indices, tile, isscalar, stack 

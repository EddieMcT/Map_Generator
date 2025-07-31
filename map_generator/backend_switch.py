### Experimental code
# to quickly test gpu acceleration, uses cupy instead of numpy
# not expected to work on windows systems, can be enabled for WSL systems but is untested

gpu_enabled = False
if gpu_enabled:
    from cupy import *
else:
    from numpy import *
    # from numpy import power, minimum, maximum,min,max, exp, sqrt, log, isnan, isinf, log2, pi,sin, cos, abs, ceil, floor, where
    # from numpy import tile,broadcast_to, ones_like, nan_to_num,  clip, zeros_like,  array, clip, where, linspace, meshgrid, asarray
    # from numpy import float32, float64, ndarray, arange, argmin, argmax, concatenate, isscalar
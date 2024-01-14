import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np

# Define a custom exception for our library-related errors
class RustEnhancerError(Exception):
    pass

# Define a wrapper function for the Rust `enhance_data` function
def enhance_data(data):
    if not isinstance(data, np.ndarray) or data.dtype != np.float32:
        raise ValueError("Data must be a numpy array of type float32")

    data_p = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result = rust_lib.enhance_data(data_p, len(data))
    if result == -1:  # Assuming -1 is the error code from Rust
        raise RustEnhancerError("Rust function `enhance_data` failed.")
    return result

# Attempt to load the shared library
try:
    rust_lib = ctypes.CDLL('../../Framework/lib/rust_enhancer.dll')
    rust_lib.enhance_data.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)
    rust_lib.enhance_data.restype = ctypes.c_float
    rust_lib.enhance_data.errcheck = lambda result, func, arguments: (
        RustEnhancerError("Rust function `enhance_data` failed.") if int(result) == -1 else result
    )
except OSError as e:
    raise RustEnhancerError(f"Could not load the shared library: {e}")

# Prepare data
data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

try:
    result = enhance_data(data)
    print(f'Result: {result}')
except RustEnhancerError as e:
    print(f'An error occurred: {e}')
except ValueError as e:
    print(f'Invalid input: {e}')
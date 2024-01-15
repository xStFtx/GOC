import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

class RustEnhancerError(Exception):
    pass

# Define the Rust function signatures
def setup_rust_functions():
    rust_lib.enhance_data.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_uint)
    rust_lib.enhance_data.restype = ctypes.POINTER(ctypes.c_float)

    rust_lib.free_enhanced_data.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)
    rust_lib.free_enhanced_data.restype = None

# Wrapper function for the Rust `enhance_data` function
def enhance_data(data, operation):
    if not isinstance(data, np.ndarray) or data.dtype != np.float32:
        raise ValueError("Data must be a numpy array of type float32")

    data_p = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    len_data = len(data)

    result_ptr = rust_lib.enhance_data(data_p, len_data, operation)
    if not result_ptr:
        raise RustEnhancerError("Rust function `enhance_data` failed to allocate memory.")

    # Convert the result back to a numpy array
    result = np.ctypeslib.as_array(result_ptr, shape=(len_data,))
    # Copy the data to a new numpy array, as the memory will be freed
    result_copy = np.array(result)

    # Free the memory allocated by Rust
    rust_lib.free_enhanced_data(result_ptr, len_data)

    return result_copy

try:
    rust_lib = ctypes.CDLL('./rust_enhancer.dll')
    setup_rust_functions()
except OSError as e:
    raise RustEnhancerError(f"Could not load the shared library: {e}")

# Prepare data
data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
operation = 1  # Choose the operation (1: scale, 2: normalize, 3: apply model)

try:
    result = enhance_data(data, operation)
    print(f'Result: {result}')
except RustEnhancerError as e:
    print(f'An error occurred: {e}')
except ValueError as e:
    print(f'Invalid input: {e}')

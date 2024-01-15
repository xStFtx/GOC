import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import os
import logging
import configparser
import threading
import queue

config = configparser.ConfigParser()
config.read('settings.ini')

log_file = config.get('Logging', 'LogFile', fallback='app.log')
log_level = config.get('Logging', 'Level', fallback='INFO')
logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

class RustEnhancerError(Exception):
    pass

class RustLibraryNotFound(Exception):
    pass

def load_rust_library(lib_path: str) -> ctypes.CDLL:
    if not os.path.exists(lib_path):
        raise RustLibraryNotFound(f"Rust library not found at {lib_path}")
    logging.info("Rust library loaded successfully.")
    return ctypes.CDLL(lib_path)

def setup_rust_functions(rust_lib):
    rust_lib.enhance_data.argtypes = (ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_uint)
    rust_lib.enhance_data.restype = ctypes.POINTER(ctypes.c_float)
    rust_lib.free_enhanced_data.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_size_t)
    rust_lib.free_enhanced_data.restype = None

class RustEnhancer:
    def __init__(self, rust_lib, operation=1):
        self.rust_lib = rust_lib
        self.operation = operation
        self.data_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.process_data, daemon=True)
        self.worker_thread.start()

    def enhance_data(self, data: np.ndarray):
        self.data_queue.put(data)

    def get_result(self) -> np.ndarray:
        return self.result_queue.get()

    def process_data(self):
        while True:
            data = self.data_queue.get()
            if data is None:
                break
            enhanced_data = self.enhance_data_internal(data)
            self.result_queue.put(enhanced_data)

    def enhance_data_internal(self, data: np.ndarray) -> np.ndarray:
        self.validate_data(data)
        len_data = data.size
        result_ptr = self.rust_lib.enhance_data(data, len_data, self.operation)
        if not result_ptr:
            raise RustEnhancerError("Rust function `enhance_data` failed to allocate memory.")
        return self.copy_result_data(result_ptr, len_data)

    @staticmethod
    def validate_data(data: np.ndarray):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        if data.ndim > 1:
            data = data.flatten()
        if data.dtype != np.float32:
            try:
                data = data.astype(np.float32)
            except Exception as e:
                raise ValueError(f"Data conversion to float32 failed: {e}")

    @staticmethod
    def copy_result_data(result_ptr, len_data):
        try:
            result = np.ctypeslib.as_array(result_ptr, shape=(len_data,))
            result_copy = np.array(result)
        finally:
            rust_lib.free_enhanced_data(result_ptr, len_data)
        return result_copy

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data_queue.put(None)
        self.worker_thread.join()
        logging.info("Rust Enhancer resources are being cleaned up.")

# Unit tests for RustEnhancer
import unittest

class TestRustEnhancer(unittest.TestCase):
    def setUp(self):
        self.rust_lib_path = './rust_enhancer.dll'
        self.rust_lib = load_rust_library(self.rust_lib_path)
        setup_rust_functions(self.rust_lib)
        self.enhancer = RustEnhancer(self.rust_lib)

    def test_enhance_data_valid_input(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        operation = 1
        result = self.enhancer.enhance_data(data, operation)
        self.assertIsNotNone(result)
        self.assertEqual(result.size, data.size)

    def test_enhance_data_invalid_input(self):
        with self.assertRaises(ValueError):
            self.enhancer.enhance_data([1, 2, 3], 1)  # Not a numpy array

# Example usage
if __name__ == "__main__":
    try:
        rust_lib_path = './rust_enhancer.dll'
        rust_lib = load_rust_library(rust_lib_path)
        setup_rust_functions(rust_lib)

        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        operation = 1  # Choose the operation

        with RustEnhancer(rust_lib) as enhancer:
            result = enhancer.enhance_data(data, operation)
            print(f'Result: {result}')
    except RustEnhancerError as e:
        print(f'An error occurred: {e}')
    except ValueError as e:
        print(f'Invalid input: {e}')
    except RustLibraryNotFound as e:
        print(f'Library loading error: {e}')
    except Exception as e:
        print(f'Unexpected error: {e}')

    unittest.main()
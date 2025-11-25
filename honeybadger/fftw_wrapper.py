"""
FFTW wrapper module - provides FFT operations using the FFTW C library.

This module wraps FFTW3 (Fastest Fourier Transform in the West) using ctypes
to provide FFT and inverse FFT operations for numerical data.
"""

import ctypes
from ctypes import POINTER, c_double, c_int, c_void_p
from typing import List, Optional, Tuple, Union
import os


# FFTW plan flags
FFTW_ESTIMATE = 64
FFTW_MEASURE = 0
FFTW_PATIENT = 32
FFTW_EXHAUSTIVE = 8

# FFTW sign constants for transform direction
FFTW_FORWARD = -1
FFTW_BACKWARD = 1


def _load_fftw() -> Optional[ctypes.CDLL]:
    """
    Load the FFTW3 library.
    
    Returns:
        Loaded FFTW library or None if not found.
    """
    lib_names = [
        "libfftw3.so.3",
        "libfftw3.so",
        "libfftw3.dylib",
        "/usr/lib/x86_64-linux-gnu/libfftw3.so.3",
        "/usr/local/lib/libfftw3.so",
    ]
    
    for name in lib_names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    
    return None


# Load FFTW library
_fftw = _load_fftw()


class FFTWError(Exception):
    """Exception raised for FFTW-related errors."""
    pass


def _check_fftw_available() -> None:
    """Check if FFTW is available and raise an error if not."""
    if _fftw is None:
        raise FFTWError(
            "FFTW library not found. Please install libfftw3: "
            "apt-get install libfftw3-dev (Linux) or brew install fftw (macOS)"
        )


def _setup_fftw_functions() -> None:
    """Set up FFTW function signatures."""
    if _fftw is None:
        return
    
    # fftw_malloc
    _fftw.fftw_malloc.argtypes = [ctypes.c_size_t]
    _fftw.fftw_malloc.restype = c_void_p
    
    # fftw_free
    _fftw.fftw_free.argtypes = [c_void_p]
    _fftw.fftw_free.restype = None
    
    # fftw_plan_dft_1d - for complex-to-complex transforms
    _fftw.fftw_plan_dft_1d.argtypes = [c_int, c_void_p, c_void_p, c_int, c_int]
    _fftw.fftw_plan_dft_1d.restype = c_void_p
    
    # fftw_plan_dft_r2c_1d - for real-to-complex transforms
    _fftw.fftw_plan_dft_r2c_1d.argtypes = [c_int, c_void_p, c_void_p, c_int]
    _fftw.fftw_plan_dft_r2c_1d.restype = c_void_p
    
    # fftw_plan_dft_c2r_1d - for complex-to-real transforms
    _fftw.fftw_plan_dft_c2r_1d.argtypes = [c_int, c_void_p, c_void_p, c_int]
    _fftw.fftw_plan_dft_c2r_1d.restype = c_void_p
    
    # fftw_execute
    _fftw.fftw_execute.argtypes = [c_void_p]
    _fftw.fftw_execute.restype = None
    
    # fftw_destroy_plan
    _fftw.fftw_destroy_plan.argtypes = [c_void_p]
    _fftw.fftw_destroy_plan.restype = None


# Set up function signatures if library is loaded
_setup_fftw_functions()


def fft(data: List[Union[float, complex]]) -> List[complex]:
    """
    Compute the Fast Fourier Transform of the input data.
    
    Uses FFTW3 library for high-performance FFT computation.
    
    Args:
        data: Input data as a list of real or complex numbers.
    
    Returns:
        List of complex numbers representing the FFT result.
    
    Raises:
        FFTWError: If FFTW library is not available.
        ValueError: If input data is empty.
    """
    _check_fftw_available()
    
    if not data:
        raise ValueError("Input data cannot be empty")
    
    n = len(data)
    
    # Allocate memory for input and output (complex: 2 doubles per element)
    in_ptr = _fftw.fftw_malloc(n * 2 * ctypes.sizeof(c_double))
    out_ptr = _fftw.fftw_malloc(n * 2 * ctypes.sizeof(c_double))
    
    if not in_ptr or not out_ptr:
        if in_ptr:
            _fftw.fftw_free(in_ptr)
        if out_ptr:
            _fftw.fftw_free(out_ptr)
        raise FFTWError("Failed to allocate memory for FFT")
    
    try:
        # Copy input data to FFTW array
        in_array = (c_double * (n * 2)).from_address(in_ptr)
        for i, val in enumerate(data):
            if isinstance(val, complex):
                in_array[i * 2] = val.real
                in_array[i * 2 + 1] = val.imag
            else:
                in_array[i * 2] = float(val)
                in_array[i * 2 + 1] = 0.0
        
        # Create plan for forward DFT
        plan = _fftw.fftw_plan_dft_1d(n, in_ptr, out_ptr, FFTW_FORWARD, FFTW_ESTIMATE)
        
        if not plan:
            raise FFTWError("Failed to create FFTW plan")
        
        try:
            # Execute the transform
            _fftw.fftw_execute(plan)
            
            # Extract output
            out_array = (c_double * (n * 2)).from_address(out_ptr)
            result = []
            for i in range(n):
                real = out_array[i * 2]
                imag = out_array[i * 2 + 1]
                result.append(complex(real, imag))
            
            return result
        
        finally:
            _fftw.fftw_destroy_plan(plan)
    
    finally:
        _fftw.fftw_free(in_ptr)
        _fftw.fftw_free(out_ptr)


def ifft(data: List[complex]) -> List[complex]:
    """
    Compute the Inverse Fast Fourier Transform of the input data.
    
    Uses FFTW3 library for high-performance IFFT computation.
    
    Args:
        data: Input data as a list of complex numbers.
    
    Returns:
        List of complex numbers representing the IFFT result.
    
    Raises:
        FFTWError: If FFTW library is not available.
        ValueError: If input data is empty.
    """
    _check_fftw_available()
    
    if not data:
        raise ValueError("Input data cannot be empty")
    
    n = len(data)
    
    # Allocate memory for input and output
    in_ptr = _fftw.fftw_malloc(n * 2 * ctypes.sizeof(c_double))
    out_ptr = _fftw.fftw_malloc(n * 2 * ctypes.sizeof(c_double))
    
    if not in_ptr or not out_ptr:
        if in_ptr:
            _fftw.fftw_free(in_ptr)
        if out_ptr:
            _fftw.fftw_free(out_ptr)
        raise FFTWError("Failed to allocate memory for IFFT")
    
    try:
        # Copy input data to FFTW array
        in_array = (c_double * (n * 2)).from_address(in_ptr)
        for i, val in enumerate(data):
            if isinstance(val, complex):
                in_array[i * 2] = val.real
                in_array[i * 2 + 1] = val.imag
            else:
                in_array[i * 2] = float(val)
                in_array[i * 2 + 1] = 0.0
        
        # Create plan for backward DFT (inverse)
        plan = _fftw.fftw_plan_dft_1d(n, in_ptr, out_ptr, FFTW_BACKWARD, FFTW_ESTIMATE)
        
        if not plan:
            raise FFTWError("Failed to create FFTW plan")
        
        try:
            # Execute the transform
            _fftw.fftw_execute(plan)
            
            # Extract output and normalize (FFTW doesn't normalize)
            out_array = (c_double * (n * 2)).from_address(out_ptr)
            result = []
            for i in range(n):
                real = out_array[i * 2] / n
                imag = out_array[i * 2 + 1] / n
                result.append(complex(real, imag))
            
            return result
        
        finally:
            _fftw.fftw_destroy_plan(plan)
    
    finally:
        _fftw.fftw_free(in_ptr)
        _fftw.fftw_free(out_ptr)


def fft_real(data: List[float]) -> List[complex]:
    """
    Compute the FFT of real-valued input data.
    
    This is an optimized version for real input, using real-to-complex transform.
    
    Args:
        data: Input data as a list of real numbers.
    
    Returns:
        List of complex numbers representing the positive frequency components.
        Length is n//2 + 1 where n is the input length.
    
    Raises:
        FFTWError: If FFTW library is not available.
        ValueError: If input data is empty.
    """
    _check_fftw_available()
    
    if not data:
        raise ValueError("Input data cannot be empty")
    
    n = len(data)
    n_complex = n // 2 + 1
    
    # Allocate memory
    in_ptr = _fftw.fftw_malloc(n * ctypes.sizeof(c_double))
    out_ptr = _fftw.fftw_malloc(n_complex * 2 * ctypes.sizeof(c_double))
    
    if not in_ptr or not out_ptr:
        if in_ptr:
            _fftw.fftw_free(in_ptr)
        if out_ptr:
            _fftw.fftw_free(out_ptr)
        raise FFTWError("Failed to allocate memory for real FFT")
    
    try:
        # Copy input data
        in_array = (c_double * n).from_address(in_ptr)
        for i, val in enumerate(data):
            in_array[i] = float(val)
        
        # Create plan for real-to-complex DFT
        plan = _fftw.fftw_plan_dft_r2c_1d(n, in_ptr, out_ptr, FFTW_ESTIMATE)
        
        if not plan:
            raise FFTWError("Failed to create FFTW plan for real FFT")
        
        try:
            # Execute the transform
            _fftw.fftw_execute(plan)
            
            # Extract output
            out_array = (c_double * (n_complex * 2)).from_address(out_ptr)
            result = []
            for i in range(n_complex):
                real = out_array[i * 2]
                imag = out_array[i * 2 + 1]
                result.append(complex(real, imag))
            
            return result
        
        finally:
            _fftw.fftw_destroy_plan(plan)
    
    finally:
        _fftw.fftw_free(in_ptr)
        _fftw.fftw_free(out_ptr)


def is_fftw_available() -> bool:
    """
    Check if FFTW library is available.
    
    Returns:
        True if FFTW is available, False otherwise.
    """
    return _fftw is not None

"""
GSL wrapper module - provides statistical functions using the GSL C library.

This module wraps GSL (GNU Scientific Library) using ctypes
to provide statistical operations for numerical data.
"""

import ctypes
from ctypes import POINTER, c_double, c_size_t, c_int
from typing import List, Optional, Union


def _load_gsl() -> tuple:
    """
    Load the GSL library and CBLAS dependency.
    
    Returns:
        Tuple of (GSL library, CBLAS library) or (None, None) if not found.
    """
    gsl_lib_names = [
        "libgsl.so.27",
        "libgsl.so",
        "libgsl.dylib",
        "/usr/lib/x86_64-linux-gnu/libgsl.so.27",
        "/usr/local/lib/libgsl.so",
    ]
    
    cblas_lib_names = [
        "libgslcblas.so.0",
        "libgslcblas.so",
        "libgslcblas.dylib",
        "/usr/lib/x86_64-linux-gnu/libgslcblas.so.0",
        "/usr/local/lib/libgslcblas.so",
    ]
    
    cblas = None
    for name in cblas_lib_names:
        try:
            cblas = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue
    
    gsl = None
    for name in gsl_lib_names:
        try:
            gsl = ctypes.CDLL(name)
            break
        except OSError:
            continue
    
    return gsl, cblas


# Load GSL library
_gsl, _cblas = _load_gsl()


class GSLError(Exception):
    """Exception raised for GSL-related errors."""
    pass


def _check_gsl_available() -> None:
    """Check if GSL is available and raise an error if not."""
    if _gsl is None:
        raise GSLError(
            "GSL library not found. Please install libgsl: "
            "apt-get install libgsl-dev (Linux) or brew install gsl (macOS)"
        )


def _setup_gsl_functions() -> None:
    """Set up GSL function signatures."""
    if _gsl is None:
        return
    
    # gsl_stats_mean
    _gsl.gsl_stats_mean.argtypes = [POINTER(c_double), c_size_t, c_size_t]
    _gsl.gsl_stats_mean.restype = c_double
    
    # gsl_stats_variance
    _gsl.gsl_stats_variance.argtypes = [POINTER(c_double), c_size_t, c_size_t]
    _gsl.gsl_stats_variance.restype = c_double
    
    # gsl_stats_variance_m (variance with known mean)
    _gsl.gsl_stats_variance_m.argtypes = [POINTER(c_double), c_size_t, c_size_t, c_double]
    _gsl.gsl_stats_variance_m.restype = c_double
    
    # gsl_stats_sd (standard deviation)
    _gsl.gsl_stats_sd.argtypes = [POINTER(c_double), c_size_t, c_size_t]
    _gsl.gsl_stats_sd.restype = c_double
    
    # gsl_stats_sd_m (standard deviation with known mean)
    _gsl.gsl_stats_sd_m.argtypes = [POINTER(c_double), c_size_t, c_size_t, c_double]
    _gsl.gsl_stats_sd_m.restype = c_double
    
    # gsl_stats_median (requires sorted data)
    _gsl.gsl_stats_median.argtypes = [POINTER(c_double), c_size_t, c_size_t]
    _gsl.gsl_stats_median.restype = c_double
    
    # gsl_stats_min
    _gsl.gsl_stats_min.argtypes = [POINTER(c_double), c_size_t, c_size_t]
    _gsl.gsl_stats_min.restype = c_double
    
    # gsl_stats_max
    _gsl.gsl_stats_max.argtypes = [POINTER(c_double), c_size_t, c_size_t]
    _gsl.gsl_stats_max.restype = c_double
    
    # gsl_stats_skew
    _gsl.gsl_stats_skew.argtypes = [POINTER(c_double), c_size_t, c_size_t]
    _gsl.gsl_stats_skew.restype = c_double
    
    # gsl_stats_kurtosis
    _gsl.gsl_stats_kurtosis.argtypes = [POINTER(c_double), c_size_t, c_size_t]
    _gsl.gsl_stats_kurtosis.restype = c_double
    
    # gsl_sort
    _gsl.gsl_sort.argtypes = [POINTER(c_double), c_size_t, c_size_t]
    _gsl.gsl_sort.restype = None


# Set up function signatures if library is loaded
_setup_gsl_functions()


def _to_c_array(data: List[Union[int, float]]) -> tuple:
    """
    Convert Python list to C array.
    
    Args:
        data: Input list of numbers.
    
    Returns:
        Tuple of (ctypes array, length).
    """
    n = len(data)
    c_array = (c_double * n)(*[float(x) for x in data])
    return c_array, n


def mean(data: List[Union[int, float]]) -> float:
    """
    Calculate the arithmetic mean of the data.
    
    Uses GSL's gsl_stats_mean function for efficient computation.
    
    Args:
        data: List of numeric values.
    
    Returns:
        The arithmetic mean.
    
    Raises:
        GSLError: If GSL library is not available.
        ValueError: If data is empty.
    """
    _check_gsl_available()
    
    if not data:
        raise ValueError("Data cannot be empty")
    
    c_array, n = _to_c_array(data)
    return _gsl.gsl_stats_mean(c_array, 1, n)


def variance(data: List[Union[int, float]], mean_val: Optional[float] = None) -> float:
    """
    Calculate the sample variance of the data.
    
    Uses GSL's gsl_stats_variance function for efficient computation.
    The variance is calculated with N-1 in the denominator (sample variance).
    
    Args:
        data: List of numeric values.
        mean_val: Optional pre-computed mean value.
    
    Returns:
        The sample variance.
    
    Raises:
        GSLError: If GSL library is not available.
        ValueError: If data is empty or has fewer than 2 elements.
    """
    _check_gsl_available()
    
    if not data:
        raise ValueError("Data cannot be empty")
    
    if len(data) < 2:
        raise ValueError("Variance requires at least 2 data points")
    
    c_array, n = _to_c_array(data)
    
    if mean_val is not None:
        return _gsl.gsl_stats_variance_m(c_array, 1, n, c_double(mean_val))
    else:
        return _gsl.gsl_stats_variance(c_array, 1, n)


def std_dev(data: List[Union[int, float]], mean_val: Optional[float] = None) -> float:
    """
    Calculate the sample standard deviation of the data.
    
    Uses GSL's gsl_stats_sd function for efficient computation.
    The standard deviation is calculated with N-1 in the denominator.
    
    Args:
        data: List of numeric values.
        mean_val: Optional pre-computed mean value.
    
    Returns:
        The sample standard deviation.
    
    Raises:
        GSLError: If GSL library is not available.
        ValueError: If data is empty or has fewer than 2 elements.
    """
    _check_gsl_available()
    
    if not data:
        raise ValueError("Data cannot be empty")
    
    if len(data) < 2:
        raise ValueError("Standard deviation requires at least 2 data points")
    
    c_array, n = _to_c_array(data)
    
    if mean_val is not None:
        return _gsl.gsl_stats_sd_m(c_array, 1, n, c_double(mean_val))
    else:
        return _gsl.gsl_stats_sd(c_array, 1, n)


def median(data: List[Union[int, float]]) -> float:
    """
    Calculate the median of the data.
    
    Uses GSL's gsl_stats_median function for efficient computation.
    The data is sorted internally before computing the median.
    
    Args:
        data: List of numeric values.
    
    Returns:
        The median value.
    
    Raises:
        GSLError: If GSL library is not available.
        ValueError: If data is empty.
    """
    _check_gsl_available()
    
    if not data:
        raise ValueError("Data cannot be empty")
    
    # GSL requires sorted data for median
    sorted_data = sorted(data)
    c_array, n = _to_c_array(sorted_data)
    
    return _gsl.gsl_stats_median(c_array, 1, n)


def min_val(data: List[Union[int, float]]) -> float:
    """
    Find the minimum value in the data.
    
    Uses GSL's gsl_stats_min function.
    
    Args:
        data: List of numeric values.
    
    Returns:
        The minimum value.
    
    Raises:
        GSLError: If GSL library is not available.
        ValueError: If data is empty.
    """
    _check_gsl_available()
    
    if not data:
        raise ValueError("Data cannot be empty")
    
    c_array, n = _to_c_array(data)
    return _gsl.gsl_stats_min(c_array, 1, n)


def max_val(data: List[Union[int, float]]) -> float:
    """
    Find the maximum value in the data.
    
    Uses GSL's gsl_stats_max function.
    
    Args:
        data: List of numeric values.
    
    Returns:
        The maximum value.
    
    Raises:
        GSLError: If GSL library is not available.
        ValueError: If data is empty.
    """
    _check_gsl_available()
    
    if not data:
        raise ValueError("Data cannot be empty")
    
    c_array, n = _to_c_array(data)
    return _gsl.gsl_stats_max(c_array, 1, n)


def skewness(data: List[Union[int, float]]) -> float:
    """
    Calculate the skewness of the data.
    
    Uses GSL's gsl_stats_skew function.
    
    Args:
        data: List of numeric values.
    
    Returns:
        The skewness.
    
    Raises:
        GSLError: If GSL library is not available.
        ValueError: If data is empty or has fewer than 2 elements.
    """
    _check_gsl_available()
    
    if not data:
        raise ValueError("Data cannot be empty")
    
    if len(data) < 3:
        raise ValueError("Skewness requires at least 3 data points")
    
    c_array, n = _to_c_array(data)
    return _gsl.gsl_stats_skew(c_array, 1, n)


def kurtosis(data: List[Union[int, float]]) -> float:
    """
    Calculate the kurtosis of the data.
    
    Uses GSL's gsl_stats_kurtosis function.
    Returns excess kurtosis (Fisher's definition), which is 0 for a normal distribution.
    
    Args:
        data: List of numeric values.
    
    Returns:
        The excess kurtosis.
    
    Raises:
        GSLError: If GSL library is not available.
        ValueError: If data is empty or has fewer than 2 elements.
    """
    _check_gsl_available()
    
    if not data:
        raise ValueError("Data cannot be empty")
    
    if len(data) < 4:
        raise ValueError("Kurtosis requires at least 4 data points")
    
    c_array, n = _to_c_array(data)
    return _gsl.gsl_stats_kurtosis(c_array, 1, n)


def is_gsl_available() -> bool:
    """
    Check if GSL library is available.
    
    Returns:
        True if GSL is available, False otherwise.
    """
    return _gsl is not None

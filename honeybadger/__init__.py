"""
HoneyBadger - DataFrame with CSV reading, basic operations, filtering, and C library wrappers.

This library provides:
- DataFrame class for data manipulation
- CSV reading functionality
- C library wrappers for FFTW (FFT) and GSL (statistics)
"""

from honeybadger.dataframe import DataFrame
from honeybadger.fftw_wrapper import fft, ifft
from honeybadger.gsl_wrapper import mean, variance, std_dev, median, min_val, max_val

__version__ = "0.1.0"
__all__ = [
    "DataFrame",
    "fft",
    "ifft",
    "mean",
    "variance",
    "std_dev",
    "median",
    "min_val",
    "max_val",
]

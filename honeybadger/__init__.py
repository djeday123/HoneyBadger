"""
HoneyBadger - DataFrame with CSV reading, basic operations, filtering,
and C library wrappers for FFTW (FFT) and GSL (statistics).
"""

from .dataframe import DataFrame
from .fftw_wrapper import FFTWrapper
from .gsl_wrapper import GSLStats

__version__ = "0.1.0"
__all__ = ["DataFrame", "FFTWrapper", "GSLStats"]

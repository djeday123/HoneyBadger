"""
GSL wrapper for statistical operations using cffi.
"""

from typing import List, Optional, Tuple
import math

try:
    from cffi import FFI
    _ffi = FFI()
    
    # Define GSL statistics interface
    _ffi.cdef("""
        double gsl_stats_mean(const double data[], size_t stride, size_t n);
        double gsl_stats_variance(const double data[], size_t stride, size_t n);
        double gsl_stats_variance_m(const double data[], size_t stride, size_t n, double mean);
        double gsl_stats_sd(const double data[], size_t stride, size_t n);
        double gsl_stats_sd_m(const double data[], size_t stride, size_t n, double mean);
        double gsl_stats_variance_with_fixed_mean(const double data[], size_t stride, size_t n, double mean);
        double gsl_stats_sd_with_fixed_mean(const double data[], size_t stride, size_t n, double mean);
        double gsl_stats_absdev(const double data[], size_t stride, size_t n);
        double gsl_stats_absdev_m(const double data[], size_t stride, size_t n, double mean);
        double gsl_stats_skew(const double data[], size_t stride, size_t n);
        double gsl_stats_skew_m_sd(const double data[], size_t stride, size_t n, double mean, double sd);
        double gsl_stats_kurtosis(const double data[], size_t stride, size_t n);
        double gsl_stats_kurtosis_m_sd(const double data[], size_t stride, size_t n, double mean, double sd);
        double gsl_stats_lag1_autocorrelation(const double data[], const size_t stride, const size_t n);
        double gsl_stats_lag1_autocorrelation_m(const double data[], const size_t stride, const size_t n, const double mean);
        double gsl_stats_covariance(const double data1[], size_t stride1, const double data2[], size_t stride2, size_t n);
        double gsl_stats_covariance_m(const double data1[], size_t stride1, const double data2[], size_t stride2, size_t n, double mean1, double mean2);
        double gsl_stats_correlation(const double data1[], size_t stride1, const double data2[], size_t stride2, size_t n);
        double gsl_stats_max(const double data[], size_t stride, size_t n);
        double gsl_stats_min(const double data[], size_t stride, size_t n);
        void gsl_stats_minmax(double *min, double *max, const double data[], size_t stride, size_t n);
        size_t gsl_stats_max_index(const double data[], size_t stride, size_t n);
        size_t gsl_stats_min_index(const double data[], size_t stride, size_t n);
        void gsl_stats_minmax_index(size_t *min_index, size_t *max_index, const double data[], size_t stride, size_t n);
        double gsl_stats_median_from_sorted_data(const double sorted_data[], size_t stride, size_t n);
        double gsl_stats_quantile_from_sorted_data(const double sorted_data[], size_t stride, size_t n, double f);
    """)
    
    try:
        _lib = _ffi.dlopen("libgsl.so.27")
        _GSL_AVAILABLE = True
    except OSError:
        try:
            _lib = _ffi.dlopen("libgsl.so")
            _GSL_AVAILABLE = True
        except OSError:
            _lib = None
            _GSL_AVAILABLE = False
except ImportError:
    _ffi = None
    _lib = None
    _GSL_AVAILABLE = False


class GSLStats:
    """
    Wrapper for GSL (GNU Scientific Library) statistical functions.
    
    Falls back to pure Python implementations if GSL is not available.
    """
    
    def __init__(self):
        """Initialize the GSL statistics wrapper."""
        self._gsl_available = _GSL_AVAILABLE
    
    @property
    def is_gsl_available(self) -> bool:
        """Check if GSL library is available."""
        return self._gsl_available
    
    def _to_c_array(self, data: List[float]) -> tuple:
        """Convert Python list to C array."""
        arr = _ffi.new(f"double[{len(data)}]", data)
        return arr, len(data)
    
    def mean(self, data: List[float], use_gsl: bool = True) -> float:
        """
        Compute the arithmetic mean of the data.
        
        Args:
            data: Input data.
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The arithmetic mean.
        """
        if not data:
            raise ValueError("Cannot compute mean of empty data")
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(data)
            return _lib.gsl_stats_mean(arr, 1, n)
        
        return sum(data) / len(data)
    
    def variance(
        self,
        data: List[float],
        mean: Optional[float] = None,
        use_gsl: bool = True
    ) -> float:
        """
        Compute the variance of the data.
        
        Args:
            data: Input data.
            mean: Pre-computed mean (optional).
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The variance (using N-1 denominator for sample variance).
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for variance")
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(data)
            if mean is not None:
                return _lib.gsl_stats_variance_m(arr, 1, n, mean)
            return _lib.gsl_stats_variance(arr, 1, n)
        
        if mean is None:
            mean = self.mean(data, use_gsl=False)
        
        n = len(data)
        return sum((x - mean) ** 2 for x in data) / (n - 1)
    
    def std(
        self,
        data: List[float],
        mean: Optional[float] = None,
        use_gsl: bool = True
    ) -> float:
        """
        Compute the standard deviation of the data.
        
        Args:
            data: Input data.
            mean: Pre-computed mean (optional).
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The standard deviation.
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for standard deviation")
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(data)
            if mean is not None:
                return _lib.gsl_stats_sd_m(arr, 1, n, mean)
            return _lib.gsl_stats_sd(arr, 1, n)
        
        return math.sqrt(self.variance(data, mean, use_gsl=False))
    
    def absdev(
        self,
        data: List[float],
        mean: Optional[float] = None,
        use_gsl: bool = True
    ) -> float:
        """
        Compute the absolute deviation from the mean.
        
        Args:
            data: Input data.
            mean: Pre-computed mean (optional).
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The mean absolute deviation.
        """
        if not data:
            raise ValueError("Cannot compute absolute deviation of empty data")
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(data)
            if mean is not None:
                return _lib.gsl_stats_absdev_m(arr, 1, n, mean)
            return _lib.gsl_stats_absdev(arr, 1, n)
        
        if mean is None:
            mean = self.mean(data, use_gsl=False)
        
        return sum(abs(x - mean) for x in data) / len(data)
    
    def skewness(
        self,
        data: List[float],
        mean: Optional[float] = None,
        sd: Optional[float] = None,
        use_gsl: bool = True
    ) -> float:
        """
        Compute the skewness of the data.
        
        Args:
            data: Input data.
            mean: Pre-computed mean (optional).
            sd: Pre-computed standard deviation (optional).
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The skewness.
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for skewness")
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(data)
            if mean is not None and sd is not None:
                return _lib.gsl_stats_skew_m_sd(arr, 1, n, mean, sd)
            return _lib.gsl_stats_skew(arr, 1, n)
        
        if mean is None:
            mean = self.mean(data, use_gsl=False)
        if sd is None:
            sd = self.std(data, mean, use_gsl=False)
        
        if sd == 0:
            return 0.0
        
        n = len(data)
        return sum(((x - mean) / sd) ** 3 for x in data) / n
    
    def kurtosis(
        self,
        data: List[float],
        mean: Optional[float] = None,
        sd: Optional[float] = None,
        use_gsl: bool = True
    ) -> float:
        """
        Compute the excess kurtosis of the data.
        
        Args:
            data: Input data.
            mean: Pre-computed mean (optional).
            sd: Pre-computed standard deviation (optional).
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The excess kurtosis (kurtosis - 3).
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for kurtosis")
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(data)
            if mean is not None and sd is not None:
                return _lib.gsl_stats_kurtosis_m_sd(arr, 1, n, mean, sd)
            return _lib.gsl_stats_kurtosis(arr, 1, n)
        
        if mean is None:
            mean = self.mean(data, use_gsl=False)
        if sd is None:
            sd = self.std(data, mean, use_gsl=False)
        
        if sd == 0:
            return 0.0
        
        n = len(data)
        return sum(((x - mean) / sd) ** 4 for x in data) / n - 3
    
    def autocorrelation(
        self,
        data: List[float],
        mean: Optional[float] = None,
        use_gsl: bool = True
    ) -> float:
        """
        Compute the lag-1 autocorrelation.
        
        Args:
            data: Input data.
            mean: Pre-computed mean (optional).
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The lag-1 autocorrelation coefficient.
        """
        if len(data) < 2:
            raise ValueError("Need at least 2 data points for autocorrelation")
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(data)
            if mean is not None:
                return _lib.gsl_stats_lag1_autocorrelation_m(arr, 1, n, mean)
            return _lib.gsl_stats_lag1_autocorrelation(arr, 1, n)
        
        if mean is None:
            mean = self.mean(data, use_gsl=False)
        
        n = len(data)
        numerator = sum((data[i] - mean) * (data[i + 1] - mean) for i in range(n - 1))
        denominator = sum((x - mean) ** 2 for x in data)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def covariance(
        self,
        data1: List[float],
        data2: List[float],
        mean1: Optional[float] = None,
        mean2: Optional[float] = None,
        use_gsl: bool = True
    ) -> float:
        """
        Compute the covariance between two datasets.
        
        Args:
            data1: First dataset.
            data2: Second dataset.
            mean1: Pre-computed mean of first dataset (optional).
            mean2: Pre-computed mean of second dataset (optional).
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The covariance.
        """
        if len(data1) != len(data2):
            raise ValueError("Data arrays must have the same length")
        
        if len(data1) < 2:
            raise ValueError("Need at least 2 data points for covariance")
        
        if use_gsl and self._gsl_available:
            arr1, n = self._to_c_array(data1)
            arr2, _ = self._to_c_array(data2)
            if mean1 is not None and mean2 is not None:
                return _lib.gsl_stats_covariance_m(arr1, 1, arr2, 1, n, mean1, mean2)
            return _lib.gsl_stats_covariance(arr1, 1, arr2, 1, n)
        
        if mean1 is None:
            mean1 = self.mean(data1, use_gsl=False)
        if mean2 is None:
            mean2 = self.mean(data2, use_gsl=False)
        
        n = len(data1)
        return sum((data1[i] - mean1) * (data2[i] - mean2) for i in range(n)) / (n - 1)
    
    def correlation(
        self,
        data1: List[float],
        data2: List[float],
        use_gsl: bool = True
    ) -> float:
        """
        Compute the Pearson correlation coefficient.
        
        Args:
            data1: First dataset.
            data2: Second dataset.
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The Pearson correlation coefficient.
        """
        if len(data1) != len(data2):
            raise ValueError("Data arrays must have the same length")
        
        if len(data1) < 2:
            raise ValueError("Need at least 2 data points for correlation")
        
        if use_gsl and self._gsl_available:
            arr1, n = self._to_c_array(data1)
            arr2, _ = self._to_c_array(data2)
            return _lib.gsl_stats_correlation(arr1, 1, arr2, 1, n)
        
        mean1 = self.mean(data1, use_gsl=False)
        mean2 = self.mean(data2, use_gsl=False)
        
        cov = self.covariance(data1, data2, mean1, mean2, use_gsl=False)
        std1 = self.std(data1, mean1, use_gsl=False)
        std2 = self.std(data2, mean2, use_gsl=False)
        
        if std1 == 0 or std2 == 0:
            return 0.0
        
        return cov / (std1 * std2)
    
    def max(self, data: List[float], use_gsl: bool = True) -> float:
        """
        Find the maximum value in the data.
        
        Args:
            data: Input data.
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The maximum value.
        """
        if not data:
            raise ValueError("Cannot find max of empty data")
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(data)
            return _lib.gsl_stats_max(arr, 1, n)
        
        return max(data)
    
    def min(self, data: List[float], use_gsl: bool = True) -> float:
        """
        Find the minimum value in the data.
        
        Args:
            data: Input data.
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The minimum value.
        """
        if not data:
            raise ValueError("Cannot find min of empty data")
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(data)
            return _lib.gsl_stats_min(arr, 1, n)
        
        return min(data)
    
    def minmax(self, data: List[float], use_gsl: bool = True) -> Tuple[float, float]:
        """
        Find both minimum and maximum values.
        
        Args:
            data: Input data.
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            Tuple of (min, max).
        """
        if not data:
            raise ValueError("Cannot find minmax of empty data")
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(data)
            min_val = _ffi.new("double *")
            max_val = _ffi.new("double *")
            _lib.gsl_stats_minmax(min_val, max_val, arr, 1, n)
            return (min_val[0], max_val[0])
        
        return (min(data), max(data))
    
    def max_index(self, data: List[float], use_gsl: bool = True) -> int:
        """
        Find the index of the maximum value.
        
        Args:
            data: Input data.
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The index of the maximum value.
        """
        if not data:
            raise ValueError("Cannot find max_index of empty data")
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(data)
            return _lib.gsl_stats_max_index(arr, 1, n)
        
        return data.index(max(data))
    
    def min_index(self, data: List[float], use_gsl: bool = True) -> int:
        """
        Find the index of the minimum value.
        
        Args:
            data: Input data.
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The index of the minimum value.
        """
        if not data:
            raise ValueError("Cannot find min_index of empty data")
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(data)
            return _lib.gsl_stats_min_index(arr, 1, n)
        
        return data.index(min(data))
    
    def median(self, data: List[float], use_gsl: bool = True) -> float:
        """
        Compute the median of the data.
        
        Args:
            data: Input data.
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The median value.
        """
        if not data:
            raise ValueError("Cannot compute median of empty data")
        
        sorted_data = sorted(data)
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(sorted_data)
            return _lib.gsl_stats_median_from_sorted_data(arr, 1, n)
        
        n = len(sorted_data)
        if n % 2 == 0:
            return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        return sorted_data[n // 2]
    
    def quantile(
        self,
        data: List[float],
        q: float,
        use_gsl: bool = True
    ) -> float:
        """
        Compute a quantile of the data.
        
        Args:
            data: Input data.
            q: Quantile to compute (0 <= q <= 1).
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The q-th quantile.
        """
        if not data:
            raise ValueError("Cannot compute quantile of empty data")
        
        if q < 0 or q > 1:
            raise ValueError("Quantile must be between 0 and 1")
        
        sorted_data = sorted(data)
        
        if use_gsl and self._gsl_available:
            arr, n = self._to_c_array(sorted_data)
            return _lib.gsl_stats_quantile_from_sorted_data(arr, 1, n, q)
        
        n = len(sorted_data)
        index = q * (n - 1)
        lower = int(index)
        upper = lower + 1
        weight = index - lower
        
        if upper >= n:
            return sorted_data[-1]
        
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    def percentile(
        self,
        data: List[float],
        p: float,
        use_gsl: bool = True
    ) -> float:
        """
        Compute a percentile of the data.
        
        Args:
            data: Input data.
            p: Percentile to compute (0 <= p <= 100).
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            The p-th percentile.
        """
        if p < 0 or p > 100:
            raise ValueError("Percentile must be between 0 and 100")
        
        return self.quantile(data, p / 100, use_gsl)
    
    def describe(self, data: List[float], use_gsl: bool = True) -> dict:
        """
        Compute summary statistics for the data.
        
        Args:
            data: Input data.
            use_gsl: Use GSL if available (default: True).
            
        Returns:
            Dictionary with count, mean, std, min, 25%, 50%, 75%, max.
        """
        if not data:
            raise ValueError("Cannot describe empty data")
        
        return {
            "count": len(data),
            "mean": self.mean(data, use_gsl),
            "std": self.std(data, use_gsl=use_gsl) if len(data) > 1 else 0.0,
            "min": self.min(data, use_gsl),
            "25%": self.quantile(data, 0.25, use_gsl),
            "50%": self.median(data, use_gsl),
            "75%": self.quantile(data, 0.75, use_gsl),
            "max": self.max(data, use_gsl),
        }

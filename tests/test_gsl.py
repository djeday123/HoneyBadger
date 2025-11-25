"""Tests for GSL statistics wrapper."""

import math
import pytest

from honeybadger import GSLStats


class TestGSLBasicStats:
    """Test basic statistical functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stats = GSLStats()
        self.data = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    def test_mean(self):
        """Test mean computation."""
        result = self.stats.mean(self.data)
        assert abs(result - 3.0) < 1e-10
    
    def test_mean_empty(self):
        """Test mean of empty data."""
        with pytest.raises(ValueError):
            self.stats.mean([])
    
    def test_variance(self):
        """Test variance computation."""
        result = self.stats.variance(self.data)
        # Sample variance with n-1 denominator
        expected = 2.5  # (10) / 4
        assert abs(result - expected) < 1e-10
    
    def test_variance_with_mean(self):
        """Test variance with pre-computed mean."""
        mean = 3.0
        result = self.stats.variance(self.data, mean=mean)
        assert abs(result - 2.5) < 1e-10
    
    def test_variance_too_few(self):
        """Test variance with too few data points."""
        with pytest.raises(ValueError):
            self.stats.variance([1.0])
    
    def test_std(self):
        """Test standard deviation."""
        result = self.stats.std(self.data)
        expected = math.sqrt(2.5)
        assert abs(result - expected) < 1e-10
    
    def test_absdev(self):
        """Test absolute deviation."""
        result = self.stats.absdev(self.data)
        # Mean is 3, deviations are 2, 1, 0, 1, 2 -> mean = 1.2
        expected = 1.2
        assert abs(result - expected) < 1e-10
    
    def test_skewness(self):
        """Test skewness of symmetric data."""
        # Symmetric data should have skewness close to 0
        result = self.stats.skewness(self.data)
        assert abs(result) < 1e-10
    
    def test_kurtosis(self):
        """Test kurtosis."""
        result = self.stats.kurtosis(self.data)
        # Uniform-ish distribution has negative excess kurtosis
        assert isinstance(result, float)


class TestGSLMinMax:
    """Test min/max functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stats = GSLStats()
        self.data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    
    def test_max(self):
        """Test maximum value."""
        assert self.stats.max(self.data) == 9.0
    
    def test_min(self):
        """Test minimum value."""
        assert self.stats.min(self.data) == 1.0
    
    def test_minmax(self):
        """Test minmax tuple."""
        min_val, max_val = self.stats.minmax(self.data)
        assert min_val == 1.0
        assert max_val == 9.0
    
    def test_max_index(self):
        """Test index of maximum."""
        assert self.stats.max_index(self.data) == 5
    
    def test_min_index(self):
        """Test index of minimum."""
        assert self.stats.min_index(self.data) == 1  # First occurrence
    
    def test_empty_data(self):
        """Test with empty data."""
        with pytest.raises(ValueError):
            self.stats.max([])
        with pytest.raises(ValueError):
            self.stats.min([])
        with pytest.raises(ValueError):
            self.stats.minmax([])


class TestGSLQuantiles:
    """Test quantile/percentile functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stats = GSLStats()
        self.data = list(range(1, 101))  # 1 to 100
    
    def test_median(self):
        """Test median computation."""
        result = self.stats.median(self.data)
        assert result == 50.5  # Average of 50 and 51
    
    def test_median_odd(self):
        """Test median with odd number of elements."""
        result = self.stats.median([1.0, 2.0, 3.0])
        assert result == 2.0
    
    def test_quantile_0(self):
        """Test 0th quantile (minimum)."""
        result = self.stats.quantile(self.data, 0)
        assert result == 1.0
    
    def test_quantile_1(self):
        """Test 1st quantile (maximum)."""
        result = self.stats.quantile(self.data, 1)
        assert result == 100.0
    
    def test_quantile_0_5(self):
        """Test 0.5 quantile (median)."""
        result = self.stats.quantile(self.data, 0.5)
        median = self.stats.median(self.data)
        assert abs(result - median) < 1e-10
    
    def test_percentile(self):
        """Test percentile computation."""
        result = self.stats.percentile(self.data, 50)
        median = self.stats.median(self.data)
        assert abs(result - median) < 1e-10
    
    def test_percentile_invalid(self):
        """Test invalid percentile."""
        with pytest.raises(ValueError):
            self.stats.percentile(self.data, 101)
        with pytest.raises(ValueError):
            self.stats.percentile(self.data, -1)


class TestGSLCovariance:
    """Test covariance and correlation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stats = GSLStats()
    
    def test_covariance(self):
        """Test covariance computation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]  # y = 2x
        
        result = self.stats.covariance(x, y)
        # Should be positive since x and y increase together
        assert result > 0
    
    def test_covariance_negative(self):
        """Test negative covariance."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]  # y decreases as x increases
        
        result = self.stats.covariance(x, y)
        assert result < 0
    
    def test_correlation_perfect(self):
        """Test perfect correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]  # y = 2x
        
        result = self.stats.correlation(x, y)
        assert abs(result - 1.0) < 1e-10
    
    def test_correlation_negative(self):
        """Test perfect negative correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]  # y decreases linearly
        
        result = self.stats.correlation(x, y)
        assert abs(result - (-1.0)) < 1e-10
    
    def test_correlation_length_mismatch(self):
        """Test correlation with mismatched lengths."""
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0]
        
        with pytest.raises(ValueError):
            self.stats.correlation(x, y)


class TestGSLAutocorrelation:
    """Test autocorrelation function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stats = GSLStats()
    
    def test_autocorrelation(self):
        """Test lag-1 autocorrelation."""
        # Highly correlated sequential data
        data = [float(i) for i in range(10)]
        result = self.stats.autocorrelation(data)
        
        # Should be positive and close to 1
        assert result > 0.5
    
    def test_autocorrelation_random(self):
        """Test autocorrelation of random-like data."""
        # Alternating pattern has negative autocorrelation
        data = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        result = self.stats.autocorrelation(data)
        
        assert result < 0


class TestGSLDescribe:
    """Test describe function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stats = GSLStats()
    
    def test_describe(self):
        """Test summary statistics."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.stats.describe(data)
        
        assert result["count"] == 5
        assert result["mean"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert "50%" in result  # median
        assert "25%" in result  # first quartile
        assert "75%" in result  # third quartile
    
    def test_describe_empty(self):
        """Test describe with empty data."""
        with pytest.raises(ValueError):
            self.stats.describe([])


class TestGSLComparison:
    """Compare GSL and Python implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stats = GSLStats()
        self.data = [2.3, 5.7, 1.2, 8.4, 3.9, 6.1, 4.5]
    
    def test_mean_comparison(self):
        """Compare mean implementations."""
        gsl = self.stats.mean(self.data, use_gsl=True)
        python = self.stats.mean(self.data, use_gsl=False)
        assert abs(gsl - python) < 1e-10
    
    def test_variance_comparison(self):
        """Compare variance implementations."""
        gsl = self.stats.variance(self.data, use_gsl=True)
        python = self.stats.variance(self.data, use_gsl=False)
        assert abs(gsl - python) < 1e-10
    
    def test_std_comparison(self):
        """Compare std implementations."""
        gsl = self.stats.std(self.data, use_gsl=True)
        python = self.stats.std(self.data, use_gsl=False)
        assert abs(gsl - python) < 1e-10
    
    def test_median_comparison(self):
        """Compare median implementations."""
        gsl = self.stats.median(self.data, use_gsl=True)
        python = self.stats.median(self.data, use_gsl=False)
        assert abs(gsl - python) < 1e-10

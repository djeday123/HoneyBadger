"""Tests for the GSL wrapper."""

import math
import pytest
from honeybadger.gsl_wrapper import (
    mean, variance, std_dev, median, min_val, max_val,
    skewness, kurtosis, is_gsl_available, GSLError
)


# Skip all tests if GSL is not available
pytestmark = pytest.mark.skipif(
    not is_gsl_available(),
    reason="GSL library not available"
)


class TestMean:
    """Tests for mean function."""
    
    def test_mean_simple(self):
        """Test mean of simple data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = mean(data)
        
        assert abs(result - 3.0) < 1e-10
    
    def test_mean_single_value(self):
        """Test mean of single value."""
        data = [42.0]
        result = mean(data)
        
        assert abs(result - 42.0) < 1e-10
    
    def test_mean_negative_values(self):
        """Test mean with negative values."""
        data = [-5.0, 0.0, 5.0]
        result = mean(data)
        
        assert abs(result) < 1e-10
    
    def test_mean_integers(self):
        """Test mean with integer input."""
        data = [1, 2, 3, 4, 5]
        result = mean(data)
        
        assert abs(result - 3.0) < 1e-10
    
    def test_mean_empty_raises_error(self):
        """Test that mean of empty data raises ValueError."""
        with pytest.raises(ValueError):
            mean([])


class TestVariance:
    """Tests for variance function."""
    
    def test_variance_simple(self):
        """Test variance of simple data."""
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = variance(data)
        
        # Sample variance = sum((x - mean)^2) / (n-1)
        # Expected: 4.571428...
        expected = 4.571428571428571
        assert abs(result - expected) < 1e-6
    
    def test_variance_with_known_mean(self):
        """Test variance with pre-computed mean."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        m = mean(data)
        result = variance(data, mean_val=m)
        
        # Should be same as computing without mean
        result2 = variance(data)
        assert abs(result - result2) < 1e-10
    
    def test_variance_constant(self):
        """Test variance of constant data."""
        data = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = variance(data)
        
        assert abs(result) < 1e-10
    
    def test_variance_empty_raises_error(self):
        """Test that variance of empty data raises ValueError."""
        with pytest.raises(ValueError):
            variance([])
    
    def test_variance_single_value_raises_error(self):
        """Test that variance of single value raises ValueError."""
        with pytest.raises(ValueError):
            variance([42.0])


class TestStdDev:
    """Tests for standard deviation function."""
    
    def test_std_dev_simple(self):
        """Test standard deviation of simple data."""
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = std_dev(data)
        
        # std_dev should be sqrt(variance)
        expected = math.sqrt(4.571428571428571)
        assert abs(result - expected) < 1e-6
    
    def test_std_dev_with_known_mean(self):
        """Test standard deviation with pre-computed mean."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        m = mean(data)
        result = std_dev(data, mean_val=m)
        
        # Should be same as computing without mean
        result2 = std_dev(data)
        assert abs(result - result2) < 1e-10
    
    def test_std_dev_empty_raises_error(self):
        """Test that std_dev of empty data raises ValueError."""
        with pytest.raises(ValueError):
            std_dev([])
    
    def test_std_dev_single_value_raises_error(self):
        """Test that std_dev of single value raises ValueError."""
        with pytest.raises(ValueError):
            std_dev([42.0])


class TestMedian:
    """Tests for median function."""
    
    def test_median_odd_count(self):
        """Test median with odd number of elements."""
        data = [1.0, 3.0, 2.0, 5.0, 4.0]
        result = median(data)
        
        assert abs(result - 3.0) < 1e-10
    
    def test_median_even_count(self):
        """Test median with even number of elements."""
        data = [1.0, 2.0, 3.0, 4.0]
        result = median(data)
        
        # Median of even count is average of two middle values
        assert abs(result - 2.5) < 1e-10
    
    def test_median_single_value(self):
        """Test median of single value."""
        data = [42.0]
        result = median(data)
        
        assert abs(result - 42.0) < 1e-10
    
    def test_median_unsorted_input(self):
        """Test that median works with unsorted input."""
        data = [5.0, 1.0, 3.0, 2.0, 4.0]
        result = median(data)
        
        assert abs(result - 3.0) < 1e-10
    
    def test_median_empty_raises_error(self):
        """Test that median of empty data raises ValueError."""
        with pytest.raises(ValueError):
            median([])


class TestMinMax:
    """Tests for min and max functions."""
    
    def test_min_val_simple(self):
        """Test min_val of simple data."""
        data = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
        result = min_val(data)
        
        assert abs(result - 1.0) < 1e-10
    
    def test_max_val_simple(self):
        """Test max_val of simple data."""
        data = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
        result = max_val(data)
        
        assert abs(result - 9.0) < 1e-10
    
    def test_min_val_negative(self):
        """Test min_val with negative values."""
        data = [-5.0, -10.0, -1.0, 0.0, 5.0]
        result = min_val(data)
        
        assert abs(result - (-10.0)) < 1e-10
    
    def test_max_val_negative(self):
        """Test max_val with negative values."""
        data = [-5.0, -10.0, -1.0]
        result = max_val(data)
        
        assert abs(result - (-1.0)) < 1e-10
    
    def test_min_val_single(self):
        """Test min_val with single value."""
        data = [42.0]
        result = min_val(data)
        
        assert abs(result - 42.0) < 1e-10
    
    def test_max_val_single(self):
        """Test max_val with single value."""
        data = [42.0]
        result = max_val(data)
        
        assert abs(result - 42.0) < 1e-10
    
    def test_min_val_empty_raises_error(self):
        """Test that min_val of empty data raises ValueError."""
        with pytest.raises(ValueError):
            min_val([])
    
    def test_max_val_empty_raises_error(self):
        """Test that max_val of empty data raises ValueError."""
        with pytest.raises(ValueError):
            max_val([])


class TestSkewness:
    """Tests for skewness function."""
    
    def test_skewness_symmetric(self):
        """Test skewness of symmetric data (should be near zero)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = skewness(data)
        
        # Symmetric data should have skewness close to 0
        assert abs(result) < 1e-10
    
    def test_skewness_positive(self):
        """Test skewness of right-skewed data."""
        # Right tail is longer
        data = [1.0, 2.0, 2.0, 3.0, 10.0]
        result = skewness(data)
        
        # Positive skewness
        assert result > 0
    
    def test_skewness_empty_raises_error(self):
        """Test that skewness of empty data raises ValueError."""
        with pytest.raises(ValueError):
            skewness([])
    
    def test_skewness_insufficient_data_raises_error(self):
        """Test that skewness with fewer than 3 data points raises ValueError."""
        with pytest.raises(ValueError):
            skewness([1.0, 2.0])


class TestKurtosis:
    """Tests for kurtosis function."""
    
    def test_kurtosis_uniform(self):
        """Test kurtosis of uniform-ish data."""
        # Uniform distribution has negative excess kurtosis
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = kurtosis(data)
        
        # This is excess kurtosis, uniform has negative
        assert result < 0
    
    def test_kurtosis_empty_raises_error(self):
        """Test that kurtosis of empty data raises ValueError."""
        with pytest.raises(ValueError):
            kurtosis([])
    
    def test_kurtosis_insufficient_data_raises_error(self):
        """Test that kurtosis with fewer than 4 data points raises ValueError."""
        with pytest.raises(ValueError):
            kurtosis([1.0, 2.0, 3.0])


class TestGSLAvailability:
    """Tests for GSL availability check."""
    
    def test_is_gsl_available(self):
        """Test that is_gsl_available returns correct value."""
        # Since we're in a test that requires GSL, it should be available
        assert is_gsl_available() is True

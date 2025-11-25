"""Tests for the FFTW wrapper."""

import math
import pytest
from honeybadger.fftw_wrapper import fft, ifft, fft_real, is_fftw_available, FFTWError


# Skip all tests if FFTW is not available
pytestmark = pytest.mark.skipif(
    not is_fftw_available(),
    reason="FFTW library not available"
)


class TestFFT:
    """Tests for FFT function."""
    
    def test_fft_real_signal(self):
        """Test FFT of a simple real signal."""
        # Simple signal: [1, 0, 0, 0]
        # Expected: DC component = 1, all others = 1
        data = [1.0, 0.0, 0.0, 0.0]
        result = fft(data)
        
        assert len(result) == 4
        # All components should be 1+0j
        for val in result:
            assert abs(val.real - 1.0) < 1e-10
            assert abs(val.imag) < 1e-10
    
    def test_fft_constant_signal(self):
        """Test FFT of a constant signal."""
        # Constant signal should have all energy at DC
        data = [1.0, 1.0, 1.0, 1.0]
        result = fft(data)
        
        # DC component should equal sum of samples
        assert abs(result[0].real - 4.0) < 1e-10
        assert abs(result[0].imag) < 1e-10
        
        # All other components should be zero
        for i in range(1, len(result)):
            assert abs(result[i].real) < 1e-10
            assert abs(result[i].imag) < 1e-10
    
    def test_fft_sine_wave(self):
        """Test FFT of a sine wave."""
        # Create a sine wave at frequency 1
        n = 8
        data = [math.sin(2 * math.pi * i / n) for i in range(n)]
        result = fft(data)
        
        # The magnitude should peak at index 1 (and n-1 due to symmetry)
        magnitudes = [abs(c) for c in result]
        
        # Index 1 and 7 should have the highest magnitudes (excluding DC)
        assert magnitudes[1] > magnitudes[2]
        assert magnitudes[7] > magnitudes[6]
    
    def test_fft_complex_input(self):
        """Test FFT with complex input."""
        data = [1+1j, 2+2j, 3+3j, 4+4j]
        result = fft(data)
        
        assert len(result) == 4
        # Just verify we get complex output
        for val in result:
            assert isinstance(val, complex)
    
    def test_fft_empty_raises_error(self):
        """Test that FFT of empty data raises ValueError."""
        with pytest.raises(ValueError):
            fft([])


class TestIFFT:
    """Tests for IFFT function."""
    
    def test_ifft_roundtrip(self):
        """Test that FFT followed by IFFT returns original data."""
        data = [1.0, 2.0, 3.0, 4.0]
        
        fft_result = fft(data)
        ifft_result = ifft(fft_result)
        
        assert len(ifft_result) == len(data)
        for orig, recovered in zip(data, ifft_result):
            assert abs(recovered.real - orig) < 1e-10
            assert abs(recovered.imag) < 1e-10
    
    def test_ifft_complex_roundtrip(self):
        """Test IFFT roundtrip with complex input."""
        data = [1+1j, 2-1j, 3+2j, 4-2j]
        
        fft_result = fft(data)
        ifft_result = ifft(fft_result)
        
        for orig, recovered in zip(data, ifft_result):
            assert abs(recovered - orig) < 1e-10
    
    def test_ifft_empty_raises_error(self):
        """Test that IFFT of empty data raises ValueError."""
        with pytest.raises(ValueError):
            ifft([])


class TestFFTReal:
    """Tests for real FFT function."""
    
    def test_fft_real_output_length(self):
        """Test that real FFT returns correct number of components."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = fft_real(data)
        
        # For n real points, output should be n//2 + 1 complex points
        expected_len = len(data) // 2 + 1
        assert len(result) == expected_len
    
    def test_fft_real_constant(self):
        """Test real FFT of constant signal."""
        data = [1.0, 1.0, 1.0, 1.0]
        result = fft_real(data)
        
        # DC component should equal sum
        assert abs(result[0].real - 4.0) < 1e-10
        
        # Other components should be zero
        for i in range(1, len(result)):
            assert abs(result[i].real) < 1e-10
            assert abs(result[i].imag) < 1e-10
    
    def test_fft_real_empty_raises_error(self):
        """Test that real FFT of empty data raises ValueError."""
        with pytest.raises(ValueError):
            fft_real([])


class TestFFTWAvailability:
    """Tests for FFTW availability check."""
    
    def test_is_fftw_available(self):
        """Test that is_fftw_available returns correct value."""
        # Since we're in a test that requires FFTW, it should be available
        assert is_fftw_available() is True

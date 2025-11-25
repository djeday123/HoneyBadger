"""Tests for FFTW wrapper."""

import math
import pytest

from honeybadger import FFTWrapper


class TestFFTBasics:
    """Test basic FFT functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fft = FFTWrapper()
    
    def test_fft_simple(self):
        """Test FFT of simple signal."""
        # DC signal
        data = [complex(1)] * 4
        result = self.fft.fft(data)
        
        # DC component should be 4, others 0
        assert abs(result[0] - 4) < 1e-10
        for i in range(1, 4):
            assert abs(result[i]) < 1e-10
    
    def test_fft_cosine(self):
        """Test FFT of cosine wave."""
        n = 8
        # Cosine at frequency 1
        data = [complex(math.cos(2 * math.pi * i / n)) for i in range(n)]
        result = self.fft.fft(data)
        
        # Peaks at frequency 1 and n-1
        assert abs(result[1]) > 3
        assert abs(result[n - 1]) > 3
    
    def test_fft_power_of_2(self):
        """Test FFT with power of 2 length."""
        for n in [2, 4, 8, 16]:
            data = [complex(i) for i in range(n)]
            result = self.fft.fft(data)
            assert len(result) == n
    
    def test_fft_non_power_of_2(self):
        """Test FFT with non-power-of-2 length (uses DFT)."""
        data = [complex(i) for i in range(5)]
        result = self.fft.fft(data)
        assert len(result) == 5
    
    def test_ifft_inverse(self):
        """Test that IFFT is inverse of FFT."""
        data = [complex(1, 2), complex(3, 4), complex(5, 6), complex(7, 8)]
        
        fft_result = self.fft.fft(data)
        ifft_result = self.fft.ifft(fft_result)
        
        for i in range(len(data)):
            assert abs(data[i] - ifft_result[i]) < 1e-10
    
    def test_rfft(self):
        """Test real FFT."""
        data = [1.0, 2.0, 3.0, 4.0]
        result = self.fft.rfft(data)
        
        # Output should have n/2 + 1 elements
        assert len(result) == 3
    
    def test_irfft(self):
        """Test inverse real FFT."""
        data = [1.0, 2.0, 3.0, 4.0]
        
        fft_result = self.fft.rfft(data)
        ifft_result = self.fft.irfft(fft_result, len(data))
        
        for i in range(len(data)):
            assert abs(data[i] - ifft_result[i]) < 1e-10


class TestFFTHelpers:
    """Test FFT helper functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fft = FFTWrapper()
    
    def test_magnitude(self):
        """Test magnitude computation."""
        data = [complex(3, 4), complex(0, 1), complex(1, 0)]
        result = self.fft.magnitude(data)
        
        assert abs(result[0] - 5) < 1e-10
        assert abs(result[1] - 1) < 1e-10
        assert abs(result[2] - 1) < 1e-10
    
    def test_phase(self):
        """Test phase computation."""
        data = [complex(1, 0), complex(0, 1), complex(-1, 0)]
        result = self.fft.phase(data)
        
        assert abs(result[0] - 0) < 1e-10
        assert abs(result[1] - math.pi / 2) < 1e-10
        assert abs(abs(result[2]) - math.pi) < 1e-10
    
    def test_power_spectrum(self):
        """Test power spectrum computation."""
        data = [1.0, 2.0, 3.0, 4.0]
        result = self.fft.power_spectrum(data)
        
        # Power spectrum should have n/2 + 1 elements
        assert len(result) == 3
        # All values should be non-negative
        assert all(p >= 0 for p in result)
    
    def test_frequencies(self):
        """Test frequency computation."""
        freqs = FFTWrapper.frequencies(4, sample_rate=1.0)
        assert len(freqs) == 4
        assert freqs[0] == 0.0
        assert freqs[1] == 0.25
        assert freqs[2] == 0.5
        assert freqs[3] == -0.25
    
    def test_rfft_frequencies(self):
        """Test real FFT frequency computation."""
        freqs = FFTWrapper.rfft_frequencies(4, sample_rate=1.0)
        assert len(freqs) == 3
        assert freqs[0] == 0.0
        assert freqs[1] == 0.25
        assert freqs[2] == 0.5


class TestFFTWithFFTW:
    """Test FFT with FFTW backend if available."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fft = FFTWrapper()
    
    def test_fftw_availability(self):
        """Test FFTW availability check."""
        # This should not raise
        _ = self.fft.is_fftw_available
    
    def test_fft_fftw_vs_python(self):
        """Compare FFTW and Python implementations."""
        data = [complex(i, i + 1) for i in range(8)]
        
        result_fftw = self.fft.fft(data, use_fftw=True)
        result_python = self.fft.fft(data, use_fftw=False)
        
        for i in range(len(data)):
            assert abs(result_fftw[i] - result_python[i]) < 1e-10
    
    def test_ifft_fftw_vs_python(self):
        """Compare IFFT FFTW and Python implementations."""
        data = [complex(i, i + 1) for i in range(8)]
        
        result_fftw = self.fft.ifft(data, use_fftw=True)
        result_python = self.fft.ifft(data, use_fftw=False)
        
        for i in range(len(data)):
            assert abs(result_fftw[i] - result_python[i]) < 1e-10
    
    def test_rfft_fftw_vs_python(self):
        """Compare real FFT FFTW and Python implementations."""
        data = [float(i) for i in range(8)]
        
        result_fftw = self.fft.rfft(data, use_fftw=True)
        result_python = self.fft.rfft(data, use_fftw=False)
        
        for i in range(len(result_fftw)):
            assert abs(result_fftw[i] - result_python[i]) < 1e-10

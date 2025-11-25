"""
FFTW wrapper for Fast Fourier Transform operations using cffi.
"""

from typing import List, Optional, Tuple
import math

try:
    from cffi import FFI
    _ffi = FFI()
    
    # Define FFTW3 interface
    _ffi.cdef("""
        typedef double fftw_complex[2];
        
        typedef struct fftw_plan_s *fftw_plan;
        
        void *fftw_malloc(size_t n);
        void fftw_free(void *p);
        
        fftw_plan fftw_plan_dft_1d(int n, fftw_complex *in, fftw_complex *out,
                                   int sign, unsigned flags);
        fftw_plan fftw_plan_dft_r2c_1d(int n, double *in, fftw_complex *out,
                                        unsigned flags);
        fftw_plan fftw_plan_dft_c2r_1d(int n, fftw_complex *in, double *out,
                                        unsigned flags);
        
        void fftw_execute(const fftw_plan plan);
        void fftw_destroy_plan(fftw_plan plan);
        
        #define FFTW_FORWARD ...
        #define FFTW_BACKWARD ...
        #define FFTW_MEASURE ...
        #define FFTW_ESTIMATE ...
    """)
    
    try:
        _lib = _ffi.dlopen("libfftw3.so.3")
        _FFTW_AVAILABLE = True
    except OSError:
        try:
            _lib = _ffi.dlopen("libfftw3.so")
            _FFTW_AVAILABLE = True
        except OSError:
            _lib = None
            _FFTW_AVAILABLE = False
except ImportError:
    _ffi = None
    _lib = None
    _FFTW_AVAILABLE = False


class FFTWrapper:
    """
    Wrapper for FFTW3 library providing Fast Fourier Transform operations.
    
    Falls back to a pure Python implementation if FFTW3 is not available.
    """
    
    # FFTW flags
    FFTW_FORWARD = -1
    FFTW_BACKWARD = 1
    FFTW_ESTIMATE = 64  # FFTW_ESTIMATE flag value
    
    def __init__(self):
        """Initialize the FFT wrapper."""
        self._fftw_available = _FFTW_AVAILABLE
    
    @property
    def is_fftw_available(self) -> bool:
        """Check if FFTW library is available."""
        return self._fftw_available
    
    def fft(
        self,
        data: List[complex],
        use_fftw: bool = True
    ) -> List[complex]:
        """
        Compute the 1D Fast Fourier Transform.
        
        Args:
            data: Input complex data.
            use_fftw: Use FFTW if available (default: True).
            
        Returns:
            FFT of the input data.
        """
        if use_fftw and self._fftw_available:
            return self._fft_fftw(data)
        return self._fft_python(data)
    
    def ifft(
        self,
        data: List[complex],
        use_fftw: bool = True
    ) -> List[complex]:
        """
        Compute the 1D Inverse Fast Fourier Transform.
        
        Args:
            data: Input complex frequency data.
            use_fftw: Use FFTW if available (default: True).
            
        Returns:
            IFFT of the input data.
        """
        if use_fftw and self._fftw_available:
            return self._ifft_fftw(data)
        return self._ifft_python(data)
    
    def rfft(
        self,
        data: List[float],
        use_fftw: bool = True
    ) -> List[complex]:
        """
        Compute the 1D FFT of real-valued data.
        
        Args:
            data: Input real data.
            use_fftw: Use FFTW if available (default: True).
            
        Returns:
            FFT of the input data (only non-redundant frequencies).
        """
        if use_fftw and self._fftw_available:
            return self._rfft_fftw(data)
        return self._rfft_python(data)
    
    def irfft(
        self,
        data: List[complex],
        n: Optional[int] = None,
        use_fftw: bool = True
    ) -> List[float]:
        """
        Compute the 1D inverse FFT to produce real-valued output.
        
        Args:
            data: Input complex frequency data.
            n: Length of the output (default: 2 * (len(data) - 1)).
            use_fftw: Use FFTW if available (default: True).
            
        Returns:
            Real-valued inverse FFT of the input data.
        """
        if n is None:
            n = 2 * (len(data) - 1)
        
        if use_fftw and self._fftw_available:
            return self._irfft_fftw(data, n)
        return self._irfft_python(data, n)
    
    def magnitude(self, data: List[complex]) -> List[float]:
        """
        Compute the magnitude (absolute value) of complex data.
        
        Args:
            data: Input complex data.
            
        Returns:
            Magnitude of each element.
        """
        return [abs(z) for z in data]
    
    def phase(self, data: List[complex]) -> List[float]:
        """
        Compute the phase (argument) of complex data.
        
        Args:
            data: Input complex data.
            
        Returns:
            Phase of each element in radians.
        """
        import cmath
        return [cmath.phase(z) for z in data]
    
    def power_spectrum(self, data: List[float]) -> List[float]:
        """
        Compute the power spectrum of real-valued data.
        
        Args:
            data: Input real data.
            
        Returns:
            Power spectrum (magnitude squared of FFT).
        """
        fft_result = self.rfft(data)
        return [abs(z) ** 2 for z in fft_result]
    
    def _fft_fftw(self, data: List[complex]) -> List[complex]:
        """Compute FFT using FFTW."""
        n = len(data)
        
        # Allocate input and output arrays
        in_arr = _ffi.cast("fftw_complex *", _lib.fftw_malloc(n * _ffi.sizeof("fftw_complex")))
        out_arr = _ffi.cast("fftw_complex *", _lib.fftw_malloc(n * _ffi.sizeof("fftw_complex")))
        
        try:
            # Copy input data
            for i, z in enumerate(data):
                in_arr[i][0] = z.real
                in_arr[i][1] = z.imag
            
            # Create and execute plan
            plan = _lib.fftw_plan_dft_1d(n, in_arr, out_arr, self.FFTW_FORWARD, self.FFTW_ESTIMATE)
            _lib.fftw_execute(plan)
            _lib.fftw_destroy_plan(plan)
            
            # Extract results
            result = [complex(out_arr[i][0], out_arr[i][1]) for i in range(n)]
        finally:
            _lib.fftw_free(in_arr)
            _lib.fftw_free(out_arr)
        
        return result
    
    def _ifft_fftw(self, data: List[complex]) -> List[complex]:
        """Compute inverse FFT using FFTW."""
        n = len(data)
        
        # Allocate input and output arrays
        in_arr = _ffi.cast("fftw_complex *", _lib.fftw_malloc(n * _ffi.sizeof("fftw_complex")))
        out_arr = _ffi.cast("fftw_complex *", _lib.fftw_malloc(n * _ffi.sizeof("fftw_complex")))
        
        try:
            # Copy input data
            for i, z in enumerate(data):
                in_arr[i][0] = z.real
                in_arr[i][1] = z.imag
            
            # Create and execute plan
            plan = _lib.fftw_plan_dft_1d(n, in_arr, out_arr, self.FFTW_BACKWARD, self.FFTW_ESTIMATE)
            _lib.fftw_execute(plan)
            _lib.fftw_destroy_plan(plan)
            
            # Extract and normalize results
            result = [complex(out_arr[i][0] / n, out_arr[i][1] / n) for i in range(n)]
        finally:
            _lib.fftw_free(in_arr)
            _lib.fftw_free(out_arr)
        
        return result
    
    def _rfft_fftw(self, data: List[float]) -> List[complex]:
        """Compute real-to-complex FFT using FFTW."""
        n = len(data)
        n_out = n // 2 + 1
        
        # Allocate input and output arrays
        in_arr = _ffi.cast("double *", _lib.fftw_malloc(n * _ffi.sizeof("double")))
        out_arr = _ffi.cast("fftw_complex *", _lib.fftw_malloc(n_out * _ffi.sizeof("fftw_complex")))
        
        try:
            # Copy input data
            for i, v in enumerate(data):
                in_arr[i] = v
            
            # Create and execute plan
            plan = _lib.fftw_plan_dft_r2c_1d(n, in_arr, out_arr, self.FFTW_ESTIMATE)
            _lib.fftw_execute(plan)
            _lib.fftw_destroy_plan(plan)
            
            # Extract results
            result = [complex(out_arr[i][0], out_arr[i][1]) for i in range(n_out)]
        finally:
            _lib.fftw_free(in_arr)
            _lib.fftw_free(out_arr)
        
        return result
    
    def _irfft_fftw(self, data: List[complex], n: int) -> List[float]:
        """Compute complex-to-real inverse FFT using FFTW."""
        n_in = n // 2 + 1
        
        # Allocate input and output arrays
        in_arr = _ffi.cast("fftw_complex *", _lib.fftw_malloc(n_in * _ffi.sizeof("fftw_complex")))
        out_arr = _ffi.cast("double *", _lib.fftw_malloc(n * _ffi.sizeof("double")))
        
        try:
            # Copy input data
            for i in range(min(len(data), n_in)):
                z = data[i]
                in_arr[i][0] = z.real
                in_arr[i][1] = z.imag
            
            # Zero-pad if necessary
            for i in range(len(data), n_in):
                in_arr[i][0] = 0.0
                in_arr[i][1] = 0.0
            
            # Create and execute plan
            plan = _lib.fftw_plan_dft_c2r_1d(n, in_arr, out_arr, self.FFTW_ESTIMATE)
            _lib.fftw_execute(plan)
            _lib.fftw_destroy_plan(plan)
            
            # Extract and normalize results
            result = [out_arr[i] / n for i in range(n)]
        finally:
            _lib.fftw_free(in_arr)
            _lib.fftw_free(out_arr)
        
        return result
    
    def _fft_python(self, data: List[complex]) -> List[complex]:
        """Pure Python Cooley-Tukey FFT implementation."""
        n = len(data)
        
        if n <= 1:
            return list(data)
        
        if n & (n - 1) != 0:
            # Not a power of 2, use DFT
            return self._dft(data)
        
        # Cooley-Tukey radix-2 FFT
        even = self._fft_python(data[0::2])
        odd = self._fft_python(data[1::2])
        
        result = [complex(0)] * n
        for k in range(n // 2):
            t = math.e ** complex(0, -2 * math.pi * k / n) * odd[k]
            result[k] = even[k] + t
            result[k + n // 2] = even[k] - t
        
        return result
    
    def _ifft_python(self, data: List[complex]) -> List[complex]:
        """Pure Python inverse FFT implementation."""
        n = len(data)
        
        # Conjugate, FFT, conjugate, scale
        conjugated = [z.conjugate() for z in data]
        fft_result = self._fft_python(conjugated)
        result = [z.conjugate() / n for z in fft_result]
        
        return result
    
    def _rfft_python(self, data: List[float]) -> List[complex]:
        """Pure Python real-to-complex FFT implementation."""
        complex_data = [complex(x) for x in data]
        fft_result = self._fft_python(complex_data)
        n_out = len(data) // 2 + 1
        return fft_result[:n_out]
    
    def _irfft_python(self, data: List[complex], n: int) -> List[float]:
        """Pure Python complex-to-real inverse FFT implementation."""
        # Reconstruct full spectrum using Hermitian symmetry
        full_spectrum = list(data)
        for i in range(len(data), n):
            j = n - i
            if j < len(data):
                full_spectrum.append(data[j].conjugate())
            else:
                full_spectrum.append(complex(0))
        
        result = self._ifft_python(full_spectrum)
        return [z.real for z in result]
    
    def _dft(self, data: List[complex]) -> List[complex]:
        """Direct DFT computation for non-power-of-2 lengths."""
        n = len(data)
        result = []
        
        for k in range(n):
            total = complex(0)
            for j, x in enumerate(data):
                angle = -2 * math.pi * k * j / n
                total += x * complex(math.cos(angle), math.sin(angle))
            result.append(total)
        
        return result
    
    @staticmethod
    def frequencies(n: int, sample_rate: float = 1.0) -> List[float]:
        """
        Return the frequencies for FFT output.
        
        Args:
            n: Number of samples.
            sample_rate: Sample rate in Hz (default: 1.0).
            
        Returns:
            List of frequencies in Hz.
        """
        freqs = []
        for i in range(n):
            if i <= n // 2:
                freqs.append(i * sample_rate / n)
            else:
                freqs.append((i - n) * sample_rate / n)
        return freqs
    
    @staticmethod
    def rfft_frequencies(n: int, sample_rate: float = 1.0) -> List[float]:
        """
        Return the frequencies for real FFT output.
        
        Args:
            n: Number of samples in the original signal.
            sample_rate: Sample rate in Hz (default: 1.0).
            
        Returns:
            List of frequencies in Hz.
        """
        n_out = n // 2 + 1
        return [i * sample_rate / n for i in range(n_out)]

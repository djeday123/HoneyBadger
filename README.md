# HoneyBadger

DataFrame — CSV reading, basic operations, filtering, and C library wrappers — FFTW for FFT, GSL for statistics

## Installation

```bash
pip install .
```

### System Dependencies

For optimal performance with C library wrappers, install FFTW3 and GSL:

```bash
# Ubuntu/Debian
sudo apt-get install libfftw3-dev libgsl-dev

# macOS
brew install fftw gsl
```

## Usage

### DataFrame

```python
from honeybadger import DataFrame

# Create from dictionary
df = DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "score": [90.5, 85.0, 92.5]
})

# Read from CSV
df = DataFrame.from_csv("data.csv")

# Basic operations
print(df.shape)       # (3, 3)
print(df.columns)     # ['name', 'age', 'score']
print(df.head(2))     # First 2 rows
print(df.tail(2))     # Last 2 rows

# Access columns
names = df["name"]    # ['Alice', 'Bob', 'Charlie']

# Filtering
adults = df.where("age", ">=", 18)
high_scorers = df.filter(lambda row: row["score"] > 90)

# Sorting
sorted_df = df.sort("age", ascending=False)

# Column operations
df2 = df.apply("score", lambda x: x * 1.1)
df3 = df.add_column("passed", lambda row: row["score"] >= 60)

# Statistics
stats = df.describe()

# Write to CSV
df.to_csv("output.csv")
```

### FFT with FFTW

```python
from honeybadger import FFTWrapper
import math

fft = FFTWrapper()

# Check if FFTW is available
print(fft.is_fftw_available)

# Complex FFT
data = [complex(math.cos(2 * math.pi * i / 8)) for i in range(8)]
spectrum = fft.fft(data)

# Inverse FFT
original = fft.ifft(spectrum)

# Real FFT (for real-valued signals)
real_data = [1.0, 2.0, 3.0, 4.0]
rfft_result = fft.rfft(real_data)

# Power spectrum
power = fft.power_spectrum(real_data)

# Get frequencies
freqs = FFTWrapper.rfft_frequencies(len(real_data), sample_rate=44100)
```

### Statistics with GSL

```python
from honeybadger import GSLStats

stats = GSLStats()

# Check if GSL is available
print(stats.is_gsl_available)

data = [1.0, 2.0, 3.0, 4.0, 5.0]

# Basic statistics
mean = stats.mean(data)
variance = stats.variance(data)
std = stats.std(data)

# Quantiles
median = stats.median(data)
q1 = stats.quantile(data, 0.25)
p90 = stats.percentile(data, 90)

# Min/Max
min_val = stats.min(data)
max_val = stats.max(data)
min_val, max_val = stats.minmax(data)

# Higher moments
skewness = stats.skewness(data)
kurtosis = stats.kurtosis(data)

# Two-variable statistics
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.0, 4.0, 6.0, 8.0, 10.0]
covariance = stats.covariance(x, y)
correlation = stats.correlation(x, y)

# Summary statistics
summary = stats.describe(data)
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## License

MIT

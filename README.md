# HoneyBadger

DataFrame — CSV reading, basic operations, filtering, and C library wrappers — FFTW for FFT, GSL for statistics

## Installation

```bash
pip install -e .
```

### Dependencies

The library requires FFTW and GSL C libraries to be installed on your system:

**Ubuntu/Debian:**
```bash
sudo apt-get install libfftw3-dev libgsl-dev
```

**macOS:**
```bash
brew install fftw gsl
```

## Usage

### DataFrame

```python
from honeybadger import DataFrame

# Create from dictionary
df = DataFrame(data={
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'score': [95.5, 87.3, 92.1]
})

# Read from CSV
df = DataFrame.from_csv('data.csv')

# Basic operations
print(df.head(5))          # First 5 rows
print(df.tail(5))          # Last 5 rows
print(df['name'])          # Get column
print(df[0])               # Get row
print(df[1:3])             # Slice rows

# Filtering
filtered = df.filter('age', lambda x: x >= 30)
filtered = df.filter_by_value('age', 30, 'ge')  # age >= 30

# Column operations
df = df.apply('age', lambda x: x + 1, new_column='age_next_year')
df = df.select(['name', 'score'])
df = df.drop('age')

# Export
df.to_csv('output.csv')
```

### FFT (FFTW Wrapper)

```python
from honeybadger import fft, ifft

# Forward FFT
data = [1.0, 2.0, 3.0, 4.0]
result = fft(data)

# Inverse FFT
original = ifft(result)
```

### Statistics (GSL Wrapper)

```python
from honeybadger import mean, variance, std_dev, median, min_val, max_val

data = [1.0, 2.0, 3.0, 4.0, 5.0]

print(mean(data))      # 3.0
print(variance(data))  # Sample variance
print(std_dev(data))   # Sample standard deviation
print(median(data))    # 3.0
print(min_val(data))   # 1.0
print(max_val(data))   # 5.0
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```

## License

MIT

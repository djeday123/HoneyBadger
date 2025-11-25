"""
DataFrame implementation for CSV reading, basic operations, and filtering.
"""

import csv
from typing import Any, Callable, Dict, Iterator, List, Optional, Union


class DataFrame:
    """
    A simple DataFrame class for handling tabular data.
    
    Supports CSV reading, basic operations, and filtering.
    """
    
    def __init__(
        self,
        data: Optional[Dict[str, List[Any]]] = None,
        columns: Optional[List[str]] = None
    ):
        """
        Initialize a DataFrame.
        
        Args:
            data: Dictionary mapping column names to lists of values.
            columns: Optional list of column names to specify order.
        """
        if data is None:
            data = {}
        
        self._data: Dict[str, List[Any]] = {}
        self._columns: List[str] = []
        
        if columns is not None:
            self._columns = list(columns)
            for col in columns:
                self._data[col] = list(data.get(col, []))
        else:
            self._columns = list(data.keys())
            for col in self._columns:
                self._data[col] = list(data[col])
        
        # Validate that all columns have the same length
        if self._columns:
            lengths = [len(self._data[col]) for col in self._columns]
            if len(set(lengths)) > 1:
                raise ValueError("All columns must have the same length")
    
    @classmethod
    def from_csv(
        cls,
        filepath: str,
        delimiter: str = ",",
        has_header: bool = True,
        encoding: str = "utf-8"
    ) -> "DataFrame":
        """
        Read a DataFrame from a CSV file.
        
        Args:
            filepath: Path to the CSV file.
            delimiter: Column delimiter (default: comma).
            has_header: Whether the first row contains column names.
            encoding: File encoding (default: utf-8).
            
        Returns:
            A new DataFrame containing the CSV data.
        """
        data: Dict[str, List[Any]] = {}
        columns: List[str] = []
        
        with open(filepath, "r", encoding=encoding, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            
            if has_header:
                columns = next(reader, [])
                for col in columns:
                    data[col] = []
            
            for row_idx, row in enumerate(reader):
                if not has_header and row_idx == 0:
                    # Generate column names if no header
                    columns = [f"col_{i}" for i in range(len(row))]
                    for col in columns:
                        data[col] = []
                
                for col_idx, value in enumerate(row):
                    if col_idx < len(columns):
                        # Try to convert to numeric types
                        parsed_value = cls._parse_value(value)
                        data[columns[col_idx]].append(parsed_value)
        
        return cls(data, columns)
    
    @staticmethod
    def _parse_value(value: str) -> Any:
        """
        Parse a string value to its appropriate type.
        
        Args:
            value: The string value to parse.
            
        Returns:
            The parsed value (int, float, or string).
        """
        value = value.strip()
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def to_csv(
        self,
        filepath: str,
        delimiter: str = ",",
        include_header: bool = True,
        encoding: str = "utf-8"
    ) -> None:
        """
        Write the DataFrame to a CSV file.
        
        Args:
            filepath: Path to the output CSV file.
            delimiter: Column delimiter (default: comma).
            include_header: Whether to include column names as first row.
            encoding: File encoding (default: utf-8).
        """
        with open(filepath, "w", encoding=encoding, newline="") as f:
            writer = csv.writer(f, delimiter=delimiter)
            
            if include_header:
                writer.writerow(self._columns)
            
            for i in range(len(self)):
                row = [self._data[col][i] for col in self._columns]
                writer.writerow(row)
    
    @property
    def columns(self) -> List[str]:
        """Return the list of column names."""
        return list(self._columns)
    
    @property
    def shape(self) -> tuple:
        """Return the shape of the DataFrame as (rows, columns)."""
        n_rows = len(self)
        n_cols = len(self._columns)
        return (n_rows, n_cols)
    
    def __len__(self) -> int:
        """Return the number of rows in the DataFrame."""
        if not self._columns:
            return 0
        return len(self._data[self._columns[0]])
    
    def __getitem__(self, key: Union[str, List[str]]) -> Union[List[Any], "DataFrame"]:
        """
        Get a column or subset of columns.
        
        Args:
            key: Column name (string) or list of column names.
            
        Returns:
            List of values for a single column, or a new DataFrame for multiple columns.
        """
        if isinstance(key, str):
            if key not in self._data:
                raise KeyError(f"Column '{key}' not found")
            return list(self._data[key])
        elif isinstance(key, list):
            for k in key:
                if k not in self._data:
                    raise KeyError(f"Column '{k}' not found")
            subset = {k: self._data[k] for k in key}
            return DataFrame(subset, key)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def __setitem__(self, key: str, value: List[Any]) -> None:
        """
        Set a column's values.
        
        Args:
            key: Column name.
            value: List of values for the column.
        """
        if not isinstance(key, str):
            raise TypeError("Column name must be a string")
        
        if len(self) > 0 and len(value) != len(self):
            raise ValueError(
                f"Value length ({len(value)}) does not match DataFrame length ({len(self)})"
            )
        
        if key not in self._columns:
            self._columns.append(key)
        
        self._data[key] = list(value)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over rows as dictionaries."""
        for i in range(len(self)):
            yield self.row(i)
    
    def row(self, index: int) -> Dict[str, Any]:
        """
        Get a row as a dictionary.
        
        Args:
            index: Row index.
            
        Returns:
            Dictionary mapping column names to values.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Row index {index} out of range")
        return {col: self._data[col][index] for col in self._columns}
    
    def head(self, n: int = 5) -> "DataFrame":
        """
        Return the first n rows.
        
        Args:
            n: Number of rows to return (default: 5).
            
        Returns:
            A new DataFrame with the first n rows.
        """
        n = min(n, len(self))
        data = {col: self._data[col][:n] for col in self._columns}
        return DataFrame(data, self._columns)
    
    def tail(self, n: int = 5) -> "DataFrame":
        """
        Return the last n rows.
        
        Args:
            n: Number of rows to return (default: 5).
            
        Returns:
            A new DataFrame with the last n rows.
        """
        n = min(n, len(self))
        data = {col: self._data[col][-n:] for col in self._columns}
        return DataFrame(data, self._columns)
    
    def filter(self, condition: Callable[[Dict[str, Any]], bool]) -> "DataFrame":
        """
        Filter rows based on a condition.
        
        Args:
            condition: A function that takes a row dictionary and returns True/False.
            
        Returns:
            A new DataFrame with only the rows where condition is True.
        """
        indices = []
        for i in range(len(self)):
            row = self.row(i)
            if condition(row):
                indices.append(i)
        
        data = {col: [self._data[col][i] for i in indices] for col in self._columns}
        return DataFrame(data, self._columns)
    
    def select(self, column: str, value: Any) -> "DataFrame":
        """
        Select rows where a column equals a specific value.
        
        Args:
            column: Column name to filter on.
            value: Value to match.
            
        Returns:
            A new DataFrame with matching rows.
        """
        return self.filter(lambda row: row.get(column) == value)
    
    def where(
        self,
        column: str,
        op: str,
        value: Any
    ) -> "DataFrame":
        """
        Filter rows based on a comparison operation.
        
        Args:
            column: Column name to filter on.
            op: Comparison operator ('==', '!=', '<', '<=', '>', '>=').
            value: Value to compare against.
            
        Returns:
            A new DataFrame with matching rows.
        """
        ops = {
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
        }
        
        if op not in ops:
            raise ValueError(f"Invalid operator: {op}")
        
        compare = ops[op]
        return self.filter(lambda row: compare(row.get(column), value))
    
    def sort(
        self,
        column: str,
        ascending: bool = True
    ) -> "DataFrame":
        """
        Sort the DataFrame by a column.
        
        Args:
            column: Column name to sort by.
            ascending: Sort order (default: True for ascending).
            
        Returns:
            A new sorted DataFrame.
        """
        if column not in self._data:
            raise KeyError(f"Column '{column}' not found")
        
        indices = list(range(len(self)))
        indices.sort(key=lambda i: self._data[column][i], reverse=not ascending)
        
        data = {col: [self._data[col][i] for i in indices] for col in self._columns}
        return DataFrame(data, self._columns)
    
    def drop(self, columns: Union[str, List[str]]) -> "DataFrame":
        """
        Drop one or more columns.
        
        Args:
            columns: Column name or list of column names to drop.
            
        Returns:
            A new DataFrame without the specified columns.
        """
        if isinstance(columns, str):
            columns = [columns]
        
        new_columns = [col for col in self._columns if col not in columns]
        data = {col: self._data[col] for col in new_columns}
        return DataFrame(data, new_columns)
    
    def rename(self, mapping: Dict[str, str]) -> "DataFrame":
        """
        Rename columns.
        
        Args:
            mapping: Dictionary mapping old names to new names.
            
        Returns:
            A new DataFrame with renamed columns.
        """
        new_columns = [mapping.get(col, col) for col in self._columns]
        data = {
            mapping.get(col, col): self._data[col]
            for col in self._columns
        }
        return DataFrame(data, new_columns)
    
    def apply(
        self,
        column: str,
        func: Callable[[Any], Any]
    ) -> "DataFrame":
        """
        Apply a function to a column.
        
        Args:
            column: Column name to apply the function to.
            func: Function to apply to each value.
            
        Returns:
            A new DataFrame with the transformed column.
        """
        if column not in self._data:
            raise KeyError(f"Column '{column}' not found")
        
        data = dict(self._data)
        data[column] = [func(v) for v in self._data[column]]
        return DataFrame(data, self._columns)
    
    def add_column(
        self,
        name: str,
        values: Union[List[Any], Callable[[Dict[str, Any]], Any]]
    ) -> "DataFrame":
        """
        Add a new column.
        
        Args:
            name: Name of the new column.
            values: List of values or a function that computes values from rows.
            
        Returns:
            A new DataFrame with the added column.
        """
        data = dict(self._data)
        columns = list(self._columns) + [name]
        
        if callable(values):
            computed = [values(self.row(i)) for i in range(len(self))]
            data[name] = computed
        else:
            if len(values) != len(self) and len(self) > 0:
                raise ValueError(
                    f"Value length ({len(values)}) does not match DataFrame length ({len(self)})"
                )
            data[name] = list(values)
        
        return DataFrame(data, columns)
    
    def describe(self) -> Dict[str, Dict[str, float]]:
        """
        Compute basic statistics for numeric columns.
        
        Returns:
            Dictionary mapping column names to statistics.
        """
        result = {}
        
        for col in self._columns:
            values = self._data[col]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            
            if not numeric_values:
                continue
            
            n = len(numeric_values)
            total = sum(numeric_values)
            mean = total / n
            
            sorted_values = sorted(numeric_values)
            if n % 2 == 0:
                median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
            else:
                median = sorted_values[n // 2]
            
            variance = sum((v - mean) ** 2 for v in numeric_values) / n
            std = variance ** 0.5
            
            result[col] = {
                "count": n,
                "mean": mean,
                "std": std,
                "min": min(numeric_values),
                "max": max(numeric_values),
                "median": median,
            }
        
        return result
    
    def __repr__(self) -> str:
        """Return a string representation of the DataFrame."""
        if len(self) == 0:
            return "DataFrame(empty)"
        
        lines = []
        lines.append("DataFrame:")
        lines.append(f"  Shape: {self.shape}")
        lines.append(f"  Columns: {self._columns}")
        
        # Show first few rows
        n_show = min(5, len(self))
        for i in range(n_show):
            row = self.row(i)
            lines.append(f"  [{i}] {row}")
        
        if len(self) > n_show:
            lines.append(f"  ... ({len(self) - n_show} more rows)")
        
        return "\n".join(lines)
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another DataFrame."""
        if not isinstance(other, DataFrame):
            return False
        return self._data == other._data and self._columns == other._columns

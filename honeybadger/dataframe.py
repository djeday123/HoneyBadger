"""
DataFrame module - provides a simple DataFrame implementation with CSV reading,
basic operations, and filtering capabilities.
"""

import csv
from typing import Any, Callable, Dict, List, Optional, Union


class DataFrame:
    """
    A simple DataFrame implementation for data manipulation.
    
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
            columns: Optional list of column names (for ordering).
        """
        if data is None:
            data = {}
        
        self._data: Dict[str, List[Any]] = data
        self._columns: List[str] = columns if columns else list(data.keys())
        
        # Validate that all columns have the same length
        if self._data:
            lengths = [len(v) for v in self._data.values()]
            if len(set(lengths)) > 1:
                raise ValueError("All columns must have the same length")
    
    @classmethod
    def from_csv(
        cls,
        filepath: str,
        delimiter: str = ",",
        has_header: bool = True,
        dtypes: Optional[Dict[str, type]] = None
    ) -> "DataFrame":
        """
        Read a DataFrame from a CSV file.
        
        Args:
            filepath: Path to the CSV file.
            delimiter: Column delimiter character.
            has_header: Whether the first row contains column headers.
            dtypes: Optional dictionary mapping column names to types for conversion.
        
        Returns:
            DataFrame instance with the CSV data.
        """
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
        
        if not rows:
            return cls()
        
        if has_header:
            columns = rows[0]
            data_rows = rows[1:]
        else:
            # Generate column names if no header
            num_cols = len(rows[0]) if rows else 0
            columns = [f"col_{i}" for i in range(num_cols)]
            data_rows = rows
        
        # Initialize data dictionary
        data: Dict[str, List[Any]] = {col: [] for col in columns}
        
        # Populate data
        for row in data_rows:
            for i, col in enumerate(columns):
                value = row[i] if i < len(row) else None
                # Try to convert to appropriate type
                if dtypes and col in dtypes:
                    try:
                        value = dtypes[col](value)
                    except (ValueError, TypeError):
                        pass
                else:
                    # Auto-detect numeric types
                    value = cls._auto_convert(value)
                data[col].append(value)
        
        return cls(data=data, columns=columns)
    
    @staticmethod
    def _auto_convert(value: str) -> Union[int, float, str]:
        """
        Attempt to convert a string value to a numeric type.
        
        Args:
            value: String value to convert.
        
        Returns:
            Converted value (int, float, or original string).
        """
        if value is None:
            return None
        
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
        
        return value
    
    @property
    def columns(self) -> List[str]:
        """Get the list of column names."""
        return self._columns.copy()
    
    @property
    def shape(self) -> tuple:
        """Get the shape of the DataFrame (rows, columns)."""
        num_rows = len(self._data[self._columns[0]]) if self._columns else 0
        num_cols = len(self._columns)
        return (num_rows, num_cols)
    
    def __len__(self) -> int:
        """Return the number of rows in the DataFrame."""
        if not self._columns:
            return 0
        return len(self._data[self._columns[0]])
    
    def __getitem__(self, key: Union[str, List[str], int, slice]) -> Any:
        """
        Get column(s) or row(s) from the DataFrame.
        
        Args:
            key: Column name(s), row index, or slice.
        
        Returns:
            Column data, subset DataFrame, or row.
        """
        if isinstance(key, str):
            # Single column access
            if key not in self._data:
                raise KeyError(f"Column '{key}' not found")
            return self._data[key].copy()
        
        elif isinstance(key, list):
            # Multiple column access
            new_data = {}
            for col in key:
                if col not in self._data:
                    raise KeyError(f"Column '{col}' not found")
                new_data[col] = self._data[col].copy()
            return DataFrame(data=new_data, columns=key)
        
        elif isinstance(key, int):
            # Single row access
            if key < 0:
                key = len(self) + key
            if key < 0 or key >= len(self):
                raise IndexError(f"Row index {key} out of range")
            return {col: self._data[col][key] for col in self._columns}
        
        elif isinstance(key, slice):
            # Slice access for rows
            new_data = {col: self._data[col][key] for col in self._columns}
            return DataFrame(data=new_data, columns=self._columns)
        
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def __setitem__(self, key: str, value: List[Any]) -> None:
        """
        Set a column in the DataFrame.
        
        Args:
            key: Column name.
            value: List of values for the column.
        """
        if not isinstance(key, str):
            raise TypeError("Column name must be a string")
        
        if self._columns and len(value) != len(self):
            raise ValueError(f"Length mismatch: expected {len(self)}, got {len(value)}")
        
        if key not in self._columns:
            self._columns.append(key)
        
        self._data[key] = list(value)
    
    def head(self, n: int = 5) -> "DataFrame":
        """
        Return the first n rows of the DataFrame.
        
        Args:
            n: Number of rows to return.
        
        Returns:
            DataFrame with first n rows.
        """
        return self[:n]
    
    def tail(self, n: int = 5) -> "DataFrame":
        """
        Return the last n rows of the DataFrame.
        
        Args:
            n: Number of rows to return.
        
        Returns:
            DataFrame with last n rows.
        """
        return self[-n:]
    
    def filter(
        self,
        column: str,
        condition: Callable[[Any], bool]
    ) -> "DataFrame":
        """
        Filter the DataFrame based on a condition applied to a column.
        
        Args:
            column: Column name to apply the condition to.
            condition: Callable that takes a value and returns a boolean.
        
        Returns:
            Filtered DataFrame.
        """
        if column not in self._data:
            raise KeyError(f"Column '{column}' not found")
        
        mask = [condition(val) for val in self._data[column]]
        new_data = {
            col: [val for val, keep in zip(self._data[col], mask) if keep]
            for col in self._columns
        }
        return DataFrame(data=new_data, columns=self._columns)
    
    def filter_by_value(
        self,
        column: str,
        value: Any,
        comparison: str = "eq"
    ) -> "DataFrame":
        """
        Filter the DataFrame by comparing column values.
        
        Args:
            column: Column name to filter by.
            value: Value to compare against.
            comparison: Comparison type - "eq", "ne", "gt", "lt", "ge", "le".
        
        Returns:
            Filtered DataFrame.
        """
        comparisons = {
            "eq": lambda x: x == value,
            "ne": lambda x: x != value,
            "gt": lambda x: x > value,
            "lt": lambda x: x < value,
            "ge": lambda x: x >= value,
            "le": lambda x: x <= value,
        }
        
        if comparison not in comparisons:
            raise ValueError(f"Invalid comparison: {comparison}")
        
        return self.filter(column, comparisons[comparison])
    
    def select(self, columns: List[str]) -> "DataFrame":
        """
        Select specific columns from the DataFrame.
        
        Args:
            columns: List of column names to select.
        
        Returns:
            DataFrame with only the selected columns.
        """
        return self[columns]
    
    def drop(self, columns: Union[str, List[str]]) -> "DataFrame":
        """
        Drop columns from the DataFrame.
        
        Args:
            columns: Column name or list of column names to drop.
        
        Returns:
            DataFrame without the dropped columns.
        """
        if isinstance(columns, str):
            columns = [columns]
        
        new_columns = [col for col in self._columns if col not in columns]
        new_data = {col: self._data[col].copy() for col in new_columns}
        return DataFrame(data=new_data, columns=new_columns)
    
    def apply(
        self,
        column: str,
        func: Callable[[Any], Any],
        new_column: Optional[str] = None
    ) -> "DataFrame":
        """
        Apply a function to a column.
        
        Args:
            column: Column to apply the function to.
            func: Function to apply to each value.
            new_column: Name for the new column. If None, modifies in place.
        
        Returns:
            DataFrame with the applied transformation.
        """
        if column not in self._data:
            raise KeyError(f"Column '{column}' not found")
        
        result_col = new_column if new_column else column
        new_data = {col: self._data[col].copy() for col in self._columns}
        new_columns = self._columns.copy()
        
        new_data[result_col] = [func(val) for val in self._data[column]]
        
        if new_column and new_column not in self._columns:
            new_columns.append(new_column)
        
        return DataFrame(data=new_data, columns=new_columns)
    
    def to_csv(
        self,
        filepath: str,
        delimiter: str = ",",
        include_header: bool = True
    ) -> None:
        """
        Write the DataFrame to a CSV file.
        
        Args:
            filepath: Path to the output CSV file.
            delimiter: Column delimiter character.
            include_header: Whether to include column headers.
        """
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=delimiter)
            
            if include_header:
                writer.writerow(self._columns)
            
            for i in range(len(self)):
                row = [self._data[col][i] for col in self._columns]
                writer.writerow(row)
    
    def to_dict(self) -> Dict[str, List[Any]]:
        """
        Convert the DataFrame to a dictionary.
        
        Returns:
            Dictionary mapping column names to lists of values.
        """
        return {col: self._data[col].copy() for col in self._columns}
    
    def to_list(self) -> List[Dict[str, Any]]:
        """
        Convert the DataFrame to a list of row dictionaries.
        
        Returns:
            List of dictionaries, one per row.
        """
        return [
            {col: self._data[col][i] for col in self._columns}
            for i in range(len(self))
        ]
    
    def __repr__(self) -> str:
        """Return a string representation of the DataFrame."""
        if not self._columns:
            return "DataFrame(empty)"
        
        lines = []
        lines.append(f"DataFrame({self.shape[0]} rows x {self.shape[1]} columns)")
        lines.append("Columns: " + ", ".join(self._columns))
        
        # Show first few rows
        preview_rows = min(5, len(self))
        if preview_rows > 0:
            lines.append("-" * 40)
            # Header
            lines.append(" | ".join(str(col)[:15].ljust(15) for col in self._columns))
            lines.append("-" * 40)
            # Data
            for i in range(preview_rows):
                row = [str(self._data[col][i])[:15].ljust(15) for col in self._columns]
                lines.append(" | ".join(row))
            
            if len(self) > preview_rows:
                lines.append(f"... ({len(self) - preview_rows} more rows)")
        
        return "\n".join(lines)
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another DataFrame."""
        if not isinstance(other, DataFrame):
            return False
        return self._data == other._data and self._columns == other._columns

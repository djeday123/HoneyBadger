"""Tests for the DataFrame class."""

import os
import tempfile
import pytest
from honeybadger.dataframe import DataFrame


class TestDataFrameCreation:
    """Tests for DataFrame creation."""
    
    def test_create_empty_dataframe(self):
        """Test creating an empty DataFrame."""
        df = DataFrame()
        assert len(df) == 0
        assert df.columns == []
        assert df.shape == (0, 0)
    
    def test_create_dataframe_from_dict(self):
        """Test creating a DataFrame from a dictionary."""
        data = {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [95.5, 87.3, 92.1]
        }
        df = DataFrame(data=data)
        
        assert len(df) == 3
        assert set(df.columns) == {"name", "age", "score"}
        assert df.shape == (3, 3)
    
    def test_create_dataframe_with_column_order(self):
        """Test creating a DataFrame with specific column order."""
        data = {
            "name": ["Alice", "Bob"],
            "age": [25, 30]
        }
        df = DataFrame(data=data, columns=["age", "name"])
        
        assert df.columns == ["age", "name"]
    
    def test_create_dataframe_mismatched_lengths(self):
        """Test that mismatched column lengths raise an error."""
        data = {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30]  # Different length
        }
        with pytest.raises(ValueError):
            DataFrame(data=data)


class TestDataFrameCSV:
    """Tests for CSV reading and writing."""
    
    def test_read_csv(self):
        """Test reading a CSV file."""
        csv_content = "name,age,score\nAlice,25,95.5\nBob,30,87.3\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            df = DataFrame.from_csv(temp_path)
            
            assert len(df) == 2
            assert df.columns == ["name", "age", "score"]
            assert df["name"] == ["Alice", "Bob"]
            assert df["age"] == [25, 30]
            assert df["score"] == [95.5, 87.3]
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_no_header(self):
        """Test reading a CSV file without headers."""
        csv_content = "Alice,25,95.5\nBob,30,87.3\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            df = DataFrame.from_csv(temp_path, has_header=False)
            
            assert len(df) == 2
            assert df.columns == ["col_0", "col_1", "col_2"]
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_custom_delimiter(self):
        """Test reading a CSV file with custom delimiter."""
        csv_content = "name;age;score\nAlice;25;95.5\nBob;30;87.3\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            df = DataFrame.from_csv(temp_path, delimiter=";")
            
            assert len(df) == 2
            assert df.columns == ["name", "age", "score"]
        finally:
            os.unlink(temp_path)
    
    def test_write_csv(self):
        """Test writing a DataFrame to CSV."""
        data = {
            "name": ["Alice", "Bob"],
            "age": [25, 30]
        }
        df = DataFrame(data=data, columns=["name", "age"])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            df.to_csv(temp_path)
            
            # Read it back
            df2 = DataFrame.from_csv(temp_path)
            assert df == df2
        finally:
            os.unlink(temp_path)


class TestDataFrameAccess:
    """Tests for DataFrame data access."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.df = DataFrame(
            data={
                "name": ["Alice", "Bob", "Charlie", "Diana"],
                "age": [25, 30, 35, 28],
                "score": [95.5, 87.3, 92.1, 88.9]
            },
            columns=["name", "age", "score"]
        )
    
    def test_get_single_column(self):
        """Test getting a single column."""
        names = self.df["name"]
        assert names == ["Alice", "Bob", "Charlie", "Diana"]
    
    def test_get_multiple_columns(self):
        """Test getting multiple columns."""
        subset = self.df[["name", "age"]]
        assert isinstance(subset, DataFrame)
        assert subset.columns == ["name", "age"]
        assert len(subset) == 4
    
    def test_get_single_row(self):
        """Test getting a single row by index."""
        row = self.df[0]
        assert row == {"name": "Alice", "age": 25, "score": 95.5}
    
    def test_get_negative_row_index(self):
        """Test getting a row with negative index."""
        row = self.df[-1]
        assert row == {"name": "Diana", "age": 28, "score": 88.9}
    
    def test_get_slice(self):
        """Test getting rows by slice."""
        subset = self.df[1:3]
        assert isinstance(subset, DataFrame)
        assert len(subset) == 2
        assert subset["name"] == ["Bob", "Charlie"]
    
    def test_set_column(self):
        """Test setting a column."""
        df = DataFrame(data={"a": [1, 2, 3]})
        df["b"] = [4, 5, 6]
        
        assert "b" in df.columns
        assert df["b"] == [4, 5, 6]
    
    def test_column_not_found(self):
        """Test accessing non-existent column raises KeyError."""
        with pytest.raises(KeyError):
            _ = self.df["nonexistent"]
    
    def test_row_index_out_of_range(self):
        """Test accessing out-of-range row raises IndexError."""
        with pytest.raises(IndexError):
            _ = self.df[100]


class TestDataFrameOperations:
    """Tests for DataFrame operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.df = DataFrame(
            data={
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "age": [25, 30, 35, 28, 22],
                "score": [95.5, 87.3, 92.1, 88.9, 91.0]
            },
            columns=["name", "age", "score"]
        )
    
    def test_head(self):
        """Test head() method."""
        result = self.df.head(2)
        assert len(result) == 2
        assert result["name"] == ["Alice", "Bob"]
    
    def test_head_default(self):
        """Test head() with default value."""
        result = self.df.head()
        assert len(result) == 5  # All rows since we only have 5
    
    def test_tail(self):
        """Test tail() method."""
        result = self.df.tail(2)
        assert len(result) == 2
        assert result["name"] == ["Diana", "Eve"]
    
    def test_select(self):
        """Test select() method."""
        result = self.df.select(["name", "score"])
        assert result.columns == ["name", "score"]
        assert "age" not in result.columns
    
    def test_drop(self):
        """Test drop() method."""
        result = self.df.drop("age")
        assert "age" not in result.columns
        assert "name" in result.columns
        assert "score" in result.columns
    
    def test_drop_multiple(self):
        """Test drop() with multiple columns."""
        result = self.df.drop(["age", "score"])
        assert result.columns == ["name"]
    
    def test_apply(self):
        """Test apply() method."""
        result = self.df.apply("age", lambda x: x * 2, new_column="double_age")
        
        assert "double_age" in result.columns
        assert result["double_age"] == [50, 60, 70, 56, 44]
    
    def test_apply_in_place(self):
        """Test apply() method modifying column in place."""
        result = self.df.apply("age", lambda x: x + 1)
        assert result["age"] == [26, 31, 36, 29, 23]
    
    def test_to_dict(self):
        """Test to_dict() method."""
        df = DataFrame(data={"a": [1, 2], "b": [3, 4]}, columns=["a", "b"])
        d = df.to_dict()
        
        assert d == {"a": [1, 2], "b": [3, 4]}
    
    def test_to_list(self):
        """Test to_list() method."""
        df = DataFrame(data={"a": [1, 2], "b": [3, 4]}, columns=["a", "b"])
        lst = df.to_list()
        
        assert lst == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


class TestDataFrameFiltering:
    """Tests for DataFrame filtering."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.df = DataFrame(
            data={
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "age": [25, 30, 35, 28, 22],
                "score": [95.5, 87.3, 92.1, 88.9, 91.0]
            },
            columns=["name", "age", "score"]
        )
    
    def test_filter_with_lambda(self):
        """Test filtering with a lambda function."""
        result = self.df.filter("age", lambda x: x >= 30)
        
        assert len(result) == 2
        assert result["name"] == ["Bob", "Charlie"]
    
    def test_filter_by_value_eq(self):
        """Test filter_by_value with equality."""
        result = self.df.filter_by_value("name", "Alice", "eq")
        
        assert len(result) == 1
        assert result["name"] == ["Alice"]
    
    def test_filter_by_value_gt(self):
        """Test filter_by_value with greater than."""
        result = self.df.filter_by_value("age", 28, "gt")
        
        assert len(result) == 2
        assert set(result["name"]) == {"Bob", "Charlie"}
    
    def test_filter_by_value_lt(self):
        """Test filter_by_value with less than."""
        result = self.df.filter_by_value("score", 90, "lt")
        
        assert len(result) == 2
        assert set(result["name"]) == {"Bob", "Diana"}
    
    def test_filter_by_value_ge(self):
        """Test filter_by_value with greater than or equal."""
        result = self.df.filter_by_value("age", 30, "ge")
        
        assert len(result) == 2
        assert result["name"] == ["Bob", "Charlie"]
    
    def test_filter_by_value_le(self):
        """Test filter_by_value with less than or equal."""
        result = self.df.filter_by_value("age", 25, "le")
        
        assert len(result) == 2
        assert set(result["name"]) == {"Alice", "Eve"}
    
    def test_filter_by_value_ne(self):
        """Test filter_by_value with not equal."""
        result = self.df.filter_by_value("name", "Alice", "ne")
        
        assert len(result) == 4
        assert "Alice" not in result["name"]
    
    def test_filter_invalid_column(self):
        """Test filtering with invalid column raises KeyError."""
        with pytest.raises(KeyError):
            self.df.filter("nonexistent", lambda x: True)
    
    def test_filter_invalid_comparison(self):
        """Test filter_by_value with invalid comparison raises ValueError."""
        with pytest.raises(ValueError):
            self.df.filter_by_value("age", 25, "invalid")
    
    def test_chained_filters(self):
        """Test chaining multiple filters."""
        result = (
            self.df
            .filter_by_value("age", 25, "ge")
            .filter_by_value("score", 90, "gt")
        )
        
        assert len(result) == 2
        assert set(result["name"]) == {"Alice", "Charlie"}


class TestDataFrameRepr:
    """Tests for DataFrame string representation."""
    
    def test_repr_empty(self):
        """Test repr of empty DataFrame."""
        df = DataFrame()
        assert "empty" in repr(df)
    
    def test_repr_with_data(self):
        """Test repr of DataFrame with data."""
        df = DataFrame(
            data={"a": [1, 2, 3], "b": [4, 5, 6]},
            columns=["a", "b"]
        )
        rep = repr(df)
        
        assert "3 rows" in rep
        assert "2 columns" in rep
        assert "a" in rep
        assert "b" in rep

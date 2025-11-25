"""Tests for DataFrame class."""

import os
import tempfile
import pytest

from honeybadger import DataFrame


class TestDataFrameBasics:
    """Test basic DataFrame functionality."""
    
    def test_empty_dataframe(self):
        """Test creating an empty DataFrame."""
        df = DataFrame()
        assert len(df) == 0
        assert df.shape == (0, 0)
        assert df.columns == []
    
    def test_dataframe_from_dict(self):
        """Test creating DataFrame from dictionary."""
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        df = DataFrame(data)
        
        assert len(df) == 3
        assert df.shape == (3, 2)
        assert set(df.columns) == {"a", "b"}
    
    def test_column_access(self):
        """Test accessing columns."""
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        df = DataFrame(data)
        
        assert df["a"] == [1, 2, 3]
        assert df["b"] == [4, 5, 6]
    
    def test_column_not_found(self):
        """Test accessing non-existent column."""
        df = DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(KeyError):
            _ = df["nonexistent"]
    
    def test_column_subset(self):
        """Test selecting multiple columns."""
        data = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
        df = DataFrame(data)
        
        subset = df[["a", "c"]]
        assert isinstance(subset, DataFrame)
        assert subset.columns == ["a", "c"]
    
    def test_set_column(self):
        """Test setting a column."""
        df = DataFrame({"a": [1, 2, 3]})
        df["b"] = [4, 5, 6]
        
        assert df["b"] == [4, 5, 6]
        assert "b" in df.columns
    
    def test_row_access(self):
        """Test accessing rows."""
        data = {"a": [1, 2], "b": [3, 4]}
        df = DataFrame(data)
        
        assert df.row(0) == {"a": 1, "b": 3}
        assert df.row(1) == {"a": 2, "b": 4}
    
    def test_row_out_of_range(self):
        """Test accessing row out of range."""
        df = DataFrame({"a": [1, 2]})
        
        with pytest.raises(IndexError):
            df.row(5)
    
    def test_iteration(self):
        """Test iterating over rows."""
        data = {"a": [1, 2], "b": [3, 4]}
        df = DataFrame(data)
        
        rows = list(df)
        assert rows == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


class TestDataFrameCSV:
    """Test CSV reading and writing."""
    
    def test_read_csv(self):
        """Test reading a CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age,score\n")
            f.write("Alice,25,95.5\n")
            f.write("Bob,30,88.0\n")
            csv_path = f.name
        
        try:
            df = DataFrame.from_csv(csv_path)
            
            assert df.columns == ["name", "age", "score"]
            assert df["name"] == ["Alice", "Bob"]
            assert df["age"] == [25, 30]
            assert df["score"] == [95.5, 88.0]
        finally:
            os.unlink(csv_path)
    
    def test_read_csv_no_header(self):
        """Test reading CSV without header."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Alice,25,95.5\n")
            f.write("Bob,30,88.0\n")
            csv_path = f.name
        
        try:
            df = DataFrame.from_csv(csv_path, has_header=False)
            
            assert df.columns == ["col_0", "col_1", "col_2"]
            assert len(df) == 2
        finally:
            os.unlink(csv_path)
    
    def test_write_csv(self):
        """Test writing a CSV file."""
        data = {"name": ["Alice", "Bob"], "age": [25, 30]}
        df = DataFrame(data, columns=["name", "age"])
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name
        
        try:
            df.to_csv(csv_path)
            
            with open(csv_path, "r") as f:
                lines = f.readlines()
            
            assert lines[0].strip() == "name,age"
            assert lines[1].strip() == "Alice,25"
            assert lines[2].strip() == "Bob,30"
        finally:
            os.unlink(csv_path)
    
    def test_csv_roundtrip(self):
        """Test reading and writing CSV preserves data."""
        original = DataFrame({
            "x": [1, 2, 3],
            "y": [4.5, 5.5, 6.5],
            "z": ["a", "b", "c"]
        }, columns=["x", "y", "z"])
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name
        
        try:
            original.to_csv(csv_path)
            loaded = DataFrame.from_csv(csv_path)
            
            assert loaded.columns == original.columns
            assert loaded["x"] == original["x"]
            assert loaded["y"] == original["y"]
            assert loaded["z"] == original["z"]
        finally:
            os.unlink(csv_path)


class TestDataFrameOperations:
    """Test DataFrame operations."""
    
    def test_head(self):
        """Test head operation."""
        df = DataFrame({"a": list(range(10))})
        
        head3 = df.head(3)
        assert len(head3) == 3
        assert head3["a"] == [0, 1, 2]
        
        # Default is 5
        head5 = df.head()
        assert len(head5) == 5
    
    def test_tail(self):
        """Test tail operation."""
        df = DataFrame({"a": list(range(10))})
        
        tail3 = df.tail(3)
        assert len(tail3) == 3
        assert tail3["a"] == [7, 8, 9]
    
    def test_sort(self):
        """Test sorting."""
        df = DataFrame({"a": [3, 1, 2], "b": ["x", "y", "z"]})
        
        sorted_df = df.sort("a")
        assert sorted_df["a"] == [1, 2, 3]
        assert sorted_df["b"] == ["y", "z", "x"]
        
        # Descending
        desc_df = df.sort("a", ascending=False)
        assert desc_df["a"] == [3, 2, 1]
    
    def test_drop(self):
        """Test dropping columns."""
        df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        
        dropped = df.drop("b")
        assert dropped.columns == ["a", "c"]
        
        dropped_multi = df.drop(["a", "c"])
        assert dropped_multi.columns == ["b"]
    
    def test_rename(self):
        """Test renaming columns."""
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        
        renamed = df.rename({"a": "x", "b": "y"})
        assert renamed.columns == ["x", "y"]
        assert renamed["x"] == [1, 2]
    
    def test_apply(self):
        """Test applying a function to a column."""
        df = DataFrame({"a": [1, 2, 3]})
        
        doubled = df.apply("a", lambda x: x * 2)
        assert doubled["a"] == [2, 4, 6]
    
    def test_add_column(self):
        """Test adding a new column."""
        df = DataFrame({"a": [1, 2, 3]})
        
        # With values
        df2 = df.add_column("b", [4, 5, 6])
        assert df2["b"] == [4, 5, 6]
        
        # With function
        df3 = df.add_column("c", lambda row: row["a"] * 10)
        assert df3["c"] == [10, 20, 30]
    
    def test_describe(self):
        """Test describe statistics."""
        df = DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "a", "b"]})
        
        stats = df.describe()
        assert "a" in stats
        assert "b" not in stats  # Non-numeric column
        assert stats["a"]["count"] == 5
        assert stats["a"]["mean"] == 3.0


class TestDataFrameFiltering:
    """Test DataFrame filtering functionality."""
    
    def test_filter(self):
        """Test filter with custom function."""
        df = DataFrame({"a": [1, 2, 3, 4, 5]})
        
        filtered = df.filter(lambda row: row["a"] > 2)
        assert filtered["a"] == [3, 4, 5]
    
    def test_select(self):
        """Test select by value."""
        df = DataFrame({
            "name": ["Alice", "Bob", "Alice"],
            "score": [90, 85, 95]
        })
        
        alice = df.select("name", "Alice")
        assert len(alice) == 2
        assert alice["score"] == [90, 95]
    
    def test_where_eq(self):
        """Test where with equality."""
        df = DataFrame({"a": [1, 2, 3]})
        
        result = df.where("a", "==", 2)
        assert result["a"] == [2]
    
    def test_where_gt(self):
        """Test where with greater than."""
        df = DataFrame({"a": [1, 2, 3, 4, 5]})
        
        result = df.where("a", ">", 3)
        assert result["a"] == [4, 5]
    
    def test_where_lt(self):
        """Test where with less than."""
        df = DataFrame({"a": [1, 2, 3, 4, 5]})
        
        result = df.where("a", "<", 3)
        assert result["a"] == [1, 2]
    
    def test_where_lte(self):
        """Test where with less than or equal."""
        df = DataFrame({"a": [1, 2, 3, 4, 5]})
        
        result = df.where("a", "<=", 3)
        assert result["a"] == [1, 2, 3]
    
    def test_where_gte(self):
        """Test where with greater than or equal."""
        df = DataFrame({"a": [1, 2, 3, 4, 5]})
        
        result = df.where("a", ">=", 3)
        assert result["a"] == [3, 4, 5]
    
    def test_where_ne(self):
        """Test where with not equal."""
        df = DataFrame({"a": [1, 2, 3]})
        
        result = df.where("a", "!=", 2)
        assert result["a"] == [1, 3]
    
    def test_where_invalid_op(self):
        """Test where with invalid operator."""
        df = DataFrame({"a": [1, 2, 3]})
        
        with pytest.raises(ValueError):
            df.where("a", ">>>", 2)
    
    def test_chained_filters(self):
        """Test chaining multiple filters."""
        df = DataFrame({
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "age": [25, 30, 35, 28],
            "score": [90, 85, 92, 88]
        })
        
        result = df.where("age", ">", 26).where("score", ">=", 88)
        assert len(result) == 2
        assert result["name"] == ["Charlie", "Diana"]

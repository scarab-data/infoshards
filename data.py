"""
Data loading and processing utilities for Vextra
Handles loading, cleaning, and preparing datasets for analysis
"""

# pylint: disable=bare-except,broad-exception-caught
# pylint: disable=logging-fstring-interpolation
# pylint: disable=redefined-outer-name
# pylint: disable=line-too-long
# pylint: disable=protected-access
# pylint: disable=invalid-name
# pylint: disable=global-statement

import os
import logging
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default data paths
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app/data", "UCD_1999-2020.txt")
DATA_CACHE = {}  # Simple in-memory cache for loaded datasets

class DataLoader:
    """Data loading and processing for tab-separated datasets"""

    def __init__(self, data_path=DEFAULT_DATA_PATH):
        """Initialize the data loader with a path to the dataset"""
        self.data_path = data_path
        self.data = None
        self.metadata = {
            "filename": os.path.basename(data_path),
            "path": data_path,
            "size": 0,
            "last_modified": None,
            "num_rows": 0,
            "num_columns": 0,
            "columns": [],
            "loaded_at": None,
        }
        self.loaded = False

    def load_data(self, force_reload=False) -> bool:
        """
        Load the tab-separated dataset into a pandas DataFrame

        Args:
            force_reload: Whether to force reload even if already loaded

        Returns:
            bool: True if loading was successful, False otherwise
        """
        # Check if already loaded and not forcing reload
        if self.loaded and not force_reload:
            logger.info(f"Dataset already loaded: {self.data_path}")
            return True

        # Check if in cache
        if self.data_path in DATA_CACHE and not force_reload:
            logger.info(f"Loading dataset from cache: {self.data_path}")
            self.data = DATA_CACHE[self.data_path]["data"]
            self.metadata = DATA_CACHE[self.data_path]["metadata"]
            self.loaded = True
            return True

        # Check if file exists
        if not os.path.exists(self.data_path):
            logger.error(f"Dataset file not found: {self.data_path}")
            return False

        try:
            # Get file metadata
            file_stats = os.stat(self.data_path)
            self.metadata["size"] = file_stats.st_size
            self.metadata["last_modified"] = datetime.fromtimestamp(file_stats.st_mtime)

            # Load the dataset
            logger.info(f"Loading dataset from file: {self.data_path}")
            self.data = pd.read_csv(self.data_path, sep="\t")

            # Update metadata
            self.metadata["num_rows"] = len(self.data)
            self.metadata["num_columns"] = len(self.data.columns)
            self.metadata["columns"] = list(self.data.columns)
            self.metadata["loaded_at"] = datetime.now()

            # Store in cache
            DATA_CACHE[self.data_path] = {"data": self.data, "metadata": self.metadata}

            self.loaded = True
            logger.info(
                f"Successfully loaded dataset with {self.metadata['num_rows']} rows and {self.metadata['num_columns']} columns"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the loaded dataset

        Returns:
            DataFrame or None if not loaded
        """
        if not self.loaded:
            logger.warning("Dataset not loaded. Call load_data() first.")
            return None

        return self.data

    def get_metadata(self) -> Dict:
        """
        Get metadata about the dataset

        Returns:
            Dict containing metadata
        """
        return self.metadata

    def get_summary(self) -> Dict:
        """
        Get a summary of the dataset including basic statistics

        Returns:
            Dict containing summary information
        """
        if not self.loaded:
            logger.warning("Dataset not loaded. Call load_data() first.")
            return {}

        summary = {"metadata": self.metadata, "statistics": {}}

        try:
            # Generate basic statistics for numeric columns
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                summary["statistics"][col] = {
                    "min": float(self.data[col].min()),
                    "max": float(self.data[col].max()),
                    "mean": float(self.data[col].mean()),
                    "median": float(self.data[col].median()),
                    "std": float(self.data[col].std()),
                    "missing": int(self.data[col].isna().sum()),
                }

            # Generate basic statistics for categorical columns
            categorical_columns = self.data.select_dtypes(include=["object"]).columns
            for col in categorical_columns:
                value_counts = self.data[col].value_counts().head(10).to_dict()
                summary["statistics"][col] = {
                    "unique_values": int(self.data[col].nunique()),
                    "top_values": value_counts,
                    "missing": int(self.data[col].isna().sum()),
                }

        except Exception as e:
            logger.error(f"Error generating summary statistics: {str(e)}")

        return summary

    def query_data(self, query: str) -> Optional[pd.DataFrame]:
        """
        Query the dataset using pandas query syntax

        Args:
            query: Query string in pandas query format

        Returns:
            DataFrame with query results or None if error
        """
        if not self.loaded:
            logger.warning("Dataset not loaded. Call load_data() first.")
            return None

        try:
            result = self.data.query(query)
            logger.info(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return None

    def get_subset(self, columns=None, rows=None) -> Optional[pd.DataFrame]:
        """
        Get a subset of the dataset

        Args:
            columns: List of column names to include
            rows: Number of rows to include (from the beginning)

        Returns:
            DataFrame subset or None if error
        """
        if not self.loaded:
            logger.warning("Dataset not loaded. Call load_data() first.")
            return None

        try:
            result = self.data

            if columns:
                # Filter to only include specified columns
                valid_columns = [col for col in columns if col in self.data.columns]
                if not valid_columns:
                    logger.warning("No valid columns specified")
                    return None
                result = result[valid_columns]

            if rows and rows > 0:
                # Limit to specified number of rows
                result = result.head(rows)

            return result
        except Exception as e:
            logger.error(f"Error getting subset: {str(e)}")
            return None

    def to_csv_string(self, subset=None) -> str:
        """
        Convert dataset or subset to CSV string

        Args:
            subset: DataFrame subset to convert

        Returns:
            CSV string representation
        """
        if subset is None:
            if not self.loaded:
                logger.warning("Dataset not loaded. Call load_data() first.")
                return ""
            subset = self.data

        try:
            return subset.to_csv(index=False)
        except Exception as e:
            logger.error(f"Error converting to CSV: {str(e)}")
            return ""

    def to_json_string(self, subset=None, orient="records") -> str:
        """
        Convert dataset or subset to JSON string

        Args:
            subset: DataFrame subset to convert
            orient: JSON orientation (records, split, index, etc.)

        Returns:
            JSON string representation
        """
        if subset is None:
            if not self.loaded:
                logger.warning("Dataset not loaded. Call load_data() first.")
                return ""
            subset = self.data

        try:
            return subset.to_json(orient=orient)
        except Exception as e:
            logger.error(f"Error converting to JSON: {str(e)}")
            return ""


# Create a singleton instance
data_loader = DataLoader()


# Function to get the DataLoader instance
def get_data_loader(data_path=DEFAULT_DATA_PATH):
    """
    Get the DataLoader instance

    Args:
        data_path: Optional path to a different dataset

    Returns:
        DataLoader instance
    """
    global data_loader

    # If requesting a different dataset, create a new loader
    if data_path != data_loader.data_path:
        data_loader = DataLoader(data_path)

    return data_loader


# Test data loading if run directly
if __name__ == "__main__":
    loader = get_data_loader()

    if loader.load_data():
        print(f"Successfully loaded dataset: {loader.data_path}")

        # Print metadata
        metadata = loader.get_metadata()
        print("\nDataset Metadata:")
        print(f"  Filename: {metadata['filename']}")
        print(f"  Size: {metadata['size']} bytes")
        print(f"  Rows: {metadata['num_rows']}")
        print(f"  Columns: {metadata['num_columns']}")
        print(f"  Column names: {', '.join(metadata['columns'][:5])}...")

        # Print first few rows
        data = loader.get_data()
        print("\nFirst 5 rows:")
        print(data.head(5))

        # Print summary statistics for a few columns
        summary = loader.get_summary()
        print("\nSummary Statistics (first 3 numeric columns):")
        numeric_cols = [
            col for col in summary["statistics"] if "mean" in summary["statistics"][col]
        ]
        for col in numeric_cols[:3]:
            stats = summary["statistics"][col]
            print(f"  {col}:")
            print(f"    Min: {stats['min']}")
            print(f"    Max: {stats['max']}")
            print(f"    Mean: {stats['mean']}")
            print(f"    Median: {stats['median']}")

        # Test query
        print("\nSample Query:")
        query_result = loader.query_data(f"{metadata['columns'][0]} > 2000")
        if query_result is not None:
            print(f"  Query returned {len(query_result)} rows")
            if len(query_result) > 0:
                print(f"  First row of result: {query_result.iloc[0].to_dict()}")
    else:
        print(f"Failed to load dataset: {loader.data_path}")

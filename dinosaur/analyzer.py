import pandas as pd
import numpy as np

class Analyzer:
    """
    Core analysis engine for Dinosaur.
    Provides summary statistics and data quality reports.
    """
    
    @staticmethod
    def quick_summary(df):
        """
        Returns a comprehensive dictionary with dataset overview.
        """
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "numeric_stats": df.describe().to_dict()
        }
        return summary

    @staticmethod
    def quality_report(df):
        """
        Prints a clean report about data quality.
        """
        print("-" * 30)
        print("DINOSAUR DATA QUALITY REPORT")
        print("-" * 30)
        print(f"Total Rows: {df.shape[0]}")
        print(f"Total Columns: {df.shape[1]}")
        print(f"Duplicates: {df.duplicated().sum()}")
        
        nulls = df.isnull().sum()
        if nulls.sum() > 0:
            print("\nMissing Values:")
            for col, count in nulls[nulls > 0].items():
                pct = (count / len(df)) * 100
                print(f" - {col}: {count} ({pct:.2f}%)")
        else:
            print("\nNo missing values found! Great job.")
        print("-" * 30)

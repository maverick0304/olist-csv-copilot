"""
Pandas Tool - Light data transformations and post-aggregations
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PandasTool:
    """Tool for pandas-based data transformations"""
    
    def __init__(self):
        """Initialize pandas tool"""
        pass
    
    def calculate_summary_stats(self, data: pd.DataFrame, column: str) -> Dict[str, float]:
        """
        Calculate summary statistics for a column
        
        Args:
            data: DataFrame
            column: Column name
            
        Returns:
            Dictionary with statistics
        """
        if column not in data.columns:
            logger.error(f"Column '{column}' not found in data")
            return {}
        
        col_data = data[column]
        
        # Filter numeric data
        if not pd.api.types.is_numeric_dtype(col_data):
            logger.warning(f"Column '{column}' is not numeric")
            return {}
        
        stats = {
            'count': int(col_data.count()),
            'mean': float(col_data.mean()),
            'median': float(col_data.median()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'q25': float(col_data.quantile(0.25)),
            'q75': float(col_data.quantile(0.75)),
        }
        
        return stats
    
    def aggregate_by_column(
        self,
        data: pd.DataFrame,
        group_by: str,
        agg_column: str,
        agg_func: str = 'sum',
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Aggregate data by a column
        
        Args:
            data: DataFrame
            group_by: Column to group by
            agg_column: Column to aggregate
            agg_func: Aggregation function ('sum', 'mean', 'count', 'min', 'max')
            top_n: Optional limit to top N results
            
        Returns:
            Aggregated DataFrame
        """
        try:
            # Perform aggregation
            if agg_func == 'sum':
                result = data.groupby(group_by)[agg_column].sum().reset_index()
            elif agg_func == 'mean':
                result = data.groupby(group_by)[agg_column].mean().reset_index()
            elif agg_func == 'count':
                result = data.groupby(group_by)[agg_column].count().reset_index()
            elif agg_func == 'min':
                result = data.groupby(group_by)[agg_column].min().reset_index()
            elif agg_func == 'max':
                result = data.groupby(group_by)[agg_column].max().reset_index()
            else:
                logger.error(f"Unknown aggregation function: {agg_func}")
                return data
            
            # Sort by aggregated column
            result = result.sort_values(agg_column, ascending=False)
            
            # Limit to top N
            if top_n:
                result = result.head(top_n)
            
            return result
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return data
    
    def pivot_table(
        self,
        data: pd.DataFrame,
        index: str,
        columns: str,
        values: str,
        aggfunc: str = 'sum'
    ) -> pd.DataFrame:
        """
        Create pivot table
        
        Args:
            data: DataFrame
            index: Column for index
            columns: Column for columns
            values: Column for values
            aggfunc: Aggregation function
            
        Returns:
            Pivoted DataFrame
        """
        try:
            result = data.pivot_table(
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc,
                fill_value=0
            )
            return result.reset_index()
        except Exception as e:
            logger.error(f"Pivot failed: {e}")
            return data
    
    def calculate_growth_rate(
        self,
        data: pd.DataFrame,
        time_col: str,
        value_col: str,
        periods: int = 1
    ) -> pd.DataFrame:
        """
        Calculate period-over-period growth rate
        
        Args:
            data: DataFrame (must be sorted by time)
            time_col: Time column name
            value_col: Value column name
            periods: Number of periods for comparison
            
        Returns:
            DataFrame with growth_rate column added
        """
        try:
            # Ensure sorted by time
            data = data.sort_values(time_col)
            
            # Calculate percentage change
            data['growth_rate'] = data[value_col].pct_change(periods=periods) * 100
            
            return data
        except Exception as e:
            logger.error(f"Growth rate calculation failed: {e}")
            return data
    
    def calculate_moving_average(
        self,
        data: pd.DataFrame,
        value_col: str,
        window: int = 7
    ) -> pd.DataFrame:
        """
        Calculate moving average
        
        Args:
            data: DataFrame
            value_col: Column to calculate MA for
            window: Window size
            
        Returns:
            DataFrame with moving_average column added
        """
        try:
            data[f'{value_col}_ma{window}'] = data[value_col].rolling(window=window).mean()
            return data
        except Exception as e:
            logger.error(f"Moving average calculation failed: {e}")
            return data
    
    def format_currency(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Format columns as currency
        
        Args:
            data: DataFrame
            columns: List of column names to format
            
        Returns:
            DataFrame with formatted columns
        """
        result = data.copy()
        
        for col in columns:
            if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
        
        return result
    
    def format_percentage(self, data: pd.DataFrame, columns: list, decimals: int = 2) -> pd.DataFrame:
        """
        Format columns as percentage
        
        Args:
            data: DataFrame
            columns: List of column names to format
            decimals: Number of decimal places
            
        Returns:
            DataFrame with formatted columns
        """
        result = data.copy()
        
        for col in columns:
            if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].apply(lambda x: f"{x:.{decimals}f}%" if pd.notna(x) else "0%")
        
        return result
    
    def add_rank_column(
        self,
        data: pd.DataFrame,
        rank_by: str,
        ascending: bool = False,
        rank_column_name: str = 'rank'
    ) -> pd.DataFrame:
        """
        Add rank column based on a value column
        
        Args:
            data: DataFrame
            rank_by: Column to rank by
            ascending: Rank in ascending order
            rank_column_name: Name for rank column
            
        Returns:
            DataFrame with rank column added
        """
        result = data.copy()
        result[rank_column_name] = result[rank_by].rank(ascending=ascending, method='dense')
        return result
    
    def filter_outliers(
        self,
        data: pd.DataFrame,
        column: str,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Filter outliers from data
        
        Args:
            data: DataFrame
            column: Column to check for outliers
            method: Method ('iqr' or 'zscore')
            threshold: Threshold value
            
        Returns:
            Filtered DataFrame
        """
        if column not in data.columns:
            return data
        
        try:
            if method == 'iqr':
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                result = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                result = data[z_scores < threshold]
            
            else:
                logger.error(f"Unknown outlier detection method: {method}")
                result = data
            
            logger.info(f"Filtered {len(data) - len(result)} outliers from {len(data)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Outlier filtering failed: {e}")
            return data
    
    def create_date_features(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Extract date features from datetime column
        
        Args:
            data: DataFrame
            date_column: Name of datetime column
            
        Returns:
            DataFrame with additional date feature columns
        """
        result = data.copy()
        
        try:
            # Ensure datetime type
            result[date_column] = pd.to_datetime(result[date_column])
            
            # Extract features
            result[f'{date_column}_year'] = result[date_column].dt.year
            result[f'{date_column}_quarter'] = result[date_column].dt.quarter
            result[f'{date_column}_month'] = result[date_column].dt.month
            result[f'{date_column}_week'] = result[date_column].dt.isocalendar().week
            result[f'{date_column}_day'] = result[date_column].dt.day
            result[f'{date_column}_dayofweek'] = result[date_column].dt.dayofweek
            result[f'{date_column}_dayname'] = result[date_column].dt.day_name()
            
            return result
            
        except Exception as e:
            logger.error(f"Date feature extraction failed: {e}")
            return data



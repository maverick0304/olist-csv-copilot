"""
Visualization Tool - Auto-generate charts from data
Smart chart type selection based on data characteristics
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class VizTool:
    """Tool for automatic chart generation"""
    
    # Chart type thresholds
    MAX_CATEGORIES_BAR = 50
    MAX_CATEGORIES_PIE = 10
    TIME_SERIES_KEYWORDS = {'date', 'time', 'year', 'month', 'quarter', 'day', 'week'}
    
    def __init__(self):
        """Initialize visualization tool"""
        self.default_template = "plotly_white"
        self.default_height = 500
        self.color_sequence = px.colors.qualitative.Set2
    
    def create_chart(
        self,
        data: pd.DataFrame,
        chart_type: Optional[str] = None,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create chart with auto-detection or specified type
        
        Args:
            data: DataFrame with data
            chart_type: Optional chart type ('bar', 'line', 'pie', 'scatter', 'heatmap')
            x_col: X-axis column name
            y_col: Y-axis column name
            title: Chart title
            **kwargs: Additional arguments for plotly
            
        Returns:
            Plotly Figure object
        """
        if data.empty:
            return self._create_empty_chart("No data to display")
        
        # Auto-detect columns if not specified
        if not x_col or not y_col:
            x_col, y_col = self._detect_columns(data)
        
        # Auto-detect chart type if not specified
        if not chart_type:
            chart_type = self._detect_chart_type(data, x_col, y_col)
        
        # Generate title if not specified
        if not title:
            title = self._generate_title(data, x_col, y_col, chart_type)
        
        # Create chart based on type
        try:
            if chart_type == 'bar':
                fig = self._create_bar_chart(data, x_col, y_col, title, **kwargs)
            elif chart_type == 'line':
                fig = self._create_line_chart(data, x_col, y_col, title, **kwargs)
            elif chart_type == 'pie':
                fig = self._create_pie_chart(data, x_col, y_col, title, **kwargs)
            elif chart_type == 'scatter':
                fig = self._create_scatter_chart(data, x_col, y_col, title, **kwargs)
            elif chart_type == 'heatmap':
                fig = self._create_heatmap(data, title, **kwargs)
            else:
                # Default to bar chart
                fig = self._create_bar_chart(data, x_col, y_col, title, **kwargs)
            
            return fig
            
        except Exception as e:
            logger.error(f"Chart creation failed: {e}")
            return self._create_empty_chart(f"Error creating chart: {e}")
    
    def _detect_columns(self, data: pd.DataFrame) -> tuple:
        """Auto-detect X and Y columns"""
        cols = data.columns.tolist()
        
        if len(cols) < 2:
            return cols[0] if cols else None, None
        
        # Assume first column is X, second is Y (or last if more than 2)
        x_col = cols[0]
        y_col = cols[1] if len(cols) == 2 else cols[-1]
        
        # Prefer numeric columns for Y axis
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols and y_col not in numeric_cols:
            y_col = numeric_cols[0]
        
        return x_col, y_col
    
    def _detect_chart_type(self, data: pd.DataFrame, x_col: str, y_col: str) -> str:
        """Auto-detect appropriate chart type"""
        if not x_col or not y_col:
            return 'bar'
        
        x_data = data[x_col]
        y_data = data[y_col]
        
        # Check if X is time-series
        if self._is_time_series(x_col, x_data):
            return 'line'
        
        # Check if categorical with few categories
        if pd.api.types.is_object_dtype(x_data) or pd.api.types.is_categorical_dtype(x_data):
            n_categories = x_data.nunique()
            
            if n_categories <= self.MAX_CATEGORIES_PIE:
                return 'pie'
            elif n_categories <= self.MAX_CATEGORIES_BAR:
                return 'bar'
            else:
                return 'scatter'
        
        # Both numeric - scatter plot
        if pd.api.types.is_numeric_dtype(x_data) and pd.api.types.is_numeric_dtype(y_data):
            return 'scatter'
        
        # Default to bar
        return 'bar'
    
    def _is_time_series(self, col_name: str, data: pd.Series) -> bool:
        """Check if column represents time series"""
        # Check column name
        col_lower = col_name.lower()
        if any(keyword in col_lower for keyword in self.TIME_SERIES_KEYWORDS):
            return True
        
        # Check data type
        if pd.api.types.is_datetime64_any_dtype(data):
            return True
        
        return False
    
    def _generate_title(self, data: pd.DataFrame, x_col: str, y_col: str, chart_type: str) -> str:
        """Generate descriptive chart title"""
        if not x_col or not y_col:
            return "Data Visualization"
        
        # Clean column names
        y_clean = y_col.replace('_', ' ').title()
        x_clean = x_col.replace('_', ' ').title()
        
        if chart_type == 'line':
            return f"{y_clean} Over {x_clean}"
        elif chart_type == 'pie':
            return f"{y_clean} Distribution by {x_clean}"
        else:
            return f"{y_clean} by {x_clean}"
    
    def _create_bar_chart(self, data: pd.DataFrame, x_col: str, y_col: str, title: str, **kwargs) -> go.Figure:
        """Create bar chart"""
        # Limit number of bars if too many
        if len(data) > self.MAX_CATEGORIES_BAR:
            data = data.nlargest(self.MAX_CATEGORIES_BAR, y_col)
        
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title,
            template=self.default_template,
            color_discrete_sequence=self.color_sequence,
            **kwargs
        )
        
        fig.update_layout(
            height=self.default_height,
            xaxis_tickangle=-45,
            showlegend=False,
        )
        
        # Format Y axis for currency if applicable
        if any(keyword in y_col.lower() for keyword in ['value', 'price', 'revenue', 'gmv', 'payment']):
            fig.update_yaxes(tickprefix="$")
        
        return fig
    
    def _create_line_chart(self, data: pd.DataFrame, x_col: str, y_col: str, title: str, **kwargs) -> go.Figure:
        """Create line chart"""
        fig = px.line(
            data,
            x=x_col,
            y=y_col,
            title=title,
            template=self.default_template,
            markers=True,
            **kwargs
        )
        
        fig.update_layout(
            height=self.default_height,
            hovermode='x unified',
        )
        
        # Format Y axis
        if any(keyword in y_col.lower() for keyword in ['value', 'price', 'revenue', 'gmv', 'payment']):
            fig.update_yaxes(tickprefix="$")
        
        return fig
    
    def _create_pie_chart(self, data: pd.DataFrame, x_col: str, y_col: str, title: str, **kwargs) -> go.Figure:
        """Create pie chart"""
        fig = px.pie(
            data,
            names=x_col,
            values=y_col,
            title=title,
            template=self.default_template,
            **kwargs
        )
        
        fig.update_layout(height=self.default_height)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def _create_scatter_chart(self, data: pd.DataFrame, x_col: str, y_col: str, title: str, **kwargs) -> go.Figure:
        """Create scatter plot"""
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            title=title,
            template=self.default_template,
            **kwargs
        )
        
        fig.update_layout(height=self.default_height)
        
        return fig
    
    def _create_heatmap(self, data: pd.DataFrame, title: str, **kwargs) -> go.Figure:
        """Create heatmap"""
        # Assume data is already in matrix form or pivot
        fig = px.imshow(
            data,
            title=title,
            template=self.default_template,
            aspect='auto',
            **kwargs
        )
        
        fig.update_layout(height=self.default_height)
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        
        fig.update_layout(
            height=self.default_height,
            template=self.default_template,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        
        return fig
    
    def create_multi_chart(
        self,
        data_list: list,
        titles: list,
        chart_types: Optional[list] = None,
        layout: str = 'vertical'
    ) -> go.Figure:
        """
        Create multiple charts in subplots
        
        Args:
            data_list: List of DataFrames
            titles: List of chart titles
            chart_types: Optional list of chart types
            layout: 'vertical', 'horizontal', or 'grid'
            
        Returns:
            Plotly Figure with subplots
        """
        n_charts = len(data_list)
        
        if layout == 'vertical':
            rows, cols = n_charts, 1
        elif layout == 'horizontal':
            rows, cols = 1, n_charts
        else:  # grid
            cols = 2
            rows = (n_charts + 1) // 2
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Add traces for each chart
        for idx, data in enumerate(data_list):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            x_col, y_col = self._detect_columns(data)
            
            fig.add_trace(
                go.Bar(x=data[x_col], y=data[y_col], name=titles[idx]),
                row=row,
                col=col
            )
        
        fig.update_layout(
            height=self.default_height * rows,
            template=self.default_template,
            showlegend=False
        )
        
        return fig



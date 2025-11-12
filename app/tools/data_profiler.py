"""
Data Profiling Tool - Analyze data quality and generate insights
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Provides comprehensive data profiling including quality metrics,
    statistical analysis, and automated insights.
    """
    
    def __init__(self, csv_analyzer):
        """
        Initialize Data Profiler
        
        Args:
            csv_analyzer: CSVAnalyzer instance with loaded tables
        """
        self.analyzer = csv_analyzer
        self.profiles = {}
    
    def profile_table(self, table_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive profile for a table
        
        Returns:
            Dict with quality metrics, statistics, and insights
        """
        logger.info(f"Profiling table: {table_name}")
        
        if table_name not in self.analyzer.tables:
            raise ValueError(f"Table {table_name} not found")
        
        schema_info = self.analyzer.tables[table_name]
        
        profile = {
            'table_name': table_name,
            'row_count': schema_info['row_count'],
            'column_count': len(schema_info['columns']),
            'quality_score': 0.0,
            'issues': [],
            'column_profiles': {},
            'insights': []
        }
        
        # Profile each column
        for col_info in schema_info['columns']:
            col_name = col_info['column_name']
            col_profile = self._profile_column(table_name, col_name, col_info, schema_info)
            profile['column_profiles'][col_name] = col_profile
            
            # Collect issues
            profile['issues'].extend(col_profile.get('issues', []))
        
        # Calculate overall quality score
        profile['quality_score'] = self._calculate_quality_score(profile)
        
        # Generate insights
        profile['insights'] = self._generate_insights(profile)
        
        self.profiles[table_name] = profile
        
        logger.info(f"Profile complete for {table_name}. Quality score: {profile['quality_score']:.1%}")
        
        return profile
    
    def _profile_column(self, table_name: str, col_name: str, 
                       col_info: Dict, schema_info: Dict) -> Dict[str, Any]:
        """
        Profile individual column with statistics and quality checks
        """
        col_type = col_info['data_type']
        stats = schema_info['column_stats'].get(col_name, {})
        
        profile = {
            'name': col_name,
            'type': col_type,
            'distinct_count': stats.get('distinct_count', 0),
            'null_count': stats.get('null_count', 0),
            'null_percentage': stats.get('null_percentage', 0),
            'cardinality_ratio': stats.get('cardinality_ratio', 0),
            'issues': []
        }
        
        # Get column data for detailed analysis
        try:
            data = self.analyzer.conn.execute(
                f'SELECT "{col_name}" FROM {table_name}'
            ).df()[col_name]
            
            # Numeric column analysis
            if self._is_numeric_type(col_type):
                profile.update(self._analyze_numeric_column(data, col_name))
            
            # Text column analysis
            elif self._is_text_type(col_type):
                profile.update(self._analyze_text_column(data, col_name))
            
            # Date/Time column analysis
            elif self._is_temporal_type(col_type):
                profile.update(self._analyze_temporal_column(data, col_name))
            
            # Quality checks
            profile['issues'].extend(self._check_column_quality(data, col_name, col_type, stats))
            
        except Exception as e:
            logger.warning(f"Could not profile column {col_name}: {e}")
            profile['error'] = str(e)
        
        return profile
    
    def _analyze_numeric_column(self, data: pd.Series, col_name: str) -> Dict[str, Any]:
        """Analyze numeric column statistics"""
        numeric_data = pd.to_numeric(data, errors='coerce').dropna()
        
        if len(numeric_data) == 0:
            return {'stats': None}
        
        return {
            'stats': {
                'min': float(numeric_data.min()),
                'max': float(numeric_data.max()),
                'mean': float(numeric_data.mean()),
                'median': float(numeric_data.median()),
                'std': float(numeric_data.std()) if len(numeric_data) > 1 else 0,
                'q25': float(numeric_data.quantile(0.25)),
                'q75': float(numeric_data.quantile(0.75)),
                'zeros': int((numeric_data == 0).sum()),
                'negatives': int((numeric_data < 0).sum())
            }
        }
    
    def _analyze_text_column(self, data: pd.Series, col_name: str) -> Dict[str, Any]:
        """Analyze text column patterns"""
        text_data = data.dropna().astype(str)
        
        if len(text_data) == 0:
            return {'text_stats': None}
        
        lengths = text_data.str.len()
        
        return {
            'text_stats': {
                'min_length': int(lengths.min()),
                'max_length': int(lengths.max()),
                'avg_length': float(lengths.mean()),
                'empty_strings': int((text_data == '').sum()),
                'most_common': text_data.value_counts().head(5).to_dict()
            }
        }
    
    def _analyze_temporal_column(self, data: pd.Series, col_name: str) -> Dict[str, Any]:
        """Analyze date/time column patterns"""
        try:
            date_data = pd.to_datetime(data, errors='coerce').dropna()
            
            if len(date_data) == 0:
                return {'temporal_stats': None}
            
            return {
                'temporal_stats': {
                    'min_date': str(date_data.min()),
                    'max_date': str(date_data.max()),
                    'date_range_days': (date_data.max() - date_data.min()).days,
                    'unique_dates': int(date_data.nunique())
                }
            }
        except Exception:
            return {'temporal_stats': None}
    
    def _check_column_quality(self, data: pd.Series, col_name: str, 
                             col_type: str, stats: Dict) -> List[Dict[str, str]]:
        """
        Check for data quality issues
        
        Returns:
            List of issues with severity levels
        """
        issues = []
        null_pct = stats.get('null_percentage', 0)
        
        # High null percentage
        if null_pct > 50:
            issues.append({
                'column': col_name,
                'type': 'high_null_percentage',
                'severity': 'high',
                'message': f"{col_name} has {null_pct:.1f}% null values"
            })
        elif null_pct > 20:
            issues.append({
                'column': col_name,
                'type': 'moderate_null_percentage',
                'severity': 'medium',
                'message': f"{col_name} has {null_pct:.1f}% null values"
            })
        
        # Single value (no variance)
        if stats.get('distinct_count', 0) == 1:
            issues.append({
                'column': col_name,
                'type': 'no_variance',
                'severity': 'medium',
                'message': f"{col_name} has only one unique value"
            })
        
        # Check for suspicious patterns in numeric columns
        if self._is_numeric_type(col_type):
            numeric_data = pd.to_numeric(data, errors='coerce').dropna()
            
            if len(numeric_data) > 0:
                # All zeros
                if (numeric_data == 0).all():
                    issues.append({
                        'column': col_name,
                        'type': 'all_zeros',
                        'severity': 'medium',
                        'message': f"{col_name} contains only zeros"
                    })
                
                # Outliers (simple Z-score method)
                if len(numeric_data) > 10:
                    z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
                    outlier_count = (z_scores > 3).sum()
                    
                    if outlier_count > len(numeric_data) * 0.05:  # More than 5% outliers
                        issues.append({
                            'column': col_name,
                            'type': 'many_outliers',
                            'severity': 'low',
                            'message': f"{col_name} has {outlier_count} potential outliers"
                        })
        
        return issues
    
    def _calculate_quality_score(self, profile: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score (0-1)
        
        Based on:
        - Completeness (null percentages)
        - Uniqueness (appropriate cardinality)
        - Validity (data type consistency)
        """
        scores = []
        
        for col_name, col_profile in profile['column_profiles'].items():
            col_score = 1.0
            
            # Penalize high null percentages
            null_pct = col_profile.get('null_percentage', 0)
            if null_pct > 0:
                col_score -= (null_pct / 100) * 0.5
            
            # Penalize no variance
            if col_profile.get('distinct_count', 0) <= 1:
                col_score -= 0.3
            
            # Penalize errors
            if col_profile.get('error'):
                col_score -= 0.2
            
            scores.append(max(col_score, 0))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_insights(self, profile: Dict[str, Any]) -> List[str]:
        """
        Generate automated insights from profile
        """
        insights = []
        
        # Overall quality
        quality_score = profile['quality_score']
        if quality_score > 0.9:
            insights.append(f"✓ Excellent data quality ({quality_score:.0%})")
        elif quality_score > 0.7:
            insights.append(f"⚠ Good data quality ({quality_score:.0%}) with minor issues")
        else:
            insights.append(f"⚠ Data quality needs attention ({quality_score:.0%})")
        
        # Completeness
        high_null_cols = [
            col for col, prof in profile['column_profiles'].items()
            if prof.get('null_percentage', 0) > 30
        ]
        if high_null_cols:
            insights.append(f"⚠ {len(high_null_cols)} columns have >30% missing values: {', '.join(high_null_cols[:3])}")
        
        # Data size
        row_count = profile['row_count']
        if row_count < 100:
            insights.append(f"⚠ Small dataset ({row_count:,} rows) - results may not be statistically significant")
        elif row_count > 1_000_000:
            insights.append(f"✓ Large dataset ({row_count:,} rows) - good for analysis")
        
        # Column types
        metric_cols = [
            col for col, prof in profile['column_profiles'].items()
            if self._is_numeric_type(prof.get('type', ''))
        ]
        if metric_cols:
            insights.append(f"✓ Found {len(metric_cols)} numeric columns for quantitative analysis")
        
        return insights
    
    def _is_numeric_type(self, col_type: str) -> bool:
        """Check if column type is numeric"""
        return any(t in col_type.upper() for t in ['INT', 'FLOAT', 'DOUBLE', 'DECIMAL', 'NUMERIC'])
    
    def _is_text_type(self, col_type: str) -> bool:
        """Check if column type is text"""
        return any(t in col_type.upper() for t in ['VARCHAR', 'CHAR', 'TEXT', 'STRING'])
    
    def _is_temporal_type(self, col_type: str) -> bool:
        """Check if column type is date/time"""
        return any(t in col_type.upper() for t in ['DATE', 'TIME', 'TIMESTAMP'])
    
    def get_summary_report(self) -> str:
        """
        Generate human-readable summary report for all profiled tables
        """
        if not self.profiles:
            return "No tables have been profiled yet."
        
        report = []
        report.append("# DATA QUALITY REPORT\n")
        
        for table_name, profile in self.profiles.items():
            report.append(f"## Table: {table_name}")
            report.append(f"Quality Score: {profile['quality_score']:.0%}")
            report.append(f"Rows: {profile['row_count']:,} | Columns: {profile['column_count']}")
            
            # Insights
            if profile['insights']:
                report.append("\n### Key Insights:")
                for insight in profile['insights']:
                    report.append(f"  {insight}")
            
            # Issues
            if profile['issues']:
                report.append(f"\n### Issues Found ({len(profile['issues'])}):")
                # Group by severity
                for severity in ['high', 'medium', 'low']:
                    severity_issues = [i for i in profile['issues'] if i.get('severity') == severity]
                    if severity_issues:
                        report.append(f"\n  {severity.upper()}:")
                        for issue in severity_issues[:5]:  # Limit to 5 per severity
                            report.append(f"    • {issue['message']}")
            
            report.append("\n")
        
        return "\n".join(report)


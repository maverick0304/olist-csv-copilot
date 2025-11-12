"""
Anomaly Detection Tool - Detect unusual patterns and outliers in data
Uses statistical methods and AI to identify anomalies
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class AnomalyTool:
    """Tool for detecting anomalies and unusual patterns in data"""
    
    def __init__(self, sensitivity: float = 2.0):
        """
        Initialize anomaly detection tool
        
        Args:
            sensitivity: Z-score threshold (default: 2.0 = ~95% confidence)
                        Lower = more sensitive (more alerts)
                        Higher = less sensitive (fewer alerts)
        """
        self.sensitivity = sensitivity
        self.methods = {
            'zscore': self._detect_zscore,
            'iqr': self._detect_iqr,
            'percent_change': self._detect_percent_change,
            'missing_values': self._detect_missing_values
        }
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies using multiple methods
        
        Args:
            data: DataFrame to analyze
            methods: List of methods to use (default: all)
            
        Returns:
            Dictionary with anomaly information
        """
        if data.empty:
            return {
                'has_anomalies': False,
                'anomalies': [],
                'summary': "No data to analyze"
            }
        
        # Use all methods if none specified
        if methods is None:
            methods = list(self.methods.keys())
        
        all_anomalies = []
        
        # Run each detection method
        for method_name in methods:
            if method_name in self.methods:
                try:
                    anomalies = self.methods[method_name](data)
                    all_anomalies.extend(anomalies)
                except Exception as e:
                    logger.warning(f"Anomaly detection method '{method_name}' failed: {e}")
        
        # Deduplicate and sort by severity
        unique_anomalies = self._deduplicate_anomalies(all_anomalies)
        unique_anomalies.sort(key=lambda x: x.get('severity_score', 0), reverse=True)
        
        return {
            'has_anomalies': len(unique_anomalies) > 0,
            'anomaly_count': len(unique_anomalies),
            'anomalies': unique_anomalies,
            'summary': self._generate_summary(unique_anomalies)
        }
    
    def _detect_zscore(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect outliers using Z-score method
        
        Identifies values that are more than N standard deviations from mean
        """
        anomalies = []
        
        # Check numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if len(data[col].dropna()) < 3:
                continue  # Need at least 3 values
            
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            
            # Find outliers
            outlier_indices = np.where(z_scores > self.sensitivity)[0]
            
            if len(outlier_indices) > 0:
                for idx in outlier_indices[:5]:  # Limit to top 5
                    actual_idx = data[col].dropna().index[idx]
                    value = data.loc[actual_idx, col]
                    z_score = z_scores[idx]
                    
                    anomalies.append({
                        'type': 'outlier',
                        'method': 'zscore',
                        'column': col,
                        'row_index': int(actual_idx),
                        'value': float(value),
                        'z_score': float(z_score),
                        'severity_score': float(z_score),
                        'description': f"{col} value {value:,.2f} is {z_score:.1f} standard deviations from mean",
                        'severity': self._get_severity_level(z_score)
                    })
        
        return anomalies
    
    def _detect_iqr(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect outliers using Interquartile Range (IQR) method
        
        More robust to extreme values than Z-score
        """
        anomalies = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if len(data[col].dropna()) < 4:
                continue
            
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            if len(outliers) > 0:
                for idx in outliers.head(5).index:  # Limit to top 5
                    value = data.loc[idx, col]
                    distance = max(abs(value - lower_bound), abs(value - upper_bound)) / IQR
                    
                    anomalies.append({
                        'type': 'outlier',
                        'method': 'iqr',
                        'column': col,
                        'row_index': int(idx),
                        'value': float(value),
                        'iqr_distance': float(distance),
                        'severity_score': float(distance),
                        'description': f"{col} value {value:,.2f} is outside normal range [{lower_bound:,.2f}, {upper_bound:,.2f}]",
                        'severity': self._get_severity_level(distance)
                    })
        
        return anomalies
    
    def _detect_percent_change(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect large percent changes between consecutive rows
        
        Good for time-series data
        """
        anomalies = []
        
        # Check if data has a time-like column
        time_cols = [col for col in data.columns if any(
            keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'quarter']
        )]
        
        if not time_cols:
            return anomalies  # Not time-series data
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if len(data[col].dropna()) < 2:
                continue
            
            # Calculate percent change
            pct_change = data[col].pct_change().abs()
            
            # Find large changes (>50% by default)
            threshold = 0.5
            large_changes = pct_change[pct_change > threshold]
            
            if len(large_changes) > 0:
                for idx in large_changes.head(3).index:  # Limit to top 3
                    change = pct_change.loc[idx]
                    current_value = data.loc[idx, col]
                    prev_value = data.loc[idx - 1, col] if idx > 0 else None
                    
                    if prev_value is not None and not pd.isna(prev_value):
                        anomalies.append({
                            'type': 'sudden_change',
                            'method': 'percent_change',
                            'column': col,
                            'row_index': int(idx),
                            'current_value': float(current_value),
                            'previous_value': float(prev_value),
                            'percent_change': float(change * 100),
                            'severity_score': float(change * 2),  # Amplify for severity
                            'description': f"{col} changed {change*100:.1f}% from {prev_value:,.2f} to {current_value:,.2f}",
                            'severity': 'high' if change > 1.0 else 'medium'
                        })
        
        return anomalies
    
    def _detect_missing_values(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect columns with high percentage of missing values
        """
        anomalies = []
        
        threshold = 0.1  # 10% missing is noteworthy
        
        for col in data.columns:
            missing_pct = data[col].isna().sum() / len(data)
            
            if missing_pct > threshold:
                anomalies.append({
                    'type': 'data_quality',
                    'method': 'missing_values',
                    'column': col,
                    'missing_count': int(data[col].isna().sum()),
                    'missing_percent': float(missing_pct * 100),
                    'severity_score': float(missing_pct * 3),  # Amplify for severity
                    'description': f"{col} has {missing_pct*100:.1f}% missing values",
                    'severity': 'high' if missing_pct > 0.3 else 'medium'
                })
        
        return anomalies
    
    def _deduplicate_anomalies(self, anomalies: List[Dict]) -> List[Dict]:
        """Remove duplicate anomalies (same row/column)"""
        seen = set()
        unique = []
        
        for anomaly in anomalies:
            key = (
                anomaly.get('column'),
                anomaly.get('row_index'),
                anomaly.get('type')
            )
            
            if key not in seen:
                seen.add(key)
                unique.append(anomaly)
        
        return unique
    
    def _get_severity_level(self, score: float) -> str:
        """Convert numeric score to severity level"""
        if score > 4:
            return 'critical'
        elif score > 3:
            return 'high'
        elif score > 2:
            return 'medium'
        else:
            return 'low'
    
    def _generate_summary(self, anomalies: List[Dict]) -> str:
        """Generate human-readable summary of anomalies"""
        if not anomalies:
            return "No anomalies detected - data looks normal! ✅"
        
        # Count by type
        type_counts = {}
        for anomaly in anomalies:
            anom_type = anomaly['type']
            type_counts[anom_type] = type_counts.get(anom_type, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'low')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Build summary
        parts = [f"Found {len(anomalies)} anomalies:"]
        
        if 'outlier' in type_counts:
            parts.append(f"• {type_counts['outlier']} outliers")
        if 'sudden_change' in type_counts:
            parts.append(f"• {type_counts['sudden_change']} sudden changes")
        if 'data_quality' in type_counts:
            parts.append(f"• {type_counts['data_quality']} data quality issues")
        
        # Add severity info
        if 'critical' in severity_counts or 'high' in severity_counts:
            critical_high = severity_counts.get('critical', 0) + severity_counts.get('high', 0)
            parts.append(f"⚠️ {critical_high} need immediate attention!")
        
        return " ".join(parts)
    
    def format_anomalies_for_display(self, anomalies: List[Dict]) -> str:
        """
        Format anomalies as readable text for UI display
        
        Returns:
            Formatted string with emoji indicators
        """
        if not anomalies:
            return "✅ No anomalies detected"
        
        lines = []
        emoji_map = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        
        for i, anomaly in enumerate(anomalies[:10], 1):  # Limit to top 10
            emoji = emoji_map.get(anomaly.get('severity', 'low'), '🔵')
            desc = anomaly.get('description', 'Unknown anomaly')
            lines.append(f"{emoji} {desc}")
        
        if len(anomalies) > 10:
            lines.append(f"\n... and {len(anomalies) - 10} more")
        
        return "\n".join(lines)




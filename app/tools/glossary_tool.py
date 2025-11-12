"""
Glossary Tool - Provides definitions for business metrics
"""

import logging
from typing import Dict, List, Optional
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class GlossaryTool:
    """Tool for accessing metric definitions and business glossary"""
    
    def __init__(self, metrics_path: Optional[Path] = None):
        """
        Initialize glossary tool
        
        Args:
            metrics_path: Path to metrics.yaml file
        """
        if metrics_path is None:
            metrics_path = Path(__file__).parent.parent / "semantic" / "metrics.yaml"
        
        self.metrics_path = metrics_path
        self.metrics = {}
        
        if metrics_path.exists():
            self._load_metrics()
        else:
            logger.warning(f"Metrics file not found: {metrics_path}")
    
    def _load_metrics(self):
        """Load metrics from YAML file"""
        try:
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.metrics = data.get('metrics', {})
            logger.info(f"Loaded {len(self.metrics)} metrics from {self.metrics_path}")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            self.metrics = {}
    
    def get_metric(self, metric_name: str) -> Optional[Dict]:
        """
        Get definition for a specific metric
        
        Args:
            metric_name: Name of the metric (case-insensitive)
            
        Returns:
            Dictionary with metric definition or None
        """
        # Normalize name
        metric_key = metric_name.lower().replace(' ', '_')
        
        # Try exact match
        if metric_key in self.metrics:
            return self.metrics[metric_key]
        
        # Try partial match
        for key, value in self.metrics.items():
            if metric_key in key or key in metric_key:
                return value
        
        return None
    
    def list_metrics(self) -> List[str]:
        """
        Get list of all available metric names
        
        Returns:
            List of metric names
        """
        return sorted(self.metrics.keys())
    
    def search_metrics(self, query: str) -> List[Dict]:
        """
        Search metrics by keyword
        
        Args:
            query: Search query
            
        Returns:
            List of matching metric definitions
        """
        query_lower = query.lower()
        results = []
        
        for name, definition in self.metrics.items():
            # Search in name
            if query_lower in name.lower():
                results.append({'name': name, **definition})
                continue
            
            # Search in description
            if 'description' in definition and query_lower in definition['description'].lower():
                results.append({'name': name, **definition})
                continue
            
            # Search in aliases
            if 'aliases' in definition:
                for alias in definition['aliases']:
                    if query_lower in alias.lower():
                        results.append({'name': name, **definition})
                        break
        
        return results
    
    def get_metric_sql(self, metric_name: str, **kwargs) -> Optional[str]:
        """
        Get SQL template for a metric
        
        Args:
            metric_name: Name of the metric
            **kwargs: Template variables (e.g., date_filter, category_filter)
            
        Returns:
            SQL query string or None
        """
        metric = self.get_metric(metric_name)
        
        if not metric or 'sql_template' not in metric:
            return None
        
        sql_template = metric['sql_template']
        
        # Replace template variables
        try:
            sql = sql_template.format(**kwargs)
            return sql
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return sql_template
    
    def explain_metric(self, metric_name: str) -> str:
        """
        Get human-readable explanation of a metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Formatted explanation string
        """
        metric = self.get_metric(metric_name)
        
        if not metric:
            return f"Metric '{metric_name}' not found. Available metrics: {', '.join(self.list_metrics())}"
        
        explanation = []
        explanation.append(f"**{metric_name.upper()}**")
        explanation.append("")
        
        if 'description' in metric:
            explanation.append(metric['description'])
            explanation.append("")
        
        if 'formula' in metric:
            explanation.append(f"**Formula:** {metric['formula']}")
            explanation.append("")
        
        if 'grain' in metric:
            explanation.append(f"**Grain:** {metric['grain']}")
        
        if 'time_window' in metric:
            explanation.append(f"**Default Time Window:** {metric['time_window']}")
        
        if 'aliases' in metric:
            explanation.append(f"**Also known as:** {', '.join(metric['aliases'])}")
        
        if 'example' in metric:
            explanation.append("")
            explanation.append(f"**Example:** {metric['example']}")
        
        return "\n".join(explanation)
    
    def get_all_metrics_summary(self) -> str:
        """
        Get summary of all available metrics
        
        Returns:
            Formatted string with all metrics
        """
        if not self.metrics:
            return "No metrics available."
        
        lines = ["**Available Metrics:**", ""]
        
        for name, definition in sorted(self.metrics.items()):
            desc = definition.get('description', 'No description')
            lines.append(f"- **{name}**: {desc}")
        
        return "\n".join(lines)
    
    def get_related_tables(self, metric_name: str) -> List[str]:
        """
        Get tables required for calculating a metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of table names
        """
        metric = self.get_metric(metric_name)
        
        if not metric:
            return []
        
        return metric.get('required_tables', [])



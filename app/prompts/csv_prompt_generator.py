"""
Dynamic Prompt Generator - Create LLM prompts from CSV schema
"""
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class CSVPromptGenerator:
    """
    Generates intelligent prompts for LLM based on dynamically loaded CSV schema
    """
    
    def __init__(self, csv_analyzer, data_profiler=None):
        """
        Initialize prompt generator
        
        Args:
            csv_analyzer: CSVAnalyzer instance with loaded tables
            data_profiler: Optional DataProfiler for quality insights
        """
        self.analyzer = csv_analyzer
        self.profiler = data_profiler
    
    def generate_schema_prompt(self) -> str:
        """
        Generate comprehensive schema documentation for LLM
        
        Returns:
            Formatted prompt string with schema details
        """
        prompt_parts = []
        
        # Header
        prompt_parts.append("# DATABASE SCHEMA DOCUMENTATION")
        prompt_parts.append("\nYou are analyzing a custom dataset with the following structure:\n")
        
        # Table summaries
        prompt_parts.append("## TABLES OVERVIEW\n")
        for table_name, info in self.analyzer.tables.items():
            prompt_parts.append(f"### {table_name}")
            prompt_parts.append(f"- **Rows**: {info['row_count']:,}")
            prompt_parts.append(f"- **Columns**: {len(info['columns'])}")
            
            # Add data quality if available
            if self.profiler and table_name in self.profiler.profiles:
                quality = self.profiler.profiles[table_name]['quality_score']
                prompt_parts.append(f"- **Data Quality**: {quality:.0%}")
            
            prompt_parts.append("")
        
        # Detailed column information
        for table_name, info in self.analyzer.tables.items():
            prompt_parts.append(f"\n## TABLE: {table_name}\n")
            
            # Group columns by purpose
            grouped = self._group_columns_by_purpose(table_name, info)
            
            # Identifiers
            if grouped['identifiers']:
                prompt_parts.append("### Identifiers (Primary/Foreign Keys):")
                for col in grouped['identifiers']:
                    prompt_parts.append(self._format_column_doc(table_name, col, info))
                prompt_parts.append("")
            
            # Temporal fields
            if grouped['temporal']:
                prompt_parts.append("### Time-Related Fields:")
                for col in grouped['temporal']:
                    prompt_parts.append(self._format_column_doc(table_name, col, info))
                prompt_parts.append("")
            
            # Metrics
            if grouped['metrics']:
                prompt_parts.append("### Metrics (Quantitative Fields):")
                for col in grouped['metrics']:
                    prompt_parts.append(self._format_column_doc(table_name, col, info))
                prompt_parts.append("")
            
            # Dimensions
            if grouped['dimensions']:
                prompt_parts.append("### Dimensions (Categorical Fields):")
                for col in grouped['dimensions']:
                    prompt_parts.append(self._format_column_doc(table_name, col, info))
                prompt_parts.append("")
            
            # Other fields
            if grouped['others']:
                prompt_parts.append("### Other Fields:")
                for col in grouped['others']:
                    prompt_parts.append(self._format_column_doc(table_name, col, info))
                prompt_parts.append("")
            
            # Sample data
            prompt_parts.append("### Sample Data Preview:")
            prompt_parts.append("```")
            prompt_parts.append(info['sample'].head(3).to_string(index=False))
            prompt_parts.append("```\n")
        
        # Relationships
        if self.analyzer.relationships:
            prompt_parts.append("\n## TABLE RELATIONSHIPS\n")
            prompt_parts.append("The following relationships have been detected:\n")
            
            for rel in self.analyzer.relationships:
                confidence_emoji = "✓" if rel['confidence'] > 0.8 else "~"
                prompt_parts.append(
                    f"{confidence_emoji} **{rel['from_table']}.{rel['from_column']}** → "
                    f"**{rel['to_table']}.{rel['to_column']}** "
                    f"(confidence: {rel['confidence']:.0%})"
                )
            
            prompt_parts.append("\n**IMPORTANT**: Use these relationships for JOIN operations.\n")
        
        # Add SQL generation rules
        prompt_parts.append(self._generate_sql_rules())
        
        return "\n".join(prompt_parts)
    
    def _group_columns_by_purpose(self, table_name: str, info: Dict) -> Dict[str, List[str]]:
        """Group columns by their detected purpose"""
        grouped = {
            'identifiers': [],
            'temporal': [],
            'metrics': [],
            'dimensions': [],
            'others': []
        }
        
        for col_info in info['columns']:
            col_name = col_info['column_name']
            key = f"{table_name}.{col_name}"
            meta = self.analyzer.column_metadata.get(key, {})
            purpose = meta.get('purpose', 'unknown')
            
            if purpose == 'identifier':
                grouped['identifiers'].append(col_name)
            elif purpose == 'temporal':
                grouped['temporal'].append(col_name)
            elif purpose == 'metric':
                grouped['metrics'].append(col_name)
            elif purpose == 'dimension':
                grouped['dimensions'].append(col_name)
            else:
                grouped['others'].append(col_name)
        
        return grouped
    
    def _format_column_doc(self, table_name: str, col_name: str, info: Dict) -> str:
        """Format column documentation with details"""
        col_info = next((c for c in info['columns'] if c['column_name'] == col_name), None)
        if not col_info:
            return f"- **{col_name}**"
        
        col_type = col_info['data_type']
        stats = info['column_stats'].get(col_name, {})
        
        # Build documentation line
        doc = f"- **{col_name}** ({col_type})"
        
        # Add statistics
        details = []
        
        distinct = stats.get('distinct_count', 0)
        if distinct > 0:
            details.append(f"{distinct:,} unique")
        
        null_pct = stats.get('null_percentage', 0)
        if null_pct > 10:
            details.append(f"{null_pct:.0f}% null")
        
        if details:
            doc += f" - {', '.join(details)}"
        
        # Add purpose hint
        key = f"{table_name}.{col_name}"
        meta = self.analyzer.column_metadata.get(key, {})
        purpose = meta.get('purpose', '')
        
        if purpose == 'metric':
            doc += " [use for aggregations like SUM, AVG]"
        elif purpose == 'dimension':
            doc += " [use for GROUP BY]"
        elif purpose == 'temporal':
            doc += " [use for time-based analysis]"
        
        return doc
    
    def _generate_sql_rules(self) -> str:
        """Generate SQL generation rules specific to this dataset"""
        rules = []
        
        rules.append("\n## SQL GENERATION RULES\n")
        rules.append("When generating SQL queries for this dataset:\n")
        
        # Rule 1: Table names
        table_list = "`, `".join(self.analyzer.get_table_list())
        rules.append(f"1. **Available Tables**: `{table_list}`")
        rules.append("   - Use these exact table names in your queries")
        rules.append("   - Wrap table/column names with special characters in double quotes\n")
        
        # Rule 2: Joins
        if self.analyzer.relationships:
            rules.append("2. **JOIN Operations**:")
            rules.append("   - Use the detected relationships listed above")
            rules.append("   - When combining data from multiple tables, always use proper JOIN syntax")
            rules.append("   - Example: `SELECT ... FROM table1 JOIN table2 ON table1.id = table2.table1_id`\n")
        
        # Rule 3: Aggregations
        metric_tables = {}
        for table_name, info in self.analyzer.tables.items():
            metrics = []
            for col in info['columns']:
                key = f"{table_name}.{col['column_name']}"
                if self.analyzer.column_metadata.get(key, {}).get('is_metric'):
                    metrics.append(col['column_name'])
            if metrics:
                metric_tables[table_name] = metrics
        
        if metric_tables:
            rules.append("3. **Aggregations**:")
            for table, metrics in metric_tables.items():
                rules.append(f"   - In `{table}`: Use {', '.join(f'`{m}`' for m in metrics)} for SUM, AVG, COUNT")
            rules.append("   - Always include GROUP BY when using aggregation functions\n")
        
        # Rule 4: Time-based analysis
        temporal_cols = {}
        for table_name, info in self.analyzer.tables.items():
            temporal = []
            for col in info['columns']:
                key = f"{table_name}.{col['column_name']}"
                if self.analyzer.column_metadata.get(key, {}).get('is_temporal'):
                    temporal.append(col['column_name'])
            if temporal:
                temporal_cols[table_name] = temporal
        
        if temporal_cols:
            rules.append("4. **Time-Based Analysis**:")
            for table, cols in temporal_cols.items():
                rules.append(f"   - In `{table}`: Use {', '.join(f'`{c}`' for c in cols)} for trends over time")
            rules.append("   - Extract year, month, quarter using date functions\n")
        
        # Rule 5: Safety
        rules.append("5. **Query Safety**:")
        rules.append("   - Only use SELECT statements (no INSERT, UPDATE, DELETE)")
        rules.append("   - Add LIMIT clause for large result sets")
        rules.append("   - Handle NULL values appropriately\n")
        
        # Rule 6: Best Practices
        rules.append("6. **Best Practices**:")
        rules.append("   - Use meaningful column aliases")
        rules.append("   - Format numbers appropriately (ROUND for decimals)")
        rules.append("   - Sort results in a logical order (ORDER BY)")
        rules.append("   - For 'top N' queries, use ORDER BY with LIMIT\n")
        
        return "\n".join(rules)
    
    def generate_example_questions(self, count: int = 10) -> List[str]:
        """
        Generate example questions based on the schema
        
        Args:
            count: Number of example questions to generate
        
        Returns:
            List of natural language questions
        """
        questions = []
        
        for table_name, info in self.analyzer.tables.items():
            # Find metrics, dimensions, and temporal columns
            metrics = []
            dimensions = []
            temporal = []
            
            for col in info['columns']:
                key = f"{table_name}.{col['column_name']}"
                meta = self.analyzer.column_metadata.get(key, {})
                
                if meta.get('is_metric'):
                    metrics.append(col['column_name'])
                elif meta.get('is_dimension'):
                    dimensions.append(col['column_name'])
                elif meta.get('is_temporal'):
                    temporal.append(col['column_name'])
            
            # Generate questions based on available columns
            
            # Aggregation questions
            if metrics and dimensions:
                questions.append(f"What is the total {metrics[0]} by {dimensions[0]}?")
                if len(metrics) > 1:
                    questions.append(f"Show me average {metrics[1]} grouped by {dimensions[0]}")
            
            # Trend questions
            if metrics and temporal:
                questions.append(f"Show me {metrics[0]} trend over time")
                questions.append(f"What's the {metrics[0]} breakdown by month?")
            
            # Top N questions
            if metrics and dimensions:
                questions.append(f"What are the top 10 {dimensions[0]} by {metrics[0]}?")
            
            # Count questions
            if dimensions:
                questions.append(f"How many unique {dimensions[0]} values are there?")
            
            # Comparison questions
            if len(dimensions) > 1 and metrics:
                questions.append(f"Compare {metrics[0]} across different {dimensions[0]}")
            
            # Summary questions
            questions.append(f"Give me a summary of the {table_name} table")
            
            if len(questions) >= count:
                break
        
        # Limit to requested count
        return questions[:count]
    
    def generate_few_shot_examples(self) -> List[Dict[str, str]]:
        """
        Generate few-shot examples for SQL generation
        
        Returns:
            List of {question, sql, reasoning} dictionaries
        """
        examples = []
        
        for table_name, info in self.analyzer.tables.items():
            # Find first metric and dimension
            first_metric = None
            first_dimension = None
            
            for col in info['columns']:
                key = f"{table_name}.{col['column_name']}"
                meta = self.analyzer.column_metadata.get(key, {})
                
                if not first_metric and meta.get('is_metric'):
                    first_metric = col['column_name']
                if not first_dimension and meta.get('is_dimension'):
                    first_dimension = col['column_name']
                
                if first_metric and first_dimension:
                    break
            
            # Example 1: Simple aggregation
            if first_metric and first_dimension:
                examples.append({
                    'question': f"What is the total {first_metric} by {first_dimension}?",
                    'sql': f'SELECT "{first_dimension}", SUM("{first_metric}") as total_{first_metric}\nFROM {table_name}\nGROUP BY "{first_dimension}"\nORDER BY total_{first_metric} DESC',
                    'reasoning': f"Aggregate {first_metric} using SUM, group by {first_dimension}, and sort by total"
                })
            
            # Example 2: Top N
            if first_metric and first_dimension:
                examples.append({
                    'question': f"Show me top 5 {first_dimension} by {first_metric}",
                    'sql': f'SELECT "{first_dimension}", SUM("{first_metric}") as total\nFROM {table_name}\nGROUP BY "{first_dimension}"\nORDER BY total DESC\nLIMIT 5',
                    'reasoning': f"Top N query requires GROUP BY, ORDER BY DESC, and LIMIT"
                })
            
            # Example 3: Count
            if first_dimension:
                examples.append({
                    'question': f"How many rows in {table_name}?",
                    'sql': f'SELECT COUNT(*) as row_count FROM {table_name}',
                    'reasoning': "Simple count of all rows"
                })
            
            # Limit examples
            if len(examples) >= 6:
                break
        
        return examples[:6]


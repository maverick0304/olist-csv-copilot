"""
CSV Analysis Tool - Load and analyze custom CSV files
"""
import duckdb
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class CSVAnalyzer:
    """
    Analyzes CSV files and provides intelligent schema detection,
    relationship inference, and data profiling.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize CSV Analyzer
        
        Args:
            db_path: Optional path to persistent DuckDB file. If None, uses in-memory DB.
        """
        if db_path:
            self.conn = duckdb.connect(db_path)
        else:
            self.conn = duckdb.connect(':memory:')
        
        self.tables = {}
        self.relationships = []
        self.column_metadata = {}
        logger.info("CSV Analyzer initialized")
    
    def load_csv(self, file_path: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load CSV file into DuckDB with automatic schema detection
        
        Args:
            file_path: Path to CSV file
            table_name: Optional table name. If None, uses filename.
        
        Returns:
            Dictionary with schema information
        """
        try:
            # Generate table name from filename if not provided
            if table_name is None:
                table_name = Path(file_path).stem.lower()
                # Clean table name (remove special chars)
                table_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in table_name)
            
            logger.info(f"Loading CSV: {file_path} as table '{table_name}'")
            
            # Use DuckDB's read_csv_auto for intelligent type detection
            self.conn.execute(f"""
                CREATE TABLE {table_name} AS 
                SELECT * FROM read_csv_auto('{file_path}', 
                    header=true,
                    auto_detect=true,
                    ignore_errors=false,
                    sample_size=10000
                )
            """)
            
            # Get comprehensive schema information
            schema_info = self._get_table_schema(table_name)
            
            # Store in tables registry
            self.tables[table_name] = schema_info
            
            # Analyze column types and purposes
            self._analyze_column_metadata(table_name, schema_info)
            
            logger.info(f"Successfully loaded {table_name}: {schema_info['row_count']:,} rows, {len(schema_info['columns'])} columns")
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Failed to load CSV {file_path}: {e}")
            raise ValueError(f"Error loading CSV: {str(e)}")
    
    def _get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Extract detailed schema information for a table
        
        Returns:
            Dict with columns, types, stats, and sample data
        """
        # Get column information
        columns_query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """
        columns_df = self.conn.execute(columns_query).df()
        
        # Get row count
        row_count = self.conn.execute(f"SELECT COUNT(*) as cnt FROM {table_name}").fetchone()[0]
        
        # Get sample data
        sample_df = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 10").df()
        
        # Get column statistics
        column_stats = {}
        for col in columns_df['column_name']:
            try:
                # Get distinct count and null count
                stats_query = f"""
                    SELECT 
                        COUNT(DISTINCT "{col}") as distinct_count,
                        COUNT(*) - COUNT("{col}") as null_count,
                        COUNT("{col}") as non_null_count
                    FROM {table_name}
                """
                stats = self.conn.execute(stats_query).fetchone()
                
                column_stats[col] = {
                    'distinct_count': stats[0],
                    'null_count': stats[1],
                    'non_null_count': stats[2],
                    'null_percentage': (stats[1] / row_count * 100) if row_count > 0 else 0,
                    'cardinality_ratio': stats[0] / row_count if row_count > 0 else 0
                }
            except Exception as e:
                logger.warning(f"Could not get stats for column {col}: {e}")
                column_stats[col] = {}
        
        return {
            'table_name': table_name,
            'columns': columns_df.to_dict('records'),
            'row_count': row_count,
            'sample': sample_df,
            'column_stats': column_stats
        }
    
    def _analyze_column_metadata(self, table_name: str, schema_info: Dict[str, Any]):
        """
        Analyze columns to determine their semantic meaning and purpose
        """
        for col_info in schema_info['columns']:
            col_name = col_info['column_name']
            col_type = col_info['data_type']
            stats = schema_info['column_stats'].get(col_name, {})
            
            # Get sample values for analysis
            sample_values = schema_info['sample'][col_name].dropna().tolist()[:100]
            
            # Determine column purpose
            purpose = self._detect_column_purpose(col_name, col_type, stats, sample_values)
            
            # Store metadata
            key = f"{table_name}.{col_name}"
            self.column_metadata[key] = {
                'purpose': purpose,
                'type': col_type,
                'stats': stats,
                'is_identifier': purpose == 'identifier',
                'is_metric': purpose == 'metric',
                'is_dimension': purpose == 'dimension',
                'is_temporal': purpose == 'temporal'
            }
    
    def _detect_column_purpose(self, col_name: str, col_type: str, 
                               stats: Dict, sample_values: List) -> str:
        """
        Intelligently detect what a column represents
        
        Returns:
            One of: 'identifier', 'metric', 'dimension', 'temporal', 'text', 'unknown'
        """
        col_lower = col_name.lower()
        
        # Temporal detection
        temporal_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'year', 'month', 'day']
        if any(kw in col_lower for kw in temporal_keywords) or 'DATE' in col_type.upper() or 'TIME' in col_type.upper():
            return 'temporal'
        
        # Identifier detection (IDs, keys)
        if col_lower.endswith('_id') or col_lower == 'id' or col_lower.endswith('_key'):
            return 'identifier'
        
        # Check for high cardinality (likely identifier)
        cardinality = stats.get('cardinality_ratio', 0)
        if cardinality > 0.95:  # 95%+ unique values
            return 'identifier'
        
        # Metric detection (numeric values)
        metric_keywords = ['amount', 'price', 'value', 'total', 'revenue', 'cost', 'profit', 
                          'count', 'quantity', 'qty', 'rate', 'percent', 'score', 'rating']
        if any(kw in col_lower for kw in metric_keywords):
            if 'INT' in col_type.upper() or 'FLOAT' in col_type.upper() or 'DOUBLE' in col_type.upper() or 'DECIMAL' in col_type.upper():
                return 'metric'
        
        # Dimension detection (categorical)
        dimension_keywords = ['type', 'category', 'status', 'name', 'city', 'state', 
                             'country', 'region', 'segment', 'class', 'group']
        if any(kw in col_lower for kw in dimension_keywords):
            return 'dimension'
        
        # Check cardinality for dimensions (low cardinality, not too low)
        if 0.01 < cardinality < 0.3:  # 1-30% unique values
            return 'dimension'
        
        # Text fields (high length strings)
        if 'VARCHAR' in col_type.upper() or 'TEXT' in col_type.upper():
            if sample_values:
                avg_length = sum(len(str(v)) for v in sample_values) / len(sample_values)
                if avg_length > 50:  # Long text
                    return 'text'
                elif cardinality < 0.3:  # Short text with low cardinality
                    return 'dimension'
        
        return 'unknown'
    
    def detect_relationships(self) -> List[Dict[str, Any]]:
        """
        Automatically detect potential foreign key relationships between tables
        
        Returns:
            List of detected relationships with confidence scores
        """
        relationships = []
        
        if len(self.tables) < 2:
            logger.info("Need at least 2 tables to detect relationships")
            return relationships
        
        table_names = list(self.tables.keys())
        
        # Compare each pair of tables
        for i, table1 in enumerate(table_names):
            for table2 in table_names[i+1:]:
                # Find potential FK relationships
                potential_rels = self._find_column_matches(table1, table2)
                relationships.extend(potential_rels)
        
        # Store and return
        self.relationships = relationships
        logger.info(f"Detected {len(relationships)} potential relationships")
        
        return relationships
    
    def _find_column_matches(self, table1: str, table2: str) -> List[Dict[str, Any]]:
        """
        Find matching columns between two tables that could be foreign keys
        """
        matches = []
        
        table1_cols = {col['column_name']: col for col in self.tables[table1]['columns']}
        table2_cols = {col['column_name']: col for col in self.tables[table2]['columns']}
        
        # Strategy 1: Exact column name matches
        common_cols = set(table1_cols.keys()) & set(table2_cols.keys())
        
        for col in common_cols:
            # Check if it's a potential FK (identifier type)
            key1 = f"{table1}.{col}"
            key2 = f"{table2}.{col}"
            
            meta1 = self.column_metadata.get(key1, {})
            meta2 = self.column_metadata.get(key2, {})
            
            # Both should be identifiers or one is identifier and other is dimension
            if meta1.get('is_identifier') or meta2.get('is_identifier'):
                confidence = self._calculate_relationship_confidence(table1, table2, col, col)
                
                if confidence > 0.5:  # Threshold for acceptance
                    matches.append({
                        'from_table': table1,
                        'from_column': col,
                        'to_table': table2,
                        'to_column': col,
                        'confidence': confidence,
                        'type': 'exact_match'
                    })
        
        # Strategy 2: Pattern-based matches (e.g., user_id in orders → id in users)
        for col1 in table1_cols:
            for col2 in table2_cols:
                if col1 == col2:
                    continue  # Already covered in strategy 1
                
                # Check patterns like: orders.customer_id → customers.id
                if self._is_fk_pattern(col1, col2, table1, table2):
                    confidence = self._calculate_relationship_confidence(table1, table2, col1, col2)
                    
                    if confidence > 0.6:  # Higher threshold for pattern matches
                        matches.append({
                            'from_table': table1,
                            'from_column': col1,
                            'to_table': table2,
                            'to_column': col2,
                            'confidence': confidence,
                            'type': 'pattern_match'
                        })
        
        return matches
    
    def _is_fk_pattern(self, col1: str, col2: str, table1: str, table2: str) -> bool:
        """
        Check if two column names follow FK naming patterns
        """
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        # Pattern: customer_id → id (where table2 is 'customers')
        if col1_lower.endswith('_id') and col2_lower == 'id':
            prefix = col1_lower[:-3]  # Remove '_id'
            # Check if prefix matches table2 name (singular or plural)
            if prefix in table2.lower() or table2.lower() in prefix:
                return True
        
        # Pattern: id → customer_id (reverse)
        if col1_lower == 'id' and col2_lower.endswith('_id'):
            prefix = col2_lower[:-3]
            if prefix in table1.lower() or table1.lower() in prefix:
                return True
        
        return False
    
    def _calculate_relationship_confidence(self, table1: str, table2: str, 
                                          col1: str, col2: str) -> float:
        """
        Calculate confidence score (0-1) for a potential relationship
        """
        confidence = 0.5  # Base score
        
        try:
            # Check data type compatibility
            type1 = next((c['data_type'] for c in self.tables[table1]['columns'] if c['column_name'] == col1), None)
            type2 = next((c['data_type'] for c in self.tables[table2]['columns'] if c['column_name'] == col2), None)
            
            if type1 == type2:
                confidence += 0.2
            
            # Check value overlap
            query = f"""
                SELECT 
                    COUNT(DISTINCT t1."{col1}") as count1,
                    COUNT(DISTINCT t2."{col2}") as count2,
                    COUNT(DISTINCT CASE WHEN t1."{col1}" = t2."{col2}" THEN t1."{col1}" END) as overlap
                FROM {table1} t1
                CROSS JOIN {table2} t2
                LIMIT 1000
            """
            
            result = self.conn.execute(query).fetchone()
            if result and result[0] > 0:
                overlap_ratio = result[2] / result[0]
                confidence += overlap_ratio * 0.3
            
        except Exception as e:
            logger.debug(f"Could not calculate confidence for {table1}.{col1} → {table2}.{col2}: {e}")
        
        return min(confidence, 1.0)
    
    def get_table_list(self) -> List[str]:
        """Get list of loaded tables"""
        return list(self.tables.keys())
    
    def get_schema_summary(self) -> str:
        """
        Generate a human-readable schema summary
        """
        summary = []
        
        for table_name, info in self.tables.items():
            summary.append(f"\n## Table: {table_name}")
            summary.append(f"Rows: {info['row_count']:,}")
            summary.append(f"Columns: {len(info['columns'])}\n")
            
            # Group columns by purpose
            identifiers = []
            metrics = []
            dimensions = []
            temporal = []
            others = []
            
            for col in info['columns']:
                col_name = col['column_name']
                key = f"{table_name}.{col_name}"
                meta = self.column_metadata.get(key, {})
                purpose = meta.get('purpose', 'unknown')
                
                col_desc = f"  - {col_name} ({col['data_type']})"
                
                if purpose == 'identifier':
                    identifiers.append(col_desc)
                elif purpose == 'metric':
                    metrics.append(col_desc)
                elif purpose == 'dimension':
                    dimensions.append(col_desc)
                elif purpose == 'temporal':
                    temporal.append(col_desc)
                else:
                    others.append(col_desc)
            
            if identifiers:
                summary.append("### Identifiers:")
                summary.extend(identifiers)
            if temporal:
                summary.append("\n### Time Fields:")
                summary.extend(temporal)
            if metrics:
                summary.append("\n### Metrics:")
                summary.extend(metrics)
            if dimensions:
                summary.append("\n### Dimensions:")
                summary.extend(dimensions)
            if others:
                summary.append("\n### Other Fields:")
                summary.extend(others)
        
        if self.relationships:
            summary.append("\n\n## Detected Relationships:")
            for rel in self.relationships:
                summary.append(
                    f"  - {rel['from_table']}.{rel['from_column']} → "
                    f"{rel['to_table']}.{rel['to_column']} "
                    f"(confidence: {rel['confidence']:.0%})"
                )
        
        return "\n".join(summary)
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        """
        try:
            return self.conn.execute(sql).df()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("CSV Analyzer closed")


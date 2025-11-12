"""
SQL Tool - Safe read-only SQL execution on DuckDB
Includes whitelist validation and auto-repair
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import duckdb
import pandas as pd
import sqlparse
from sqlparse.sql import Token, Identifier, Function
from sqlparse.tokens import Keyword, DML

logger = logging.getLogger(__name__)


class SQLValidationError(Exception):
    """Raised when SQL validation fails"""
    pass


class SQLTool:
    """Tool for executing read-only SQL queries on DuckDB"""
    
    # Whitelist of allowed views
    ALLOWED_TABLES = {
        "order_facts",
        "order_item_facts",
        "product_dim",
        "customer_dim",
        "seller_dim",
        "order_summary",
    }
    
    # Forbidden keywords/patterns
    FORBIDDEN_KEYWORDS = {
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
        "TRUNCATE", "REPLACE", "MERGE", "PRAGMA", "ATTACH",
        "DETACH", "IMPORT", "EXPORT", "COPY", "LOAD"
    }
    
    def __init__(self, db_path: Path, read_only: bool = True, max_rows: int = 1000):
        """
        Initialize SQL tool
        
        Args:
            db_path: Path to DuckDB database
            read_only: Open in read-only mode (default: True)
            max_rows: Maximum rows to return (default: 1000)
        """
        self.db_path = db_path
        self.read_only = read_only
        self.max_rows = max_rows
        self.conn = None
        
        # Initialize allowed_tables as instance variable (can be overridden for CSV mode)
        self.allowed_tables = self.ALLOWED_TABLES.copy()
        
        # Only connect if db_path is provided (for Olist mode)
        # In CSV mode, db_path will be None and connection will be set externally
        if db_path is not None:
            if not db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")
            self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = duckdb.connect(
                str(self.db_path),
                read_only=self.read_only
            )
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query for safety
        
        Args:
            sql: SQL query string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Remove comments and normalize whitespace
        sql_clean = self._clean_sql(sql)
        
        # Check for statement chaining (semicolons)
        if sql_clean.count(';') > 1:
            return False, "Multiple statements not allowed (remove extra semicolons)"
        
        # Parse SQL
        try:
            parsed = sqlparse.parse(sql_clean)
            if not parsed:
                return False, "Invalid SQL syntax"
            
            statement = parsed[0]
            
            # Check if it's a SELECT statement
            if not self._is_select_statement(statement):
                return False, "Only SELECT statements allowed"
            
            # Check for forbidden keywords
            sql_upper = sql_clean.upper()
            for keyword in self.FORBIDDEN_KEYWORDS:
                if re.search(r'\b' + keyword + r'\b', sql_upper):
                    return False, f"Forbidden keyword: {keyword}"
            
            # Extract and validate table names
            tables = self._extract_tables(statement)
            invalid_tables = tables - self.allowed_tables
            
            if invalid_tables:
                return False, f"Invalid tables: {', '.join(invalid_tables)}. Allowed: {', '.join(sorted(self.allowed_tables))}"
            
            return True, None
            
        except Exception as e:
            return False, f"SQL parsing error: {e}"
    
    def _clean_sql(self, sql: str) -> str:
        """Clean and normalize SQL"""
        # Remove comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        # Normalize whitespace
        sql = ' '.join(sql.split())
        
        return sql.strip()
    
    def _is_select_statement(self, statement) -> bool:
        """Check if statement is a SELECT"""
        for token in statement.tokens:
            if token.ttype is DML and token.value.upper() == 'SELECT':
                return True
        return False
    
    def _extract_tables(self, statement) -> set:
        """Extract table names from parsed SQL statement using regex as fallback"""
        tables = set()
        
        # Convert statement to string for regex parsing (more reliable for our use case)
        sql_str = str(statement).upper()
        
        # Remove common SQL functions and keywords that might confuse parsing
        # Use regex to find table names after FROM and JOIN
        import re
        
        # Pattern 1: FROM table_name
        from_pattern = r'\bFROM\s+(\w+)'
        from_matches = re.findall(from_pattern, sql_str)
        for match in from_matches:
            tables.add(match.lower())
        
        # Pattern 2: JOIN table_name
        join_pattern = r'\bJOIN\s+(\w+)'
        join_matches = re.findall(join_pattern, sql_str)
        for match in join_matches:
            tables.add(match.lower())
        
        # Pattern 3: FROM table_name alias (with alias)
        from_alias_pattern = r'\bFROM\s+(\w+)\s+(?:AS\s+)?(\w+)'
        from_alias_matches = re.findall(from_alias_pattern, sql_str)
        for match in from_alias_matches:
            tables.add(match[0].lower())
        
        # Pattern 4: JOIN table_name alias (with alias)
        join_alias_pattern = r'\bJOIN\s+(\w+)\s+(?:AS\s+)?(\w+)'
        join_alias_matches = re.findall(join_alias_pattern, sql_str)
        for match in join_alias_matches:
            tables.add(match[0].lower())
        
        # Remove SQL keywords that might have been captured
        sql_keywords = {
            'select', 'where', 'group', 'order', 'having', 'limit', 
            'by', 'on', 'and', 'or', 'as', 'in', 'is', 'not', 'null',
            'distinct', 'count', 'sum', 'avg', 'min', 'max', 'case', 'when', 'then', 'else', 'end'
        }
        tables = {t for t in tables if t not in sql_keywords}
        
        return tables
    
    def execute(self, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results
        
        Args:
            sql: SQL query string
            params: Optional query parameters
            
        Returns:
            pandas DataFrame with results
            
        Raises:
            SQLValidationError: If SQL fails validation
            Exception: If execution fails
        """
        # Validate SQL
        is_valid, error_msg = self.validate_sql(sql)
        if not is_valid:
            logger.warning(f"SQL validation failed: {error_msg}")
            raise SQLValidationError(error_msg)
        
        # Add LIMIT if not present
        sql_with_limit = self._ensure_limit(sql)
        
        # Execute query
        try:
            logger.info(f"Executing SQL: {sql_with_limit[:200]}...")
            
            if params:
                result = self.conn.execute(sql_with_limit, params).fetchdf()
            else:
                result = self.conn.execute(sql_with_limit).fetchdf()
            
            logger.info(f"Query returned {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            raise
    
    def _ensure_limit(self, sql: str) -> str:
        """Add LIMIT clause if not present"""
        sql_upper = sql.upper()
        
        # Check if LIMIT already exists
        if 'LIMIT' in sql_upper:
            return sql
        
        # Add LIMIT
        sql_clean = sql.rstrip(';').strip()
        return f"{sql_clean} LIMIT {self.max_rows}"
    
    def execute_safe(self, sql: str, params: Optional[Dict] = None) -> Dict:
        """
        Execute SQL with error handling and return structured result
        
        Args:
            sql: SQL query string
            params: Optional query parameters
            
        Returns:
            Dictionary with:
                - success (bool): Whether execution succeeded
                - data (pd.DataFrame): Results if successful
                - error (str): Error message if failed
                - sql (str): Executed SQL
        """
        try:
            data = self.execute(sql, params)
            return {
                "success": True,
                "data": data,
                "error": None,
                "sql": sql,
                "row_count": len(data),
            }
        except SQLValidationError as e:
            return {
                "success": False,
                "data": None,
                "error": f"Validation error: {e}",
                "sql": sql,
                "row_count": 0,
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Execution error: {e}",
                "sql": sql,
                "row_count": 0,
            }
    
    def get_schema_info(self) -> Dict[str, List[Dict]]:
        """
        Get schema information for all allowed tables
        
        Returns:
            Dictionary mapping table names to list of column info
        """
        schema_info = {}
        
        for table in self.allowed_tables:
            try:
                columns_query = f"""
                    SELECT 
                        column_name,
                        data_type
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position
                """
                columns = self.conn.execute(columns_query).fetchdf()
                schema_info[table] = columns.to_dict('records')
            except Exception as e:
                logger.error(f"Failed to get schema for {table}: {e}")
                schema_info[table] = []
        
        return schema_info
    
    def repair_sql(self, sql: str, error_message: str) -> Optional[str]:
        """
        Attempt to repair SQL based on error message
        
        Args:
            sql: Original SQL query
            error_message: Error message from execution
            
        Returns:
            Repaired SQL or None if can't repair
        """
        # Common repairs
        sql_repaired = sql
        
        # Fix table name case sensitivity
        for table in self.allowed_tables:
            pattern = re.compile(r'\b' + table + r'\b', re.IGNORECASE)
            sql_repaired = pattern.sub(table, sql_repaired)
        
        # Fix common column name typos
        if "column" in error_message.lower() and "not" in error_message.lower():
            # Try to extract the problematic column name
            # This is a simplified repair - a production system would be more sophisticated
            pass
        
        # Verify the repair is different and valid
        if sql_repaired != sql:
            is_valid, _ = self.validate_sql(sql_repaired)
            if is_valid:
                return sql_repaired
        
        return None
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


"""
Tests for SQL generation, validation, and security
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from app.tools.sql_tool import SQLTool, SQLValidationError


class TestSQLValidation:
    """Test SQL validation and security"""
    
    def test_select_allowed(self):
        """Test that SELECT queries are allowed"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        valid_queries = [
            "SELECT * FROM order_summary",
            "SELECT order_id, customer_id FROM order_facts",
            "SELECT COUNT(*) FROM order_item_facts WHERE category_name_en = 'Electronics'",
        ]
        
        for sql in valid_queries:
            is_valid, error = sql_tool.validate_sql(sql)
            assert is_valid, f"Query should be valid: {sql}. Error: {error}"
    
    def test_ddl_denied(self):
        """Test that DDL statements are denied"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        invalid_queries = [
            "CREATE TABLE test (id INT)",
            "DROP TABLE order_summary",
            "ALTER TABLE order_facts ADD COLUMN test VARCHAR",
            "TRUNCATE TABLE order_summary",
        ]
        
        for sql in invalid_queries:
            is_valid, error = sql_tool.validate_sql(sql)
            assert not is_valid, f"DDL should be denied: {sql}"
            assert error is not None
    
    def test_dml_denied(self):
        """Test that DML statements are denied"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        invalid_queries = [
            "INSERT INTO order_summary VALUES (1, 2, 3)",
            "UPDATE order_summary SET payment_value = 100",
            "DELETE FROM order_summary WHERE order_id = 1",
            "MERGE INTO order_summary USING other_table",
        ]
        
        for sql in invalid_queries:
            is_valid, error = sql_tool.validate_sql(sql)
            assert not is_valid, f"DML should be denied: {sql}"
    
    def test_system_commands_denied(self):
        """Test that system commands are denied"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        invalid_queries = [
            "PRAGMA table_info(order_summary)",
            "ATTACH DATABASE 'other.db' AS other",
            "IMPORT DATABASE 'data.csv'",
            "EXPORT DATABASE TO 'output.csv'",
            "COPY order_summary TO 'file.csv'",
        ]
        
        for sql in invalid_queries:
            is_valid, error = sql_tool.validate_sql(sql)
            assert not is_valid, f"System command should be denied: {sql}"
    
    def test_table_whitelist(self):
        """Test that only whitelisted tables are allowed"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        # Valid tables
        valid_query = "SELECT * FROM order_summary"
        is_valid, error = sql_tool.validate_sql(valid_query)
        assert is_valid, f"Whitelisted table should be allowed: {valid_query}"
        
        # Invalid tables
        invalid_queries = [
            "SELECT * FROM users",
            "SELECT * FROM raw_orders",
            "SELECT * FROM sensitive_data",
        ]
        
        for sql in invalid_queries:
            is_valid, error = sql_tool.validate_sql(sql)
            assert not is_valid, f"Non-whitelisted table should be denied: {sql}"
            assert "Invalid tables" in error
    
    def test_multiple_statements_denied(self):
        """Test that statement chaining is denied"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        invalid_queries = [
            "SELECT * FROM order_summary; DROP TABLE order_summary;",
            "SELECT 1; SELECT 2; SELECT 3;",
        ]
        
        for sql in invalid_queries:
            is_valid, error = sql_tool.validate_sql(sql)
            assert not is_valid, f"Multiple statements should be denied: {sql}"
            assert "Multiple statements" in error
    
    def test_sql_cleaning(self):
        """Test SQL cleaning and normalization"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        # SQL with comments
        sql_with_comments = """
        -- This is a comment
        SELECT * FROM order_summary
        /* Multi-line
           comment */
        WHERE order_id = 1
        """
        
        cleaned = sql_tool._clean_sql(sql_with_comments)
        assert "--" not in cleaned
        assert "/*" not in cleaned
        assert "SELECT" in cleaned
    
    def test_limit_injection(self):
        """Test that LIMIT is properly added"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        sql_without_limit = "SELECT * FROM order_summary"
        sql_with_limit = sql_tool._ensure_limit(sql_without_limit)
        
        assert "LIMIT" in sql_with_limit.upper()
        assert str(sql_tool.max_rows) in sql_with_limit
        
        # Should not add LIMIT if already present
        sql_already_limited = "SELECT * FROM order_summary LIMIT 10"
        sql_result = sql_tool._ensure_limit(sql_already_limited)
        assert sql_result.upper().count("LIMIT") == 1


class TestSQLExamples:
    """Test realistic SQL examples"""
    
    def test_top_categories_query(self):
        """Test top categories by GMV query"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        sql = """
        SELECT 
            category_name_en as category,
            SUM(total_value) as gmv,
            COUNT(DISTINCT order_id) as orders
        FROM order_item_facts
        GROUP BY category_name_en
        ORDER BY gmv DESC
        LIMIT 5
        """
        
        is_valid, error = sql_tool.validate_sql(sql)
        assert is_valid, f"Valid query rejected: {error}"
    
    def test_aov_by_quarter(self):
        """Test AOV by quarter query"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        sql = """
        SELECT 
            purchase_year,
            purchase_quarter,
            ROUND(AVG(order_gmv), 2) as aov,
            COUNT(DISTINCT order_id) as order_count
        FROM order_summary
        GROUP BY purchase_year, purchase_quarter
        ORDER BY purchase_year, purchase_quarter
        """
        
        is_valid, error = sql_tool.validate_sql(sql)
        assert is_valid, f"Valid query rejected: {error}"
    
    def test_on_time_delivery_rate(self):
        """Test on-time delivery rate query"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        sql = """
        SELECT 
            seller_id,
            COUNT(*) as total_orders,
            COUNT(CASE WHEN is_on_time = true THEN 1 END) as on_time_orders,
            ROUND(COUNT(CASE WHEN is_on_time = true THEN 1 END) * 100.0 / COUNT(*), 2) as on_time_rate
        FROM order_summary os
        JOIN order_item_facts oif ON os.order_id = oif.order_id
        WHERE delivered_ts IS NOT NULL
        GROUP BY seller_id
        HAVING COUNT(*) >= 10
        ORDER BY on_time_rate ASC
        LIMIT 10
        """
        
        is_valid, error = sql_tool.validate_sql(sql)
        assert is_valid, f"Valid query rejected: {error}"
    
    def test_repeat_rate_calculation(self):
        """Test repeat purchase rate query"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        sql = """
        WITH customer_order_counts AS (
            SELECT 
                customer_id,
                COUNT(DISTINCT order_id) as order_count
            FROM order_summary
            GROUP BY customer_id
        )
        SELECT 
            COUNT(CASE WHEN order_count > 1 THEN 1 END) as repeat_customers,
            COUNT(*) as total_customers,
            ROUND(COUNT(CASE WHEN order_count > 1 THEN 1 END) * 100.0 / COUNT(*), 2) as repeat_rate
        FROM customer_order_counts
        """
        
        is_valid, error = sql_tool.validate_sql(sql)
        assert is_valid, f"Valid query rejected: {error}"


class TestEdgeCases:
    """Test edge cases and potential vulnerabilities"""
    
    def test_sql_injection_attempts(self):
        """Test that SQL injection attempts are blocked"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        injection_attempts = [
            "SELECT * FROM order_summary WHERE order_id = 1; DROP TABLE order_summary; --",
            "SELECT * FROM order_summary WHERE customer_id = '1' OR '1'='1'",  # Actually valid SELECT
            "SELECT * FROM order_summary; EXEC xp_cmdshell('calc.exe')",
        ]
        
        for sql in injection_attempts:
            is_valid, error = sql_tool.validate_sql(sql)
            # The second one is actually a valid SELECT, others should fail
            if "DROP" in sql.upper() or "EXEC" in sql.upper():
                assert not is_valid, f"Injection attempt should be blocked: {sql}"
    
    def test_case_insensitivity(self):
        """Test that validation is case-insensitive for keywords"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        queries = [
            "select * from order_summary",
            "SELECT * FROM order_summary",
            "SeLeCt * FrOm order_summary",
        ]
        
        for sql in queries:
            is_valid, error = sql_tool.validate_sql(sql)
            assert is_valid, f"Case should not matter for valid queries: {sql}"
    
    def test_empty_query(self):
        """Test handling of empty queries"""
        sql_tool = SQLTool(Path("dummy.db"), read_only=True)
        
        is_valid, error = sql_tool.validate_sql("")
        assert not is_valid
        
        is_valid, error = sql_tool.validate_sql("   ")
        assert not is_valid


def test_glossary_metrics():
    """Test that metrics are properly defined"""
    from app.tools.glossary_tool import GlossaryTool
    
    # Create metrics.yaml if not exists
    metrics_path = Path(__file__).parent.parent / "app" / "semantic" / "metrics.yaml"
    
    if not metrics_path.exists():
        pytest.skip("metrics.yaml not found")
    
    glossary = GlossaryTool(metrics_path)
    
    # Test that key metrics exist
    key_metrics = ["gmv", "aov", "repeat_rate", "on_time_delivery_rate"]
    
    for metric_name in key_metrics:
        metric = glossary.get_metric(metric_name)
        assert metric is not None, f"Metric {metric_name} should be defined"
        assert "description" in metric, f"Metric {metric_name} should have description"
        assert "formula" in metric, f"Metric {metric_name} should have formula"


def test_viz_tool_chart_detection():
    """Test chart type auto-detection"""
    from app.tools.viz_tool import VizTool
    import pandas as pd
    
    viz_tool = VizTool()
    
    # Test bar chart detection (categorical data)
    data_bar = pd.DataFrame({
        "category": ["A", "B", "C"],
        "value": [100, 200, 150]
    })
    
    chart_type = viz_tool._detect_chart_type(data_bar, "category", "value")
    assert chart_type in ["bar", "pie"]  # Should suggest bar or pie
    
    # Test line chart detection (time series)
    data_line = pd.DataFrame({
        "month": [1, 2, 3, 4, 5],
        "revenue": [1000, 1200, 1100, 1300, 1400]
    })
    
    chart_type = viz_tool._detect_chart_type(data_line, "month", "revenue")
    # Month might be detected as time series or numeric
    assert chart_type in ["line", "scatter", "bar"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])



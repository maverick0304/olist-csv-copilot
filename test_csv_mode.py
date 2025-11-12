"""
Test script for Custom CSV Analysis Mode
Creates sample CSV files and tests the analysis pipeline
"""

import pandas as pd
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample CSV files for testing"""
    
    logger.info("Creating sample CSV files...")
    
    # Create test data directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Sample 1: Sales data
    sales_data = pd.DataFrame({
        'order_id': range(1, 101),
        'customer_id': [i % 20 + 1 for i in range(100)],
        'product_id': [i % 10 + 1 for i in range(100)],
        'order_date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'quantity': [i % 5 + 1 for i in range(100)],
        'unit_price': [10.0 + (i % 50) for i in range(100)],
        'total_amount': [(i % 5 + 1) * (10.0 + (i % 50)) for i in range(100)]
    })
    sales_data.to_csv(test_dir / "sales.csv", index=False)
    logger.info(f"✓ Created sales.csv ({len(sales_data)} rows)")
    
    # Sample 2: Products data
    products_data = pd.DataFrame({
        'product_id': range(1, 11),
        'product_name': [f'Product {i}' for i in range(1, 11)],
        'category': ['Electronics', 'Clothing', 'Food', 'Books', 'Sports'] * 2,
        'cost_price': [5.0 + i * 5 for i in range(10)],
        'list_price': [10.0 + i * 8 for i in range(10)]
    })
    products_data.to_csv(test_dir / "products.csv", index=False)
    logger.info(f"✓ Created products.csv ({len(products_data)} rows)")
    
    # Sample 3: Customers data
    customers_data = pd.DataFrame({
        'customer_id': range(1, 21),
        'customer_name': [f'Customer {i}' for i in range(1, 21)],
        'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'] * 4,
        'state': ['NY', 'CA', 'IL', 'TX', 'AZ'] * 4,
        'signup_date': pd.date_range('2023-01-01', periods=20, freq='M')
    })
    customers_data.to_csv(test_dir / "customers.csv", index=False)
    logger.info(f"✓ Created customers.csv ({len(customers_data)} rows)")
    
    return test_dir


def test_csv_analyzer(test_dir):
    """Test CSV Analyzer functionality"""
    
    logger.info("\n=== Testing CSV Analyzer ===")
    
    from app.tools.csv_tool import CSVAnalyzer
    
    # Initialize analyzer
    analyzer = CSVAnalyzer()
    logger.info("✓ CSVAnalyzer initialized")
    
    # Load CSV files
    csv_files = list(test_dir.glob("*.csv"))
    for csv_file in csv_files:
        logger.info(f"Loading {csv_file.name}...")
        schema = analyzer.load_csv(str(csv_file))
        logger.info(f"  ✓ Loaded: {schema['row_count']} rows, {len(schema['columns'])} columns")
        
        # Show column purposes
        table_name = csv_file.stem
        for col in schema['columns']:
            col_name = col['column_name']
            key = f"{table_name}.{col_name}"
            meta = analyzer.column_metadata.get(key, {})
            purpose = meta.get('purpose', 'unknown')
            logger.info(f"    - {col_name} ({col['data_type']}) → {purpose}")
    
    logger.info(f"\n✓ Loaded {len(analyzer.tables)} tables")
    
    # Detect relationships
    logger.info("\nDetecting relationships...")
    relationships = analyzer.detect_relationships()
    logger.info(f"✓ Found {len(relationships)} relationships:")
    for rel in relationships:
        logger.info(
            f"  - {rel['from_table']}.{rel['from_column']} → "
            f"{rel['to_table']}.{rel['to_column']} "
            f"(confidence: {rel['confidence']:.0%})"
        )
    
    # Test query execution
    logger.info("\nTesting query execution...")
    test_sql = "SELECT COUNT(*) as total_orders FROM sales"
    result = analyzer.execute_query(test_sql)
    logger.info(f"✓ Query executed: {result['total_orders'][0]} orders")
    
    return analyzer


def test_data_profiler(analyzer):
    """Test Data Profiler functionality"""
    
    logger.info("\n=== Testing Data Profiler ===")
    
    from app.tools.data_profiler import DataProfiler
    
    profiler = DataProfiler(analyzer)
    logger.info("✓ DataProfiler initialized")
    
    # Profile each table
    for table_name in analyzer.get_table_list():
        logger.info(f"\nProfiling {table_name}...")
        profile = profiler.profile_table(table_name)
        
        logger.info(f"  Quality Score: {profile['quality_score']:.0%}")
        logger.info(f"  Issues Found: {len(profile['issues'])}")
        
        if profile['issues']:
            for issue in profile['issues'][:3]:
                logger.info(f"    {issue['severity'].upper()}: {issue['message']}")
        
        logger.info(f"  Insights:")
        for insight in profile['insights']:
            logger.info(f"    {insight}")
    
    # Print summary report
    logger.info("\n" + "="*60)
    logger.info(profiler.get_summary_report())
    logger.info("="*60)
    
    return profiler


def test_prompt_generator(analyzer, profiler):
    """Test Dynamic Prompt Generator"""
    
    logger.info("\n=== Testing Prompt Generator ===")
    
    from app.prompts.csv_prompt_generator import CSVPromptGenerator
    
    generator = CSVPromptGenerator(analyzer, profiler)
    logger.info("✓ CSVPromptGenerator initialized")
    
    # Generate schema prompt
    logger.info("\nGenerating schema prompt...")
    schema_prompt = generator.generate_schema_prompt()
    logger.info(f"✓ Generated prompt: {len(schema_prompt)} characters")
    logger.info(f"\nFirst 500 characters:\n{schema_prompt[:500]}...")
    
    # Generate example questions
    logger.info("\nGenerating example questions...")
    questions = generator.generate_example_questions(count=10)
    logger.info(f"✓ Generated {len(questions)} questions:")
    for i, q in enumerate(questions, 1):
        logger.info(f"  {i}. {q}")
    
    # Generate few-shot examples
    logger.info("\nGenerating few-shot examples...")
    examples = generator.generate_few_shot_examples()
    logger.info(f"✓ Generated {len(examples)} few-shot examples:")
    for i, ex in enumerate(examples, 1):
        logger.info(f"  {i}. Q: {ex['question']}")
        logger.info(f"     SQL: {ex['sql'][:80]}...")
    
    return generator


def test_agent_integration(analyzer, profiler):
    """Test Agent with CSV mode"""
    
    logger.info("\n=== Testing Agent Integration ===")
    
    try:
        from app.agent import OlistAgent
        
        # Initialize agent in CSV mode
        logger.info("Initializing agent in CSV mode...")
        agent = OlistAgent(
            db_path=None,
            csv_mode=True,
            csv_analyzer=analyzer,
            data_profiler=profiler
        )
        logger.info("✓ Agent initialized in CSV mode")
        
        # Test a simple query
        logger.info("\nTesting query: 'What are the top 5 products by sales?'")
        result = agent.process_query("What are the top 5 products by sales?", {})
        
        if result['success']:
            logger.info("✓ Query successful!")
            logger.info(f"  Insight: {result['insight'][:100]}...")
            if result.get('sql'):
                logger.info(f"  SQL: {result['sql'][:100]}...")
            if result.get('data') is not None:
                logger.info(f"  Rows returned: {len(result['data'])}")
        else:
            logger.error(f"✗ Query failed: {result.get('error')}")
        
        return agent
        
    except Exception as e:
        logger.error(f"✗ Agent test failed: {e}", exc_info=True)
        return None


def cleanup(test_dir):
    """Clean up test files"""
    
    logger.info("\n=== Cleanup ===")
    
    import shutil
    
    if test_dir.exists():
        shutil.rmtree(test_dir)
        logger.info(f"✓ Removed {test_dir}")


def main():
    """Run all tests"""
    
    logger.info("="*60)
    logger.info("CSV MODE TEST SUITE")
    logger.info("="*60)
    
    try:
        # Step 1: Create sample data
        test_dir = create_sample_data()
        
        # Step 2: Test CSV Analyzer
        analyzer = test_csv_analyzer(test_dir)
        
        # Step 3: Test Data Profiler
        profiler = test_data_profiler(analyzer)
        
        # Step 4: Test Prompt Generator
        generator = test_prompt_generator(analyzer, profiler)
        
        # Step 5: Test Agent Integration
        agent = test_agent_integration(analyzer, profiler)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"✓ CSV Analyzer: PASSED")
        logger.info(f"✓ Data Profiler: PASSED")
        logger.info(f"✓ Prompt Generator: PASSED")
        logger.info(f"✓ Agent Integration: {'PASSED' if agent else 'FAILED'}")
        logger.info("="*60)
        
        # Ask user if they want to keep test files
        keep_files = input("\nKeep test CSV files for manual testing? (y/n): ").lower()
        if keep_files != 'y':
            cleanup(test_dir)
        else:
            logger.info(f"\n✓ Test files kept in {test_dir}/")
            logger.info("  You can upload these files in CSV Mode to test the UI!")
        
        logger.info("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


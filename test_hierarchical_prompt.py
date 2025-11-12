"""
Test the new hierarchical prompt structure
"""

import sys
from pathlib import Path

def test_prompt_loading():
    """Test that all prompt layers load correctly"""
    
    print("=" * 70)
    print("🧪 Testing Hierarchical Prompt Structure")
    print("=" * 70)
    
    # Check all prompt files exist
    prompts = {
        "Context Wrapper": Path("app/prompts/context_wrapper.txt"),
        "Complete Schema": Path("app/prompts/complete_schema.txt"),
        "SQL Rules & Examples": Path("app/prompts/sql_generation_v2.txt")
    }
    
    total_chars = 0
    total_words = 0
    
    for prompt_name, file_path in prompts.items():
        if file_path.exists():
            content = file_path.read_text(encoding='utf-8')
            chars = len(content)
            words = len(content.split())
            lines = len(content.split('\n'))
            
            total_chars += chars
            total_words += words
            
            print(f"\n✅ {prompt_name}")
            print(f"   Path: {file_path}")
            print(f"   Lines: {lines}")
            print(f"   Words: {words:,}")
            print(f"   Characters: {chars:,}")
        else:
            print(f"\n❌ {prompt_name}")
            print(f"   Path: {file_path} NOT FOUND!")
    
    print("\n" + "=" * 70)
    print("📊 COMBINED PROMPT STATISTICS")
    print("=" * 70)
    print(f"Total Words: {total_words:,}")
    print(f"Total Characters: {total_chars:,}")
    print(f"Estimated Tokens: ~{total_chars // 4:,} (rough estimate)")
    
    # Test loading with agent
    print("\n" + "=" * 70)
    print("🤖 Testing Agent Initialization")
    print("=" * 70)
    
    try:
        from app.agent import OlistAgent
        
        db_path = Path("data/duckdb/olist.duckdb")
        if not db_path.exists():
            print("⚠️  Database not found, skipping agent test")
            print(f"   Expected at: {db_path.absolute()}")
            return
        
        print("Initializing agent...")
        agent = OlistAgent(db_path=db_path)
        
        print("✅ Agent initialized successfully!")
        print(f"   Using LLM: {agent.llm.__class__.__name__}")
        
        # Test a query
        print("\n" + "=" * 70)
        print("🧪 Testing Query: 'Revenue trend by month for Electronics'")
        print("=" * 70)
        
        result = agent.process_query("Revenue trend by month for Electronics category")
        
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ Query executed successfully!")
            print(f"\n📝 Generated SQL:")
            print("-" * 70)
            print(result.get('sql', 'N/A'))
            print("-" * 70)
            
            if result.get('data') is not None:
                import pandas as pd
                df = result['data']
                if isinstance(df, pd.DataFrame):
                    print(f"\n📊 Results: {len(df)} rows")
                    print(df.head())
                    
                    # Check if it has monthly breakdown
                    if 'purchase_month' in df.columns or 'month' in df.columns:
                        print("\n✅ SUCCESS! Query has monthly breakdown (trend)")
                    else:
                        print("\n⚠️  Warning: Query missing monthly breakdown")
        
    except Exception as e:
        print(f"❌ Error testing agent: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_prompt_loading()

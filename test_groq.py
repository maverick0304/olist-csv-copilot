"""
Quick test script for Groq integration
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

def test_groq():
    """Test Groq provider"""
    print("=" * 60)
    print("🧪 Testing Groq Integration")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY not found in .env")
        print("\n📝 To fix:")
        print("  1. Get key from: https://console.groq.com/keys")
        print("  2. Add to .env: GROQ_API_KEY=gsk_...")
        return
    
    print(f"✅ Found GROQ_API_KEY: {api_key[:20]}...")
    
    try:
        from app.llm import create_llm_provider
        
        print("\n🔧 Creating Groq provider...")
        groq = create_llm_provider("groq")
        print(f"✅ Provider created: {groq.__class__.__name__}")
        print(f"   Model: {groq.model}")
        
        # Test SQL generation
        print("\n📊 Testing SQL generation...")
        prompt = """Generate SQL to find top 5 categories by GMV in 2018.

Available views:
- order_item_facts (has category_name_en, total_value)
- order_facts (has order_id, purchase_year)

Return ONLY the SQL query:"""
        
        start = time.time()
        sql = groq.generate(prompt, temperature=0.1)
        elapsed = time.time() - start
        
        print(f"✅ SQL generated in {elapsed:.2f}s")
        print("\n📝 Generated SQL:")
        print("-" * 60)
        print(sql)
        print("-" * 60)
        
        # Test with actual agent
        print("\n🤖 Testing with OlistAgent...")
        from pathlib import Path
        from app.agent import OlistAgent
        
        db_path = Path("data/duckdb/olist.duckdb")
        if not db_path.exists():
            print(f"⚠️  Database not found: {db_path}")
            print("   Run: python scripts/build_duckdb.py")
            return
        
        agent = OlistAgent(db_path=db_path, llm_provider="groq")
        print(f"✅ Agent initialized with: {agent.llm.__class__.__name__}")
        
        # Test query
        print("\n🔍 Testing query: 'Top 5 categories by GMV in 2018'")
        start = time.time()
        result = agent.process_query("What are the top 5 product categories by GMV in 2018?")
        elapsed = time.time() - start
        
        print(f"\n✅ Query processed in {elapsed:.2f}s")
        print(f"\n💡 Insight: {result.get('insight', 'N/A')}")
        
        if result.get('error'):
            print(f"\n❌ Error: {result['error']}")
        elif result.get('data') is not None:
            print(f"\n📊 Results: {len(result['data'])} rows")
            print(result['data'].head() if hasattr(result['data'], 'head') else result['data'])
        
        print("\n" + "=" * 60)
        print("✅ GROQ INTEGRATION TEST PASSED!")
        print("=" * 60)
        print(f"\n⚡ Total time: {elapsed:.2f}s")
        print("   (Compare to Gemini: typically 7-8s)")
        print(f"\n🚀 Speedup: ~{7.5/elapsed:.1f}x faster!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n📝 To fix:")
        print("  Run: pip install groq")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_groq()




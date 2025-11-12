"""
Explore Olist Database Schema and Relationships
Run this to understand how CSV files are connected
"""

import duckdb
from pathlib import Path

def main():
    db_path = Path("data/duckdb/olist.duckdb")
    
    if not db_path.exists():
        print("❌ Database not found. Run: python scripts/build_duckdb.py")
        return
    
    conn = duckdb.connect(str(db_path), read_only=True)
    
    print("=" * 70)
    print("🗂️  OLIST DATABASE SCHEMA EXPLORER")
    print("=" * 70)
    
    # 1. Show all views
    print("\n📊 AVAILABLE VIEWS:\n")
    views = conn.execute("""
        SELECT table_name, COUNT(*) as row_count
        FROM (
            SELECT 'order_summary' as table_name, COUNT(*) as cnt FROM order_summary
            UNION ALL SELECT 'order_facts', COUNT(*) FROM order_facts
            UNION ALL SELECT 'order_item_facts', COUNT(*) FROM order_item_facts
            UNION ALL SELECT 'customer_dim', COUNT(*) FROM customer_dim
            UNION ALL SELECT 'seller_dim', COUNT(*) FROM seller_dim
            UNION ALL SELECT 'product_dim', COUNT(*) FROM product_dim
        ) t
        GROUP BY table_name
    """).fetchall()
    
    for view, count in views:
        print(f"  ✓ {view:20s} {count:>10,} rows")
    
    # 2. Show sample join
    print("\n" + "=" * 70)
    print("🔗 EXAMPLE: How order_item_facts is built")
    print("=" * 70)
    print("""
This view JOINS multiple CSV files:

order_items.csv (raw_order_items)
    ├─ JOIN products.csv (raw_products) ON product_id
    │   └─ JOIN translation.csv ON category_name
    └─ JOIN sellers.csv (raw_sellers) ON seller_id

Result: Enriched item data with category names and seller locations
""")
    
    # 3. Show a real example
    print("\n" + "=" * 70)
    print("📦 SAMPLE: Order #00143d0f86d6fbd9f9b38ab440ac16f5")
    print("=" * 70)
    
    sample_order = conn.execute("""
        SELECT 
            oif.order_id,
            oif.category_name_en as category,
            oif.price,
            oif.seller_state,
            of.purchase_year,
            of.customer_id
        FROM order_item_facts oif
        JOIN order_facts of ON oif.order_id = of.order_id
        LIMIT 1
    """).fetchone()
    
    if sample_order:
        order_id, cat, price, seller_state, year, cust_id = sample_order
        print(f"""
Order ID: {order_id}
├─ From order_item_facts:
│  ├─ Category: {cat}
│  ├─ Price: R$ {price:.2f}
│  └─ Seller State: {seller_state}
├─ From order_facts (joined on order_id):
│  ├─ Purchase Year: {year}
│  └─ Customer ID: {cust_id}

This shows how we JOIN two views to get complete information!
        """)
    
    # 4. Show category breakdown
    print("=" * 70)
    print("📊 TOP 5 CATEGORIES (Demonstrating GROUP BY)")
    print("=" * 70)
    
    top_cats = conn.execute("""
        SELECT 
            category_name_en,
            COUNT(DISTINCT order_id) as orders,
            ROUND(SUM(total_value), 2) as gmv
        FROM order_item_facts
        GROUP BY category_name_en
        ORDER BY gmv DESC
        LIMIT 5
    """).fetchall()
    
    for i, (cat, orders, gmv) in enumerate(top_cats, 1):
        print(f"{i}. {cat:30s} {orders:>6,} orders  R$ {gmv:>12,.2f}")
    
    # 5. Show relationship diagram
    print("\n" + "=" * 70)
    print("🗺️  RELATIONSHIP DIAGRAM")
    print("=" * 70)
    print("""
CSV Files → Raw Tables → Views with Relationships

orders.csv ─────────► raw_orders ────┐
                                      ├─► order_facts ──┐
payments.csv ───────► raw_payments ──┤                  │
                                      │                  │
reviews.csv ────────► raw_reviews ───┘                  │
                                                         ├─► order_summary
order_items.csv ────► raw_order_items ──┐               │   (final view)
                                         ├─► order_item_facts ┘
products.csv ───────► raw_products ─────┤
                                         │
translation.csv ────► raw_translation ──┤
                                         │
sellers.csv ────────► raw_sellers ──────┘

Key Joins:
• order_id links orders → items → payments → reviews
• product_id links items → products
• seller_id links items → sellers
• customer_id links orders → customers
""")
    
    # 6. Show query example
    print("=" * 70)
    print("💡 EXAMPLE QUERY: Top Categories in 2018")
    print("=" * 70)
    print("""
SQL Query:
----------
SELECT 
    oif.category_name_en as category,
    SUM(oif.total_value) as gmv
FROM order_item_facts oif
JOIN order_facts of ON oif.order_id = of.order_id
WHERE of.purchase_year = 2018
GROUP BY oif.category_name_en
ORDER BY gmv DESC
LIMIT 5

This query demonstrates:
1. JOIN between two views (order_item_facts + order_facts)
2. Using order_id as the relationship key
3. Filtering on year from order_facts
4. Aggregating revenue from order_item_facts
5. Grouping by category (dimension analysis)
""")
    
    result = conn.execute("""
        SELECT 
            oif.category_name_en as category,
            ROUND(SUM(oif.total_value), 2) as gmv
        FROM order_item_facts oif
        JOIN order_facts of ON oif.order_id = of.order_id
        WHERE of.purchase_year = 2018
        GROUP BY oif.category_name_en
        ORDER BY gmv DESC
        LIMIT 5
    """).fetchall()
    
    print("\nResult:")
    print("-" * 50)
    for i, (cat, gmv) in enumerate(result, 1):
        print(f"{i}. {cat:30s} R$ {gmv:>12,.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Schema exploration complete!")
    print(f"📖 See SCHEMA_RELATIONSHIPS.md for detailed documentation")
    print("=" * 70)
    
    conn.close()


if __name__ == "__main__":
    main()




"""
Generate schema documentation from DuckDB database
Creates schema.md with table/column descriptions and sample data
"""

import sys
from pathlib import Path
from datetime import datetime
import duckdb

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))


def generate_schema_docs(db_path: Path, output_path: Path):
    """Generate markdown documentation for database schema"""
    
    print("📚 Generating schema documentation...")
    
    conn = duckdb.connect(str(db_path), read_only=True)
    
    # Get all views
    views_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main' 
        AND table_type = 'VIEW'
        ORDER BY table_name
    """
    views = [row[0] for row in conn.execute(views_query).fetchall()]
    
    markdown_lines = [
        "# Olist Database Schema",
        "",
        f"*Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Overview",
        "",
        "This database contains e-commerce data from Olist, organized into analytical views.",
        "",
        f"**Total Views:** {len(views)}",
        "",
        "## Views",
        "",
    ]
    
    for view_name in views:
        print(f"  Documenting {view_name}...")
        
        # Get column info
        columns_query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_name = '{view_name}'
            ORDER BY ordinal_position
        """
        columns = conn.execute(columns_query).fetchall()
        
        # Get row count
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
        except:
            count = "N/A"
        
        # Add view documentation
        markdown_lines.extend([
            f"### `{view_name}`",
            "",
            f"**Rows:** {count:,}" if isinstance(count, int) else f"**Rows:** {count}",
            "",
            "| Column | Type | Nullable | Description |",
            "|--------|------|----------|-------------|",
        ])
        
        for col_name, data_type, nullable in columns:
            nullable_str = "✓" if nullable == "YES" else "✗"
            description = get_column_description(view_name, col_name)
            markdown_lines.append(
                f"| `{col_name}` | {data_type} | {nullable_str} | {description} |"
            )
        
        # Add sample data
        try:
            sample_query = f"SELECT * FROM {view_name} LIMIT 3"
            samples = conn.execute(sample_query).fetchdf()
            
            markdown_lines.extend([
                "",
                "**Sample Data:**",
                "",
                "```",
                samples.to_string(index=False, max_colwidth=50),
                "```",
                "",
            ])
        except:
            markdown_lines.extend(["", "_Sample data unavailable_", ""])
        
        markdown_lines.append("")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    
    conn.close()
    
    print(f"✓ Schema documentation saved: {output_path}")
    print(f"  Total views documented: {len(views)}")


def get_column_description(view_name: str, column_name: str) -> str:
    """Get human-readable description for a column"""
    
    descriptions = {
        # Common columns
        "order_id": "Unique order identifier",
        "customer_id": "Unique customer identifier",
        "product_id": "Unique product identifier",
        "seller_id": "Unique seller identifier",
        "purchase_ts": "Order purchase timestamp",
        "approved_ts": "Order approval timestamp",
        "delivered_ts": "Actual delivery timestamp",
        "estimated_delivery_ts": "Estimated delivery timestamp",
        "payment_value": "Total payment amount",
        "review_score": "Customer review score (1-5)",
        "order_status": "Order status (delivered, shipped, etc.)",
        "price": "Item price",
        "freight_value": "Shipping cost",
        "total_value": "Price + freight",
        "category_name_en": "Product category (English)",
        "product_category_name": "Product category (Portuguese)",
        "is_on_time": "Whether order delivered on time",
        "delivery_days_delta": "Days difference from estimated delivery",
        "purchase_year": "Year of purchase",
        "purchase_quarter": "Quarter of purchase (1-4)",
        "purchase_month": "Month of purchase (1-12)",
        "order_gmv": "Gross merchandise value for order",
        "item_count": "Number of items in order",
        "unique_products": "Number of unique products",
        "unique_sellers": "Number of unique sellers",
        "customer_city": "Customer city",
        "customer_state": "Customer state (BR)",
        "seller_city": "Seller city",
        "seller_state": "Seller state (BR)",
        "payment_type": "Payment method",
        "payment_installments": "Number of payment installments",
    }
    
    return descriptions.get(column_name, "")


def main():
    db_path = Path(__file__).parent.parent / "data" / "duckdb" / "olist.duckdb"
    output_path = Path(__file__).parent.parent / "app" / "semantic" / "schema.md"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("Run: python scripts/build_duckdb.py")
        sys.exit(1)
    
    generate_schema_docs(db_path, output_path)


if __name__ == "__main__":
    main()



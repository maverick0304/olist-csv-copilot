"""
Build DuckDB database from Olist CSV files
Creates optimized views, indexes, and data types
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import duckdb
from tqdm import tqdm

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))


class OlistDBBuilder:
    """Builds DuckDB database from Olist CSV files"""
    
    def __init__(self, raw_dir: Path, output_path: Path):
        self.raw_dir = raw_dir
        self.output_path = output_path
        self.conn = None
        
        # Expected CSV files
        self.expected_files = {
            "orders": "olist_orders_dataset.csv",
            "order_items": "olist_order_items_dataset.csv",
            "products": "olist_products_dataset.csv",
            "customers": "olist_customers_dataset.csv",
            "sellers": "olist_sellers_dataset.csv",
            "payments": "olist_order_payments_dataset.csv",
            "reviews": "olist_order_reviews_dataset.csv",
            "geolocation": "olist_geolocation_dataset.csv",
            "category_translation": "product_category_name_translation.csv",
        }
    
    def validate_files(self):
        """Check that all required CSV files exist"""
        print("🔍 Validating CSV files...")
        missing = []
        
        for name, filename in self.expected_files.items():
            filepath = self.raw_dir / filename
            if not filepath.exists():
                missing.append(filename)
                print(f"  ❌ Missing: {filename}")
            else:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  ✓ Found: {filename} ({size_mb:.2f} MB)")
        
        if missing:
            print(f"\n❌ Missing {len(missing)} required files!")
            print(f"Download from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/")
            print(f"Place CSV files in: {self.raw_dir}")
            sys.exit(1)
        
        print("✓ All CSV files found!\n")
    
    def connect(self):
        """Create DuckDB connection"""
        print(f"📁 Creating database: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing database
        if self.output_path.exists():
            self.output_path.unlink()
            print("  (Removed existing database)")
        
        self.conn = duckdb.connect(str(self.output_path))
        print("✓ Database connected\n")
    
    def load_raw_tables(self):
        """Load CSV files into raw tables"""
        print("📊 Loading raw tables...")
        
        for name, filename in tqdm(self.expected_files.items(), desc="Loading CSVs"):
            filepath = self.raw_dir / filename
            table_name = f"raw_{name}"
            
            try:
                # Load CSV with auto-detection
                self.conn.execute(f"""
                    CREATE TABLE {table_name} AS 
                    SELECT * FROM read_csv_auto('{filepath}', 
                        header=true, 
                        sample_size=100000,
                        all_varchar=false,
                        dateformat='%Y-%m-%d %H:%M:%S'
                    )
                """)
                
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                print(f"  ✓ {table_name}: {count:,} rows")
                
            except Exception as e:
                print(f"  ❌ Error loading {name}: {e}")
                raise
        
        print("✓ All raw tables loaded\n")
    
    def create_views(self):
        """Create optimized analytical views"""
        print("🔧 Creating analytical views...")
        
        # Order facts view
        print("  Creating order_facts...")
        self.conn.execute("""
            CREATE VIEW order_facts AS
            SELECT 
                o.order_id,
                o.customer_id,
                o.order_status,
                CAST(o.order_purchase_timestamp AS TIMESTAMP) as purchase_ts,
                CAST(o.order_approved_at AS TIMESTAMP) as approved_ts,
                CAST(o.order_delivered_carrier_date AS TIMESTAMP) as carrier_ts,
                CAST(o.order_delivered_customer_date AS TIMESTAMP) as delivered_ts,
                CAST(o.order_estimated_delivery_date AS TIMESTAMP) as estimated_delivery_ts,
                COALESCE(p.payment_value, 0.0) as payment_value,
                p.payment_type,
                p.payment_installments,
                COALESCE(r.review_score, 0) as review_score,
                r.review_comment_message,
                -- Derived date parts
                YEAR(CAST(o.order_purchase_timestamp AS TIMESTAMP)) as purchase_year,
                QUARTER(CAST(o.order_purchase_timestamp AS TIMESTAMP)) as purchase_quarter,
                MONTH(CAST(o.order_purchase_timestamp AS TIMESTAMP)) as purchase_month,
                DAYOFWEEK(CAST(o.order_purchase_timestamp AS TIMESTAMP)) as purchase_dayofweek,
                -- Delivery metrics
                CASE 
                    WHEN o.order_delivered_customer_date IS NOT NULL 
                         AND o.order_estimated_delivery_date IS NOT NULL
                    THEN CAST(o.order_delivered_customer_date AS TIMESTAMP) <= CAST(o.order_estimated_delivery_date AS TIMESTAMP)
                    ELSE NULL
                END as is_on_time,
                CASE 
                    WHEN o.order_delivered_customer_date IS NOT NULL 
                         AND o.order_estimated_delivery_date IS NOT NULL
                    THEN DATEDIFF('day', 
                        CAST(o.order_delivered_customer_date AS TIMESTAMP),
                        CAST(o.order_estimated_delivery_date AS TIMESTAMP))
                    ELSE NULL
                END as delivery_days_delta
            FROM raw_orders o
            LEFT JOIN (
                SELECT 
                    order_id,
                    SUM(payment_value) as payment_value,
                    MAX(payment_type) as payment_type,
                    MAX(payment_installments) as payment_installments
                FROM raw_payments
                GROUP BY order_id
            ) p ON o.order_id = p.order_id
            LEFT JOIN raw_reviews r ON o.order_id = r.order_id
        """)
        
        # Order item facts view
        print("  Creating order_item_facts...")
        self.conn.execute("""
            CREATE VIEW order_item_facts AS
            SELECT 
                oi.order_id,
                oi.order_item_id,
                oi.product_id,
                oi.seller_id,
                oi.price,
                oi.freight_value,
                oi.price + oi.freight_value as total_value,
                p.product_category_name,
                COALESCE(t.product_category_name_english, p.product_category_name) as category_name_en,
                p.product_name_lenght as product_name_length,
                p.product_description_lenght as product_description_length,
                p.product_photos_qty as product_photos_qty,
                p.product_weight_g,
                p.product_length_cm,
                p.product_height_cm,
                p.product_width_cm,
                s.seller_city,
                s.seller_state
            FROM raw_order_items oi
            LEFT JOIN raw_products p ON oi.product_id = p.product_id
            LEFT JOIN raw_category_translation t ON p.product_category_name = t.product_category_name
            LEFT JOIN raw_sellers s ON oi.seller_id = s.seller_id
        """)
        
        # Product dimension
        print("  Creating product_dim...")
        self.conn.execute("""
            CREATE VIEW product_dim AS
            SELECT DISTINCT
                p.product_id,
                p.product_category_name,
                COALESCE(t.product_category_name_english, p.product_category_name) as category_name_en,
                p.product_name_lenght as product_name_length,
                p.product_description_lenght as product_description_length,
                p.product_photos_qty,
                p.product_weight_g,
                p.product_length_cm,
                p.product_height_cm,
                p.product_width_cm,
                COALESCE(p.product_weight_g * p.product_length_cm * p.product_height_cm * p.product_width_cm / 1000000.0, 0) as volume_cubic_cm
            FROM raw_products p
            LEFT JOIN raw_category_translation t ON p.product_category_name = t.product_category_name
        """)
        
        # Customer dimension
        print("  Creating customer_dim...")
        self.conn.execute("""
            CREATE VIEW customer_dim AS
            SELECT 
                customer_id,
                customer_unique_id,
                customer_zip_code_prefix,
                customer_city,
                customer_state
            FROM raw_customers
        """)
        
        # Seller dimension
        print("  Creating seller_dim...")
        self.conn.execute("""
            CREATE VIEW seller_dim AS
            SELECT 
                seller_id,
                seller_zip_code_prefix,
                seller_city,
                seller_state
            FROM raw_sellers
        """)
        
        # Summary view - order level aggregation
        print("  Creating order_summary...")
        self.conn.execute("""
            CREATE VIEW order_summary AS
            SELECT 
                of.*,
                c.customer_city,
                c.customer_state,
                c.customer_zip_code_prefix,
                COALESCE(SUM(oif.total_value), 0) as order_gmv,
                COUNT(oif.order_item_id) as item_count,
                COUNT(DISTINCT oif.product_id) as unique_products,
                COUNT(DISTINCT oif.seller_id) as unique_sellers,
                STRING_AGG(DISTINCT oif.category_name_en, ', ') as categories
            FROM order_facts of
            LEFT JOIN customer_dim c ON of.customer_id = c.customer_id
            LEFT JOIN order_item_facts oif ON of.order_id = oif.order_id
            GROUP BY 
                of.order_id, of.customer_id, of.order_status,
                of.purchase_ts, of.approved_ts, of.carrier_ts, 
                of.delivered_ts, of.estimated_delivery_ts,
                of.payment_value, of.payment_type, of.payment_installments,
                of.review_score, of.review_comment_message,
                of.purchase_year, of.purchase_quarter, of.purchase_month, of.purchase_dayofweek,
                of.is_on_time, of.delivery_days_delta,
                c.customer_city, c.customer_state, c.customer_zip_code_prefix
        """)
        
        print("✓ All views created\n")
    
    def create_indexes(self):
        """Create indexes for common queries (note: views don't support indexes in DuckDB)"""
        print("⚡ Optimizing tables...")
        
        # DuckDB doesn't support indexes on views, but we can optimize the underlying tables
        # Run ANALYZE to collect statistics
        self.conn.execute("ANALYZE")
        
        print("✓ Statistics collected\n")
    
    def validate_data(self):
        """Run validation queries"""
        print("✅ Validating data integrity...")
        
        validations = [
            ("Total orders", "SELECT COUNT(*) FROM order_facts"),
            ("Total order items", "SELECT COUNT(*) FROM order_item_facts"),
            ("Total customers", "SELECT COUNT(DISTINCT customer_id) FROM customer_dim"),
            ("Total sellers", "SELECT COUNT(*) FROM seller_dim"),
            ("Total products", "SELECT COUNT(*) FROM product_dim"),
            ("Date range", "SELECT MIN(purchase_ts), MAX(purchase_ts) FROM order_facts"),
            ("Total GMV", "SELECT ROUND(SUM(order_gmv), 2) FROM order_summary"),
            ("Avg order value", "SELECT ROUND(AVG(order_gmv), 2) FROM order_summary"),
        ]
        
        for name, query in validations:
            try:
                result = self.conn.execute(query).fetchone()
                if len(result) == 1:
                    print(f"  ✓ {name}: {result[0]:,}")
                else:
                    print(f"  ✓ {name}: {result}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
        
        print("✓ Validation complete\n")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print(f"✓ Database saved: {self.output_path}")
            size_mb = self.output_path.stat().st_size / (1024 * 1024)
            print(f"  Size: {size_mb:.2f} MB")
    
    def build(self):
        """Run full build pipeline"""
        start_time = datetime.now()
        print("=" * 60)
        print("🚀 Olist DuckDB Builder")
        print("=" * 60)
        print()
        
        try:
            self.validate_files()
            self.connect()
            self.load_raw_tables()
            self.create_views()
            self.create_indexes()
            self.validate_data()
            self.close()
            
            duration = (datetime.now() - start_time).total_seconds()
            print()
            print("=" * 60)
            print(f"✨ Build completed in {duration:.1f} seconds!")
            print("=" * 60)
            print()
            print("Next steps:")
            print("  1. Set GEMINI_API_KEY in .env")
            print("  2. Run: streamlit run app/main.py")
            print()
            
        except Exception as e:
            print(f"\n❌ Build failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Build DuckDB database from Olist CSV files"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw",
        help="Directory containing CSV files (default: data/raw/)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "duckdb" / "olist.duckdb",
        help="Output database path (default: data/duckdb/olist.duckdb)"
    )
    
    args = parser.parse_args()
    
    builder = OlistDBBuilder(args.raw_dir, args.output)
    builder.build()


if __name__ == "__main__":
    main()



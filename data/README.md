# Data Directory

This directory contains the Olist e-commerce dataset and the generated DuckDB database.

## Structure

```
data/
├── raw/           # Place Kaggle CSV files here
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   └── product_category_name_translation.csv
└── duckdb/        # Generated database (auto-created)
    └── olist.duckdb
```

## Dataset Information

**Source:** Olist Brazilian E-commerce Public Dataset  
**Link:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/  
**Size:** ~100,000 orders from 2016-2018  
**License:** CC BY-NC-SA 4.0

### Tables Overview

1. **orders** (~99K rows) - Order master data with timestamps and status
2. **order_items** (~112K rows) - Individual items within orders
3. **products** (~32K rows) - Product catalog with dimensions and categories
4. **customers** (~99K rows) - Customer information and location
5. **sellers** (~3K rows) - Seller information and location
6. **payments** (~103K rows) - Payment transactions
7. **reviews** (~99K rows) - Customer reviews and ratings
8. **geolocation** (~1M rows) - Brazilian zip code geolocation data
9. **category_translation** (71 rows) - Portuguese to English category names

## Download Instructions

### Option 1: Manual Download

1. Go to https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/
2. Click "Download" (requires Kaggle login)
3. Extract all CSV files to `data/raw/`

### Option 2: Kaggle CLI

```bash
# Install Kaggle CLI
pip install kaggle

# Setup credentials (download kaggle.json from https://www.kaggle.com/settings)
# Place in ~/.kaggle/ (Linux/Mac) or C:\Users\<user>\.kaggle\ (Windows)

# Download and extract
kaggle datasets download -d olistbr/brazilian-ecommerce
unzip brazilian-ecommerce.zip -d data/raw/
```

### Option 3: Direct wget (if available)

```bash
# Requires Kaggle API token
kaggle datasets download -d olistbr/brazilian-ecommerce --path data/raw/ --unzip
```

## Build Database

After placing CSV files in `data/raw/`, run:

```bash
python scripts/build_duckdb.py
```

This will:
- ✅ Validate all CSV files are present
- ✅ Load data into DuckDB
- ✅ Create optimized analytical views
- ✅ Generate statistics and indexes
- ✅ Save to `data/duckdb/olist.duckdb`

Expected build time: 1-2 minutes

## Data Quality

The dataset has been cleaned by Olist but may contain:
- Missing values in some optional fields (reviews, delivery dates)
- Some products without categories
- Geographic data at zip code prefix level (not full address)
- Dates in Brazilian timezone (UTC-3)

## Privacy

All personal information has been anonymized:
- Customer and seller IDs are hashed
- No names, emails, or phone numbers
- Locations only show city/state
- Product details are generic categories

## Usage Notes

1. **Date Range:** Dataset covers September 2016 to October 2018
2. **Currency:** All values in Brazilian Reals (BRL)
3. **Geography:** Brazil only, 27 states
4. **Language:** Category names in Portuguese (translation table provided)

## Analytical Views

The build script creates these views for easy querying:

- **order_facts** - Complete order information with derived metrics
- **order_item_facts** - Line items with product and seller details
- **product_dim** - Product master data
- **customer_dim** - Customer master data
- **seller_dim** - Seller master data
- **order_summary** - Pre-aggregated order metrics

See `app/semantic/schema.md` for detailed column descriptions.

## File Size Estimates

- CSV files: ~150 MB total
- DuckDB database: ~50 MB
- With indexes: ~60 MB

## Troubleshooting

**"Missing CSV files" error:**
- Ensure all 9 CSV files are in `data/raw/`
- Check filenames match exactly (case-sensitive)

**"Database build failed":**
- Check CSV files are not corrupted
- Ensure sufficient disk space (~200 MB free)
- Verify Python packages installed: `pip install -r requirements.txt`

**"Database not found" in app:**
- Run `python scripts/build_duckdb.py` first
- Check `data/duckdb/olist.duckdb` exists

## Citation

If using this dataset in research or publications:

```bibtex
@misc{olist2018,
  author = {Olist},
  title = {Brazilian E-Commerce Public Dataset by Olist},
  year = {2018},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/olistbr/brazilian-ecommerce}},
}
```

## Resources

- **Dataset Documentation:** https://www.kaggle.com/olistbr/brazilian-ecommerce
- **Olist Website:** https://olist.com/
- **Data Dictionary:** Included in Kaggle download
- **Schema ERD:** See dataset page on Kaggle

---

Need help? See [QUICKSTART.md](../QUICKSTART.md) for setup instructions.




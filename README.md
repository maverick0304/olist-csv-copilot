# Olist Copilot 🛒

A production-ready GenAI agentic system for natural-language analytics. Ask questions in **any language**, get SQL-powered insights with auto-generated charts.

## 🎬 Walkthrough

Video link: 

```
https://drive.google.com/drive/folders/1K3GFHg33HvE-udz3jVCpZYD67yr9skao?usp=sharing
```

## ✨ Features

### Core Capabilities
- 🗣️ **Natural Language Q&A** - "What were the top 5 categories by revenue in 2018?"
- 🤖 **Autonomous Agent** - Plans → Generates SQL → Executes → Visualizes
- 📊 **Smart Visualizations** - Auto-generated charts based on query type
- 💾 **Session Memory** - Follow-ups remember context
- 🔒 **Safe Execution** - Read-only SQL with strict validation
- 📥 **Export Ready** - Download results as CSV

### Advanced Features
- 🌍 **Multilingual** - Ask in English, Portuguese, Hindi, Spanish, French, Arabic, Chinese, Japanese, Korean, etc.
- 📚 **Semantic Layer** - Pre-defined business metrics (GMV, AOV, repeat rate)
- 🧠 **RAG-Enhanced** - Retrieves schema and past queries for better accuracy
- 💡 **Contextual Suggestions** - AI-powered follow-up questions
- 📂 **Custom CSV Mode** - Upload your own data and ask questions instantly

### Two Modes

**1. Olist Mode** (Default)
- Analyze Brazilian e-commerce dataset
- Pre-configured schema and metrics
- Optimized for retail analytics

**2. CSV Mode** 
- Upload any CSV files
- Auto-detects schema and relationships
- Generates custom prompts dynamically

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google Gemini API Key ([Free](https://makersuite.google.com/app/apikey)) or Groq API Key ([Free](https://console.groq.com))

### Installation

```bash
# 1. Clone and install dependencies
git clone https://github.com/yourusername/olist-copilot.git
cd olist-copilot
pip install -r requirements.txt

# 2. Configure API keys
# Edit .env and add:
#   GEMINI_API_KEY=your_key_here
#   OR
#   GROQ_API_KEY=your_key_here

# 3. Download Olist dataset (for Olist mode)
# Option A: Manual from Kaggle
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/
# Extract all CSVs to data/raw/

# Option B: Kaggle CLI
pip install kaggle
kaggle datasets download -d olistbr/brazilian-ecommerce
unzip brazilian-ecommerce.zip -d data/raw/

# 4. Build database (for Olist mode)
python scripts/build_duckdb.py

# 5. Launch!
streamlit run app/main.py
```

Open http://localhost:8501 🎉

---



---

## 💬 Example Questions

### Olist Mode
```
"What are the top 5 product categories by GMV in 2018?"
"Show me monthly revenue trends for Electronics"
"Which sellers have the worst on-time delivery rate?"
"Compare payment methods by transaction value"
"What's the customer repeat purchase rate?"
```

### CSV Mode
```
"How many unique cities are there?"
"What are the top 10 products by sales?"
"Show me monthly sales trends"
"Which customers spent the most?"
```

### Multilingual
```
"Mostre-me as vendas totais por categoria" (Portuguese)
"मुझे श्रेणी के अनुसार बिक्री दिखाएं" (Hindi)
"Muéstrame las ventas por categoría" (Spanish)
```

---

## 📂 Project Structure

```
olist-copilot/
├── app/
│   ├── main.py                 # Main Streamlit UI (Olist mode)
│   ├── agent.py                # Core agent orchestration
│   ├── pages/
│   │   └── csv_mode.py         # CSV upload & analysis UI
│   ├── tools/
│   │   ├── sql_tool.py         # SQL execution (DuckDB)
│   │   ├── viz_tool.py         # Chart generation (Plotly)
│   │   ├── glossary_tool.py    # Semantic metrics
│   │   ├── rag_tool.py         # RAG for schema retrieval
│   │   ├── csv_tool.py         # CSV analysis & profiling
│   │   ├── data_profiler.py    # Data quality checks
│   │   └── translate_tool.py   # Language detection
│   ├── llm/
│   │   └── provider.py         # LLM abstraction (Gemini/Groq/Ollama/HF)
│   ├── prompts/
│   │   ├── system.txt          # Legacy prompt
│   │   ├── sql_generation_v2.txt  # Enhanced SQL prompt
│   │   ├── context_wrapper.txt    # Domain context layer
│   │   ├── complete_schema.txt    # Detailed schema docs
│   │   ├── few_shots.json         # Example queries (Olist)
│   │   └── csv_prompt_generator.py # Dynamic prompts (CSV mode)
│   ├── semantic/
│   │   ├── metrics.yaml        # Business metric definitions
│   │   └── schema.md           # Auto-generated schema reference
│   └── memory/
│       └── sessions.sqlite     # Session storage
├── data/
│   ├── raw/                    # Kaggle CSVs (user provides)
│   ├── duckdb/
│   │   └── olist.duckdb        # Built database
│   └── README.md
├── test_data/                  # Sample CSVs for testing
│   ├── customers.csv
│   ├── sales.csv
│   └── products.csv
├── scripts/
│   ├── build_duckdb.py         # Data ingestion pipeline
│   └── profile_schema.py       # Schema doc generator
├── tests/
│   └── test_sql_generation.py  # SQL safety tests
├── .env.example                # Environment template
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── ARCHITECTURE.md             # Technical docs
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────┐
│        Streamlit UI                      │
│  [Chat] [Charts] [CSV Export]            │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│           Agent (agent.py)               │
│  Plan → Generate SQL → Execute → Insight │
│                                           │
│  Powered by: Gemini / Groq / Ollama      │
└──────┬──────┬────────┬──────┬────────────┘
       │      │        │      │
  ┌────▼──┐ ┌▼────┐ ┌─▼──┐ ┌▼─────┐
  │SQL    │ │Viz  │ │RAG │ │Gloss.│
  │Tool   │ │Tool │ │Tool│ │Tool  │
  └───┬───┘ └─────┘ └────┘ └──────┘
      │
  ┌───▼────────────────┐
  │  DuckDB Database   │
  │  or In-Memory CSV  │
  └────────────────────┘
```

For detailed architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 🔒 Security & Guardrails

✅ **Read-Only Database** - DuckDB opened in read-only mode  
✅ **SQL Validation** - Only SELECT statements allowed  
✅ **Table Whitelist** - Only pre-defined tables accessible  
✅ **No Secrets in Code** - All keys from .env  
✅ **Auto-Repair** - Self-corrects common SQL errors

**Forbidden:**
- DDL (CREATE, ALTER, DROP)
- DML (INSERT, UPDATE, DELETE)
- System commands (PRAGMA, ATTACH)
- SQL injection patterns

---

## 🌍 Multilingual Support

**Supported Languages:**
- English
- Portuguese (Brazilian)
- Spanish
- Hindi
- French
- German
- Chinese (Simplified)
- Japanese
- Korean
- Arabic
- And 45+ more via auto-detection

**How it works:**
1. User asks in any language
2. `langdetect` identifies language with confidence check
3. LLM generates SQL (universal)
4. LLM generates insight in user's language
5. Follow-up questions also in user's language

**No configuration needed** - just ask naturally!

---

## 🤖 LLM Providers

### Supported Models

**1. Google Gemini (Default)**
- Free tier: 15 requests/minute
- Model: `gemini-2.5-flash`
- Best for: General queries

**2. Groq (Fastest)**
- Free tier: 30 requests/minute
- Model: `llama-3.3-70b-versatile`
- Best for: Speed (10x faster than Gemini)

**3. Ollama (Local)**
- Models: `llama3.2`, `codellama`, `mistral`, `phi3`
- Best for: Privacy, no API costs

**4. HuggingFace (Free)**
- Models: Various open-source
- Best for: Experimentation

### Switch Providers

Edit `app/llm/provider.py`:
```python
# Use Groq (fast, free)
from app.llm.provider import GroqProvider
llm = GroqProvider(api_key=os.getenv("GROQ_API_KEY"))

# Use Ollama (local)
from app.llm.provider import OllamaProvider
llm = OllamaProvider(model="llama3.2")
```

---

## 📊 Semantic Metrics

Pre-defined business metrics with SQL templates:

| Metric | Description | Formula |
|--------|-------------|---------|
| **GMV** | Gross Merchandise Value | SUM(order_value) |
| **AOV** | Average Order Value | AVG(order_value) |
| **Repeat Rate** | % returning customers | customers_with_2+_orders / total_customers |
| **On-Time Delivery** | % delivered by estimate | delivered_on_time / total_delivered |
| **Category Penetration** | % buying from category | customers_in_category / total_customers |

Access via: `"Explain GMV"` or `"What metrics are available?"`

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# SQL validation tests
pytest tests/test_sql_generation.py -v

# Test CSV mode
python test_csv_mode.py
```

---

## ▶️ Run the App Anytime

Once dependencies are installed and your `.env` is configured, launch the Streamlit experience with:

```bash
streamlit run app/main.py
```

The UI will open at [http://localhost:8501](http://localhost:8501).

---

## 🐛 Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### Database Not Found
```bash
python scripts/build_duckdb.py
```

### API Key Errors
- Ensure `.env` file exists
- Check key is valid at provider website
- Verify key name: `GEMINI_API_KEY` or `GROQ_API_KEY`

### Empty Results
- Check date range filters
- Verify CSV files in `data/raw/`
- Rebuild database

### Language Detection Issues
- Short queries (<15 chars) default to English
- Non-English requires >80% confidence
- Override by being more explicit in your query

---

## 🚀 Future Enhancements

If I had more time, here's what I would add:

- **Vector Search** - Embed schema descriptions for semantic column matching
- **Multi-Modal Charts** - Scatter plots, heatmaps, geographic visualizations
- **Query History & Bookmarks** - Save favorite queries and share with team
- **Scheduled Reports** - Recurring analytics with email/Slack delivery
- **Real-Time Data Pipeline** - Streaming ingestion for live dashboards
- **Multi-User Authentication** - Role-based access and query audit logs
- **Advanced Anomaly Detection** - Statistical outlier detection in time series
- **Natural Language Explanations** - LLM-generated insights on trends and patterns

---

## 🙏 Acknowledgments

- **Dataset:** [Olist Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/)
- **Built with:** Streamlit, DuckDB, Google Gemini, Groq, Plotly
- **Inspired by:** Modern AI agent architectures

---

Built with ❤️ for production-ready AI agents

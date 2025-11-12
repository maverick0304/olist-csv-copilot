# Olist Copilot - Technical Architecture

Complete technical documentation for the Olist Copilot GenAI agentic system.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Dual-Mode Architecture](#dual-mode-architecture)
4. [LLM Provider Abstraction](#llm-provider-abstraction)
5. [Prompt Engineering](#prompt-engineering)
6. [RAG Implementation](#rag-implementation)
7. [Multilingual Support](#multilingual-support)
8. [SQL Generation & Validation](#sql-generation--validation)
9. [Data Flow](#data-flow)
10. [Security & Guardrails](#security--guardrails)
11. [Performance Optimizations](#performance-optimizations)
12. [Design Decisions](#design-decisions)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                             │
│                         (Streamlit Apps)                            │
│  ┌──────────────────────┐        ┌──────────────────────────┐      │
│  │  Olist Mode         │        │  CSV Mode                │      │
│  │  (main.py)          │        │  (pages/csv_mode.py)     │      │
│  └──────────┬───────────┘        └──────────┬───────────────┘      │
└─────────────┼──────────────────────────────┼──────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│                        AGENT ORCHESTRATOR                            │
│                           (agent.py)                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Initialization:                                             │   │
│  │  if csv_mode:                                                │   │
│  │      • Load CSV files → CSVAnalyzer                         │   │
│  │      • Profile data → DataProfiler                          │   │
│  │      • Generate prompts → CSVPromptGenerator                │   │
│  │      • No RAG, No Glossary                                  │   │
│  │  else:                                                       │   │
│  │      • Connect to DuckDB                                    │   │
│  │      • Load RAG Tool (schema + query history)               │   │
│  │      • Load Glossary Tool (semantic metrics)                │   │
│  │      • Load static prompts                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌──────────┐  ┌─────────────┐  ┌───────────┐  ┌────────────┐     │
│  │ Planner  │→ │SQL Generator│→ │ Validator │→ │  Executor  │     │
│  └──────────┘  └─────────────┘  └───────────┘  └────────────┘     │
│                        │                              │              │
│                  ┌─────▼─────┐                 ┌──────▼──────┐      │
│                  │Self-Repair│                 │   Insight   │      │
│                  └───────────┘                 │  Generator  │      │
│                                                 └─────────────┘      │
└─────────┬────────┬─────────┬─────────┬─────────────┬───────────────┘
          │        │         │         │             │
    ┌─────▼──┐ ┌──▼───┐ ┌───▼────┐ ┌─▼──────┐ ┌───▼────┐
    │SQL     │ │Viz   │ │RAG     │ │Glossary│ │Translate│
    │Tool    │ │Tool  │ │Tool    │ │Tool    │ │Tool     │
    └────┬───┘ └──────┘ └───┬────┘ └────────┘ └─────────┘
         │                   │
         │                   │ (Olist mode only)
         │                   │
    ┌────▼───────────────────▼──────────────┐
    │   DuckDB (Olist) or In-Memory (CSV)   │
    │                                        │
    │  Olist Mode:                           │
    │  • olist.duckdb (read-only)           │
    │  • Pre-built views & indexes          │
    │                                        │
    │  CSV Mode:                             │
    │  • Temporary DuckDB instance          │
    │  • CSVs loaded dynamically            │
    └────────────────────────────────────────┘
```

---

## Core Components

### 1. User Interface Layer

#### **app/main.py** - Olist Mode UI

```python
# Key Features:
- Chat-style Q&A interface
- Quick start question buttons
- Real-time query processing with spinners
- Data tables with export functionality
- Auto-generated charts (Plotly)
- Collapsible SQL preview
- Context-aware follow-up suggestions
- Multilingual auto-detection
- Session history management
```

**UI Components:**
- Sidebar: Display preferences (show SQL, auto-visualize)
- Main area: Chat history, suggested questions
- Chat input: Always visible at bottom
- Follow-up buttons: Contextual, clickable

#### **app/pages/csv_mode.py** - CSV Upload UI

```python
# Key Features:
- Multi-file CSV uploader
- Schema visualization
- Relationship graph
- Data quality dashboard
- Dynamic chat interface
- Same core features as Olist mode
```

**Upload Flow:**
1. User uploads CSV files
2. System analyzes schema & relationships
3. Generates data quality report
4. Creates dynamic prompts
5. Initializes agent in CSV mode
6. Chat interface activates

---

### 2. Agent Orchestrator (app/agent.py)

**Core Class: `OlistAgent`**

```python
class OlistAgent:
    def __init__(
        self, 
        db_path=None,           # Path to DuckDB (Olist mode)
        csv_mode=False,         # Enable CSV mode
        csv_analyzer=None,      # CSV schema analyzer
        data_profiler=None      # Data quality profiler
    ):
        # Mode-specific initialization
        if csv_mode:
            # CSV Mode: Dynamic prompts, no RAG/Glossary
            self.sql_tool = SQLTool(db_path=None, ...)
            self.rag_tool = None
            self.glossary_tool = None
            self.system_prompt = self._load_csv_system_prompt()
            self.few_shots = self._load_csv_few_shots()
        else:
            # Olist Mode: Static schema, RAG enabled
            self.sql_tool = SQLTool(db_path=db_path, ...)
            self.rag_tool = RAGTool(db_path=db_path)
            self.glossary_tool = GlossaryTool()
            self.system_prompt = self._load_system_prompt()
            self.few_shots = self._load_few_shots()
```

**Query Processing Pipeline:**

```
1. Resolve References
   └─ "Show that for 2017" → "Show GMV for 2017"

2. Plan Query (LLM)
   └─ Extract: metric, filters, aggregation, grouping, limit

3. Check Semantic Metrics (Olist only)
   └─ If recognized metric → use template

4. Generate SQL
   └─ LLM with system prompt + few-shots + RAG context

5. Validate & Repair
   └─ Check table names, add GROUP BY if missing, fix time columns

6. Execute SQL (Read-Only)
   └─ DuckDB query with retry logic

7. Save to RAG History (Olist only)
   └─ Store successful query patterns

8. Generate Insight (LLM)
   └─ Natural language explanation in user's language

9. Create Visualization
   └─ Auto-select chart type (bar, line, pie)

10. Extract Entities
    └─ Track categories, years, metrics for context

11. Generate Follow-ups (LLM)
    └─ 5 contextual questions

12. Return Response
    └─ {insight, data, chart, sql, followup_questions}
```

---

### 3. Tool Suite

#### **SQL Tool** (app/tools/sql_tool.py)

```python
class SQLTool:
    """
    Read-only SQL execution with validation
    
    Security Features:
    - Only SELECT statements allowed
    - Table whitelist enforcement
    - SQL injection prevention
    - Statement chaining blocked
    - Read-only connection
    """
    
    def execute_safe(self, sql: str) -> Dict:
        # 1. Validate SQL structure
        # 2. Check allowed tables
        # 3. Execute in read-only mode
        # 4. Return DataFrame result
```

**Validation Rules:**
- Must start with SELECT
- No DDL (CREATE, DROP, ALTER)
- No DML (INSERT, UPDATE, DELETE)
- No PRAGMA, ATTACH, IMPORT
- Only whitelisted tables
- No semicolon chaining

#### **Visualization Tool** (app/tools/viz_tool.py)

```python
class VizTool:
    """
    Auto-generate charts based on data shape
    
    Chart Selection Logic:
    - 1 dimension + 1 metric → Bar chart
    - Time column + metric → Line chart
    - Single row → Metric card
    - Categorical breakdown → Pie chart
    - 2 metrics → Scatter plot
    """
    
    def create_chart(self, data, chart_type=None, title=None):
        # Auto-detect if not specified
        # Apply consistent styling
        # Return Plotly figure
```

#### **RAG Tool** (app/tools/rag_tool.py) - Olist Mode Only

```python
class RAGTool:
    """
    Retrieval-Augmented Generation for schema understanding
    
    Storage:
    - SQLite database (rag_history.db)
    - Schema embeddings (if enabled)
    - Query history with success metrics
    
    Retrieval:
    - Keyword search (default, fast)
    - Semantic search (optional, requires embeddings)
    """
    
    def get_relevant_context(self, user_query: str) -> str:
        # 1. Search schema for relevant tables/columns
        # 2. Find similar past queries
        # 3. Format as context for LLM
        # 4. Return formatted string
```

**Schema Store:**
```sql
CREATE TABLE schema_info (
    table_name TEXT,
    column_name TEXT,
    data_type TEXT,
    description TEXT
);

CREATE TABLE query_history (
    query TEXT,
    sql TEXT,
    success BOOLEAN,
    timestamp DATETIME
);
```

#### **Glossary Tool** (app/tools/glossary_tool.py) - Olist Mode Only

```python
class GlossaryTool:
    """
    Semantic business metrics with SQL templates
    
    Metrics defined in: app/semantic/metrics.yaml
    """
    
    def get_metric(self, metric_name: str) -> Dict:
        # Returns: {name, description, sql_template, required_tables}
```

**Example Metric:**
```yaml
gmv:
  name: "Gross Merchandise Value"
  description: "Total value of all orders"
  sql_template: |
    SELECT 
      {group_by_clause}
      SUM(total_value) as gmv,
      COUNT(DISTINCT order_id) as orders
    FROM order_item_facts
    {where_clause}
    {group_by_clause}
    ORDER BY gmv DESC
  required_tables:
    - order_item_facts
```

#### **CSV Tools** (CSV Mode Only)

**CSVAnalyzer** (app/tools/csv_tool.py)
```python
class CSVAnalyzer:
    """
    Dynamic schema detection and relationship inference
    
    Features:
    - Load multiple CSVs
    - Detect column types (ID, metric, dimension, temporal)
    - Infer relationships (FK detection by naming + cardinality)
    - Generate schema summary
    """
    
    def load_csv(self, file_path, table_name) -> Dict:
        # 1. Read CSV with DuckDB
        # 2. Analyze each column:
        #    - Unique values
        #    - Data type
        #    - Purpose (ID vs metric vs dimension)
        # 3. Store metadata
```

**Relationship Detection:**
```python
def infer_relationships(self) -> List[Dict]:
    """
    Detect foreign keys by:
    1. Column name matching (e.g., customer_id in both tables)
    2. Cardinality check (1:many relationship)
    3. Value overlap (>80% of FK values exist in PK table)
    
    Returns confidence score (0-1)
    """
```

**DataProfiler** (app/tools/data_profiler.py)
```python
class DataProfiler:
    """
    Data quality analysis
    
    Checks:
    - Missing values percentage
    - Duplicate rows
    - Outliers in numeric columns
    - Data type consistency
    - Date range validity
    
    Generates quality score (0-1)
    """
```

**CSVPromptGenerator** (app/prompts/csv_prompt_generator.py)
```python
class CSVPromptGenerator:
    """
    Dynamic prompt generation based on uploaded CSVs
    
    Generates:
    1. System prompt with schema documentation
    2. Few-shot examples tailored to the data
    3. SQL generation rules
    4. Relationship instructions
    """
    
    def generate_schema_prompt(self) -> str:
        # Creates prompt like:
        """
        You are analyzing 3 CSV tables:
        
        • customers (1,000 rows, 5 columns)
          - customer_id (PK, INTEGER, unique)
          - city (dimension, TEXT, 200 unique values)
          - signup_date (temporal, DATE)
        
        • sales (5,000 rows, 4 columns)
          - sale_id (PK, INTEGER)
          - customer_id (FK → customers.customer_id, 95% confidence)
          - amount (metric, DECIMAL, use for SUM/AVG)
          - sale_date (temporal, DATE)
        
        Relationships:
        ✓ sales.customer_id → customers.customer_id (95% confidence)
        
        SQL Rules:
        - Use these exact table names
        - JOIN using detected relationships
        - GROUP BY for aggregations
        """
```

---

## Dual-Mode Architecture

### Olist Mode (Static Schema)

**Characteristics:**
- Pre-defined database schema
- RAG enabled for schema retrieval
- Glossary tool for semantic metrics
- Static prompts from files
- Historical query patterns

**Data Pipeline:**
```
Kaggle CSVs → build_duckdb.py → olist.duckdb → Agent
```

**Tables:**
- order_facts (orders with dates)
- order_item_facts (line items)
- product_dim (product catalog)
- customer_dim (customer master)
- seller_dim (seller master)

### CSV Mode (Dynamic Schema)

**Characteristics:**
- Schema detected at runtime
- No RAG (schema in prompt already)
- No glossary (no predefined metrics)
- Dynamic prompts generated
- Relationship auto-detection

**Data Pipeline:**
```
User CSVs → CSVAnalyzer → DataProfiler → CSVPromptGenerator → Agent
```

**Initialization Flow:**
```python
# 1. User uploads files
uploaded_files = st.file_uploader(...)

# 2. Analyze schema
analyzer = CSVAnalyzer()
for file in uploaded_files:
    analyzer.load_csv(file, table_name)

# 3. Detect relationships
relationships = analyzer.detect_relationships()

# 4. Profile data
profiler = DataProfiler(analyzer)
profiler.profile_table(table_name)

# 5. Generate prompts
generator = CSVPromptGenerator(analyzer, profiler)
system_prompt = generator.generate_schema_prompt()
few_shots = generator.generate_few_shot_examples()

# 6. Initialize agent
agent = OlistAgent(
    db_path=None,
    csv_mode=True,
    csv_analyzer=analyzer,
    data_profiler=profiler
)
```

---

## LLM Provider Abstraction

**Location:** `app/llm/provider.py`

```python
class LLMProvider(ABC):
    """
    Base class for all LLM providers
    
    Supported:
    - GeminiProvider (Google Gemini)
    - GroqProvider (Groq Cloud)
    - OllamaProvider (Local Ollama)
    - HuggingFaceProvider (HF Inference API)
    """
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float) -> str:
        pass
```

### Provider Comparison

| Provider | Speed | Cost | Privacy | Setup |
|----------|-------|------|---------|-------|
| **Gemini** | Medium | Free tier | Cloud | API key |
| **Groq** | Fast (10x) | Free tier | Cloud | API key |
| **Ollama** | Slow | Free | Local | Install Ollama |
| **HuggingFace** | Slow | Free | Cloud | API key |

### Groq Integration (Recommended for Speed)

**Setup:**
```bash
# 1. Get API key from console.groq.com
# 2. Add to .env
GROQ_API_KEY=gsk_xxxxxxxxxxxx

# 3. Update agent.py
from app.llm.provider import GroqProvider
self.llm = GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
```

**Benefits:**
- 10x faster than Gemini
- Same accuracy
- Free tier: 30 req/min
- Model: `llama-3.3-70b-versatile`

### Ollama (Local, Private)

**Setup:**
```bash
# 1. Install Ollama
https://ollama.ai/download

# 2. Pull model
ollama pull llama3.2

# 3. Update agent.py
from app.llm.provider import OllamaProvider
self.llm = OllamaProvider(model="llama3.2")
```

**Supported Models:**
- `llama3.2` (general purpose)
- `llama3.2:1b` (faster, lower quality)
- `codellama:7b` (better for SQL)
- `mistral:7b` (balanced)
- `phi3` (lightweight)

---

## Prompt Engineering

### Hierarchical Prompting Strategy

**Layer 1: Context Wrapper** (`app/prompts/context_wrapper.txt`)
```
You are an expert data analyst for Olist, a Brazilian e-commerce company.

Domain Context:
- Business Model: Online marketplace
- Data: Orders from 2016-2018
- Geography: Brazil (states, cities)
- Currency: Brazilian Real (BRL)
- Date Format: YYYY-MM-DD

Your task: Convert natural language questions into SQL queries.
```

**Layer 2: Complete Schema** (`app/prompts/complete_schema.txt`)
```
DATABASE SCHEMA:

Table: order_facts
- order_id (PK)
- customer_id (FK)
- purchase_ts (TIMESTAMP)
- purchase_year (INTEGER)
- purchase_month (INTEGER)
- delivery_date (DATE)
- estimated_delivery_date (DATE)
- order_status (VARCHAR)

[Full schema with all tables, columns, types, relationships]
```

**Layer 3: SQL Generation Rules** (`app/prompts/sql_generation_v2.txt`)
```
CRITICAL SQL GENERATION RULES:

Rule 1: Time-Series Queries
When user asks for "trends", "monthly", "over time":
MUST include time columns in SELECT and GROUP BY:
SELECT 
  purchase_year,
  purchase_month,
  SUM(revenue) as total
FROM order_facts
GROUP BY purchase_year, purchase_month
ORDER BY purchase_year, purchase_month

Rule 2: Ranking Queries
When user asks for "top 5", "best", "highest":
MUST include:
- GROUP BY <dimension>
- ORDER BY <metric> DESC
- LIMIT <N>

[30+ detailed rules with examples]
```

**Layer 4: Few-Shot Examples** (`app/prompts/few_shots.json`)
```json
[
  {
    "user_query": "What are the top 5 product categories by GMV in 2018?",
    "reasoning": "This is a ranking query with year filter. Need to GROUP BY category, filter by year, ORDER BY gmv DESC, LIMIT 5",
    "sql": "SELECT category_name_en, SUM(total_value) as gmv FROM order_item_facts oif JOIN order_facts of ON oif.order_id = of.order_id WHERE YEAR(of.purchase_ts) = 2018 GROUP BY category_name_en ORDER BY gmv DESC LIMIT 5"
  }
]
```

### Dynamic Prompts (CSV Mode)

**Generated on-the-fly based on uploaded data:**

```python
def generate_schema_prompt(analyzer, profiler):
    prompt = f"""
    You are analyzing {len(analyzer.tables)} CSV tables:
    
    """
    
    for table_name, info in analyzer.tables.items():
        prompt += f"\n• {table_name} ({info['row_count']} rows, {len(info['columns'])} columns)\n"
        
        # Group columns by purpose
        for col in info['columns']:
            purpose = analyzer.column_metadata[f"{table_name}.{col['name']}"]["purpose"]
            if purpose == "metric":
                prompt += f"  - {col['name']} (METRIC, use for SUM/AVG)\n"
            elif purpose == "dimension":
                prompt += f"  - {col['name']} (DIMENSION, use for GROUP BY)\n"
            elif purpose == "temporal":
                prompt += f"  - {col['name']} (TEMPORAL, use for trends)\n"
    
    # Add relationships
    if analyzer.relationships:
        prompt += "\nDetected Relationships:\n"
        for rel in analyzer.relationships:
            prompt += f"✓ {rel['from_table']}.{rel['from_column']} → "
            prompt += f"{rel['to_table']}.{rel['to_column']} "
            prompt += f"({rel['confidence']:.0%} confidence)\n"
    
    return prompt
```

---

## RAG Implementation

**Purpose:** Retrieve relevant schema info and past queries to improve SQL generation accuracy.

### Architecture

```
┌─────────────────────────────────────┐
│     User Query                      │
│  "Show me monthly sales trends"    │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│     RAG Tool                        │
│  1. Extract keywords                │
│  2. Search schema store             │
│  3. Find similar past queries       │
│  4. Format context                  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Retrieved Context:                 │
│  - Tables: order_facts, sales       │
│  - Columns: purchase_month, amount  │
│  - Past: "revenue by quarter"       │
│     SQL: SELECT quarter, SUM(...)   │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   LLM Prompt = System + Context     │
│   + Few-Shots + User Query          │
└─────────────┬───────────────────────┘
              │
              ▼
         Generated SQL
```

### Keyword Search (Default, Fast)

```python
def get_relevant_context(self, user_query: str) -> str:
    # Extract keywords
    keywords = self._extract_keywords(user_query)
    # "monthly sales trends" → ["monthly", "sales", "trends"]
    
    # Search schema
    relevant_tables = []
    relevant_columns = []
    
    for keyword in keywords:
        # Find tables with matching names
        tables = self._search_tables(keyword)
        relevant_tables.extend(tables)
        
        # Find columns with matching names or descriptions
        columns = self._search_columns(keyword)
        relevant_columns.extend(columns)
    
    # Search query history
    similar_queries = self._search_query_history(keywords)
    
    # Format context
    context = f"""
    RELEVANT SCHEMA:
    Tables: {", ".join(relevant_tables)}
    Columns: {", ".join(relevant_columns)}
    
    SIMILAR PAST QUERIES:
    {similar_queries}
    """
    
    return context
```

### Semantic Search (Optional, Requires Embeddings)

```bash
# Install dependencies
pip install sentence-transformers

# Enable in app/tools/rag_tool.py
USE_EMBEDDINGS = True
```

```python
from sentence_transformers import SentenceTransformer

class RAGTool:
    def __init__(self):
        if USE_EMBEDDINGS:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.schema_embeddings = self._embed_schema()
    
    def get_relevant_context(self, user_query):
        # Embed query
        query_embedding = self.model.encode(user_query)
        
        # Find most similar schema elements
        similarities = cosine_similarity(query_embedding, self.schema_embeddings)
        
        # Retrieve top-K
        top_indices = np.argsort(similarities)[-5:]
        
        # Format context
        return self._format_context(top_indices)
```

---

## Multilingual Support

**Library:** `langdetect` (Python port of Google's language-detection)

### Detection Algorithm

```python
def _detect_language(self, text: str) -> str:
    # Layer 1: Length check
    if len(text.strip()) < 15:
        # Too short for reliable detection
        return 'English'
    
    # Layer 2: Confidence threshold
    langs = detect_langs(text)  # Returns list with probabilities
    top_lang = langs[0]
    
    if top_lang.lang != 'en' and top_lang.prob < 0.8:
        # Low confidence for non-English → default to English
        return 'English'
    
    # Layer 3: Map to full name
    return self.lang_map.get(top_lang.lang, 'English')
```

**Confidence Thresholds:**
- English: No threshold (default)
- Other languages: >80% confidence required
- Short queries (<15 chars): Always English

### Language Processing Flow

```
1. User Query → "Mostre-me as vendas totais"

2. Language Detection
   └─ detect_langs() → Portuguese (96% confidence) ✓

3. SQL Generation (Universal)
   └─ SELECT SUM(revenue) FROM sales

4. Insight Generation (Portuguese)
   └─ LLM Prompt:
      "CRITICAL: User asked in Portuguese.
       You MUST respond in Portuguese.
       User Query: 'Mostre-me as vendas totais'
       Results: [data]
       Generate insight in Portuguese:"
   
   └─ Response: "As vendas totais foram R$ 15.843.550,00..."

5. Follow-up Questions (Portuguese)
   └─ "Compare com o ano passado"
   └─ "Mostre-me por categoria"
```

**Supported Languages (55+):**
```python
lang_map = {
    'en': 'English',
    'pt': 'Portuguese (Brazilian)',
    'es': 'Spanish',
    'hi': 'Hindi',
    'fr': 'French',
    'de': 'German',
    'zh-cn': 'Chinese (Simplified)',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    # ... 45+ more
}
```

---

## SQL Generation & Validation

### Generation Process

```python
def _generate_sql(self, user_query, plan):
    # 1. Load system prompt
    system_prompt = self.system_prompt  # From file or generated
    
    # 2. Get RAG context (Olist mode only)
    rag_context = ""
    if self.rag_tool:
        rag_context = self.rag_tool.get_relevant_context(user_query)
    
    # 3. Select relevant few-shot examples
    relevant_examples = self._select_few_shots(user_query, self.few_shots)
    
    # 4. Build final prompt
    sql_prompt = f"""
    {system_prompt}
    
    {rag_context}
    
    EXAMPLES:
    {relevant_examples}
    
    USER QUERY: "{user_query}"
    
    FILTERS:
    - Year: {plan['filters'].get('year', 'none')}
    - Category: {plan['filters'].get('category', 'none')}
    - Limit: {plan.get('limit', 10)}
    
    Generate SQL following the rules and examples above.
    Return ONLY the SQL query, no explanation.
    """
    
    # 5. Call LLM
    sql = self.llm.generate(sql_prompt, temperature=0.1)
    
    # 6. Clean up
    sql = self._clean_sql(sql)  # Remove markdown, semicolons
    
    # 7. Validate & repair
    sql = self._validate_and_repair_sql(sql, user_query, plan)
    
    return sql
```

### Auto-Repair Logic

```python
def _validate_and_repair_sql(self, sql, user_query, plan):
    """
    Automatically fix common LLM mistakes
    """
    
    # Fix 1: Time-series queries missing time columns
    if "trend" in user_query.lower() or "monthly" in user_query.lower():
        if "purchase_month" not in sql:
            # Rebuild query with time grouping
            sql = self._rebuild_with_time_columns(sql, user_query)
    
    # Fix 2: Ranking queries missing GROUP BY
    if any(word in user_query.lower() for word in ["top", "best", "highest"]):
        if "GROUP BY" not in sql.upper():
            # Add GROUP BY based on SELECT columns
            sql = self._add_group_by(sql, plan)
    
    # Fix 3: Incorrect table names
    if self.sql_tool:
        # Check against whitelist
        used_tables = self._extract_tables(sql)
        invalid = set(used_tables) - set(self.sql_tool.allowed_tables)
        if invalid:
            # Try to repair or reject
            sql = self._fix_table_names(sql, invalid)
    
    return sql
```

### Validation Rules

```python
def validate_sql(self, sql):
    # 1. Must be SELECT statement
    if not sql.strip().upper().startswith('SELECT'):
        raise ValueError("Only SELECT statements allowed")
    
    # 2. No DDL/DML keywords
    dangerous_keywords = [
        'DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE',
        'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE'
    ]
    sql_upper = sql.upper()
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            raise ValueError(f"Dangerous keyword: {keyword}")
    
    # 3. No system commands
    if 'PRAGMA' in sql_upper or 'ATTACH' in sql_upper:
        raise ValueError("System commands not allowed")
    
    # 4. No statement chaining
    if ';' in sql[:-1]:  # Allow trailing semicolon
        raise ValueError("Multiple statements not allowed")
    
    # 5. Table whitelist
    tables_used = self._extract_tables(sql)
    invalid_tables = set(tables_used) - set(self.allowed_tables)
    if invalid_tables:
        raise ValueError(f"Invalid tables: {invalid_tables}")
    
    return True
```

---

## Data Flow

### Olist Mode Query Flow

```
1. User Input
   └─ "What are the top 5 categories by GMV in 2018?"

2. Language Detection
   └─ English (default)

3. Reference Resolution
   └─ No pronouns, no changes

4. Query Planning (LLM)
   └─ {
        "metric_name": "gmv",
        "aggregation": "sum",
        "group_by": ["category"],
        "filters": {"year": 2018},
        "limit": 5,
        "should_visualize": True
      }

5. Check Semantic Metrics
   └─ GMV found in glossary
   └─ But skip template (ranking query detected)

6. RAG Context Retrieval
   └─ Tables: order_item_facts, order_facts
   └─ Similar query: "top categories by revenue"

7. SQL Generation (LLM)
   └─ SELECT category_name_en, SUM(total_value) as gmv
      FROM order_item_facts oif
      JOIN order_facts of ON oif.order_id = of.order_id
      WHERE YEAR(of.purchase_ts) = 2018
      GROUP BY category_name_en
      ORDER BY gmv DESC
      LIMIT 5

8. Validation
   └─ ✓ SELECT only
   └─ ✓ Valid tables
   └─ ✓ Has GROUP BY
   └─ ✓ Has ORDER BY + LIMIT

9. Execution (DuckDB Read-Only)
   └─ Returns DataFrame with 5 rows

10. Save to RAG
    └─ Store successful query pattern

11. Insight Generation (LLM in English)
    └─ "The top 5 product categories by GMV in 2018 were:
        1. Health & Beauty (R$ 2.1M)
        2. Bed, Bath & Table (R$ 1.8M)
        ..."

12. Visualization
    └─ Auto-detect: Bar chart (category vs GMV)

13. Entity Extraction
    └─ Track: 2018, categories mentioned

14. Follow-up Generation (LLM)
    └─ ["Show monthly trend for Health & Beauty",
        "Compare 2018 with 2017",
        "Which sellers had highest GMV in Health & Beauty?",
        ...]

15. Return Response
    └─ {
         "success": True,
         "insight": "...",
         "data": <DataFrame>,
         "chart": <Plotly Figure>,
         "sql": "...",
         "followup_questions": [...]
       }
```

### CSV Mode Query Flow

```
1. User Uploads CSVs
   └─ customers.csv, sales.csv, products.csv

2. CSVAnalyzer.load_csv()
   └─ For each CSV:
      - Detect column types
      - Identify purpose (ID, metric, dimension, temporal)
      - Store metadata

3. CSVAnalyzer.detect_relationships()
   └─ Find foreign keys:
      - sales.customer_id → customers.customer_id (95%)
      - sales.product_id → products.product_id (98%)

4. DataProfiler.profile_table()
   └─ For each table:
      - Missing values: 2%
      - Duplicates: 0
      - Quality score: 0.95

5. CSVPromptGenerator.generate_schema_prompt()
   └─ Creates custom system prompt with:
      - Table descriptions
      - Column purposes
      - Relationships
      - SQL rules

6. CSVPromptGenerator.generate_few_shot_examples()
   └─ [
        {
          "question": "How many unique cities?",
          "sql": "SELECT COUNT(DISTINCT city) FROM customers",
          "reasoning": "Simple count of unique values"
        },
        ...
      ]

7. Initialize Agent in CSV Mode
   └─ OlistAgent(
        db_path=None,
        csv_mode=True,
        csv_analyzer=analyzer,
        data_profiler=profiler
      )
   └─ self.rag_tool = None
   └─ self.glossary_tool = None
   └─ self.system_prompt = <generated>

8. User Query
   └─ "How many unique city values are there?"

9. SQL Generation (LLM with dynamic prompt)
   └─ SELECT COUNT(DISTINCT city) as unique_cities
      FROM customers

10. Validation & Execution
    └─ Same as Olist mode

11. Response
    └─ No RAG saving (rag_tool is None)
    └─ Same insight & visualization flow
```

---

## Security & Guardrails

### 1. Read-Only Database

```python
# Olist Mode
conn = duckdb.connect(str(db_path), read_only=True)

# CSV Mode
conn = duckdb.connect()  # In-memory, but still validated
```

### 2. SQL Validation

**Whitelist Approach:**
```python
ALLOWED_TABLES = [
    'order_facts',
    'order_item_facts',
    'product_dim',
    'customer_dim',
    'seller_dim'
]

# CSV Mode - dynamically set
allowed_tables = analyzer.get_table_list()
```

**Blacklist Approach:**
```python
DANGEROUS_KEYWORDS = [
    'DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER',
    'TRUNCATE', 'GRANT', 'REVOKE', 'PRAGMA', 'ATTACH',
    'IMPORT', 'EXPORT', 'COPY'
]
```

### 3. Input Sanitization

```python
def sanitize_query(self, user_query: str) -> str:
    # Remove potential SQL injection patterns
    user_query = re.sub(r'--.*$', '', user_query, flags=re.MULTILINE)
    user_query = re.sub(r'/\*.*?\*/', '', user_query, flags=re.DOTALL)
    return user_query.strip()
```

### 4. Rate Limiting (LLM APIs)

```python
# Gemini: 15 requests/minute
# Groq: 30 requests/minute
# Ollama: No limit (local)

# Implement exponential backoff
@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def generate(self, prompt, temperature):
    return self.llm.generate(prompt, temperature)
```

### 5. Error Handling

```python
try:
    result = agent.process_query(user_query)
except ValidationError as e:
    return {"success": False, "error": f"Invalid query: {e}"}
except ExecutionError as e:
    return {"success": False, "error": f"SQL error: {e}"}
except TimeoutError as e:
    return {"success": False, "error": "Query timeout"}
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    return {"success": False, "error": "An error occurred"}
```

---

## Performance Optimizations

### 1. DuckDB Optimizations

```python
# Create indexes on frequently joined columns
CREATE INDEX idx_order_id ON order_facts(order_id);
CREATE INDEX idx_customer_id ON order_facts(customer_id);
CREATE INDEX idx_product_id ON order_item_facts(product_id);

# Materialize common aggregations as views
CREATE VIEW order_summary AS
SELECT 
    purchase_year,
    purchase_quarter,
    purchase_month,
    COUNT(DISTINCT order_id) as orders,
    SUM(total_value) as gmv,
    AVG(total_value) as aov
FROM order_item_facts oif
JOIN order_facts of ON oif.order_id = of.order_id
GROUP BY purchase_year, purchase_quarter, purchase_month;
```

### 2. LLM Caching

```python
# Session-level caching for language detection
if user_query in self.language_cache:
    return self.language_cache[user_query]

# Prompt template caching
@lru_cache(maxsize=128)
def _build_system_prompt(self) -> str:
    return self._load_system_prompt()
```

### 3. Streamlit Optimizations

```python
# Cache expensive operations
@st.cache_resource
def init_agent():
    return OlistAgent(db_path)

@st.cache_data
def load_example_data():
    return pd.read_csv("examples.csv")

# Use session state to persist agent
if "agent" not in st.session_state:
    st.session_state.agent = init_agent()
```

### 4. SQL Result Pagination

```python
# Limit result size
MAX_ROWS = 10000

# Add LIMIT if missing
if "LIMIT" not in sql.upper():
    sql += f" LIMIT {MAX_ROWS}"
```

### 5. Chart Rendering

```python
# Limit data points in charts
if len(data) > 100:
    # Aggregate or sample
    data = data.sample(n=100)

# Use efficient chart types
# Line chart: up to 1000 points
# Bar chart: up to 50 categories
# Pie chart: up to 10 slices
```

---

## Design Decisions

### Why Custom Agent Over LangGraph?

**Pros:**
- Full control over execution flow
- Easy to debug (no black box)
- Minimal dependencies
- Simple to extend

**Cons:**
- More code to maintain
- No built-in graph visualization
- Manual state management

**Decision:** Custom agent for production clarity.

### Why DuckDB Over PostgreSQL?

**DuckDB Advantages:**
- Zero-config embedded database
- Fast analytical queries (columnar storage)
- Excellent pandas integration
- No server to manage
- Small file size (<100MB)

**PostgreSQL Advantages:**
- Better for concurrent writes
- More mature ecosystem
- Better for >1TB data

**Decision:** DuckDB for analytical workload.

### Why Streamlit Over React/Vue?

**Streamlit Advantages:**
- Pure Python (no JS needed)
- Built-in chat components
- Rapid prototyping
- Auto-reload on code changes

**React/Vue Advantages:**
- More UI flexibility
- Better performance at scale
- Richer component ecosystem

**Decision:** Streamlit for MVP speed.

### Why Gemini Over GPT-4?

**Gemini Advantages:**
- Free tier (15 req/min)
- Fast response times
- Good SQL generation
- Google's infrastructure

**GPT-4 Advantages:**
- Better reasoning for complex queries
- More extensive training data
- Better function calling

**Decision:** Gemini for cost + speed. (Groq for even faster.)

### Why Dual-Mode Architecture?

**Rationale:**
- Olist mode: Optimized for specific schema
- CSV mode: General-purpose, flexible
- Shared core: agent, tools, validation
- Conditional initialization: clean separation

**Alternative Considered:**
- Single mode with dynamic detection
- **Rejected:** Too complex, harder to optimize

---

## Testing Strategy

### Unit Tests

```bash
# SQL validation
pytest tests/test_sql_generation.py

# Tool functionality
pytest tests/test_tools.py

# Language detection
pytest tests/test_multilingual.py
```

### Integration Tests

```python
def test_end_to_end_query():
    agent = OlistAgent(db_path="test_data/test.duckdb")
    
    result = agent.process_query("Show me top 5 categories")
    
    assert result["success"] == True
    assert result["data"] is not None
    assert len(result["data"]) == 5
    assert "chart" in result
```

### Manual Testing

```bash
# Start app
streamlit run app/main.py

# Test scenarios:
1. Simple aggregation: "Total revenue?"
2. Time-series: "Monthly sales trend"
3. Ranking: "Top 10 sellers"
4. Complex: "Compare Q1 2017 vs Q1 2018 by category"
5. Multilingual: "Mostre-me as vendas"
6. CSV mode: Upload test CSVs
```

---

## Deployment Considerations

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your_gemini_key
# OR
GROQ_API_KEY=your_groq_key

# Optional
OLLAMA_BASE_URL=http://localhost:11434
HUGGINGFACE_API_KEY=your_hf_key
```

### Resource Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 2GB
- Disk: 500MB (includes DuckDB)

**Recommended:**
- CPU: 4 cores
- RAM: 4GB
- Disk: 1GB

### Scaling

**Vertical:**
- Increase Streamlit workers
- Use faster LLM (Groq)
- Add caching layer (Redis)

**Horizontal:**
- Deploy multiple instances
- Load balancer in front
- Shared DuckDB (read-only)

---

## Future Enhancements

### Planned Features

1. **Vector Search**
   - Embed schema descriptions
   - Semantic column matching
   - Better context retrieval

2. **Multi-Modal Charts**
   - Scatter plots (2 metrics)
   - Heat maps (2 dimensions + metric)
   - Geographic maps (lat/lon)
   - Sankey diagrams (flows)

3. **Query History**
   - Save favorite queries
   - Share with team
   - Export to dashboard

4. **Scheduling**
   - Recurring reports
   - Email delivery
   - Slack integration

5. **Fine-Tuned Models**
   - Train on e-commerce schemas
   - Better SQL accuracy
   - Domain-specific vocabulary

6. **Real-Time Data**
   - Streaming ingestion
   - Live dashboards
   - Change data capture

7. **Multi-User**
   - Authentication
   - Role-based access
   - Query audit logs

---

## Troubleshooting Guide

### Common Issues

**1. "Module not found"**
```bash
pip install -r requirements.txt
```

**2. "Database not found"**
```bash
python scripts/build_duckdb.py
```

**3. "API quota exceeded"**
- Switch to Groq (faster limits)
- Or use Ollama (no limits)

**4. "SQL validation failed"**
- Check table names in error message
- Verify against allowed_tables
- Check for typos in column names

**5. "Language detection wrong"**
- Query too short (<15 chars)
- Try more explicit language
- Check langdetect is installed

**6. "Charts not rendering"**
- Check Plotly is installed
- Verify data has values
- Try different chart type

**7. "CSV mode upload fails"**
- Check file size (<200MB)
- Verify CSV format (UTF-8)
- Check for special characters

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check agent state
agent = st.session_state.agent
print(agent.context)
print(agent.get_context())

# Inspect SQL
result = agent.process_query(query)
print(result["sql"])
```

---

## References

### Key Technologies

- **Streamlit**: https://streamlit.io/
- **DuckDB**: https://duckdb.org/
- **Google Gemini**: https://ai.google.dev/
- **Groq**: https://console.groq.com/
- **Ollama**: https://ollama.ai/
- **Plotly**: https://plotly.com/python/
- **langdetect**: https://pypi.org/project/langdetect/

### Dataset

- **Olist Brazilian E-commerce**: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/

### Papers & Articles

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG)
- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- "ReAct: Synergizing Reasoning and Acting in Language Models"

---

## Appendix

### SQL Views

**order_facts:**
```sql
CREATE VIEW order_facts AS
SELECT 
    o.order_id,
    o.customer_id,
    o.order_purchase_timestamp as purchase_ts,
    YEAR(o.order_purchase_timestamp) as purchase_year,
    QUARTER(o.order_purchase_timestamp) as purchase_quarter,
    MONTH(o.order_purchase_timestamp) as purchase_month,
    o.order_delivered_customer_date as delivery_date,
    o.order_estimated_delivery_date as estimated_delivery_date,
    o.order_status,
    c.customer_city,
    c.customer_state
FROM olist_orders_dataset o
LEFT JOIN olist_customers_dataset c ON o.customer_id = c.customer_id;
```

**order_item_facts:**
```sql
CREATE VIEW order_item_facts AS
SELECT 
    oi.order_id,
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value,
    oi.price + oi.freight_value as total_value,
    p.product_category_name,
    t.product_category_name_english as category_name_en,
    p.product_photos_qty,
    p.product_weight_g,
    s.seller_city,
    s.seller_state
FROM olist_order_items_dataset oi
LEFT JOIN olist_products_dataset p ON oi.product_id = p.product_id
LEFT JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
LEFT JOIN olist_sellers_dataset s ON oi.seller_id = s.seller_id;
```

### Metrics YAML

```yaml
gmv:
  name: "Gross Merchandise Value"
  description: "Total value of all orders including shipping"
  sql_template: |
    SELECT 
      SUM(total_value) as gmv,
      COUNT(DISTINCT order_id) as orders
    FROM order_item_facts
    {where_clause}
  required_tables:
    - order_item_facts

aov:
  name: "Average Order Value"
  description: "Average value per order"
  sql_template: |
    SELECT 
      AVG(order_total) as aov
    FROM (
      SELECT 
        order_id,
        SUM(total_value) as order_total
      FROM order_item_facts
      {where_clause}
      GROUP BY order_id
    )
  required_tables:
    - order_item_facts

repeat_rate:
  name: "Customer Repeat Purchase Rate"
  description: "Percentage of customers who made more than one purchase"
  sql_template: |
    SELECT 
      ROUND(100.0 * SUM(CASE WHEN order_count > 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as repeat_rate
    FROM (
      SELECT 
        customer_id,
        COUNT(DISTINCT order_id) as order_count
      FROM order_facts
      {where_clause}
      GROUP BY customer_id
    )
  required_tables:
    - order_facts
```

---

**End of Architecture Documentation**

For quick start guide, see [README.md](README.md).

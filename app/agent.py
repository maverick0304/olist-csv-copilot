"""
Agent - Orchestrates query planning, tool routing, and response generation
Uses Gemini for natural language understanding and SQL generation
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

from app.tools.sql_tool import SQLTool, SQLValidationError
from app.tools.viz_tool import VizTool
from app.tools.glossary_tool import GlossaryTool
from app.tools.pandas_tool import PandasTool
from app.tools.translate_tool import TranslateTool
from app.tools.rag_tool import RAGTool
from app.llm import LLMProvider, get_default_provider, create_llm_provider

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language detection for multilingual support
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available - multilingual support disabled")


class OlistAgent:
    """Main agent for orchestrating analytics queries - supports both Olist and custom CSV modes"""
    
    def __init__(
        self,
        db_path: Path = None,
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 2,
        csv_mode: bool = False,
        csv_analyzer=None,
        data_profiler=None
    ):
        """
        Initialize agent with tools and LLM
        
        Args:
            db_path: Path to DuckDB database (for Olist mode)
            llm_provider: LLM provider ("groq" or "gemini", auto-detects if None)
            api_key: API key for LLM provider (or from env)
            max_retries: Maximum SQL retry attempts
            csv_mode: If True, operates in custom CSV mode
            csv_analyzer: CSVAnalyzer instance (required if csv_mode=True)
            data_profiler: Optional DataProfiler instance for CSV mode
        """
        # Initialize LLM provider
        if llm_provider:
            self.llm = create_llm_provider(llm_provider, api_key=api_key)
        else:
            self.llm = get_default_provider()
        
        logger.info(f"Using LLM provider: {self.llm.__class__.__name__}")
        
        # Set mode
        self.csv_mode = csv_mode
        self.csv_analyzer = csv_analyzer
        self.data_profiler = data_profiler
        
        # Initialize tools
        if csv_mode:
            if not csv_analyzer:
                raise ValueError("csv_analyzer required for CSV mode")
            # Use CSV analyzer's connection for SQL tool
            self.sql_tool = SQLTool(db_path=None, read_only=True)
            self.sql_tool.conn = csv_analyzer.conn
            self.sql_tool.allowed_tables = set(csv_analyzer.get_table_list())
            logger.info(f"CSV Mode: {len(self.sql_tool.allowed_tables)} tables loaded")
        else:
            # Normal Olist mode
            self.sql_tool = SQLTool(db_path, read_only=True)
        
        self.viz_tool = VizTool()
        self.glossary_tool = GlossaryTool() if not csv_mode else None
        self.pandas_tool = PandasTool()
        self.translate_tool = TranslateTool()
        self.rag_tool = RAGTool() if not csv_mode else None  # RAG only for Olist mode
        # Anomaly detection removed - was causing issues
        
        # Load prompts (different for CSV vs Olist mode)
        if csv_mode:
            self.system_prompt = self._load_csv_system_prompt()
            self.few_shots = self._load_csv_few_shots()
        else:
            self.system_prompt = self._load_system_prompt()
            self.few_shots = self._load_few_shots()
        
        # Enhanced session context for better conversation tracking
        self.context = {
            "date_range": None,
            "category_filter": None,
            "last_metric": None,
            "conversation_history": [],
            "user_language": None,
            # New: Enhanced context tracking
            "last_query": None,
            "last_sql": None,
            "last_result": None,
            "last_data": None,
            "entities_mentioned": {
                "categories": set(),
                "years": set(),
                "metrics": set(),
                "states": set()
            },
            "conversation_summary": ""  # LLM-generated summary of conversation so far
        }

        self.max_retries = max_retries
        
        # Language mapping for multilingual support (35+ languages)
        self.lang_map = {
            'en': 'English',
            'pt': 'Portuguese (Brazilian)',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'hi': 'Hindi',
            'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'ru': 'Russian',
            'it': 'Italian',
            'nl': 'Dutch',
            'tr': 'Turkish',
            'vi': 'Vietnamese',
            'th': 'Thai',
            'id': 'Indonesian',
            'pl': 'Polish',
            'sv': 'Swedish',
            'da': 'Danish',
            'fi': 'Finnish',
            'no': 'Norwegian',
            'cs': 'Czech',
            'ro': 'Romanian',
            'el': 'Greek',
            'he': 'Hebrew',
            'bn': 'Bengali',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'pa': 'Punjabi',
            'ur': 'Urdu'
        }

        logger.info("OlistAgent initialized successfully")
    
    def _load_system_prompt(self) -> str:
        """Load hierarchical multi-layer prompt"""
        base_path = Path(__file__).parent / "prompts"
        
        # Define all layers
        context_path = base_path / "context_wrapper.txt"
        schema_path = base_path / "complete_schema.txt"
        rules_path = base_path / "sql_generation_v2.txt"
        fallback_path = base_path / "system.txt"
        
        prompt_parts = []
        
        # Layer 1: Business context wrapper
        if context_path.exists():
            prompt_parts.append(context_path.read_text(encoding='utf-8'))
            logger.info("✅ Layer 1: Business context loaded")
        
        # Layer 2: Complete schema with all fields
        if schema_path.exists():
            prompt_parts.append(schema_path.read_text(encoding='utf-8'))
            logger.info("✅ Layer 2: Complete schema loaded (87 fields)")
        
        # Layer 3: SQL rules & examples
        if rules_path.exists():
            prompt_parts.append(rules_path.read_text(encoding='utf-8'))
            logger.info("✅ Layer 3: SQL generation rules loaded")
        elif fallback_path.exists():
            prompt_parts.append(fallback_path.read_text(encoding='utf-8'))
            logger.warning("Using fallback system.txt")
        else:
            logger.warning("No prompts found, using minimal default")
            return "You are a SQL query generator for e-commerce analytics."
        
        # Combine all layers with clear separators
        separator = "\n\n" + "="*80 + "\n\n"
        combined = separator.join(prompt_parts)
        
        words = len(combined.split())
        logger.info(f"📊 Total prompt: {len(combined):,} chars, ~{words:,} words")
        
        return combined
    
    def _load_few_shots(self) -> List[Dict]:
        """Load few-shot examples from file"""
        few_shots_path = Path(__file__).parent / "prompts" / "few_shots.json"
        if few_shots_path.exists():
            with open(few_shots_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("few_shots", [])
        else:
            logger.warning("Few-shot examples not found")
            return []
    
    def _detect_language(self, text: str) -> str:
        """
        Robust language detection with confidence checks
        
        Args:
            text: User query text
            
        Returns:
            Full language name (e.g., "Portuguese (Brazilian)")
            Defaults to "English" if detection fails or confidence is low
        """
        if not LANGDETECT_AVAILABLE:
            return 'English'
        
        # Short queries often misdetect - default to English
        if len(text.strip()) < 15:
            logger.debug(f"Query too short for reliable detection, defaulting to English")
            return 'English'
        
        try:
            # Use detect_langs for confidence scores
            from langdetect import detect_langs
            
            # Get all language probabilities
            langs = detect_langs(text)
            
            if not langs:
                return 'English'
            
            # Get top detection
            top_lang = langs[0]
            confidence = top_lang.prob
            lang_code = top_lang.lang
            
            # Only accept non-English if confidence is high (>0.8)
            # This prevents misdetection of short English queries
            if lang_code != 'en' and confidence < 0.8:
                logger.debug(f"Low confidence ({confidence:.2f}) for {lang_code}, defaulting to English")
                return 'English'
            
            detected = self.lang_map.get(lang_code, 'English')
            logger.debug(f"Detected {detected} with {confidence:.2f} confidence")
            
            return detected
            
        except (LangDetectException, Exception) as e:
            logger.debug(f"Language detection failed: {e}")
            return 'English'  # Safe fallback
    
    def _get_user_language(self, user_query: str) -> str:
        """
        Get user's language - always detects to support language switching
        
        Always detects language for each query (~5ms overhead per query).
        This allows users to switch languages mid-session.
        
        Args:
            user_query: Current user query
            
        Returns:
            Full language name
        """
        # Always detect to support language switching in same session
        detected_lang = self._detect_language(user_query)
        
        # Log only if language changed from previous query
        prev_lang = self.context.get("user_language")
        if detected_lang != prev_lang:
            logger.info(f"🌍 Detected user language: {detected_lang}")
        
        # Update context (for reference, not caching)
        self.context["user_language"] = detected_lang
        
        return detected_lang
    
    def _generate_followup_questions(self, user_query: str, data, plan: Dict) -> List[str]:
        """
        Generate contextual follow-up questions based on current query and results
        
        Args:
            user_query: Original user query
            data: Query results DataFrame
            plan: Query plan
            
        Returns:
            List of 3-5 relevant follow-up questions
        """
        user_language = self._get_user_language(user_query)
        
        # Build context-aware prompt
        data_summary = data.head(5).to_string(index=False) if len(data) > 0 else "No data"
        
        followup_prompt = f"""You are an e-commerce analytics assistant.

IMPORTANT: Generate follow-up questions in {user_language}.

User just asked: "{user_query}"

Results (top 5 rows):
{data_summary}

Total rows: {len(data)}

Recent conversation context:
- Last query: {self.context.get('last_query', 'None')}
- Categories mentioned: {', '.join(list(self.context['entities_mentioned']['categories'])[:3]) if self.context['entities_mentioned']['categories'] else 'None'}
- Years mentioned: {', '.join([str(y) for y in list(self.context['entities_mentioned']['years'])[:3]]) if self.context['entities_mentioned']['years'] else 'None'}

Generate 5 relevant follow-up questions the user might want to ask next.
These should be natural continuations of the conversation.

Examples of good follow-up questions:
- "Why did this happen?"
- "Show me the trend over time"
- "Compare this with last year"
- "Which sellers contributed most?"
- "Break this down by state"

Return ONLY a JSON array of 5 question strings in {user_language}, no other text:
["question 1", "question 2", "question 3", "question 4", "question 5"]
"""
        
        try:
            response = self.llm.generate(followup_prompt, temperature=0.7).strip()
            
            # Extract JSON array
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            questions = json.loads(response)
            
            # Validate it's a list
            if isinstance(questions, list):
                return questions[:5]  # Limit to 5
            else:
                logger.warning(f"Follow-up questions not a list: {questions}")
                return self._get_default_followups(user_language)
                
        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}")
            return self._get_default_followups(user_language)
    
    def _get_default_followups(self, language: str = "English") -> List[str]:
        """Get default follow-up questions based on language"""
        defaults = {
            'English': [
                "Show me the trend over time",
                "Break this down by category",
                "Compare with last year",
                "Which states had the highest values?",
                "Show me the top sellers"
            ],
            'Portuguese (Brazilian)': [
                "Mostre-me a tendência ao longo do tempo",
                "Detalhe isso por categoria",
                "Compare com o ano passado",
                "Quais estados tiveram os valores mais altos?",
                "Mostre-me os principais vendedores"
            ],
            'Hindi': [
                "समय के साथ रुझान दिखाएं",
                "इसे श्रेणी के अनुसार विभाजित करें",
                "पिछले वर्ष से तुलना करें",
                "किन राज्यों में सबसे अधिक मूल्य थे?",
                "शीर्ष विक्रेता दिखाएं"
            ],
            'Spanish': [
                "Muéstrame la tendencia a lo largo del tiempo",
                "Desglosa esto por categoría",
                "Compara con el año pasado",
                "¿Qué estados tuvieron los valores más altos?",
                "Muéstrame los principales vendedores"
            ]
        }
        
        return defaults.get(language, defaults['English'])
    
    def _extract_entities(self, user_query: str, data, plan: Dict):
        """
        Extract entities (categories, years, metrics, states) from query and results
        Updates self.context['entities_mentioned']
        """
        query_lower = user_query.lower()
        
        # Extract years
        import re
        years = re.findall(r'\b(20\d{2})\b', user_query)
        for year in years:
            self.context['entities_mentioned']['years'].add(int(year))
        
        # Extract metrics
        metrics = ['gmv', 'revenue', 'aov', 'orders', 'customers', 'sellers']
        for metric in metrics:
            if metric in query_lower:
                self.context['entities_mentioned']['metrics'].add(metric)
        
        # Extract categories from results (if data has category column)
        if len(data) > 0:
            for col in data.columns:
                if 'category' in col.lower():
                    categories = data[col].dropna().unique()
                    for cat in categories[:10]:  # Limit to 10
                        self.context['entities_mentioned']['categories'].add(str(cat))
        
        # Extract states from results
        if len(data) > 0:
            for col in data.columns:
                if 'state' in col.lower():
                    states = data[col].dropna().unique()
                    for state in states[:10]:
                        self.context['entities_mentioned']['states'].add(str(state))
    
    def _resolve_references(self, user_query: str) -> str:
        """
        Resolve pronoun references in queries using context
        
        Examples:
        - "Show me that for 2017" → "Show me [last_metric] for 2017"
        - "Compare it with electronics" → "Compare [last_category] with electronics"
        """
        query_lower = user_query.lower()
        resolved = user_query
        
        # Resolve "that" / "it" / "this"
        if any(word in query_lower for word in ['that', 'it', 'this', 'same']):
            if self.context.get('last_metric'):
                # Replace pronouns with last metric
                resolved = resolved.replace('that', f"the {self.context['last_metric']}")
                resolved = resolved.replace('it', f"the {self.context['last_metric']}")
                resolved = resolved.replace('this', f"the {self.context['last_metric']}")
        
        # Add implicit year if mentioned "for YYYY" but no explicit comparison
        if 'for ' in query_lower and any(char.isdigit() for char in user_query):
            # Already has year filter, no need to resolve
            pass
        elif self.context['entities_mentioned']['years']:
            # User might be continuing a time-based analysis
            # But don't force it - let LLM decide
            pass
        
        if resolved != user_query:
            logger.info(f"Resolved reference: '{user_query}' → '{resolved}'")
        
        return resolved

    def process_query(
        self,
        user_query: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process user query and return results
        
        Args:
            user_query: Natural language query from user
            context: Optional context from previous queries
            
        Returns:
            Dictionary with response components
        """
        logger.info(f"Processing query: {user_query}")
        
        # Step 0: Resolve pronoun references using context
        user_query = self._resolve_references(user_query)
        
        # Update context
        if context:
            self.context.update(context)
        
        # Check for special commands
        if user_query.lower().startswith("explain "):
            return self._handle_explain_metric(user_query)
        
        if "what metrics" in user_query.lower() or "available metrics" in user_query.lower():
            return self._handle_list_metrics()
        
        # Main query processing
        try:
            # Step 1: Plan the query
            plan = self._plan_query(user_query)
            logger.info(f"Query plan: {plan}")
            
            # Step 2: Check if it matches a semantic metric
            metric_sql = None
            if plan.get("metric_name"):
                metric_sql = self._get_metric_sql(plan["metric_name"], plan.get("filters", {}), user_query=user_query, plan=plan)
            
            # Step 3: Generate SQL (use metric template or generate fresh)
            if metric_sql:
                sql = metric_sql
                logger.info("Using semantic metric template")
            else:
                sql = self._generate_sql(user_query, plan)
            
            # Step 4: Execute SQL (with retry logic)
            result = self._execute_with_retry(sql)
            
            # Save successful query to RAG history (only in Olist mode)
            if result["success"] and self.rag_tool:
                self.rag_tool.save_successful_query(user_query, sql, success=True)
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": result["error"],
                    "sql": sql,
                    "suggestion": "Try rephrasing your question or being more specific about what you want to analyze."
                }
            
            data = result["data"]
            
            # Step 5: Generate insight
            insight = self._generate_insight(user_query, data, sql)
            
            # Step 6: Create visualization
            chart = None
            if len(data) > 0 and plan.get("should_visualize", True):
                try:
                    chart = self.viz_tool.create_chart(
                        data,
                        chart_type=plan.get("chart_type"),
                        title=plan.get("chart_title")
                    )
                except Exception as e:
                    logger.error(f"Chart creation failed: {e}")
            
            # Step 7: Extract entities for context
            self._extract_entities(user_query, data, plan)
            
            # Step 8: Generate follow-up questions
            followup_questions = self._generate_followup_questions(user_query, data, plan)
            
            # Step 9: Format response
            response = {
                "success": True,
                "insight": insight,
                "data": data,
                "chart": chart,
                "sql": sql,
                "row_count": len(data),
                "timestamp": datetime.now().isoformat(),
                # Follow-up suggestions
                "followup_questions": followup_questions
            }
            
            # Update context
            self._update_context(user_query, plan, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "suggestion": "An error occurred. Please try rephrasing your question."
            }
    
    def _plan_query(self, user_query: str) -> Dict:
        """
        Use LLM to plan the query
        
        Returns:
            Dictionary with query plan
        """
        planning_prompt = f"""
Analyze this user query and extract key information:

Query: "{user_query}"

Context from previous queries:
- Date range: {self.context.get('date_range', 'None')}
- Category filter: {self.context.get('category_filter', 'None')}
- Last metric: {self.context.get('last_metric', 'None')}

Return a JSON object with:
{{
  "metric_name": "name of metric if recognized (gmv, aov, repeat_rate, etc.)",
  "aggregation": "sum, avg, count, min, max",
  "group_by": ["columns to group by"],
  "filters": {{
    "date_filter": "year, quarter, date range",
    "category": "category name if specified",
    "state": "state if specified",
    "seller": "seller if specified"
  }},
  "limit": "number for top N queries",
  "order_by": "column to order by",
  "chart_type": "bar, line, pie, or scatter",
  "should_visualize": true/false
}}

Only return the JSON, no other text.
"""
        
        try:
            plan_text = self.llm.generate(planning_prompt, temperature=0.1).strip()
            
            # Extract JSON from response (might be wrapped in markdown)
            if "```json" in plan_text:
                plan_text = plan_text.split("```json")[1].split("```")[0].strip()
            elif "```" in plan_text:
                plan_text = plan_text.split("```")[1].split("```")[0].strip()
            
            plan = json.loads(plan_text)
            return plan
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Return default plan
            return {
                "metric_name": None,
                "aggregation": "sum",
                "group_by": [],
                "filters": {},
                "limit": 10,
                "should_visualize": True
            }
    
    def _generate_sql(self, user_query: str, plan: Dict) -> str:
        """
        Generate SQL query using LLM
        
        Args:
            user_query: Original user query
            plan: Query plan from planner
            
        Returns:
            SQL query string
        """
        # Build prompt with system instructions and few-shots
        # Prioritize examples that match query type
        relevant_shots = []
        query_lower = user_query.lower()
        
        # Find most relevant examples first
        for ex in self.few_shots:
            if any(keyword in query_lower for keyword in ['top', 'category', 'categories', 'gmv', 'revenue']):
                if any(keyword in ex['user_query'].lower() for keyword in ['top', 'category', 'gmv']):
                    relevant_shots.insert(0, ex)
                else:
                    relevant_shots.append(ex)
            else:
                relevant_shots.append(ex)
        
        few_shot_examples = "\n\n".join([
            f"Example {i+1}:\nQuery: {ex.get('user_query') or ex.get('question')}\nReasoning: {ex['reasoning']}\nSQL: {ex['sql']}"
            for i, ex in enumerate(relevant_shots[:8])  # Use top 8 most relevant
        ])
        
        # Build simplified, focused prompt
        sql_prompt = f"""{self.system_prompt}

USER QUERY: "{user_query}"

FILTERS:
- Year: {plan.get('filters', {}).get('date_filter', 'none')}
- Category: {plan.get('filters', {}).get('category', 'none')}
- Limit: {plan.get('limit', 10)}

Generate SQL following the template and examples above. Return ONLY SQL:
"""
        
        try:
            sql = self.llm.generate(sql_prompt, temperature=0.1).strip()
            
            # Clean up the SQL
            sql = self._clean_sql(sql)
            
            # CRITICAL FIX: Validate and auto-repair common mistakes
            sql = self._validate_and_repair_sql(sql, user_query, plan)
            
            logger.info(f"Generated SQL: {sql[:200]}...")
            return sql
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            # Fallback to simple query
            return "SELECT * FROM order_summary LIMIT 10"
    
    def _clean_sql(self, sql: str) -> str:
        """Clean up generated SQL"""
        # Remove markdown code blocks
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0].strip()
        
        # Remove trailing semicolons (sql_tool adds them)
        sql = sql.rstrip(';').strip()
        
        return sql
    
    def _validate_and_repair_sql(self, sql: str, user_query: str, plan: Dict) -> str:
        """
        Validate SQL and auto-repair common mistakes
        
        Critical fixes:
        1. Time-series queries missing time columns (trend, monthly, etc.)
        2. Ranking queries missing GROUP BY
        """
        sql_upper = sql.upper()
        query_lower = user_query.lower()
        
        # CRITICAL FIX 1: Time-series queries MUST have time columns
        is_time_series = any(word in query_lower for word in [
            'trend', 'monthly', 'by month', 'per month', 'each month',
            'quarterly', 'by quarter', 'yearly', 'by year',
            'over time', 'time series'
        ])
        
        has_aggregation = any(func in sql_upper for func in [
            'SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN('
        ])
        
        if is_time_series and has_aggregation:
            has_time_cols = any(col in sql_upper for col in [
                'PURCHASE_YEAR', 'PURCHASE_QUARTER', 'PURCHASE_MONTH'
            ])
            
            if not has_time_cols:
                logger.warning("🔧 AUTO-REPAIR: Time-series query missing time columns!")
                
                # Extract category
                category = None
                for cat in ['electro', 'health', 'beauty', 'furniture', 'sports', 'bed', 'bath']:
                    if cat in query_lower:
                        category = cat
                        break
                
                if category:
                    logger.info(f"Rebuilding as monthly trend for '{category}'")
                    sql = f"""SELECT 
    of.purchase_year,
    of.purchase_month,
    SUM(oif.total_value) as revenue,
    COUNT(DISTINCT oif.order_id) as orders
FROM order_item_facts oif
JOIN order_facts of ON oif.order_id = of.order_id
WHERE LOWER(oif.category_name_en) LIKE '%{category}%'
GROUP BY of.purchase_year, of.purchase_month
ORDER BY of.purchase_year, of.purchase_month"""
                    logger.info("✅ SQL rebuilt with time columns")
                    return sql
        
        # CRITICAL FIX 2: Ranking queries need GROUP BY
        is_ranking_query = any(keyword in query_lower for keyword in [
            'top ', 'bottom ', 'best ', 'worst ', 'highest ', 'lowest ',
            'rank', 'compare', 'by category', 'by seller', 'by state',
            'by product', 'by payment'
        ])
        
        # CRITICAL: If it's a ranking query with aggregation but NO GROUP BY → FIX IT
        if is_ranking_query and has_aggregation and 'GROUP BY' not in sql_upper:
            logger.warning(f"SQL missing GROUP BY for ranking query! Auto-repairing...")
            
            # Detect what dimension to group by
            dimension = None
            
            if 'categor' in query_lower:
                dimension = 'category_name_en'
                table_alias = 'oif'
            elif 'seller' in query_lower:
                dimension = 'seller_id'
                table_alias = 'oif'
            elif 'state' in query_lower:
                if 'customer' in query_lower:
                    dimension = 'customer_state'
                    table_alias = 'os'
                else:
                    dimension = 'seller_state'
                    table_alias = 'oif'
            elif 'product' in query_lower:
                dimension = 'product_id'
                table_alias = 'oif'
            elif 'payment' in query_lower:
                dimension = 'payment_type'
                table_alias = 'os'
            
            if dimension:
                # Build proper SQL from scratch using few-shot template
                year_match = None
                for word in query_lower.split():
                    if word.isdigit() and len(word) == 4:
                        year_match = word
                        break
                
                # Use the template from few_shots
                if 'categor' in query_lower and 'gmv' in query_lower or 'revenue' in query_lower:
                    limit = plan.get('limit', 5)
                    
                    if year_match:
                        sql = f"""SELECT 
    oif.category_name_en as category,
    SUM(oif.total_value) as gmv,
    COUNT(DISTINCT oif.order_id) as orders
FROM order_item_facts oif
JOIN order_facts of ON oif.order_id = of.order_id
WHERE YEAR(of.purchase_ts) = {year_match}
GROUP BY oif.category_name_en
ORDER BY gmv DESC
LIMIT {limit}"""
                    else:
                        sql = f"""SELECT 
    category_name_en as category,
    SUM(total_value) as gmv,
    COUNT(DISTINCT order_id) as orders
FROM order_item_facts
GROUP BY category_name_en
ORDER BY gmv DESC
LIMIT {limit}"""
                    
                    logger.info(f"Auto-repaired SQL with GROUP BY {dimension}")
        
        return sql
    
    def _execute_with_retry(self, sql: str) -> Dict:
        """
        Execute SQL with retry logic for error recovery
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Result dictionary from sql_tool
        """
        result = self.sql_tool.execute_safe(sql)
        
        # If failed, try to repair and retry
        if not result["success"] and self.max_retries > 0:
            logger.info(f"SQL failed, attempting repair. Error: {result['error']}")
            
            repaired_sql = self.sql_tool.repair_sql(sql, result["error"])
            
            if repaired_sql:
                logger.info(f"Trying repaired SQL: {repaired_sql[:200]}...")
                result = self.sql_tool.execute_safe(repaired_sql)
        
        return result
    
    def _generate_insight(self, user_query: str, data, sql: str) -> str:
        """
        Generate natural language insight from results
        
        Automatically detects user's language and responds in that language.
        Uses cached language for speed (only 5ms overhead on first query).

        Args:
            user_query: Original query
            data: Query results DataFrame
            sql: Executed SQL

        Returns:
            Insight text in user's language
        """
        if len(data) == 0:
            # Get user's language for fallback message
            user_language = self._get_user_language(user_query)
            fallback_messages = {
                'English': "No results found for your query.",
                'Portuguese (Brazilian)': "Nenhum resultado encontrado para sua consulta.",
                'Spanish': "No se encontraron resultados para su consulta.",
                'Hindi': "आपकी क्वेरी के लिए कोई परिणाम नहीं मिला।",
                'French': "Aucun résultat trouvé pour votre requête.",
                'German': "Keine Ergebnisse für Ihre Anfrage gefunden.",
                'Chinese (Simplified)': "未找到您查询的结果。",
                'Japanese': "クエリの結果が見つかりませんでした。",
                'Korean': "쿼리에 대한 결과를 찾을 수 없습니다.",
                'Arabic': "لم يتم العثور على نتائج لاستعلامك.",
            }
            return fallback_messages.get(user_language, "No results found for your query.")

        # Detect user's language (cached after first call - 0ms overhead)
        user_language = self._get_user_language(user_query)

        # Convert data to string summary
        data_summary = data.head(10).to_string(index=False, max_colwidth=50)

        insight_prompt = f"""You are an e-commerce analytics expert.

CRITICAL: The user asked their question in {user_language}. You MUST respond in {user_language}.

User Query (in {user_language}): "{user_query}"

SQL Executed:
{sql}

Results (first 10 rows):
{data_summary}

Total rows: {len(data)}

Generate a clear, actionable insight in {user_language} with specific numbers and trends.
Start directly with the finding, no preamble.
Use the same language style and formality level as the user's question.
"""

        try:
            insight = self.llm.generate(insight_prompt, temperature=0.3).strip()
            return insight
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            # Language-aware fallback
            fallback_messages = {
                'English': f"Found {len(data)} results for your query.",
                'Portuguese (Brazilian)': f"Encontrados {len(data)} resultados para sua consulta.",
                'Spanish': f"Se encontraron {len(data)} resultados para su consulta.",
                'Hindi': f"आपकी क्वेरी के लिए {len(data)} परिणाम मिले।",
                'French': f"Trouvé {len(data)} résultats pour votre requête.",
                'German': f"{len(data)} Ergebnisse für Ihre Anfrage gefunden.",
                'Chinese (Simplified)': f"找到 {len(data)} 个结果。",
                'Japanese': f"{len(data)} 件の結果が見つかりました。",
                'Korean': f"{len(data)} 개의 결과를 찾았습니다.",
                'Arabic': f"تم العثور على {len(data)} نتيجة.",
            }
            return fallback_messages.get(user_language, f"Found {len(data)} results for your query.")
    
    def _get_metric_sql(self, metric_name: str, filters: Dict, user_query: Optional[str] = None, plan: Optional[Dict] = None) -> Optional[str]:
        """Get SQL from semantic metric definition

        Automatically skips templates when they don't fit (e.g. time-series trends).
        """
        # Glossary tool only available in Olist mode
        if not self.glossary_tool:
            return None
            
        metric = self.glossary_tool.get_metric(metric_name)
        
        if not metric or "sql_template" not in metric:
            return None

        # If query needs time grouping or trend, fallback to LLM generation
        if plan:
            group_by = plan.get("group_by") or []
            if any(g in group_by for g in ["date", "month", "quarter", "year"]):
                logger.info("Skipping metric template: time grouping required")
                return None

        if user_query:
            ql = user_query.lower()
            if any(word in ql for word in ["trend", "monthly", "by month", "per month", "quarterly", "over time", "time series"]):
                logger.info("Skipping metric template: trend/time-series query detected")
                return None
            if any(word in ql for word in ["top ", "top-", "rank", "best", "highest", "lowest", "compare"]):
                logger.info("Skipping metric template: ranking query detected")
                return None

        required_tables = metric.get("required_tables", [])

        # If date filter requested but template lacks time tables, skip
        if filters.get("date_filter"):
            if ("order_facts" not in required_tables) and ("order_summary" not in required_tables):
                if "order_item_facts" in required_tables:
                    logger.info("Skipping metric template: date filter needs time table join")
                    return None
        
        # Build WHERE clause from filters
        where_conditions = []
        
        if filters.get("date_filter"):
            # Parse date filter
            date_filter = filters["date_filter"]
            if "year" in date_filter.lower():
                year = date_filter.split()[-1]
                where_conditions.append(f"purchase_year = {year}")
            elif "quarter" in date_filter.lower():
                # Extract Q and year
                parts = date_filter.split()
                for part in parts:
                    if part.startswith("Q"):
                        quarter = part[1]
                        where_conditions.append(f"purchase_quarter = {quarter}")
        
        if filters.get("category"):
            category = filters["category"]
            where_conditions.append(f"LOWER(category_name_en) LIKE '%{category.lower()}%'")
        
        if filters.get("state"):
            state = filters["state"]
            where_conditions.append(f"customer_state = '{state}'")
        
        # Build WHERE clause
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        else:
            where_clause = ""
        
        # Format template
        try:
            sql = metric["sql_template"].format(
                where_clause=where_clause,
                and_where_clause=("AND " + " AND ".join(where_conditions)) if where_conditions else ""
            )
            return sql
        except Exception as e:
            logger.error(f"Failed to format metric SQL: {e}")
            return None
    
    def _handle_explain_metric(self, query: str) -> Dict:
        """Handle 'Explain [metric]' queries"""
        # Glossary tool only available in Olist mode
        if not self.glossary_tool:
            return {
                "success": False,
                "error": "Metric explanations are only available in Olist mode",
                "suggestion": "Try asking questions about your uploaded CSV data instead"
            }
            
        metric_name = query.lower().replace("explain", "").strip()
        explanation = self.glossary_tool.explain_metric(metric_name)
        
        return {
            "success": True,
            "insight": explanation,
            "data": None,
            "chart": None,
            "sql": None,
            "type": "explanation"
        }
    
    def _handle_list_metrics(self) -> Dict:
        """Handle 'What metrics are available' queries"""
        # Glossary tool only available in Olist mode
        if not self.glossary_tool:
            return {
                "success": False,
                "error": "Metric listing is only available in Olist mode",
                "suggestion": "Try asking questions about your uploaded CSV data instead"
            }
            
        metrics_summary = self.glossary_tool.get_all_metrics_summary()
        
        return {
            "success": True,
            "insight": metrics_summary,
            "data": None,
            "chart": None,
            "sql": None,
            "type": "metrics_list"
        }
    
    def _update_context(self, user_query: str, plan: Dict, response: Dict):
        """Update enhanced session context with query information"""
        # Update context from plan
        if plan.get("filters", {}).get("date_filter"):
            self.context["date_range"] = plan["filters"]["date_filter"]
        
        if plan.get("filters", {}).get("category"):
            self.context["category_filter"] = plan["filters"]["category"]
        
        if plan.get("metric_name"):
            self.context["last_metric"] = plan["metric_name"]
        
        # Store last query, SQL, and result for reference resolution
        self.context["last_query"] = user_query
        self.context["last_sql"] = response.get("sql")
        self.context["last_result"] = {
            "insight": response.get("insight"),
            "row_count": response.get("row_count"),
            "timestamp": response.get("timestamp")
        }
        self.context["last_data"] = response.get("data")  # Store DataFrame for "show me more" queries
        
        # Add to conversation history (enhanced with results)
        self.context["conversation_history"].append({
            "query": user_query,
            "plan": plan,
            "sql": response.get("sql"),
            "insight": response.get("insight"),
            "row_count": response.get("row_count"),
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 queries in history (increased for better context)
        if len(self.context["conversation_history"]) > 10:
            self.context["conversation_history"] = self.context["conversation_history"][-10:]
    
    def get_context(self) -> Dict:
        """Get current session context"""
        return self.context.copy()
    
    def reset_context(self):
        """Reset session context"""
        self.context = {
            "date_range": None,
            "category_filter": None,
            "last_metric": None,
            "conversation_history": [],
            "user_language": None,
            "last_query": None,
            "last_sql": None,
            "last_result": None,
            "last_data": None,
            "entities_mentioned": {
                "categories": set(),
                "years": set(),
                "metrics": set(),
                "states": set()
            },
            "conversation_summary": ""
        }
        logger.info("Context reset")
    
    def _load_csv_system_prompt(self) -> str:
        """
        Load system prompt for CSV mode (dynamically generated from schema)
        """
        if not self.csv_analyzer:
            return "You are a data analyst. Answer questions about the uploaded CSV data."
        
        try:
            from app.prompts.csv_prompt_generator import CSVPromptGenerator
            
            generator = CSVPromptGenerator(self.csv_analyzer, self.data_profiler)
            prompt = generator.generate_schema_prompt()
            
            logger.info(f"Generated CSV system prompt: {len(prompt)} characters")
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to generate CSV prompt: {e}")
            # Fallback to basic prompt
            return self.csv_analyzer.get_schema_summary()
    
    def _load_csv_few_shots(self) -> List[Dict]:
        """
        Load few-shot examples for CSV mode (dynamically generated)
        """
        if not self.csv_analyzer:
            return []
        
        try:
            from app.prompts.csv_prompt_generator import CSVPromptGenerator
            
            generator = CSVPromptGenerator(self.csv_analyzer, self.data_profiler)
            examples = generator.generate_few_shot_examples()
            
            logger.info(f"Generated {len(examples)} CSV few-shot examples")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to generate CSV few-shots: {e}")
            return []
    
    def close(self):
        """Clean up resources"""
        if self.sql_tool:
            self.sql_tool.close()
        logger.info("Agent closed")


"""
RAG Tool - Retrieval-Augmented Generation for better SQL generation
Improves schema understanding and query accuracy
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Try to import numpy, but make it optional
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not installed, some RAG features disabled")


class RAGTool:
    """
    RAG tool for retrieving relevant schema information and past queries
    Uses embeddings for semantic search
    """
    
    def __init__(
        self,
        schema_path: Optional[Path] = None,
        query_history_path: Optional[Path] = None,
        use_embeddings: bool = True
    ):
        """
        Initialize RAG tool
        
        Args:
            schema_path: Path to schema documentation
            query_history_path: Path to query history
            use_embeddings: Whether to use embeddings (requires sentence-transformers)
        """
        self.schema_path = schema_path or Path(__file__).parent.parent / "semantic" / "schema.md"
        self.query_history_path = query_history_path or Path(__file__).parent.parent / "memory" / "query_history.json"
        self.use_embeddings = use_embeddings
        
        # Initialize embedding model if available
        self.embedder = None
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("RAG embedder initialized")
            except ImportError:
                logger.warning("sentence-transformers not installed, using keyword search")
                self.use_embeddings = False
        
        # Load knowledge base
        self.schema_docs = self._load_schema_docs()
        self.query_history = self._load_query_history()
        
        # Build vector index
        if self.use_embeddings and self.embedder:
            self._build_vector_index()
    
    def _load_schema_docs(self) -> List[Dict]:
        """Load and parse schema documentation"""
        docs = []
        
        # Table/column descriptions
        schema_content = {
            "order_summary": {
                "description": "Pre-aggregated order-level data with customer, payment, delivery metrics",
                "columns": {
                    "order_id": "Unique order identifier",
                    "customer_id": "Customer identifier",
                    "purchase_year": "Year of purchase",
                    "purchase_quarter": "Quarter (1-4)",
                    "purchase_month": "Month (1-12)",
                    "order_gmv": "Gross merchandise value (order total)",
                    "payment_value": "Payment amount",
                    "review_score": "Customer review (1-5)",
                    "is_on_time": "Whether delivered on time (boolean)",
                    "customer_state": "Customer state",
                    "customer_city": "Customer city",
                },
                "use_for": ["order metrics", "AOV", "time-series", "customer analysis"]
            },
            "order_item_facts": {
                "description": "Individual line items with product, seller, and category details",
                "columns": {
                    "order_id": "Order identifier",
                    "product_id": "Product identifier",
                    "seller_id": "Seller identifier",
                    "category_name_en": "Product category (English)",
                    "price": "Item price",
                    "freight_value": "Shipping cost",
                    "total_value": "Price + freight",
                    "seller_city": "Seller city",
                    "seller_state": "Seller state",
                },
                "use_for": ["category analysis", "product analysis", "seller analysis", "GMV by category"]
            },
            "order_facts": {
                "description": "Order-level data with timestamps and delivery info",
                "columns": {
                    "order_id": "Order identifier",
                    "customer_id": "Customer identifier",
                    "purchase_ts": "Purchase timestamp",
                    "delivered_ts": "Delivery timestamp",
                    "payment_value": "Payment amount",
                    "review_score": "Review score",
                },
                "use_for": ["date filtering", "delivery analysis", "time-series with dates"]
            },
            "product_dim": {
                "description": "Product master data",
                "columns": {
                    "product_id": "Product identifier",
                    "category_name_en": "Category name",
                    "product_weight_g": "Weight in grams",
                },
                "use_for": ["product details", "physical dimensions"]
            },
            "customer_dim": {
                "description": "Customer master data",
                "columns": {
                    "customer_id": "Customer identifier",
                    "customer_city": "City",
                    "customer_state": "State",
                },
                "use_for": ["customer location", "geographic analysis"]
            },
            "seller_dim": {
                "description": "Seller master data",
                "columns": {
                    "seller_id": "Seller identifier",
                    "seller_city": "City",
                    "seller_state": "State",
                },
                "use_for": ["seller location", "seller analysis"]
            }
        }
        
        # Convert to documents
        for table_name, info in schema_content.items():
            # Table-level doc
            docs.append({
                "type": "table",
                "table": table_name,
                "text": f"Table {table_name}: {info['description']}. Use for: {', '.join(info['use_for'])}",
                "metadata": {"description": info['description'], "use_cases": info['use_for']}
            })
            
            # Column-level docs
            for col_name, col_desc in info['columns'].items():
                docs.append({
                    "type": "column",
                    "table": table_name,
                    "column": col_name,
                    "text": f"{table_name}.{col_name}: {col_desc}",
                    "metadata": {"description": col_desc}
                })
        
        logger.info(f"Loaded {len(docs)} schema documents")
        return docs
    
    def _load_query_history(self) -> List[Dict]:
        """Load past successful queries"""
        if not self.query_history_path.exists():
            return []
        
        try:
            with open(self.query_history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            logger.info(f"Loaded {len(history)} queries from history")
            return history
        except Exception as e:
            logger.error(f"Failed to load query history: {e}")
            return []
    
    def _build_vector_index(self):
        """Build vector index from documents"""
        if not self.embedder:
            return
        
        # Embed schema docs
        schema_texts = [doc['text'] for doc in self.schema_docs]
        self.schema_embeddings = self.embedder.encode(schema_texts, show_progress_bar=False)
        
        # Embed query history
        if self.query_history:
            history_texts = [q['user_query'] for q in self.query_history]
            self.history_embeddings = self.embedder.encode(history_texts, show_progress_bar=False)
        else:
            self.history_embeddings = None
        
        logger.info("Vector index built")
    
    def retrieve_schema_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve relevant schema information for a query
        
        Args:
            query: User's natural language query
            top_k: Number of results to retrieve
            
        Returns:
            Formatted context string
        """
        if self.use_embeddings and self.embedder and HAS_NUMPY:
            # Semantic search using embeddings
            query_embedding = self.embedder.encode([query], show_progress_bar=False)[0]
            
            # Calculate similarities
            similarities = np.dot(self.schema_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = [self.schema_docs[i] for i in top_indices]
        else:
            # Keyword search fallback
            results = self._keyword_search(query, self.schema_docs, top_k)
        
        # Format results
        context_parts = ["Relevant schema information:"]
        for result in results:
            if result['type'] == 'table':
                context_parts.append(f"- {result['text']}")
            elif result['type'] == 'column':
                context_parts.append(f"  • {result['text']}")
        
        return "\n".join(context_parts)
    
    def retrieve_similar_queries(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve similar past queries
        
        Args:
            query: User's natural language query
            top_k: Number of results
            
        Returns:
            Formatted similar queries with SQL
        """
        if not self.query_history:
            return ""
        
        if self.use_embeddings and self.embedder and self.history_embeddings is not None and HAS_NUMPY:
            # Semantic search
            query_embedding = self.embedder.encode([query], show_progress_bar=False)[0]
            similarities = np.dot(self.history_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = [self.query_history[i] for i in top_indices]
        else:
            # Keyword search
            results = self._keyword_search(query, self.query_history, top_k, key='user_query')
        
        if not results:
            return ""
        
        context_parts = ["Similar past queries:"]
        for i, result in enumerate(results, 1):
            context_parts.append(f"{i}. Q: {result['user_query']}")
            context_parts.append(f"   SQL: {result['sql'][:100]}...")
        
        return "\n".join(context_parts)
    
    def _keyword_search(self, query: str, documents: List[Dict], top_k: int, key: str = 'text') -> List[Dict]:
        """Simple keyword-based search fallback"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_docs = []
        for doc in documents:
            doc_text = doc.get(key, '').lower()
            doc_words = set(doc_text.split())
            
            # Simple word overlap score
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored_docs.append((overlap, doc))
        
        # Sort by score and return top k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:top_k]]
    
    def save_successful_query(self, user_query: str, sql: str, success: bool = True):
        """Save a successful query to history for future retrieval"""
        if not success:
            return
        
        # Load existing history
        history = self.query_history
        
        # Add new query
        history.append({
            "user_query": user_query,
            "sql": sql,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep last 100 queries
        history = history[-100:]
        
        # Save to file
        try:
            self.query_history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.query_history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
            
            # Update in-memory
            self.query_history = history
            
            # Rebuild embeddings if using them
            if self.use_embeddings and self.embedder:
                history_texts = [q['user_query'] for q in history]
                self.history_embeddings = self.embedder.encode(history_texts, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Failed to save query history: {e}")
    
    def get_enhanced_prompt_context(self, query: str) -> str:
        """
        Get comprehensive RAG context for a query
        
        Args:
            query: User query
            
        Returns:
            Enhanced context string to include in LLM prompt
        """
        parts = []
        
        # Schema context
        schema_context = self.retrieve_schema_context(query, top_k=5)
        if schema_context:
            parts.append(schema_context)
        
        # Similar queries
        similar_context = self.retrieve_similar_queries(query, top_k=2)
        if similar_context:
            parts.append(similar_context)
        
        return "\n\n".join(parts)


# Singleton instance
_rag_tool_instance = None

def get_rag_tool() -> RAGTool:
    """Get or create RAG tool singleton"""
    global _rag_tool_instance
    if _rag_tool_instance is None:
        _rag_tool_instance = RAGTool()
    return _rag_tool_instance


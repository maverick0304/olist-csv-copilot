"""
Olist Copilot - Streamlit UI
Natural language interface for e-commerce analytics
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Olist Copilot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sql-preview {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .example-chip {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background-color: #e3f2fd;
        border-radius: 1rem;
        cursor: pointer;
        font-size: 0.9rem;
    }
    .example-chip:hover {
        background-color: #bbdefb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if "context" not in st.session_state:
    st.session_state.context = {}

# Header with Mode Selector
col_header, col_mode = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-header">🛒 Olist Copilot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask questions about your e-commerce data in natural language</div>',
        unsafe_allow_html=True
    )
with col_mode:
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
    if st.button("📊 Switch to CSV Mode →", type="primary", use_container_width=True, help="Upload and analyze your own CSV files"):
        st.switch_page("app/pages/csv_mode.py")

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.caption("💡 The system automatically detects date ranges, granularity, and top-N from your questions!")
    st.caption("Example: \"Show me top 5 categories by revenue in 2018\"")
    
    st.divider()
    
    # User preferences (not query parameters)
    st.subheader("Display Preferences")
    show_sql = st.checkbox("Always show SQL queries", value=False, help="Show the generated SQL for every answer")
    auto_viz = st.checkbox("Auto-generate visualizations", value=True, help="Automatically create charts when appropriate")
    
    st.divider()

    # Conversation Context
    if "agent" in st.session_state:
        context = st.session_state.agent.get_context()
        
        with st.expander("🧠 Active Context"):
            # Show entities mentioned
            if context.get('entities_mentioned'):
                entities = context['entities_mentioned']
                if entities.get('years'):
                    st.caption(f"📅 Years: {', '.join([str(y) for y in list(entities['years'])[:5]])}")
                if entities.get('categories'):
                    st.caption(f"📦 Categories: {', '.join(list(entities['categories'])[:3])}")
                if entities.get('metrics'):
                    st.caption(f"📊 Metrics: {', '.join(list(entities['metrics'])[:3])}")
            
            # Show last query
            if context.get('last_query'):
                st.caption(f"💬 Last: {context['last_query'][:50]}...")

    st.divider()

    # Session info
    st.subheader("📊 Session Info")
    st.caption(f"Session ID: {st.session_state.session_id}")
    st.caption(f"Messages: {len(st.session_state.messages)}")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context = {}
        if "agent" in st.session_state:
            st.session_state.agent.reset_context()
        st.rerun()
    
    st.divider()
    
    # Help
    with st.expander("❓ Help"):
        st.markdown("""
        **How to use:**
        1. Type your question in natural language
        2. Review the SQL query and results
        3. Download data or ask follow-up questions

        **🌍 Multilingual Support:**
        - Ask in ANY language (English, Portuguese, Hindi, Spanish, etc.)
        - The system auto-detects your language and responds accordingly
        - No language selection needed!

        **Tips:**
        - Be specific about time periods
        - Ask one question at a time
        - Use follow-ups to refine results
        
        **Examples:**
        - "Top 5 categories by revenue"
        - "Average order value by quarter"
        - "Delivery performance by seller"
        """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")

with col2:
    st.subheader("📈 Quick Stats")
    
    # Placeholder metrics (will be populated by agent)
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Total Orders", "99,441", "")
        st.metric("Avg Order Value", "$137.75", "")
    with metric_col2:
        st.metric("Total Revenue", "$13.6M", "")
        st.metric("Active Sellers", "3,095", "")

# Example questions - professionally styled
st.markdown("### ⚡ Quick Start Questions")

example_questions = [
    "What are the top 5 product categories by GMV in 2018?",
    "Show me average order value by quarter",
    "Which sellers have the worst on-time delivery rate?",
    "Revenue trend by month for the Electronics category",
    "What's the customer repeat purchase rate?",
    "Compare payment methods by transaction value",
]

# Display example chips with professional styling
cols = st.columns(3)
for idx, question in enumerate(example_questions):
    with cols[idx % 3]:
        if st.button(f"▸ {question[:40]}...", key=f"example_{idx}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

st.divider()

# Chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                # Display content/insight
                st.markdown(content)
                st.markdown("")  # spacing
                
                # Display data table if present
                if "data" in message and message["data"] is not None:
                    st.dataframe(message["data"], use_container_width=True)
                    st.markdown("")  # spacing
                
                # Display chart if present
                if "chart" in message and message["chart"] is not None:
                    st.plotly_chart(
                        message["chart"], 
                        use_container_width=True,
                        key=f"chart_history_{message.get('timestamp', 0)}"
                    )
                    st.markdown("")  # spacing
                
                # Display SQL if enabled
                if "sql" in message and (show_sql or message.get("show_sql_always")):
                    with st.expander("⚙ View SQL Query"):
                        st.code(message["sql"], language="sql")
                
                # Download button
                if "download" in message:
                    st.download_button(
                        label="↓ Export as CSV",
                        data=message["download"],
                        file_name=f"olist_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key=f"download_{message.get('timestamp', 0)}"
                    )

# Handle chat input and pending queries
prompt = None

# Check for pending query from buttons
if len(st.session_state.messages) > 0:
    last_msg = st.session_state.messages[-1]
    # If last message is user and no response yet
    if last_msg["role"] == "user" and (len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "user"):
        prompt = last_msg["content"]

# Process query if we have one
if prompt:
    
    # Process with agent
    with st.chat_message("assistant"):
        with st.spinner("⚙️ Analyzing your query..."):
            try:
                # Initialize agent if not already done
                if "agent" not in st.session_state:
                    db_path = Path(__file__).parent.parent / "data" / "duckdb" / "olist.duckdb"
                    
                    if not db_path.exists():
                        st.error("Database not found. Please run: `python scripts/build_duckdb.py`")
                        st.stop()
                    
                    from app.agent import OlistAgent
                    st.session_state.agent = OlistAgent(db_path)
                
                # Get agent and process query
                agent = st.session_state.agent
                
                # Process query (system auto-detects all parameters from natural language)
                result = agent.process_query(prompt, {})
                
                if result["success"]:
                    # Display insight/description first
                    st.markdown(f"**▸ Analysis:**")
                    st.write(result['insight'])
                    st.markdown("")  # spacing
                    
                    # Display data table if present
                    if result.get("data") is not None and len(result["data"]) > 0:
                        st.markdown("**▸ Data Preview:**")
                        st.dataframe(result["data"], use_container_width=True, hide_index=True)
                        
                        # Download button with professional icon
                        csv = result["data"].to_csv(index=False)
                        st.download_button(
                            label="↓ Export as CSV",
                            data=csv,
                            file_name=f"olist_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
                        st.markdown("")  # spacing
                    
                    # Display chart if present (with description)
                    if result.get("chart") is not None and auto_viz:
                        st.markdown("**▸ Visualization:**")
                        st.plotly_chart(
                            result["chart"], 
                            use_container_width=True,
                            key=f"chart_{result.get('timestamp', datetime.now().timestamp())}"
                        )
                        st.markdown("")  # spacing
                    
                    # Display SQL if enabled
                    if result.get("sql") and (show_sql or result.get("show_sql_always", False)):
                        with st.expander("⚙ View SQL Query"):
                            st.code(result["sql"], language="sql")
                    
                    # Prepare message for history
                    message = {
                        "role": "assistant",
                        "content": result["insight"],
                        "data": result.get("data"),
                        "chart": result.get("chart"),
                        "sql": result.get("sql"),
                        "followup_questions": result.get("followup_questions"),
                        "timestamp": datetime.now().timestamp()
                    }
                    
                else:
                    # Handle error with professional styling
                    st.error(f"⚠ {result.get('error', 'An error occurred')}")
                    
                    if result.get("suggestion"):
                        st.info(f"▸ Suggestion: {result['suggestion']}")
                    
                    if result.get("sql") and show_sql:
                        with st.expander("⚙ View Failed SQL"):
                            st.code(result["sql"], language="sql")
                    
                    message = {
                        "role": "assistant",
                        "content": f"Error: {result.get('error', 'Unknown error')}",
                        "timestamp": datetime.now().timestamp()
                    }
                
                # Add to message history
                st.session_state.messages.append(message)
                
            except Exception as e:
                st.error(f"⚠ An unexpected error occurred: {e}")
                logger.error(f"Error processing query: {e}", exc_info=True)

# Show suggested follow-up questions (context-aware) above chat input
if len(st.session_state.messages) > 0:
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "assistant" and last_message.get("followup_questions"):
        st.markdown("---")
        st.markdown("**💡 Suggested Questions (based on your last query):**")
        st.caption("Click any question to continue the conversation")
        
        # Display as clickable chips
        cols = st.columns(min(3, len(last_message["followup_questions"])))
        for i, question in enumerate(last_message["followup_questions"][:6]):
            with cols[i % 3]:
                if st.button(f"▸ {question}", key=f"suggested_{last_message.get('timestamp')}_{i}", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": question
                    })
                    st.rerun()

# Chat input - always visible at the bottom
st.markdown("---")
new_prompt = st.chat_input("Ask a question about your e-commerce data...")

# If new input from chat, add it to messages and rerun
if new_prompt:
    st.session_state.messages.append({"role": "user", "content": new_prompt})
    st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Built with ❤️ using Streamlit, DuckDB, and Google Gemini | 
    <a href="https://github.com/yourusername/olist-copilot" target="_blank">GitHub</a> | 
    <a href="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce" target="_blank">Dataset</a>
</div>
""", unsafe_allow_html=True)

# Status bar
if not os.getenv("GEMINI_API_KEY"):
    st.error("⚠️ GEMINI_API_KEY not found in environment. Please configure .env file.")

db_path = Path(__file__).parent.parent / "data" / "duckdb" / "olist.duckdb"
if not db_path.exists():
    st.warning("⚠️ Database not found. Run `python scripts/build_duckdb.py` to build it.")


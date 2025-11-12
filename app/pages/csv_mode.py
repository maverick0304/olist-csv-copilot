"""
Custom CSV Analysis Mode - Upload and analyze any CSV files
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import logging
import os
from datetime import datetime
import tempfile

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="CSV Analysis | Olist Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_csv_session():
    """Initialize session state for CSV mode"""
    if 'csv_files_loaded' not in st.session_state:
        st.session_state.csv_files_loaded = False
    if 'csv_analyzer' not in st.session_state:
        st.session_state.csv_analyzer = None
    if 'csv_profiler' not in st.session_state:
        st.session_state.csv_profiler = None
    if 'csv_agent' not in st.session_state:
        st.session_state.csv_agent = None
    if 'csv_messages' not in st.session_state:
        st.session_state.csv_messages = []
    if 'uploaded_files_info' not in st.session_state:
        st.session_state.uploaded_files_info = []

def main():
    """Main CSV analysis page"""
    
    init_csv_session()
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📊 Custom CSV Analysis")
        st.caption("Upload your CSV files and ask questions in natural language")
    with col2:
        if st.button("← Back to Olist Mode", type="secondary"):
            # Clear CSV session and go back
            st.session_state.clear()
            st.switch_page("app/main.py")
    
    st.divider()
    
    # Main layout
    if not st.session_state.csv_files_loaded:
        # Step 1: File Upload Phase
        show_upload_interface()
    else:
        # Step 2: Analysis Phase
        show_analysis_interface()

def show_upload_interface():
    """Display file upload and initial processing interface"""
    
    st.markdown("### 📁 Upload Your CSV Files")
    st.info("💡 Upload one or more CSV files. We'll automatically detect the schema, relationships, and generate suggested questions.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more CSV files (max 200MB each)"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")
        
        # Show uploaded files
        with st.expander("▸ View uploaded files", expanded=True):
            for file in uploaded_files:
                file_size = file.size / (1024 * 1024)  # Convert to MB
                st.write(f"• **{file.name}** ({file_size:.2f} MB)")
        
        # Process files button
        if st.button("⚙ Analyze Files", type="primary", use_container_width=True):
            process_csv_files(uploaded_files)
    else:
        # Show example/demo option
        st.markdown("---")
        st.markdown("### 🎯 Don't have CSV files?")
        st.info("Try our demo with the Olist dataset or download sample CSV files from [Kaggle](https://www.kaggle.com/datasets)")

def process_csv_files(uploaded_files):
    """Process uploaded CSV files"""
    
    try:
        from app.tools.csv_tool import CSVAnalyzer
        from app.tools.data_profiler import DataProfiler
        from app.agent import OlistAgent
        
        with st.spinner("⚙️ Processing your files..."):
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Initialize analyzer
            status_text.text("Initializing analyzer...")
            analyzer = CSVAnalyzer()
            progress_bar.progress(10)
            
            # Step 2: Load each CSV
            temp_files = []
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Loading {uploaded_file.name}...")
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                    temp_files.append(tmp_path)
                
                # Load into analyzer
                table_name = Path(uploaded_file.name).stem
                schema_info = analyzer.load_csv(tmp_path, table_name)
                
                # Store file info
                st.session_state.uploaded_files_info.append({
                    'name': uploaded_file.name,
                    'table_name': table_name,
                    'rows': schema_info['row_count'],
                    'columns': len(schema_info['columns'])
                })
                
                progress_bar.progress(int(10 + (i + 1) * 40 / len(uploaded_files)))
            
            # Step 3: Detect relationships
            status_text.text("Detecting relationships...")
            relationships = analyzer.detect_relationships()
            progress_bar.progress(60)
            
            # Step 4: Profile data quality
            status_text.text("Analyzing data quality...")
            profiler = DataProfiler(analyzer)
            for table_name in analyzer.get_table_list():
                profiler.profile_table(table_name)
            progress_bar.progress(80)
            
            # Step 5: Initialize agent
            status_text.text("Initializing AI agent...")
            agent = OlistAgent(
                db_path=None,
                csv_mode=True,
                csv_analyzer=analyzer,
                data_profiler=profiler
            )
            progress_bar.progress(90)
            
            # Store in session
            st.session_state.csv_analyzer = analyzer
            st.session_state.csv_profiler = profiler
            st.session_state.csv_agent = agent
            st.session_state.csv_files_loaded = True
            
            progress_bar.progress(100)
            status_text.text("✓ Analysis complete!")
            
            # Clean up temp files
            for tmp_path in temp_files:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            st.success("✓ Files loaded successfully! You can now ask questions.")
            st.rerun()
            
    except Exception as e:
        st.error(f"⚠ Error processing files: {str(e)}")
        logger.error(f"CSV processing error: {e}", exc_info=True)

def show_analysis_interface():
    """Display the main analysis interface with chat"""
    
    analyzer = st.session_state.csv_analyzer
    profiler = st.session_state.csv_profiler
    agent = st.session_state.csv_agent
    
    # Sidebar with data info
    with st.sidebar:
        st.markdown("### 📊 Loaded Data")
        
        # Show loaded files
        for file_info in st.session_state.uploaded_files_info:
            with st.expander(f"▸ {file_info['name']}", expanded=False):
                st.metric("Rows", f"{file_info['rows']:,}")
                st.metric("Columns", file_info['columns'])
        
        st.divider()
        
        # Schema overview
        if st.button("▸ View Schema", use_container_width=True):
            st.session_state.show_schema = not st.session_state.get('show_schema', False)
        
        # Relationships
        if analyzer.relationships:
            st.markdown(f"### 🔗 Relationships ({len(analyzer.relationships)})")
            for rel in analyzer.relationships[:5]:
                st.caption(
                    f"• {rel['from_table']}.{rel['from_column']} → "
                    f"{rel['to_table']}.{rel['to_column']}"
                )
        
        st.divider()
        
        # Data Quality Summary
        st.markdown("### ✓ Data Quality")
        for table_name, profile in profiler.profiles.items():
            quality = profile['quality_score']
            color = "green" if quality > 0.8 else "orange" if quality > 0.6 else "red"
            st.markdown(f"**{table_name}**: :{color}[{quality:.0%}]")
        
        st.divider()
        
        # Settings
        st.markdown("### ⚙ Display Preferences")
        st.caption("💡 The AI automatically detects what to analyze from your questions!")
        show_sql = st.checkbox("Show SQL queries", value=False, help="Display the generated SQL for every answer")
        auto_viz = st.checkbox("Auto-generate charts", value=True, help="Automatically create visualizations when appropriate")
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Load New Files", use_container_width=True, type="secondary"):
            st.session_state.clear()
            st.rerun()
    
    # Main area
    if st.session_state.get('show_schema', False):
        # Show detailed schema
        st.markdown("### 📋 Database Schema")
        st.code(analyzer.get_schema_summary(), language="markdown")
        st.divider()
    
    # Suggested questions (only show if no messages yet)
    if len(st.session_state.csv_messages) == 0:
        st.markdown("### ⚡ Suggested Questions")
        
        try:
            from app.prompts.csv_prompt_generator import CSVPromptGenerator
            generator = CSVPromptGenerator(analyzer, profiler)
            example_questions = generator.generate_example_questions(count=6)
            
            cols = st.columns(3)
            for i, question in enumerate(example_questions):
                with cols[i % 3]:
                    if st.button(f"▸ {question[:45]}...", key=f"example_{i}", use_container_width=True):
                        st.session_state.csv_messages.append({"role": "user", "content": question})
                        st.rerun()
        except Exception as e:
            logger.warning(f"Could not generate example questions: {e}")
        
        st.divider()
    
    # Chat interface
    st.markdown("### 💬 Ask Questions")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.csv_messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)
                    
                    # Display data table if present
                    if message.get("data") is not None and len(message["data"]) > 0:
                        st.markdown("**▸ Data Preview:**")
                        st.dataframe(message["data"], use_container_width=True, hide_index=True)
                        
                        # Download button
                        csv = message["data"].to_csv(index=False)
                        st.download_button(
                            label="↓ Export as CSV",
                            data=csv,
                            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key=f"download_csv_{message.get('timestamp', 0)}"
                        )
                    
                    # Display chart if present
                    if message.get("chart") is not None:
                        st.plotly_chart(
                            message["chart"], 
                            use_container_width=True,
                            key=f"chart_csv_{message.get('timestamp', 0)}"
                        )
                    
                    # Display SQL if enabled
                    if message.get("sql") and message.get("show_sql"):
                        with st.expander("⚙ View SQL Query"):
                            st.code(message["sql"], language="sql")
    
    # Handle chat input and pending queries
    prompt = None
    
    # Check for pending query from buttons
    if len(st.session_state.csv_messages) > 0:
        last_msg = st.session_state.csv_messages[-1]
        if last_msg["role"] == "user" and (len(st.session_state.csv_messages) == 1 or 
                                           st.session_state.csv_messages[-2]["role"] == "user"):
            prompt = last_msg["content"]
    
    # Process query if we have one
    if prompt:
        
        # Process with agent
        with st.chat_message("assistant"):
            with st.spinner("⚙️ Analyzing your query..."):
                try:
                    # Process query
                    result = agent.process_query(prompt, {})
                    
                    if result["success"]:
                        # Display insight/analysis first
                        st.markdown(f"**▸ Analysis:**")
                        st.write(result['insight'])
                        st.markdown("")  # spacing
                        
                        # Display data table if present
                        if result.get("data") is not None and len(result["data"]) > 0:
                            st.markdown("**▸ Data Preview:**")
                            st.dataframe(result["data"], use_container_width=True, hide_index=True)
                            
                            # Download button
                            csv = result["data"].to_csv(index=False)
                            st.download_button(
                                label="↓ Export as CSV",
                                data=csv,
                                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )
                            st.markdown("")  # spacing
                        
                        # Display chart if present (with description header)
                        if result.get("chart") is not None and auto_viz:
                            st.markdown("**▸ Visualization:**")
                            st.plotly_chart(
                                result["chart"], 
                                use_container_width=True,
                                key=f"chart_csv_{datetime.now().timestamp()}"
                            )
                            st.markdown("")  # spacing
                        
                        # Display SQL if enabled
                        if result.get("sql") and show_sql:
                            with st.expander("⚙ View SQL Query"):
                                st.code(result["sql"], language="sql")
                        
                        # Prepare message for history
                        message = {
                            "role": "assistant",
                            "content": result["insight"],
                            "data": result.get("data"),
                            "chart": result.get("chart"),
                            "sql": result.get("sql"),
                            "show_sql": show_sql,
                            "followup_questions": result.get("followup_questions"),
                            "timestamp": datetime.now().timestamp()
                        }
                    else:
                        # Handle error
                        st.error(f"⚠ {result.get('error', 'An error occurred')}")
                        
                        if result.get("suggestion"):
                            st.info(f"▸ Suggestion: {result['suggestion']}")
                        
                        message = {
                            "role": "assistant",
                            "content": f"Error: {result.get('error', 'Unknown error')}",
                            "timestamp": datetime.now().timestamp()
                        }
                    
                    # Add to message history
                    st.session_state.csv_messages.append(message)
                
                except Exception as e:
                    st.error(f"⚠ An unexpected error occurred: {e}")
                    logger.error(f"Query processing error: {e}", exc_info=True)
    
    # Show suggested follow-up questions (context-aware) above chat input
    if len(st.session_state.csv_messages) > 0:
        last_message = st.session_state.csv_messages[-1]
        if last_message["role"] == "assistant" and last_message.get("followup_questions"):
            st.markdown("---")
            st.markdown("**💡 Suggested Questions (based on your last query):**")
            st.caption("Click any question to continue the conversation")
            
            # Display as clickable chips
            cols = st.columns(min(3, len(last_message["followup_questions"])))
            for i, question in enumerate(last_message["followup_questions"][:6]):
                with cols[i % 3]:
                    if st.button(f"▸ {question}", key=f"csv_suggested_{last_message.get('timestamp')}_{i}", use_container_width=True):
                        st.session_state.csv_messages.append({
                            "role": "user",
                            "content": question
                        })
                        st.rerun()
    
    # Chat input - always visible at the bottom
    st.markdown("---")
    new_prompt = st.chat_input("Ask a question about your data...")
    
    # If new input from chat, add it to messages and rerun
    if new_prompt:
        st.session_state.csv_messages.append({"role": "user", "content": new_prompt})
        st.rerun()
    
    # Footer
    st.divider()
    st.caption("💡 Tip: You can reference previous results using words like 'that', 'these', or 'them'")

if __name__ == "__main__":
    main()


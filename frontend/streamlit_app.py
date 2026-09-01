"""
Streamlit frontend for RAG pipeline
User-friendly interface for querying ISRO documents
"""
import os
import streamlit as st
import requests
import json
from typing import List, Dict, Any


# Page configuration
st.set_page_config(
    page_title="Vyom AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .stMetric label {
            color: black !important;
        }
        .stMetric span {
            color: black !important;
        }
        .stMetric p {
            color: black !important;
        }
        .stMetric div {
            color: black !important;
        }
        .retrieved-doc {
            background-color: #e8f4f8;
            padding: 1rem;
            border-left: 4px solid #0066cc;
            margin: 0.5rem 0;
            border-radius: 0.25rem;
            color: black;
        }
        .response-box {
            background-color: #f0fff4;
            padding: 1.5rem;
            border-left: 4px solid #22c55e;
            border-radius: 0.25rem;
            margin-top: 1rem;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# API configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")
QUERY_ENDPOINT = f"{API_URL}/query"
HEALTH_ENDPOINT = f"{API_URL}/health"
STATS_ENDPOINT = f"{API_URL}/vectorstore/stats"


#@st.cache_resource
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@st.cache_data
def get_vectorstore_stats():
    """Get vectorstore statistics"""
    try:
        response = requests.get(STATS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def query_rag(query: str, top_k: int, score_threshold: float) -> Dict[str, Any]:
    """Send query to RAG API"""
    payload = {
        "query": query,
        "top_k": top_k,
        "score_threshold": score_threshold
    }
    
    try:
        response = requests.post(QUERY_ENDPOINT, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The query might be processing a lot of documents."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}


# Main UI
st.title("Vyom AI")
st.markdown("*Retrieval-Augmented Generation powered by LLM*")

# Check API status
if not check_api_health():
    st.error(
        "❌ API Server is not running!\n\n"
        "Please start the API server first:\n"
        "```bash\npython api_server.py\n```"
    )
    st.stop()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    top_k = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )
    
    score_threshold = st.slider(
        "Minimum similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1
    )
    
    st.divider()
    
    # Vectorstore stats
    stats = get_vectorstore_stats()
    if stats:
        st.subheader("📊 Vectorstore Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;">
                    <p style="color: black; font-size: 14px; margin: 0;">Total Documents</p>
                    <p style="color: black; font-size: 28px; margin: 0; font-weight: bold;">{stats.get("total_documents", "N/A")}</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;">
                    <p style="color: black; font-size: 14px; margin: 0;">Collection</p>
                    <p style="color: black; font-size: 28px; margin: 0; font-weight: bold;">{stats.get("collection_name", "N/A")}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("Made by Nagula Sai Sathvik")


# Main content
st.subheader("Ask a Question")

# Query input
query_input = st.text_area(
    "Enter your question about ISRO:",
    placeholder="e.g., Tell me about Chandrayaan-2 mission...",
    height=100
)

# Query button
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    search_button = st.button("🔍 Search", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# Process query
if search_button:
    if not query_input.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("🔄 Processing your query..."):
            result = query_rag(query_input, top_k, score_threshold)
        
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            # Display results
            st.success("✓ Query processed successfully!")
            
            # Retrieved documents
            retrieved_docs = result.get("retrieved_documents", [])
            
            if retrieved_docs:
                st.subheader(f"📚 Retrieved Documents ({len(retrieved_docs)})")
                
                for doc in retrieved_docs:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Rank {doc['rank']}** | {doc['source']} (Page {doc['page']})")
                            st.caption(f"Similarity Score: {doc['similarity_score']:.4f}")
                        
                        with col2:
                            if doc['similarity_score'] > 0.7:
                                st.success("✓ High Match")
                            elif doc['similarity_score'] > 0.5:
                                st.warning("⚠ Medium Match")
                            else:
                                st.info("ℹ Low Match")
                        
                        st.markdown(f"> {doc['content'][:300]}...")
                        st.divider()
            else:
                st.info("No documents found matching your query.")
            
            # Generated response
            response_text = result.get("generated_response", "")
            if response_text:
                st.subheader("🤖 Generated Response")
                st.markdown(f"""
                    <div class="response-box">
                        {response_text}
                    </div>
                """, unsafe_allow_html=True)


# Examples section
with st.expander("📌 Example Questions"):
    examples = [
        "What is Chandrayaan-2?",
        "Tell me about ISRO's Mars mission",
        "What are the key achievements of ISRO?",
        "What is Aditya-L1?",
        "Tell me about Mangalyaan",
    ]
    
    st.write("Try asking:")
    for example in examples:
        if st.button(f"🔹 {example}", use_container_width=True):
            st.session_state.query_input = example

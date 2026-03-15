"""
Clinical Document Intelligence Platform — Interactive Frontend
A Streamlit-based interface for querying, comparing, and analyzing FDA drug labels.

Usage:
    streamlit run app.py

Requires the FastAPI backend running at http://localhost:8000
    uvicorn src.api.main:app --reload
"""
import time
import json
import streamlit as st
import requests

# ============================================================
# Configuration
# ============================================================

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Clinical Document Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
    /* Global */
    .block-container { padding-top: 2rem; max-width: 1200px; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1B3A5C 0%, #2E75B6 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #B0C4DE; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Cards */
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1B3A5C; }
    .metric-card .label { font-size: 0.85rem; color: #666; }

    /* Citation box */
    .citation-box {
        background: #EDF3F8;
        border-left: 4px solid #2E75B6;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }

    /* Safety warning */
    .safety-warning {
        background: #FFF3E0;
        border-left: 4px solid #E65100;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
    }

    /* Confidence bar */
    .confidence-high { color: #2E7D32; font-weight: 600; }
    .confidence-medium { color: #F57F17; font-weight: 600; }
    .confidence-low { color: #C62828; font-weight: 600; }

    /* Agent trace */
    .agent-step {
        background: #f5f5f5;
        border-left: 3px solid #2E75B6;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .agent-step .step-num {
        background: #1B3A5C;
        color: white;
        padding: 0.1rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }

    /* Disclaimer */
    .disclaimer {
        background: #ECEFF1;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.8rem;
        color: #546E7A;
        margin-top: 1rem;
    }

    /* Hide Streamlit footer */
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# API Helpers
# ============================================================

def api_call(endpoint: str, method: str = "GET", data: dict = None, timeout: int = 60):
    """Make an API call to the FastAPI backend."""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=data, timeout=timeout)
        else:
            resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("⚠️ Cannot connect to the API backend. Make sure it's running: `uvicorn src.api.main:app --reload`")
        return None
    except requests.Timeout:
        st.error("⚠️ Request timed out. The backend may be processing a large query.")
        return None
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"⚠️ API Error: {detail}")
        return None


def get_health():
    return api_call("/health")


def get_drugs():
    return api_call("/drugs")


def query_labels(query, drug_name=None, section_type=None, therapeutic_area=None):
    payload = {"query": query, "enable_rewrite": True}
    if drug_name and drug_name != "All drugs":
        payload["drug_name"] = drug_name
    if section_type and section_type != "All sections":
        payload["section_type"] = section_type
    if therapeutic_area and therapeutic_area != "All areas":
        payload["therapeutic_area"] = therapeutic_area
    return api_call("/query", method="POST", data=payload)


def compare_drugs(query, drug_names):
    return api_call("/compare", method="POST", data={"query": query, "drug_names": drug_names})


def agent_analyze(task):
    return api_call("/agent/analyze", method="POST", data={"task": task}, timeout=120)


def get_stats():
    return api_call("/stats")


def get_agent_stats():
    return api_call("/agent/stats")


# ============================================================
# UI Components
# ============================================================

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Clinical Document Intelligence Platform</h1>
        <p>AI-powered FDA drug label intelligence — query, compare, and analyze with citation-grounded answers</p>
    </div>
    """, unsafe_allow_html=True)


def render_confidence(score):
    if score >= 0.7:
        css_class = "confidence-high"
        label = "High"
    elif score >= 0.4:
        css_class = "confidence-medium"
        label = "Medium"
    else:
        css_class = "confidence-low"
        label = "Low"
    st.markdown(f'<span class="{css_class}">Confidence: {score:.0%} ({label})</span>', unsafe_allow_html=True)


def render_citations(citations):
    if not citations:
        return
    st.markdown("**📚 Sources**")
    for c in citations:
        drug = c.get("drug_name", "Unknown")
        section = c.get("section_display_name", c.get("section_type", "Unknown"))
        label_id = c.get("label_id", "N/A")
        score = c.get("relevance_score", 0)
        preview = c.get("chunk_content_preview", "")[:200]
        st.markdown(f"""
        <div class="citation-box">
            <strong>{drug}</strong> — {section}<br/>
            <span style="color: #888; font-size: 0.8rem;">Label: {label_id} | Relevance: {score:.2f}</span><br/>
            <span style="font-size: 0.85rem;">{preview}...</span>
        </div>
        """, unsafe_allow_html=True)


def render_agent_trace(trace):
    if not trace:
        return
    st.markdown("**🔍 Agent Execution Trace**")
    for step in trace:
        step_num = step.get("step", "?")
        action = step.get("action", "Unknown")
        tool = step.get("tool", "none")
        reasoning = step.get("reasoning", "")
        st.markdown(f"""
        <div class="agent-step">
            <span class="step-num">Step {step_num}</span>
            <strong>{action}</strong> → <code>{tool}</code><br/>
            <span style="color: #666; font-size: 0.85rem;">{reasoning}</span>
        </div>
        """, unsafe_allow_html=True)


def render_disclaimer():
    st.markdown("""
    <div class="disclaimer">
        <strong>DISCLAIMER:</strong> This information is derived from FDA drug labels and is intended
        for informational purposes only. It does not constitute medical advice.
        Always consult a qualified healthcare provider for clinical decisions.
    </div>
    """, unsafe_allow_html=True)


def render_metadata(meta):
    cols = st.columns(4)
    with cols[0]:
        st.metric("Model", meta.get("model_used", "N/A"))
    with cols[1]:
        st.metric("Latency", f"{meta.get('latency_ms', 0):.0f} ms")
    with cols[2]:
        st.metric("Tokens", f"{meta.get('total_tokens', 0):,}")
    with cols[3]:
        st.metric("Context Docs", meta.get("context_documents", 0))


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ System Status")

    health = get_health()
    if health:
        st.success(f"✅ API Connected")
        st.metric("Documents Indexed", f"{health.get('documents_indexed', 0):,}")

        drugs = health.get("available_drugs", [])
        st.metric("Unique Drugs", len(drugs))

        with st.expander("Available Drugs", expanded=False):
            for d in sorted(drugs):
                st.markdown(f"• {d}")
    else:
        st.error("❌ API Offline")
        st.markdown("Start the backend:")
        st.code("uvicorn src.api.main:app --reload", language="bash")

    st.markdown("---")
    st.markdown("### 📊 Cost Tracking")
    stats = get_stats()
    if stats and stats.get("token_usage"):
        usage = stats["token_usage"]
        st.metric("Total Requests", usage.get("request_count", 0))
        st.metric("Total Cost", f"${usage.get('total_cost_usd', 0):.4f}")
    else:
        st.markdown("_No usage data yet_")

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 0.8rem; color: #999;">
        Built by <strong>Erick K. Yegon, PhD</strong><br/>
        <a href="https://github.com/erickyegon/clinical-doc-intelligence" target="_blank">GitHub</a> |
        <a href="https://linkedin.com/in/erickyegon" target="_blank">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Main Content — Tabs
# ============================================================

render_header()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💊 Drug Q&A",
    "⚖️ Drug Comparison",
    "🤖 Agent Analysis",
    "📈 System Dashboard",
    "ℹ️ About",
])


# ====== TAB 1: Drug Q&A ======
with tab1:
    st.markdown("### Ask a Clinical Question")
    st.markdown("Query FDA drug labels with citation-grounded answers. Filter by drug, section, or therapeutic area.")

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_area(
            "Your question:",
            placeholder="e.g., What are the contraindications for empagliflozin?",
            height=80,
            key="qa_query",
        )

    with col2:
        drug_list = ["All drugs"]
        if health and health.get("available_drugs"):
            drug_list += sorted(health["available_drugs"])
        selected_drug = st.selectbox("Filter by drug:", drug_list, key="qa_drug")

        section_options = [
            "All sections", "boxed_warning", "contraindications",
            "warnings_and_cautions", "adverse_reactions",
            "dosage_and_administration", "drug_interactions",
            "indications_and_usage", "use_in_specific_populations",
            "clinical_pharmacology", "clinical_studies",
        ]
        selected_section = st.selectbox("Filter by section:", section_options, key="qa_section")

    if st.button("🔍 Search", key="qa_search", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching FDA drug labels..."):
                start = time.time()
                result = query_labels(query, selected_drug, selected_section)
                elapsed = time.time() - start

            if result:
                # Answer
                st.markdown("### Answer")
                answer = result.get("answer", "No answer generated.")
                if "⚠️" in answer or "WARNING" in answer.upper():
                    st.markdown(f'<div class="safety-warning">{answer}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(answer)

                # Warning
                warning = result.get("warning")
                if warning:
                    st.warning(warning)

                # Confidence
                meta = result.get("metadata", {})
                render_confidence(meta.get("confidence", 0))

                # Metadata
                with st.expander("📊 Response Metadata", expanded=False):
                    render_metadata(meta)
                    rewritten = meta.get("rewritten_query")
                    if rewritten:
                        st.markdown(f"**Query Rewrite:** {rewritten}")

                # Citations
                with st.expander("📚 Citations & Sources", expanded=True):
                    render_citations(result.get("citations", []))

                render_disclaimer()

    # Example queries
    with st.expander("💡 Example Queries", expanded=False):
        examples = [
            "What are the contraindications for empagliflozin?",
            "What is the recommended dosage of semaglutide for type 2 diabetes?",
            "What are the most common adverse reactions for atorvastatin?",
            "Is lisinopril safe during pregnancy?",
            "What drug interactions should I be aware of with JARDIANCE?",
            "What is the mechanism of action of pembrolizumab?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{hash(ex)}"):
                st.session_state.qa_query = ex
                st.rerun()


# ====== TAB 2: Drug Comparison ======
with tab2:
    st.markdown("### Compare Drug Labels")
    st.markdown("Side-by-side comparison of safety profiles, dosing, or indications across drugs.")

    comparison_query = st.text_input(
        "What to compare:",
        placeholder="e.g., Compare the safety profiles and contraindications",
        key="comp_query",
    )

    drug_list_for_comp = sorted(health.get("available_drugs", [])) if health else []

    selected_drugs = st.multiselect(
        "Select drugs to compare (2-5):",
        drug_list_for_comp,
        default=drug_list_for_comp[:2] if len(drug_list_for_comp) >= 2 else [],
        max_selections=5,
        key="comp_drugs",
    )

    if st.button("⚖️ Compare", key="comp_btn", type="primary", use_container_width=True):
        if not comparison_query:
            st.warning("Please enter a comparison question.")
        elif len(selected_drugs) < 2:
            st.warning("Please select at least 2 drugs to compare.")
        else:
            with st.spinner(f"Comparing {', '.join(selected_drugs)}..."):
                result = compare_drugs(comparison_query, selected_drugs)

            if result:
                st.markdown("### Comparison Results")
                st.markdown(result.get("answer", ""))

                with st.expander("📚 Citations", expanded=False):
                    render_citations(result.get("citations", []))

                render_disclaimer()


# ====== TAB 3: Agent Analysis ======
with tab3:
    st.markdown("### AI Agent Analysis")
    st.markdown("""
    The multi-agent system automatically classifies your task, routes it to the right 
    specialist agent (Drug Analysis, Safety Review, or Comparison), and executes a 
    multi-step plan with tool calls, reasoning, and synthesis.
    """)

    agent_task = st.text_area(
        "Describe your analysis task:",
        placeholder="e.g., Analyze the safety profile of Jardiance for elderly patients with renal impairment",
        height=100,
        key="agent_task",
    )

    if st.button("🤖 Run Agent Analysis", key="agent_btn", type="primary", use_container_width=True):
        if not agent_task:
            st.warning("Please describe your analysis task.")
        else:
            with st.spinner("Agent is planning and executing... (this may take 30-60 seconds)"):
                result = agent_analyze(agent_task)

            if result:
                meta = result.get("metadata", {})

                # Agent routing info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Agent Used", meta.get("agent_used", "N/A"))
                with col2:
                    st.metric("Task Type", meta.get("task_type", "N/A"))
                with col3:
                    st.metric("Steps Taken", meta.get("total_steps", 0))
                with col4:
                    render_confidence(meta.get("confidence", 0))

                # Human review flag
                if meta.get("human_review_required"):
                    st.warning(f"⚠️ **Human Review Required:** {meta.get('human_review_reason', 'Safety-critical content')}")

                # Answer
                st.markdown("### Analysis Result")
                st.markdown(result.get("answer", "No result generated."))

                # Execution trace
                with st.expander("🔍 Execution Trace", expanded=True):
                    render_agent_trace(result.get("execution_trace", []))

                # Performance
                with st.expander("📊 Performance Metrics", expanded=False):
                    st.markdown(f"**Total Latency:** {meta.get('total_latency_ms', 0):.0f} ms")
                    st.markdown(f"**Total Tokens:** {meta.get('total_tokens', 0):,}")

                render_disclaimer()

    # Example tasks
    with st.expander("💡 Example Agent Tasks", expanded=False):
        agent_examples = [
            "Tell me everything about Ozempic for type 2 diabetes",
            "Is Jardiance safe for patients with renal impairment?",
            "Compare the safety profiles of Jardiance vs Farxiga vs Invokana",
            "What are the black box warnings for SGLT2 inhibitors?",
            "Analyze pembrolizumab for non-small cell lung cancer",
        ]
        for ex in agent_examples:
            if st.button(ex, key=f"ag_{hash(ex)}"):
                st.session_state.agent_task = ex
                st.rerun()


# ====== TAB 4: System Dashboard ======
with tab4:
    st.markdown("### System Dashboard")

    if health:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{health.get('documents_indexed', 0):,}</div>
                <div class="label">Indexed Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            drugs = health.get("available_drugs", [])
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{len(drugs)}</div>
                <div class="label">Unique Drugs</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">v{health.get('version', '1.0.0')}</div>
                <div class="label">Platform Version</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Token usage
    st.markdown("#### Token Usage & Cost")
    stats = get_stats()
    if stats and stats.get("token_usage"):
        usage = stats["token_usage"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Requests", usage.get("request_count", 0))
        with col2:
            st.metric("Total Tokens", f"{usage.get('total_tokens', 0):,}")
        with col3:
            st.metric("Total Cost", f"${usage.get('total_cost_usd', 0):.4f}")

        by_model = usage.get("by_model", {})
        if by_model:
            st.markdown("**Usage by Model:**")
            for model, data in by_model.items():
                st.markdown(
                    f"• **{model}**: {data.get('requests', 0)} requests, "
                    f"{data.get('input', 0) + data.get('output', 0):,} tokens, "
                    f"${data.get('cost', 0):.4f}"
                )
    else:
        st.info("No usage data available yet. Run some queries to see metrics.")

    st.markdown("---")

    # Agent stats
    st.markdown("#### Agent System Status")
    agent_stats = get_agent_stats()
    if agent_stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Agent Tokens Used", f"{agent_stats.get('total_tokens_used', 0):,}")
            st.metric("Token Budget Remaining", f"{agent_stats.get('token_budget_remaining', 0):,}")
        with col2:
            st.markdown("**Available Tools:**")
            for tool in agent_stats.get("tools_available", []):
                st.markdown(f"• `{tool}`")

        agents = agent_stats.get("agents", {})
        if agents:
            st.markdown("**Agent States:**")
            for name, info in agents.items():
                st.markdown(f"• **{name}**: {info.get('state', 'unknown')} ({info.get('steps_executed', 0)} steps)")


# ====== TAB 5: About ======
with tab5:
    st.markdown("""
    ### About This Platform

    The **Clinical Document Intelligence Platform** is a production-grade RAG 
    (Retrieval-Augmented Generation) system for FDA drug label intelligence.

    #### Architecture Highlights
    
    **Data Sources:** Real FDA drug labels from the openFDA API (70,000+ labels) 
    and clinical trial data from ClinicalTrials.gov V2 API.

    **Section-Aware Chunking:** Black Box Warnings and Contraindications are never 
    split. Safety-critical sections are preserved intact regardless of size.

    **5-Stage Retrieval:** Dense vector search → Metadata filtering → Section priority 
    boosting → MMR diversification → Cross-encoder re-ranking.

    **Multi-Provider LLM:** Automatic routing across OpenAI, Groq, and AWS Bedrock 
    with cost tracking and fallback logic.

    **Guardrails:** PHI detection, prompt injection defense, output validation, 
    confidence thresholds, and mandatory clinical disclaimers.

    **Multi-Agent System:** Three specialized agents (Drug Analysis, Safety Review, 
    Comparison) coordinated by a supervisor pattern with human-in-the-loop gates.

    **MCP Integration:** Model Context Protocol server for Claude Desktop and 
    VS Code integration.

    #### Tech Stack
    
    Python, FastAPI, ChromaDB, OpenAI/Groq/Bedrock, Streamlit, Docker, AWS ECS

    #### Project Stats
    
    6,100+ lines of Python | 54 automated tests | 14 core modules | 3 AI agents | 6 agent tools

    ---

    **Built by [Erick K. Yegon, PhD](https://linkedin.com/in/erickyegon)**  
    Director-Level Data Science & Epidemiology | 17+ Years Global Health  
    [GitHub](https://github.com/erickyegon/clinical-doc-intelligence) | 
    [LinkedIn](https://linkedin.com/in/erickyegon) | 
    [ORCID](https://orcid.org/0000-0002-7055-4848)
    """)

"""
app.py — Sensor GPT | ELCIA Center of Excellence Internship
A RAG-powered chatbot for hardware engineers to find the right sensor.
"""

import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Avoid tqdm/transformers progress rendering issues in Streamlit on Windows.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sensor GPT — ELCIA COE",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Light background */
.stApp {
    background: linear-gradient(135deg, #f8fafc, #eef2ff, #f5f3ff);
    color: #1f2937;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(6px);
    border-right: 1px solid rgba(99,102,241,0.2);
}

/* Chat message bubbles */
.user-msg {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
}

.bot-msg {
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(99,102,241,0.2);
    color: #1f2937;
    padding: 14px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 85%;
    backdrop-filter: blur(5px);
}

.sensor-card {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.85em;
}

.header-badge {
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75em;
    font-weight: 600;
    letter-spacing: 0.05em;
}

h1, h2, h3 { color: #4338ca !important; }
.stTextInput > div > div > input { background: white !important; color: #111827 !important; border: 1px solid rgba(99,102,241,0.35) !important; border-radius: 10px !important; }
.stButton > button { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 0.5rem 1.5rem !important; font-weight: 600 !important; }
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }
</style>
""", unsafe_allow_html=True)

CHROMA_DIR = "./chroma_db"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"


# ── Load vector store ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading sensor knowledge base...")
def load_vectorstore():
    if not os.path.exists(CHROMA_DIR):
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


# ── Retrieve sensors ──────────────────────────────────────────────────────────
def retrieve_sensors(query: str, vectorstore, k: int = 4):
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


# ── Format sensor card ────────────────────────────────────────────────────────
def format_sensor_card(doc, score):
    m = doc.metadata
    return f"""
**🔩 {m.get('sensor_name', 'Unknown')}** &nbsp; `{m.get('sensor_type', '')}` &nbsp; `{m.get('protocol', '')}`  
💰 **Cost:** ${m.get('cost_usd', '?')} &nbsp;&nbsp; 🔌 **Voltage:** {m.get('voltage', '?')}  
🌍 **Use case:** {m.get('environment', '?')} &nbsp;&nbsp; 📡 **I2C:** {m.get('i2c', '?')} | **SPI:** {m.get('spi', '?')}  
📊 *Relevance score: {1 - score:.2f}*
"""


# ── Build answer using LLM ────────────────────────────────────────────────────
def generate_answer(query, contexts, llm_provider, api_key):
    context_text = "\n\n".join([doc.page_content for doc, _ in contexts])

    prompt = f"""You are Sensor GPT, an expert AI assistant helping hardware and embedded systems engineers select the right sensors.

Based on the following sensor information retrieved from a database, answer the user's question in a clear, helpful, and professional manner. Recommend specific sensors, explain the trade-offs, and justify your choice.

Sensor Database Context:
{context_text}

User Question: {query}

Answer:"""

    if llm_provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
        response = llm.invoke(prompt)
        return response.content

    elif llm_provider == "OpenAI GPT":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.3)
        response = llm.invoke(prompt)
        return response.content

    else:  # Local prototype
        return (
            "🔍 **Retrieved Sensors (Local Prototype Mode — No LLM Key Required):**\n\n"
            "The following sensors were matched from the knowledge base based on your query. "
            "Add a Gemini or OpenAI API key in the sidebar for a full AI-generated recommendation.\n\n"
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="header-badge">ELCIA COE · RIT Bengaluru</span>', unsafe_allow_html=True)
    st.markdown("## 🤖 Sensor GPT")
    st.markdown("*AI Sensor Recommendation Chatbot*")
    st.divider()

    st.markdown("### ⚙️ Configuration")
    llm_provider = st.selectbox(
        "LLM Provider",
        ["Local Prototype (No Key)", "Google Gemini", "OpenAI GPT"],
        help="Select your preferred AI provider"
    )

    api_key = ""
    if llm_provider == "Google Gemini":
        api_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
        st.caption("[Get a free key →](https://aistudio.google.com/)")
    elif llm_provider == "OpenAI GPT":
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        st.caption("[Get a key →](https://platform.openai.com/)")
    else:
        st.info("Running in local mode — no API key needed!")

    st.divider()
    st.markdown("### 💡 Try asking:")
    examples = [
        "Best low-cost I2C temp sensor?",
        "Sensor for CO2 detection indoors",
        "Which sensors work below -40°C?",
        "Proximity sensor for 3.3V systems",
        "Waterproof temperature sensor",
    ]
    for ex in examples:
        if st.button(f"→ {ex}", key=ex):
            st.session_state["pending_query"] = ex

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("**Sensor GPT** · ELCIA Internship 2026")
    st.caption("Built with LangChain · ChromaDB · Streamlit")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("# 🤖 Sensor GPT")
st.markdown("**AI-powered sensor recommendation engine for hardware engineers** · ELCIA COE Internship Project")
st.divider()

# Check vectorstore
vectorstore = load_vectorstore()
if vectorstore is None:
    st.error("""
    ⚠️ **Knowledge base not found!**  
    Please run the setup scripts first:
    ```bash
    python data_generator.py
    python ingest.py
    ```
    """)
    st.stop()

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm **Sensor GPT**, your AI assistant for sensor selection.\n\nAsk me anything like:\n- *\"What's the best low-cost humidity sensor with I2C support?\"*\n- *\"Which sensor should I use for outdoor CO2 monitoring?\"*\n- *\"Recommend a motion sensor for a 3.3V microcontroller\"*"
    })

# Handle example button queries
if "pending_query" in st.session_state:
    pending = st.session_state.pop("pending_query")
    st.session_state.messages.append({"role": "user", "content": pending})

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown(f'<div class="bot-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            if "sensors" in msg:
                with st.expander("📊 Retrieved Sensor Data", expanded=False):
                    for doc, score in msg["sensors"]:
                        st.markdown(f'<div class="sensor-card">{format_sensor_card(doc, score)}</div>', unsafe_allow_html=True)

# Chat input
query = st.chat_input("Ask about a sensor (e.g. low-cost I2C temperature sensor for high humidity)...")

# Process query
if query or st.session_state.get("pending_query"):
    if not query:
        query = st.session_state.pop("pending_query", "")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.markdown(f'<div class="user-msg">👤 {query}</div>', unsafe_allow_html=True)

        with st.spinner("🔍 Searching sensor knowledge base..."):
            sensors = retrieve_sensors(query, vectorstore, k=4)

        if llm_provider != "Local Prototype (No Key)" and not api_key:
            answer = "⚠️ Please enter your API key in the sidebar, or switch to **Local Prototype** mode."
            message = {"role": "assistant", "content": answer}
        else:
            with st.spinner("🤖 Generating recommendation..."):
                prefix = generate_answer(query, sensors, llm_provider, api_key)

            # Format sensor list
            sensor_summary = "\n\n".join([
                f"**{doc.metadata.get('sensor_name')}** — {doc.metadata.get('sensor_type')} | {doc.metadata.get('protocol')} | ${doc.metadata.get('cost_usd')} | {doc.metadata.get('environment')}"
                for doc, _ in sensors
            ])

            if llm_provider == "Local Prototype (No Key)":
                answer = prefix + sensor_summary
            else:
                answer = prefix

            message = {"role": "assistant", "content": answer, "sensors": sensors}

        st.session_state.messages.append(message)
        st.rerun()

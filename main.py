# Multi-Agent AI System for Customer Support (Full App)

import streamlit as st
import pandas as pd
import sqlite3
import json
import ollama
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from concurrent.futures import ThreadPoolExecutor

DB_PATH = 'support_agents.db'
HISTORICAL_DATA = 'C:/Users/Supriya A B/OneDrive/Desktop/customer_support/Historical_ticket_data.csv'

# === SETUP ===
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS tickets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        customer_input TEXT,
                        summary TEXT,
                        actions TEXT,
                        recommendation TEXT,
                        routed_to TEXT,
                        resolution_time TEXT
                    )''')
    conn.commit()
    conn.close()

# === OLLAMA UTILITIES ===
def query_ollama(prompt: str, model: str = "mistral") -> str:
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# === AGENT TASKS ===
def summarize_conversation(conversation: str) -> str:
    return query_ollama(f"Summarize the following customer conversation:\n{conversation}")

def extract_actions(conversation: str) -> List[str]:
    result = query_ollama(f"Extract all action items from this conversation as a list:\n{conversation}")
    return json.loads(result) if result.startswith('[') else [result]

def recommend_resolution(summary: str, actions: List[str]) -> str:
    return query_ollama(f"Based on this summary and actions, recommend a resolution:\nSummary: {summary}\nActions: {', '.join(actions)}")

def route_task(actions: List[str]) -> str:
    if any("technical" in action.lower() for action in actions):
        return "Technical Team"
    elif any("billing" in action.lower() for action in actions):
        return "Billing Department"
    else:
        return "General Support"

def estimate_resolution_time(routed_team: str, actions: List[str]) -> str:
    return query_ollama(f"Estimate resolution time for team '{routed_team}' with actions: {actions}")

# === EMBEDDING + RETRIEVAL SETUP ===
def load_historical_tickets():
    return pd.read_csv(HISTORICAL_DATA)

def embed_text(text: str, model: str = "nomic-embed-text") -> List[float]:
    response = ollama.embeddings(model=model, prompt=text)
    return response['embedding']

def get_similar_ticket(query: str, df: pd.DataFrame) -> str:
    vectorizer = TfidfVectorizer()
    corpus = df['Issue'].tolist() + [query]
    vectors = vectorizer.fit_transform(corpus)
    sim_scores = cosine_similarity(vectors[-1], vectors[:-1])
    best_idx = sim_scores.argmax()
    return df.iloc[best_idx]['Resolution']

# === STREAMLIT INTERFACE ===
st.set_page_config(page_title="Multi-Agent Support AI", layout="wide")
st.title("🤖 Multi-Agent AI System for Customer Support")

init_db()
ticket_data = load_historical_tickets()

# === CHATBOT INTERFACE ===
st.markdown("---")
st.subheader("💬 AI Chatbot (Ask a Question)")
chat_input = st.text_input("Ask your question about support or issues:")
if st.button("Ask Bot") and chat_input:
    with st.spinner("Thinking..."):
        bot_reply = query_ollama(f"Answer this customer query as a support agent:\n{chat_input}")
    st.markdown(f"**🤖 Agent Response:** {bot_reply}")

# === MULTI-AGENT EXECUTION ===
st.markdown("---")
st.subheader("🎯 Run Multi-Agent System on a Customer Conversation")
customer_input = st.text_area("Paste the customer conversation:")

if st.button("Run Agents") and customer_input:
    progress = st.progress(0, text="Starting multi-agent tasks...")

    with ThreadPoolExecutor() as executor:
        future_summary = executor.submit(summarize_conversation, customer_input)
        future_actions = executor.submit(extract_actions, customer_input)

        summary = future_summary.result()
        progress.progress(25, text="Summary complete. Extracting actions...")
        actions = future_actions.result()

        future_recommend = executor.submit(recommend_resolution, summary, actions)
        future_route = executor.submit(route_task, actions)

        recommendation = future_recommend.result()
        progress.progress(60, text="Resolution recommended. Routing task...")
        routed_to = future_route.result()

        resolution_time = estimate_resolution_time(routed_to, actions)
        progress.progress(100, text="All agents finished!")

    st.success("Multi-agent processing complete!")

    st.subheader("📝 Summary")
    st.write(summary)

    st.subheader("✅ Action Items")
    st.write(actions)

    st.subheader("💡 Recommended Resolution")
    st.write(recommendation)

    st.subheader("📍 Routed To")
    st.write(routed_to)

    st.subheader("⏱ Estimated Resolution Time")
    st.write(resolution_time)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tickets (customer_input, summary, actions, recommendation, routed_to, resolution_time) VALUES (?, ?, ?, ?, ?, ?)",
                   (customer_input, summary, json.dumps(actions), recommendation, routed_to, resolution_time))
    conn.commit()
    conn.close()
    st.success("Ticket logged to database.")

# === TICKET HISTORY ===
st.markdown("---")
st.subheader("📊 Ticket History")
if st.button("Load Past Tickets"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, summary, actions, recommendation, routed_to, resolution_time FROM tickets ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        st.markdown(f"**Ticket #{row[0]}**")
        st.markdown(f"- **Summary:** {row[1]}")
        st.markdown(f"- **Actions:** {row[2]}")
        st.markdown(f"- **Recommendation:** {row[3]}")
        st.markdown(f"- **Routed To:** {row[4]}")
        st.markdown(f"- **Estimated Time:** {row[5]}")
        st.markdown("---")

# === CURRENT PROCESS EXPLAINER ===
st.markdown("---")
with st.expander("📋 See Current Manual Process vs. AI Automation", expanded=False):
    st.markdown("""
### 🔁 Current Manual Process:
1. **Summary & Action Extraction**:
   - Agents read full conversations manually
   - Write summaries & identify action items
2. **Routing to Teams**:
   - Tasks assigned manually or via fixed rules
   - Delays happen due to misrouting
3. **Resolution Recommendation**:
   - Agents search ticket history/FAQs manually
   - Leads to inconsistencies and escalations
4. **Time Estimation**:
   - Rely on gut feeling or SLAs
   - No real-time analytics to optimize flow

---

### 🤖 Automated by Multi-Agent AI System:
- 🔍 Auto-summary using LLM
- ✅ Action items extracted instantly
- 📍 Task routed based on action context
- 💡 Resolution generated from similar tickets
- ⏱ Predictive resolution time output
    """)

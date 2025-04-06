# Multi-Agent-AI-System-for-Customer-Support
# 🤖 Multi-Agent AI System for Customer Support

## Overview
This project implements a multi-agent AI system designed to automate and optimize customer support operations. The system uses **on-premise Ollama LLMs**, **Streamlit UI**, and **SQLite** to deliver intelligent conversation summarization, action extraction, resolution recommendations, task routing, and resolution time predictions — all from a single interface.

## 🔧 Features

- 📝 **Summary Generation**: Automatic summarization of customer conversations.
- ✅ **Action Extraction**: Detect follow-ups, escalations, and other actionable items.
- 💡 **Resolution Recommendation**: Suggest best-fit solutions using historical ticket analysis.
- 📍 **Task Routing**: Direct tasks to the right support team (e.g., Technical, Billing).
- ⏱ **Resolution Time Estimation**: Predicts how long a case may take to resolve.
- 💬 **AI Chatbot**: Live interface for customer Q&A with intelligent response generation.
- ⚙️ **Multithreaded Processing**: Accelerated response through concurrent execution.
- 🗃 **Ticket History Viewer**: Review recent processed tickets from SQLite DB.
- 🔍 **Insight Detection**: Extracts insights from uploaded ticket conversations.

## 🛠 Tech Stack

| Component       | Technology             |
|----------------|------------------------|
| LLM             | Ollama (e.g., Mistral, Nomic Embed) |
| UI              | Streamlit              |
| Backend DB      | SQLite                 |
| ML/NLP Tools    | Scikit-learn (TF-IDF, Cosine Similarity) |
| Threading       | Python `concurrent.futures` |

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/support-ai-system.git
cd support-ai-system


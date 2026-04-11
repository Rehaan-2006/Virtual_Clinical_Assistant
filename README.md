# 🩺 CareAssist – AI-Powered Medical Decision Support System (RAG vs. Generic LLM)

A full-stack, containerized AI application designed to demonstrate the power of **Retrieval-Augmented Generation (RAG)** in the medical field. 

This project features a custom-built, split-screen UI that allows users to test a clinically-grounded AI (connected to a medical knowledge base) side-by-side against a standard, hallucination-prone generic LLM.

## ✨ Key Features

* **Split-Screen Compare Mode:** Toggle a side-by-side view to send the same prompt to both the RAG-enabled Clinical AI and a Generic AI simultaneously.
* **Multi-Model Support:** Seamlessly switch between Google's native Gemini models and OpenRouter's open-source models (Meta Llama 3, DeepSeek, Mistral) via a UI dropdown.
* **Transparent Citations:** The Clinical RAG assistant provides a clickable "View Sources" dropdown, displaying the exact document chunks and metadata used to generate the response.
* **Fully Dockerized:** The entire backend and frontend are packaged into a single Docker container for instant, dependency-free deployment.
* **Lightweight Frontend:** A lightning-fast, zero-dependency vanilla HTML/CSS/JS frontend styled with a clean, clinical "Bright Gemini" aesthetic.

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI
* **AI Orchestration:** Pydantic AI
* **Vector Database:** Supabase (PostgreSQL + pgvector)
* **Frontend:** Vanilla HTML5, CSS3, JavaScript (with `marked.js` for markdown parsing)
* **Deployment:** Docker

## 📁 Project Structure

```text
Virtual_Clinical_Assistant/
├── frontend/
│   ├── index.html        # UI Skeleton & Layout
│   ├── style.css         # Clinical Light-Mode Styling
│   └── script.js         # API integration & DOM manipulation
├── agent.py              # Pydantic AI logic (RAG & Generic Agents)
├── main.py               # FastAPI server routing
├── ingestion.py          # Script for populating the Supabase vector DB
├── Dockerfile            # Container blueprint
├── requirements.txt      # Python dependencies
└── .env                  # API keys and secrets (not committed)
```
## 🚀 Getting Started

### 1. Prerequisites
You will need Docker installed on your machine, along with API keys for the following services:
* Google Gemini API
* OpenRouter API (for Llama, Mistral, etc.)
* Supabase (URL and Service Role Key)

### 2. Environment Setup
Create a `.env` file in the root directory and add your credentials:

```env
GEMINI_API_KEY=your_gemini_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_SERVICE_KEY=your_supabase_service_key_here
```
### 3. Build and Run via Docker
**Build the Docker image:**
```bash
docker build -t medical-ai-agent .
```
**Run the container in the background on port 8000:**
```bash
docker run -d --name medical-api -p 8000:8000 --env-file .env medical-ai-agent
```
### 4. Access the Interface
Simply open `frontend/index.html` in your web browser. The frontend is pre-configured to communicate with the Docker container running on `localhost:8000`.

### ⚠️ Disclaimer
**This AI is intended for educational and clinical assistance demonstrations, not as a replacement for professional medical doctors.** Always consult a qualified healthcare provider for medical advice, diagnoses, or treatment.

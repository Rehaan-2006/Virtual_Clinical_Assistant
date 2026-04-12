# 🩺 CareAssist — Evidence-Based Medical Decision Support System

A full-stack, containerized clinical AI application powered by a **three-stage agentic RAG pipeline**. The system grounds every response in official WHO/NICE clinical guidelines and FDA DailyMed drug labels, with a transparent citation system and a side-by-side compare mode to demonstrate the difference between RAG-grounded and hallucination-prone generic LLM responses.

> ⚠️ **Disclaimer:** This system is intended for clinical decision support and educational purposes only. It is not a substitute for professional medical judgment. Always consult a qualified healthcare provider.

---

## ✨ Key Features

- **Three-Stage Agentic Pipeline** — The clinical agent follows a strict reasoning loop: lab interpretation → clinical pathway retrieval → drug safety verification. It cannot skip steps.
- **Evidence-Based RAG** — Retrieves grounded answers exclusively from WHO/NICE guidelines and FDA DailyMed drug labels stored in a pgvector database.
- **Lab Result Analyzer** — Interprets raw lab values (eGFR, HbA1c, sodium, potassium, etc.) against hardcoded clinical reference ranges before any LLM reasoning begins.
- **Transparent Citations** — Every RAG response includes a collapsible "Sources Retrieved" panel showing the exact document chunk and similarity score used.
- **Compare Mode** — Side-by-side split-screen view sends the same prompt to both the RAG Clinical AI and a standard LLM simultaneously.
- **Multi-Model Support** — Switch between Google Gemini, local Ollama models, and OpenRouter free-tier models via a dropdown. No restart required.
- **Dark Clinical UI** — Professional dark-theme interface built in vanilla HTML/CSS/JS with suggestion chips, animated thinking indicators, and a persistent sidebar.
- **Fully Dockerized** — Backend packaged with system dependencies, Python packages, and the BAAI embedding model pre-baked in. Single `docker run` to deploy.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| AI Orchestration | Pydantic AI |
| LLM Providers | Google Gemini, OpenRouter, Ollama (local) |
| Embeddings | BAAI/bge-base-en-v1.5 (local, via sentence-transformers) |
| Vector Database | Supabase (PostgreSQL + pgvector) |
| PDF Ingestion | Unstructured (hi_res + fast strategy, auto-detected) |
| Frontend | Vanilla HTML5, CSS3, JavaScript + marked.js |
| Deployment | Docker |

---

## 📁 Project Structure

```
clinicalai/
├── frontend/
│   ├── index.html              # UI layout and structure
│   ├── style.css               # Dark clinical precision theme
│   └── script.js               # API calls, compare mode, sources UI
│
├── data/
│   ├── guidelines/             # WHO/NICE PDF guidelines
│   │   └── manifest.json       # Metadata mapping for PDFs
│   └── dailymed/               # FDA DailyMed drug label JSON files
│
├── agent.py                    # Pydantic AI agents + 3 RAG tools
├── main.py                     # FastAPI server and endpoints
├── ingest_pipeline.py          # One-time bulk ingestion script
├── Dockerfile                  # Container blueprint
├── requirements.txt            # Python dependencies
├── .env                        # API keys and secrets (not committed)
└── .gitignore
```

---

## 🧠 How the Agent Works

When a clinical query is received, the agent follows this mandatory reasoning loop:

```
User Query
    │
    ▼
[Tool 1] analyze_lab_results
    Interprets raw lab values against hardcoded clinical
    reference ranges (eGFR, HbA1c, Na, K, WBC, etc.)
    │
    ▼
[Tool 2] query_clinical_pathway
    Embeds the patient state and performs vector similarity
    search against WHO/NICE guideline chunks in Supabase
    │
    ▼
[Tool 3] verify_drug_safety_and_dosage
    Embeds the drug name + patient context and searches
    FDA DailyMed label chunks for exact dosing,
    contraindications, and renal adjustments
    │
    ▼
Grounded Clinical Response + Cited Sources
```

The system prompt enforces this order — the agent cannot hallucinate dosages because it must retrieve the FDA label before responding.

---

## 🗄️ Knowledge Base

### Clinical Guidelines (WHO/NICE PDFs)
| Source | Topic |
|---|---|
| NICE NG136 | Hypertension |
| WHO | Hypertension |
| NICE NG28 | Type 2 Diabetes |
| WHO | Diabetes |
| NICE NG106 | Heart Failure |
| NICE NG196 | Atrial Fibrillation |
| NICE NG238 | Lipid Management |
| NICE NG203 | Chronic Kidney Disease |
| NICE NG148 | Acute Kidney Injury |
| NICE NG80 | Asthma |
| NICE NG115 | COPD |
| NICE NG138 | Community-Acquired Pneumonia |
| NICE NG112 | Urinary Tract Infection |

### Drug Labels (FDA DailyMed)
Covers major drug classes across hypertension (ACE inhibitors, ARBs, beta-blockers, calcium channel blockers, diuretics), diabetes (Metformin, SGLT2 inhibitors, insulins, sulfonylureas, statins), heart failure (Digoxin, Sacubitril/Valsartan), anticoagulation (Apixaban, Rivaroxaban, Warfarin), respiratory (SABA, ICS, LABA, LAMA), and infectious disease (antibiotics for UTI, pneumonia).

---

## 🚀 Getting Started

### Prerequisites

- Docker Desktop installed
- API keys for the services you want to use
- A Supabase project with pgvector enabled and the knowledge base already ingested

### 1. Environment Setup

Create a `.env` file in the project root:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_role_key
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
LLM_MODEL=gemini-2.5-flash
```

### 2. Supabase Setup

Run the SQL statements in databse.sql to:
1. Create the required tables
2. Enable Vector similarity search 

In Supabase, do this by going to the "SQL Editor" tab and pasting in the SQL into the editor there. Then click "Run".

### 3. Ingest the Knowledge Base (one-time)

Place your PDF guidelines in `data/guidelines/` and drug JSON files in `data/dailymed/`, then run:

```bash
python ingest_pipeline.py
```

This only needs to be run once, or when adding new documents. Progress is saved to `processed_files.log` — the script is safe to restart if interrupted.

### 4. Build and Run with Docker

```bash
# Build the image (first build takes ~10 minutes due to model download)
docker build -t clinicalai-backend .

# Run the container
docker run --rm --env-file .env -p 8000:8000 clinicalai-backend
```

### 5. Open the Frontend

Open `frontend/index.html` directly in your browser. It connects to the backend at `http://localhost:8000` automatically.

---

## 🧪 Running Locally Without Docker

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open `frontend/index.html` in your browser.

---

## 🧪 Test Queries

Use these to verify the full pipeline is working:

**Lab interpretation only (Tool 1)**
```
Patient has HbA1c of 8.4% and eGFR of 38.
```

**Full three-tool pipeline**
```
65-year-old male with HbA1c 8.4% and eGFR 38. What is the treatment pathway and is Metformin safe?
```

**Drug safety + renal adjustment (Tools 2 + 3)**
```
Patient with Stage 3b CKD and atrial fibrillation. Which anticoagulant is appropriate and at what dose?
```

**Compare mode test**
Enable Compare Mode and send the same query to both windows to see the difference between RAG-grounded and generic LLM responses.

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/api/chat` | RAG clinical agent (3-tool pipeline) |
| POST | `/api/chat/generic` | Standard LLM, no RAG |

**Request body (both endpoints):**
```json
{
    "query": "Patient case or clinical question",
    "model": "gemini-2.5-flash"
}
```

**Response:**
```json
{
    "response": "Markdown-formatted clinical response",
    "sources": [
        {
            "content": "Retrieved chunk text",
            "metadata": {
                "title": "NICE — Hypertension",
                "similarity": 0.847
            }
        }
    ]
}
```

---

## ☁️ Deployment

The Docker image is ready for deployment on any container hosting platform.

**Render (recommended):**
1. Push code to GitHub
2. Create a new Web Service on Render → select Docker
3. Add all `.env` variables in the Render dashboard
4. Deploy — Render builds the image automatically on every push to `main`

**Environment variables required on the host:**
```
SUPABASE_URL
SUPABASE_SERVICE_KEY
GEMINI_API_KEY
OPENROUTER_API_KEY
EMBEDDING_MODEL
LLM_MODEL
FRONTEND_ORIGIN   # Set to your frontend's deployed URL
```

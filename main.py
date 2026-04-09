import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import Client
from pydantic_ai import Agent
from typing import List, Dict, Any, Optional

# We now import 'get_llm_model' from your agent.py so we can dynamically change models!
# from agent import clinical_assistant, PydanticAIDeps, get_llm_model
from agent import clinical_assistant, generic_assistant, PydanticAIDeps, get_llm_model

# Initialize the FastAPI app
app = FastAPI(
    title="Medical AI Assistant API",
    description="API for querying clinical documentation via RAG and Generic LLMs"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models for the API ---
class ChatRequest(BaseModel):
    query: str
    model: str  # <-- NEW: The frontend will now tell us which model to use!

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None # <-- NEW

# --- Database Initialization ---
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

if supabase_url and supabase_key:
    supabase_client = Client(supabase_url, supabase_key)
else:
    print("WARNING: Supabase credentials missing from environment.")
    supabase_client = None


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"status": "online", "message": "Medical AI Assistant API is running!"}

# --- ENDPOINT 1: The RAG Assistant (Left Window) ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not supabase_client:
        raise HTTPException(status_code=500, detail="Database connection not configured.")

    try:
        deps = PydanticAIDeps(supabase=supabase_client)
        
        # Load the specific model chosen in the frontend dropdown
        selected_llm = get_llm_model(request.model)
        
        # Run the RAG agent and override its default model with the selected one
        result = await clinical_assistant.run(request.query, deps=deps, model=selected_llm)
        return ChatResponse(response=result.output, sources=deps.sources)
        # return ChatResponse(response=result.output)

    except Exception as e:
        print(f"API Error (RAG): {e}")
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")

# --- ENDPOINT 2: The Generic Standard LLM (Right Window) ---
@app.post("/api/chat/generic", response_model=ChatResponse)
async def generic_chat_endpoint(request: ChatRequest):
    try:
        # Load the specific model chosen in the frontend dropdown
        selected_llm = get_llm_model(request.model)
        
        # Run the generic agent we imported from agent.py
        result = await generic_assistant.run(request.query, model=selected_llm)
        
        return ChatResponse(response=result.output)

    except Exception as e:
        print(f"API Error (Generic): {e}")
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")
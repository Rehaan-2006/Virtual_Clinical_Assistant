import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent
from supabase import create_client, Client
from typing import Any, Dict, List, Optional

from agent import clinical_assistant, generic_assistant, PydanticAIDeps, get_llm_model

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ==========================================
# APP SETUP
# ==========================================
app = FastAPI(
    title="Medical AI Assistant API",
    description="API for querying clinical documentation via RAG and Generic LLMs"
)

# CORS — restrict to your actual frontend origin.
# "*" with allow_credentials=True is rejected by browsers per the CORS spec.
# Update FRONTEND_ORIGIN in your .env for production.
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[FRONTEND_ORIGIN],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # Must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==========================================
# SUPABASE CLIENT
# ==========================================
# create_client() is the correct constructor — Client is only a type annotation.
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

supabase_client: Optional[Client] = None

if supabase_url and supabase_key:
    supabase_client = create_client(supabase_url, supabase_key)
    log.info("Supabase client initialised successfully.")
else:
    log.warning("Supabase credentials missing from environment. RAG endpoint will be unavailable.")

# ==========================================
# REQUEST / RESPONSE MODELS
# ==========================================
class ChatRequest(BaseModel):
    query: str
    model: str

    @field_validator("query")
    @classmethod
    def query_must_be_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        if len(v) > 4000:
            raise ValueError("Query exceeds maximum length of 4000 characters.")
        return v

    @field_validator("model")
    @classmethod
    def model_must_be_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Model name cannot be empty.")
        return v


class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None


# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/")
async def root():
    return {"status": "online", "message": "Medical AI Assistant API is running!"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    RAG-powered clinical assistant.
    Uses WHO/NICE guidelines and FDA drug label database.
    Always passes a fresh PydanticAIDeps instance so sources don't
    bleed across requests.
    """
    if not supabase_client:
        raise HTTPException(
            status_code=503,
            detail="Database connection is not configured. RAG queries are unavailable."
        )

    try:
        # Fresh deps per request — never reuse across calls
        deps = PydanticAIDeps(supabase=supabase_client)
        selected_llm = get_llm_model(request.model)

        result = await clinical_assistant.run(
            request.query,
            deps=deps,
            model=selected_llm
        )

        log.info(f"RAG query completed. Sources retrieved: {len(deps.sources)}")
        return ChatResponse(response=result.output, sources=deps.sources or None)

    except ValueError as e:
        # Raised by get_llm_model if API keys are missing
        log.error(f"Model configuration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        log.error(f"RAG endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again."
        )


@app.post("/api/chat/generic", response_model=ChatResponse)
async def generic_chat_endpoint(request: ChatRequest):
    """
    Generic LLM assistant with no RAG or database access.
    Answers from general medical knowledge only.
    """
    try:
        selected_llm = get_llm_model(request.model)
        result = await generic_assistant.run(request.query, model=selected_llm)

        log.info("Generic query completed.")
        return ChatResponse(response=result.output)

    except ValueError as e:
        log.error(f"Model configuration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        log.error(f"Generic endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again."
        )

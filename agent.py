from __future__ import annotations as _annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import logfire
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from sentence_transformers import SentenceTransformer
from supabase import Client

load_dotenv()

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ==========================================
# EMBEDDING MODEL
# ==========================================
# Loaded once globally so it stays in memory across requests
_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
embedding_model = SentenceTransformer(_model_name)

# ==========================================
# RAG THRESHOLDS
# Higher thresholds = stricter matching = safer for clinical use
# ==========================================
GUIDELINE_MATCH_THRESHOLD = 0.68   # WHO/NICE guidelines
DRUG_LABEL_MATCH_THRESHOLD = 0.72  # FDA drug safety data — strictest


# ==========================================
# LLM ROUTER
# ==========================================
def get_llm_model(model_name: str):
    """
    Routes to the correct LLM backend based on the model name prefix.

    Supported prefixes:
      - "ollama/<model>"  → local Ollama instance
      - "gemini-*"        → Google Gemini via native API
      - anything else     → OpenRouter (OpenAI-compatible)

    NOTE: Ollama and OpenRouter both use OpenAIModel but with different
    base URLs. These are passed directly into AsyncOpenAI clients rather
    than mutating os.environ, which avoids race conditions under concurrent load.
    """
    model_name_clean = model_name.lower().strip()

    # --- Route 1: Local Ollama ---
    if model_name_clean.startswith("ollama/"):
        actual_model = model_name_clean.split("/", 1)[1]
        ollama_host = "host.docker.internal" if os.path.exists("/.dockerenv") else "localhost"
        ollama_base_url = f"http://{ollama_host}:11434/v1"

        log.info(f"Routing to Ollama: model='{actual_model}' host='{ollama_host}'")

        # Pass base_url directly — no os.environ mutation
        client = AsyncOpenAI(base_url=ollama_base_url, api_key="ollama")
        return OpenAIModel(actual_model, openai_client=client)

    # --- Route 2: Google Gemini ---
    elif "gemini" in model_name_clean:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from environment variables.")
        log.info(f"Routing to Gemini: model='{model_name_clean}'")
        return GeminiModel(model_name_clean)

    # --- Route 3: OpenRouter ---
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing from environment variables.")

        log.info(f"Routing to OpenRouter: model='{model_name_clean}'")

        # Pass base_url directly — no os.environ mutation
        # If OpenAI client raises a parameter error here in your version,
        # fall back to the os.environ approach as a last resort.
        try:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            return OpenAIModel(model_name_clean, openai_client=client)
        except TypeError:
            # Older pydantic-ai versions may not accept openai_client param
            # Fall back to env mutation with a warning
            log.warning(
                "OpenAIModel does not accept openai_client param in this pydantic-ai version. "
                "Falling back to os.environ base URL mutation. "
                "Upgrade pydantic-ai to fix potential concurrency issues."
            )
            os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
            os.environ["OPENAI_API_KEY"] = api_key
            return OpenAIModel(model_name_clean)


# Boot the model once at startup
_llm_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
model = get_llm_model(_llm_name)

logfire.configure(send_to_logfire="if-token-present")


# ==========================================
# DEPENDENCIES
# ==========================================
@dataclass
class PydanticAIDeps:
    # Using Any to avoid hard import coupling; pass a supabase.Client at runtime
    supabase: Any
    # Tracks retrieved source chunks for the frontend "View Sources" feature.
    # IMPORTANT: always pass a fresh PydanticAIDeps() per request — never reuse
    # the same instance across turns, or sources from prior turns will accumulate.
    sources: List[Dict[str, Any]] = field(default_factory=list)


# ==========================================
# EMBEDDING HELPER
# ==========================================
async def get_embedding(text: str) -> List[float]:
    """
    Generate a 768-dim embedding using the local BAAI model.
    Raises RuntimeError on failure — never returns a zero vector,
    as that would cause silent garbage results in RAG retrieval.
    """
    try:
        embedding = embedding_model.encode(text).tolist()
        if all(v == 0.0 for v in embedding):
            raise ValueError("Embedding model returned an all-zero vector.")
        return embedding
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed: {e}") from e


# ==========================================
# CLINICAL RAG AGENT
# ==========================================
clinical_assistant = Agent(
    model,
    deps_type=PydanticAIDeps,
    retries=2, #5
    system_prompt="""
You are an expert AI Medical Clinical Assistant. You have access to a secure database of
WHO/NICE clinical guidelines and FDA DailyMed drug labels.

Your primary role is to assist clinicians by providing accurate, evidence-based medical
information. You MUST follow this strict reasoning loop when evaluating a patient case:

1. LABS FIRST: If the user provides any lab values (e.g., eGFR, Sodium, HbA1c), you MUST
   use the `analyze_lab_results` tool to interpret them before doing anything else.
2. PATHWAY SECOND: Use the `query_clinical_pathway` tool to find the official diagnostic
   and treatment guidelines for the patient's condition.
3. DOSAGE THIRD: If a specific medication is recommended, you MUST use the
   `verify_drug_safety_and_dosage` tool to check the FDA label for exact dosages,
   contraindications, and renal adjustments based on the patient's specific context.

Never guess or hallucinate dosages or guidelines. If the tools do not return relevant
information, explicitly state that it is outside your verified knowledge base and do not
proceed with a recommendation.
""",
)


# ==========================================
# GENERIC ASSISTANT (no RAG, no tools)
# ==========================================
# Intentionally has no deps_type and no tools.
# Do NOT pass PydanticAIDeps to this agent at runtime.
generic_assistant = Agent(
    model,
    system_prompt=(
        "You are an AI Medical Clinical Assistant. "
        "Answer the user's questions based on your general medical knowledge. "
        "Do not reference any specific documents or databases. "
        "Format your responses clearly using markdown."
    ),
)


# ==========================================
# TOOL 1: Lab Result Analyzer
# ==========================================
@clinical_assistant.tool
async def analyze_lab_results(ctx: RunContext[PydanticAIDeps], lab_name: str, value: float) -> str:
    """
    Analyzes a patient's raw lab result against standard clinical reference ranges.
    ALWAYS use this tool first if the user provides any lab values.

    Args:
        ctx: Run context (unused here but required for consistent tool signature).
        lab_name: The name of the lab test (e.g., 'sodium', 'hba1c', 'egfr').
        value: The numeric value of the lab result.
    """
    test_key = lab_name.lower().strip()

    reference_ranges = {
        "sodium":      {"min": 135.0, "max": 145.0,  "unit": "mEq/L",          "name": "Sodium"},
        "potassium":   {"min": 3.5,   "max": 5.0,    "unit": "mEq/L",          "name": "Potassium"},
        "creatinine":  {"min": 0.74,  "max": 1.35,   "unit": "mg/dL",          "name": "Creatinine"},
        "egfr":        {"min": 90.0,  "max": 999.0,  "unit": "mL/min/1.73m2",  "name": "eGFR"},
        "hba1c":       {"min": 4.0,   "max": 5.6,    "unit": "%",              "name": "Hemoglobin A1c"},
        "wbc":         {"min": 4.5,   "max": 11.0,   "unit": "K/uL",           "name": "White Blood Cell Count"},
        "hemoglobin":  {"min": 12.0,  "max": 17.5,   "unit": "g/dL",           "name": "Hemoglobin"},
        "platelets":   {"min": 150.0, "max": 450.0,  "unit": "K/uL",           "name": "Platelets"},
    }

    if test_key not in reference_ranges:
        return (
            f"System Note: Reference range for '{lab_name}' is not in the internal database. "
            f"Proceed with standard clinical caution."
        )

    lab_data   = reference_ranges[test_key]
    min_val    = lab_data["min"]
    max_val    = lab_data["max"]
    unit       = lab_data["unit"]
    proper_name = lab_data["name"]

    # --- Specialised interpretation logic ---
    if test_key == "hba1c":
        if value < 5.7:
            interpretation = "NORMAL"
        elif value <= 6.4:
            interpretation = "PREDIABETES — Lifestyle intervention recommended."
        else:
            interpretation = "DIABETES — Medical management required."

    elif test_key == "egfr":
        if value >= 90:
            interpretation = "NORMAL (Stage 1 CKD if kidney damage present)."
        elif value >= 60:
            interpretation = "MILDLY DECREASED (Stage 2 CKD)."
        elif value >= 45:
            interpretation = "MODERATELY DECREASED (Stage 3a CKD) — Monitor renal dosing carefully."
        elif value >= 30:
            interpretation = "MODERATE TO SEVERELY DECREASED (Stage 3b CKD) — Strict renal dosing required."
        elif value >= 15:
            interpretation = "SEVERELY DECREASED (Stage 4 CKD) — High risk for drug toxicity."
        else:
            interpretation = "KIDNEY FAILURE (Stage 5 CKD) — End-stage renal disease."

    else:
        if value < min_val:
            interpretation = "LOW — Below normal reference range."
        elif value > max_val:
            interpretation = "HIGH — Above normal reference range."
        else:
            interpretation = "NORMAL — Within standard reference range."

    return (
        f"--- LAB RESULT ANALYSIS ---\n"
        f"Test:            {proper_name}\n"
        f"Patient Value:   {value} {unit}\n"
        f"Standard Range:  {min_val} – {max_val} {unit}\n"
        f"INTERPRETATION:  {interpretation}\n"
        f"---------------------------"
    )


# ==========================================
# TOOL 2: Clinical Pathway Query
# ==========================================
@clinical_assistant.tool
async def query_clinical_pathway(
    ctx: RunContext[PydanticAIDeps],
    patient_presentation: str,
    suspected_condition: str
) -> str:
    """
    Searches official WHO/NICE Clinical Guidelines for the correct treatment pathway.
    Use AFTER analyzing lab results, but BEFORE verifying drug dosages.

    Args:
        ctx: Context holding the Supabase client and source tracker.
        patient_presentation: Brief patient summary (e.g., '65yo male, Stage 3a CKD').
        suspected_condition: Primary disease to search for (e.g., 'Hypertension').
    """
    search_query = (
        f"Clinical pathway and first-line treatment for {suspected_condition} "
        f"given: {patient_presentation}"
    )

    try:
        query_embedding = await get_embedding(search_query)
    except RuntimeError as e:
        log.error(f"query_clinical_pathway embedding failed: {e}")
        return "SYSTEM ERROR: Could not process the query. Please try again."

    try:
        result = ctx.deps.supabase.rpc(
            "match_guidelines",
            {
                "query_embedding": query_embedding,
                "match_threshold": GUIDELINE_MATCH_THRESHOLD,
                "match_count": 3,
            }
        ).execute()
    except Exception as e:
        log.error(f"query_clinical_pathway Supabase RPC failed: {e}")
        return (
            "SYSTEM ERROR: Clinical guideline database is temporarily unavailable. "
            "Do not proceed with treatment recommendations until this is resolved."
        )

    if not result.data:
        return (
            "No official WHO/NICE guidelines found for this specific presentation. "
            "Do not rely on unverified sources — escalate to a senior clinician."
        )

    formatted_chunks = []
    for doc in result.data:
        formatted_chunks.append(
            f"Source: {doc['source_org']} | Topic: {doc['disease_topic']}\n"
            f"Guideline: {doc['chunk_content']}"
        )
        ctx.deps.sources.append({
            "content": doc["chunk_content"],
            "metadata": {
                "title": f"{doc['source_org']} — {doc['disease_topic']}",
                "similarity": round(doc["similarity"], 3),
            }
        })

    return (
        "--- OFFICIAL CLINICAL GUIDELINES ---\n\n"
        + "\n\n---\n\n".join(formatted_chunks)
        + "\n\nINSTRUCTION: Use these pathways to determine the recommended intervention."
    )


# ==========================================
# TOOL 3: Drug Safety & Dosage Verifier
# ==========================================
@clinical_assistant.tool
async def verify_drug_safety_and_dosage(
    ctx: RunContext[PydanticAIDeps],
    drug_name: str,
    patient_context: str
) -> str:
    """
    Searches the official DailyMed/FDA database for drug dosages, warnings, and renal adjustments.
    Use AFTER a drug has been identified from the clinical pathway.

    Args:
        ctx: Context holding the Supabase client and source tracker.
        drug_name: Medication to look up (e.g., 'Lisinopril').
        patient_context: Relevant patient factors (e.g., 'eGFR 45, elderly, no ACE inhibitor history').
    """
    search_query = (
        f"Dosage, warnings, contraindications, and renal adjustment for {drug_name} "
        f"given patient context: {patient_context}"
    )

    try:
        query_embedding = await get_embedding(search_query)
    except RuntimeError as e:
        log.error(f"verify_drug_safety_and_dosage embedding failed: {e}")
        return "SYSTEM ERROR: Could not process the drug query. Please try again."

    try:
        result = ctx.deps.supabase.rpc(
            "match_drug_labels",
            {
                "query_embedding": query_embedding,
                "match_threshold": DRUG_LABEL_MATCH_THRESHOLD,
                "match_count": 3,
            }
        ).execute()
    except Exception as e:
        log.error(f"verify_drug_safety_and_dosage Supabase RPC failed: {e}")
        return (
            f"SYSTEM ERROR: FDA drug label database is temporarily unavailable. "
            f"DO NOT recommend a dosage for {drug_name} until safety data can be verified."
        )

    if not result.data:
        return (
            f"No official DailyMed FDA safety data found for '{drug_name}'. "
            f"DO NOT recommend specific dosages — verify manually before proceeding."
        )

    formatted_chunks = []
    for doc in result.data:
        formatted_chunks.append(
            f"Drug: {doc['drug_name']}\n"
            f"Official Label Data: {doc['chunk_content']}"
        )
        ctx.deps.sources.append({
            "content": doc["chunk_content"],
            "metadata": {
                "title": f"DailyMed FDA Label — {doc['drug_name']}",
                "similarity": round(doc["similarity"], 3),
            }
        })

    return (
        f"--- OFFICIAL FDA DRUG LABEL: {drug_name.upper()} ---\n\n"
        + "\n\n---\n\n".join(formatted_chunks)
        + "\n\nINSTRUCTION: Cross-reference this label with the patient's renal function "
          "and contraindications before recommending a dose."
    )

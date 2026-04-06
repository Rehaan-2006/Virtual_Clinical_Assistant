import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import Client

# Import the agent and dependencies from your agent.py file
from agent import clinical_assistant, PydanticAIDeps

# Initialize the FastAPI app
app = FastAPI(
    title="Medical AI Assistant API",
    description="API for querying clinical documentation via RAG"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local testing. (Change to your frontend domain in production!)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models for the API ---
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str

# --- Database Initialization ---
# We initialize the Supabase client globally so it's ready for all requests
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
    """Health check endpoint to verify the server is running."""
    return {"status": "online", "message": "Medical AI Assistant API is running!"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """The main endpoint that receives a question and returns the AI's answer."""
    if not supabase_client:
        raise HTTPException(status_code=500, detail="Database connection not configured.")

    try:
        # Set up the dependencies (Supabase) for this specific request
        deps = PydanticAIDeps(supabase=supabase_client)
        
        # Run the agent using the query sent by the frontend
        result = await clinical_assistant.run(request.query, deps=deps)
        
        # Return the AI's text output
        return ChatResponse(response=result.output)

    except Exception as e:
        print(f"API Error: {e}")
        # Return a clean 500 error to the frontend if something goes wrong
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")
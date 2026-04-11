from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from dotenv import load_dotenv
import logfire
import os

from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from supabase import Client
from typing import List, Dict, Any

# Import sentence-transformers for local BAAI embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- 1. Load Local Embedding Model ---
# We load this globally so the model stays in memory and doesn't reload on every request
model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
embedding_model = SentenceTransformer(model_name)

# --- 2. Dynamic Model Router ---
def get_llm_model(model_name: str):
    model_name_clean = model_name.lower().strip() 
    
    # 1. NEW: Route to Local Ollama
    # 1. NEW: Route to Local Ollama
    if model_name_clean.startswith("ollama/"):
        # Strip the "ollama/" prefix to get the real name (e.g., "llama3.2")
        actual_model = model_name_clean.split("/")[1]
        
        from pydantic_ai.models.openai import OpenAIModel
        # This checks if you are in Docker. If yes, it uses the Docker URL. If no, it uses localhost!
        ollama_host = "host.docker.internal" if os.path.exists("/.dockerenv") else "localhost"
        os.environ["OPENAI_BASE_URL"] = f"http://{ollama_host}:11434/v1"
        os.environ["OPENAI_API_KEY"] = "ollama"
        
        return OpenAIModel(actual_model)
        
    # 2. Route to Google Native API
    elif "gemini" in model_name_clean:
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY is missing from environment variables.")
        return GeminiModel(model_name_clean)
        
    # 3. Route EVERYTHING else to OpenRouter
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing from environment variables.")
            
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        os.environ["OPENAI_API_KEY"] = api_key

        from pydantic_ai.models.openai import OpenAIModel
        return OpenAIModel(model_name_clean)
    
# Read the requested model from .env, fallback to gemini-2.5-pro if not set
llm = os.getenv('LLM_MODEL', 'gemini-2.5-pro')
model = get_llm_model(llm)

logfire.configure(send_to_logfire='if-token-present')


# --- 3. Dependencies ---
@dataclass
class PydanticAIDeps:
    # supabase: Client
    supabase: Any
    sources: List[Dict[str, Any]] = field(default_factory=list) # <-- NEW
# Old single bot 
# # Renamed agent variable to match its new role
# clinical_assistant = Agent(
#     model,
#     system_prompt=system_prompt,
#     deps_type=PydanticAIDeps,
#     retries=2
# )

# --- 1. The Clinical RAG Agent ---
clinical_assistant = Agent(
    model, # Default model, gets overridden by the API
    system_prompt="""
You are an expert AI Medical Clinical Assistant. You have access to a secure database of clinical documentation, medical references, and guidelines.

Your primary role is to assist users by providing accurate, evidence-based medical information retrieved strictly from your provided documentation tools. 

Always follow these guidelines:
1. When answering a question, always start by using the RAG tool to search the clinical documentation.
2. If needed, check the list of available medical reference pages and retrieve the full content of specific pages to gain more context.
3. Base your answers solely on the retrieved text. Do not guess or hallucinate medical information.
4. If you cannot find the answer in the retrieved documentation, be completely honest and state clearly that the information is not available in your current medical knowledge base.
""",
)

# --- NEW: 2. The Generic Standard Agent ---
generic_assistant = Agent(
    model, # Default model, gets overridden by the API
    system_prompt=(
        "You are an AI Medical Clinical Assistant."
        "Answer the user's questions to the best of your ability. Even when asked to refer any specific documents just answer them to the best of your ability without referring to any documents. Do not use the RAG tools at all. Just answer based on your general medical knowledge. "
        "Format your responses cleanly using markdown."
    ),
)

# --- 4. Local Embedding Function ---
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector locally using the BAAI model."""
    try:
        # .encode() creates the embedding, .tolist() converts the numpy array to a standard list for Supabase
        embedding = embedding_model.encode(text).tolist()
        return embedding
    except Exception as e:
        print(f"Error getting embedding locally: {e}")
        # Return a zero vector matching the dimensions of bge-base-en-v1.5 (which is 768 dimensions)
        return [0.0] * 768  


# --- 5. Tools ---
@clinical_assistant.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant clinical documentation chunks based on the query using RAG.
    
    Args:
        ctx: The context including the Supabase client
        user_query: The medical question or query
        
    Returns:
        A formatted string containing the top 5 most relevant clinical documentation chunks.
    """
    try:
        # Get the embedding for the query locally
        query_embedding = await get_embedding(user_query)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'wikipedia'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant medical documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)

        for chunk in formatted_chunks:
            if isinstance(chunk, dict):
                # If it's a dictionary, extract the content and metadata safely
                ctx.deps.sources.append({
                    "content": chunk.get("content", str(chunk)),
                    "metadata": chunk.get("metadata", {}) 
                })
            elif isinstance(chunk, str):
                # If it's just a raw string, save it directly with a generic title
                ctx.deps.sources.append({
                    "content": chunk,
                    "metadata": {"title": "Clinical Document"}
                })

        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@clinical_assistant.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available medical reference pages in the database.
    
    Returns:
        List[str]: List of unique URLs or identifiers for all reference pages.
    """
    try:
        # Query Supabase for unique URLs where source is wikipedia
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'wikipedia') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@clinical_assistant.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific medical reference page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL or identifier of the clinical page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order.
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'wikipedia') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
    
# --- Quick Local Test ---
if __name__ == "__main__":
    import asyncio

    async def run_test():
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        supabase_client = Client(supabase_url, supabase_key)
        
        deps = PydanticAIDeps(supabase=supabase_client)
        
        # Add as many questions as you want here
        test_queries = [
            "What is the primary function of the human heart?",
            "How does the structure of the heart support its function in pumping blood?",
            "What are the four chambers of the human heart, and what does each one do?",
            "How does the heart work with the circulatory system to deliver oxygen to the body?",
            "What role does the heart play in digestion?"
        ]
        
        for query in test_queries:
            print(f"\n======================================")
            print(f"Testing Query: {query}")
            print(f"======================================\n")
            
            result = await clinical_assistant.run(query, deps=deps)
            
            # FIXED: Changed result.data to result.output based on the Pydantic AI warning
            print(f"--- AI Response ---\n{result.output}\n")
            
            # THE FIX: Tell the script to pause for 20 seconds before asking the next question
            print("⏳ Pausing for 20 seconds to respect Google's free-tier speed limits...")
            await asyncio.sleep(20)

    asyncio.run(run_test())
import os
from dotenv import load_dotenv
from llm_core.gemini_provider import GeminiProvider
from llm_core.openrouter_provider import OpenRouterProvider

# Load environment variables from the .env file
load_dotenv()

def get_llm_client(model_name: str):
    """
    Dynamically routes to the correct provider based on the model name.
    """
    model_name_lower = model_name.lower()

    # Route to Google if "gemini" is in the model name
    if "gemini" in model_name_lower:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from your .env file.")
        return GeminiProvider(api_key=api_key, model_name=model_name)

    # Otherwise, route to OpenRouter for models like Llama, Mistral, Qwen, etc.
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing from your .env file.")
        return OpenRouterProvider(api_key=api_key, default_model=model_name)
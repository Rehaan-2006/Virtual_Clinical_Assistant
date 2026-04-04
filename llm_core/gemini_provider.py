import google.generativeai as genai
from llm_provider import LLMProvider

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"[Gemini Error] {e}")
            raise
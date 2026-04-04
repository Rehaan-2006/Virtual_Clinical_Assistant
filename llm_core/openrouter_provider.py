import requests
from llm_provider import LLMProvider

class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str, default_model: str):
        self.api_key = api_key
        self.default_model = default_model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_text(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.default_model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=payload)
            response.raise_for_status() 
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"[OpenRouter Error] {e}")
            raise
"""
LLM Provider Abstraction
Supports multiple LLM providers: Gemini, Groq, etc.
"""

import os
import logging
from typing import Optional, Iterator
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion from prompt"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate completion with streaming"""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        import google.generativeai as genai
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(f'models/{model}')
        logger.info(f"Initialized Gemini provider with model: {model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion"""
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 8192)
        
        response = self.model.generate_content(
            prompt,
            generation_config={
                'temperature': temperature,
                'max_output_tokens': max_tokens,
            }
        )
        return response.text
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate with streaming (not commonly used for SQL generation)"""
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 8192)
        
        response = self.model.generate_content(
            prompt,
            generation_config={
                'temperature': temperature,
                'max_output_tokens': max_tokens,
            },
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text


class GroqProvider(LLMProvider):
    """Groq provider with fast inference"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq provider
        
        Available models:
        - llama-3.3-70b-versatile (recommended, 128K context)
        - llama-3.1-70b-versatile (128K context)
        - mixtral-8x7b-32768 (32K context, very fast)
        - llama3-groq-70b-8192-tool-use-preview (function calling)
        """
        from groq import Groq
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        logger.info(f"Initialized Groq provider with model: {model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion"""
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 8192)
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False,
            stop=None
        )
        
        return completion.choices[0].message.content
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate with streaming"""
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 8192)
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=True,
            stop=None
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaProvider(LLMProvider):
    """Ollama provider - 100% FREE, runs locally, NO API KEY needed!"""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama provider
        
        Available models (download with 'ollama pull <model>'):
        - llama3.2 (3B) - Fast, good for SQL (recommended)
        - llama3.2:1b - Very fast, smaller
        - codellama:7b - Code-focused
        - mistral - Good general purpose
        - phi3 - Microsoft's small model
        
        Args:
            model: Model name
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        logger.info(f"Initialized Ollama provider with model: {model}")
        logger.info(f"Make sure Ollama is running: ollama serve")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion"""
        import requests
        
        temperature = kwargs.get('temperature', 0.1)
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            },
            timeout=120  # Longer timeout for local processing
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.text}")
        
        return response.json()["response"]
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate with streaming"""
        import requests
        
        temperature = kwargs.get('temperature', 0.1)
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                }
            },
            stream=True,
            timeout=120
        )
        
        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API - FREE tier available"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initialize HuggingFace provider
        
        Free models:
        - meta-llama/Llama-3.2-3B-Instruct (recommended)
        - mistralai/Mistral-7B-Instruct-v0.2
        - google/flan-t5-xxl
        
        Args:
            api_key: HF token (get free at huggingface.co/settings/tokens)
            model: Model name
        """
        self.api_key = api_key or os.getenv("HUGGINGFACE_TOKEN")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_TOKEN not found")
        
        self.model = model
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        logger.info(f"Initialized HuggingFace provider with model: {model}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate completion"""
        import requests
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 2048)
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "return_full_text": False,
                }
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"HuggingFace error: {response.text}")
        
        result = response.json()
        if isinstance(result, list):
            return result[0]["generated_text"]
        return result["generated_text"]
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate with streaming (not supported, falls back to regular)"""
        yield self.generate(prompt, **kwargs)


def create_llm_provider(
    provider: str = "ollama",
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMProvider:
    """
    Factory function to create LLM provider
    
    Args:
        provider: "ollama" (free!), "gemini", "groq", or "huggingface"
        api_key: API key (or from env)
        model: Model name (uses defaults if not specified)
    
    Returns:
        LLMProvider instance
    """
    provider = provider.lower()
    
    if provider == "ollama":
        model = model or "llama3.2"
        return OllamaProvider(model=model)
    
    elif provider == "gemini":
        model = model or "gemini-2.5-flash"
        return GeminiProvider(api_key=api_key, model=model)
    
    elif provider == "groq":
        model = model or "llama-3.3-70b-versatile"
        return GroqProvider(api_key=api_key, model=model)
    
    elif provider == "huggingface" or provider == "hf":
        model = model or "meta-llama/Llama-3.2-3B-Instruct"
        return HuggingFaceProvider(api_key=api_key, model=model)
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama', 'gemini', 'groq', or 'huggingface'")


# Convenience function to get provider from environment
def get_default_provider() -> LLMProvider:
    """
    Get default LLM provider based on environment variables
    
    Priority:
    1. LLM_PROVIDER env var
    2. Try Ollama (free, no key needed!)
    3. If GROQ_API_KEY exists, use Groq
    4. If GEMINI_API_KEY exists, use Gemini
    5. If HUGGINGFACE_TOKEN exists, use HuggingFace
    6. Raise error
    """
    provider_name = os.getenv("LLM_PROVIDER", "").lower()
    
    if provider_name:
        logger.info(f"Using LLM provider from LLM_PROVIDER env: {provider_name}")
        return create_llm_provider(provider_name)
    
    # Try Ollama first (free, no API key!)
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            logger.info("Auto-detected Ollama running locally, using Ollama provider (FREE!)")
            return create_llm_provider("ollama")
    except:
        pass
    
    # Auto-detect based on available API keys
    if os.getenv("GROQ_API_KEY"):
        logger.info("Auto-detected Groq API key, using Groq provider")
        return create_llm_provider("groq")
    
    if os.getenv("GEMINI_API_KEY"):
        logger.info("Auto-detected Gemini API key, using Gemini provider")
        return create_llm_provider("gemini")
    
    if os.getenv("HUGGINGFACE_TOKEN"):
        logger.info("Auto-detected HuggingFace token, using HuggingFace provider")
        return create_llm_provider("huggingface")
    
    raise ValueError(
        "No LLM provider configured. Options:\n"
        "1. Ollama (FREE, no key!) - Install: https://ollama.com\n"
        "2. Groq - Set GROQ_API_KEY\n"
        "3. Gemini - Set GEMINI_API_KEY\n"
        "4. HuggingFace - Set HUGGINGFACE_TOKEN"
    )


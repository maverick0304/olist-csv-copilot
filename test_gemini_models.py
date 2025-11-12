"""Test which Gemini models are available with your API key"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not found in .env")
    exit(1)

print(f"✓ API Key found: {api_key[:10]}...")
print(f"✓ SDK Version: {genai.__version__}")
print()

genai.configure(api_key=api_key)

print("📋 Available models that support generateContent:\n")

available_models = []
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        available_models.append(model.name)
        print(f"  ✓ {model.name}")

print(f"\n📊 Total: {len(available_models)} models available")
print()

# Test each model
print("🧪 Testing models...\n")

test_prompt = "Say 'Hello' if you can read this."

for model_name in available_models[:3]:  # Test first 3
    try:
        print(f"Testing {model_name}...", end=" ")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(test_prompt)
        print(f"✅ WORKS - Response: {response.text[:50]}")
    except Exception as e:
        print(f"❌ FAILED - {str(e)[:80]}")

print("\n" + "="*60)
print("RECOMMENDATION:")
if available_models:
    print(f"Use this model name in agent.py: '{available_models[0]}'")
else:
    print("No models available - check your API key!")




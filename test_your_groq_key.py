"""
Quick test to verify your Groq API key works
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("=" * 60)
print("🔑 Testing Your Groq API Key")
print("=" * 60)

# Check if key exists
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file!")
    print("\n📝 Make sure your .env file has:")
    print("   GROQ_API_KEY=your_groq_key_here")
    exit(1)

print(f"✅ Found GROQ_API_KEY in .env")
print(f"   Key: {api_key[:20]}...{api_key[-10:]}")

# Test the API
try:
    print("\n🧪 Testing Groq API connection...")
    from groq import Groq
    
    client = Groq(api_key=api_key)
    
    # Simple test
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": "Say 'Hello! Groq is working!' in exactly 5 words."
            }
        ],
        temperature=0.1,
        max_completion_tokens=100,
    )
    
    response = completion.choices[0].message.content
    
    print("✅ SUCCESS! Groq API is working!")
    print(f"\n💬 Response: {response}")
    print("\n" + "=" * 60)
    print("✅ YOUR GROQ API KEY IS VALID AND WORKING!")
    print("=" * 60)
    print("\n🚀 Next step: Run the app!")
    print("   streamlit run app\\main.py")
    
except ImportError:
    print("❌ ERROR: 'groq' package not installed!")
    print("\n📝 Fix it:")
    print("   pip install groq")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\n📝 Possible issues:")
    print("   - Invalid API key")
    print("   - Network connection problem")
    print("   - API key doesn't have permissions")
    print("\n🔗 Get a new key: https://console.groq.com/keys")




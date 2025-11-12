"""
Debug script to check API key configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

print("=" * 70)
print("🔍 DEBUGGING API KEY ISSUE")
print("=" * 70)

# Check if .env file exists
env_path = Path(".env")
print(f"\n1️⃣ Checking .env file location...")
print(f"   Looking for: {env_path.absolute()}")

if env_path.exists():
    print(f"   ✅ Found .env file")
    print(f"\n   📄 Contents of .env file:")
    print("   " + "-" * 60)
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Hide most of the API key for security
                if 'API_KEY' in line and '=' in line:
                    key_name, key_value = line.split('=', 1)
                    if len(key_value) > 20:
                        masked = f"{key_value[:10]}...{key_value[-5:]}"
                    else:
                        masked = "***"
                    print(f"   {key_name}={masked}")
                else:
                    print(f"   {line}")
    print("   " + "-" * 60)
else:
    print(f"   ❌ .env file NOT FOUND!")
    print(f"\n   📝 Create .env file at: {env_path.absolute()}")
    print(f"   With contents:")
    print(f"   LLM_PROVIDER=groq")
    print(f"   GROQ_API_KEY=your_actual_key_here")

# Load environment variables
print(f"\n2️⃣ Loading environment variables...")
load_dotenv()
print(f"   ✅ dotenv loaded")

# Check what Python sees
groq_key = os.getenv("GROQ_API_KEY")
llm_provider = os.getenv("LLM_PROVIDER")

print(f"\n3️⃣ Checking environment variables...")
print(f"   LLM_PROVIDER = {llm_provider}")

if groq_key:
    print(f"   ✅ GROQ_API_KEY found")
    print(f"   Key length: {len(groq_key)} characters")
    print(f"   Starts with: {groq_key[:10]}...")
    print(f"   Ends with: ...{groq_key[-5:]}")
    
    # Check for common issues
    issues = []
    if groq_key.startswith('"') or groq_key.startswith("'"):
        issues.append("❌ Key has quotes around it - remove them!")
    if groq_key.startswith(' ') or groq_key.endswith(' '):
        issues.append("❌ Key has spaces - remove them!")
    if not groq_key.startswith('gsk_'):
        issues.append("⚠️  Key doesn't start with 'gsk_' - might be invalid")
    if len(groq_key) < 50:
        issues.append("⚠️  Key seems too short - typical keys are 56+ characters")
    
    if issues:
        print(f"\n   ⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"   ✅ Key format looks good")
else:
    print(f"   ❌ GROQ_API_KEY not found in environment!")

# Try to use the key
print(f"\n4️⃣ Testing API key with Groq...")

if not groq_key:
    print(f"   ❌ Cannot test - no API key found")
else:
    try:
        from groq import Groq
        
        client = Groq(api_key=groq_key)
        print(f"   ✅ Groq client created")
        
        print(f"   🔄 Making test API call...")
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": "Reply with only the word 'SUCCESS'"
                }
            ],
            temperature=0.1,
            max_completion_tokens=10,
        )
        
        response = completion.choices[0].message.content
        print(f"   ✅ API call succeeded!")
        print(f"   Response: {response}")
        
        print(f"\n" + "=" * 70)
        print(f"✅ YOUR API KEY IS WORKING CORRECTLY!")
        print(f"=" * 70)
        
    except ImportError:
        print(f"   ❌ groq package not installed")
        print(f"   Fix: pip install groq")
        
    except Exception as e:
        print(f"   ❌ API call failed: {e}")
        
        print(f"\n" + "=" * 70)
        print(f"❌ API KEY IS INVALID OR HAS ISSUES")
        print(f"=" * 70)
        
        print(f"\n🔧 HOW TO FIX:")
        print(f"\n1. Get a NEW API key:")
        print(f"   → Go to: https://console.groq.com/keys")
        print(f"   → Click 'Create API Key'")
        print(f"   → Copy the ENTIRE key")
        
        print(f"\n2. Update your .env file:")
        print(f"   → Open: {env_path.absolute()}")
        print(f"   → Replace the line with:")
        print(f"     GROQ_API_KEY=gsk_your_new_key_here")
        print(f"   → Save the file")
        
        print(f"\n3. Restart your application")
        
        print(f"\n⚠️  IMPORTANT:")
        print(f"   - NO quotes around the key")
        print(f"   - NO spaces before or after the =")
        print(f"   - Key should start with 'gsk_'")

print(f"\n" + "=" * 70)




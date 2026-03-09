import os
from google import genai

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
)

try:
    print("Trying generate_content with deep-research...")
    response = client.models.generate_content(
        model="models/deep-research-pro-preview-12-2025",
        contents="Give me a 1 sentence summary of quantum computing.",
    )
    print("Success with generate_content!")
    print(response.text)
except Exception as e:
    print(f"Failed generate_content: {e}")

try:
    print("\nTrying client.interactions...")
    interaction = client.interactions.create(
        agent="models/deep-research-pro-preview-12-2025",
        input="Give me a 1 sentence summary of quantum computing.",
    )
    print("Success with interactions!")
except Exception as e:
    print(f"Failed interactions: {e}")

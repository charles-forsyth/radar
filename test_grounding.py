import os
from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
)

try:
    print("Testing gemini-3.1-pro-preview with search grounding...")
    response = client.models.generate_content(
        model="models/gemini-3.1-pro-preview",
        contents="What is the latest news on solid state batteries?",
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}], temperature=0.2
        ),
    )
    print("Success!")
    print(response.text[:200] + "...")

    # Check if grounding metadata exists
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
            print("Grounding metadata found!")
            print(candidate.grounding_metadata)
except Exception as e:
    print(f"Error: {e}")

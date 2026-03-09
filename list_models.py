from google import genai
import os

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

print("Listing models...")
try:
    for model in client.models.list():
        # Printing the model name and other available attributes
        print(f"- {model.name}")
except Exception as e:
    print(f"Error: {e}")

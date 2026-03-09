import os
from google import genai
import inspect

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
)

print("Signature of client.interactions.create:")
print(inspect.signature(client.interactions.create))

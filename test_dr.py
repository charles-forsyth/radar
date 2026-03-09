import os
from google import genai
import time

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
)

try:
    print("Starting deep research...")
    interaction = client.interactions.create(
        agent="models/deep-research-pro-preview-12-2025",
        input="Summary of AI in 2026.",
        background=True,
    )
    print(f"Started. ID: {interaction.id}")

    while True:
        interaction = client.interactions.get(interaction.id)
        print(f"Status: {interaction.status}")
        if interaction.status == "completed":
            print(interaction.outputs[-1].text)
            break
        elif interaction.status == "failed":
            print(f"Failed: {getattr(interaction, 'error', 'Unknown Error')}")
            break
        time.sleep(10)
except Exception as e:
    print(f"Error: {e}")

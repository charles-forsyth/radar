from ddgs import DDGS
import json


def search():
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text("Tioga County PA Emergency Monitoring", max_results=3):
                results.append(r)
    except Exception as e:
        print(f"Error: {e}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    search()

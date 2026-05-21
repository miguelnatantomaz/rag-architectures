import requests
import sys


def check_ollama(model="llama3.2"):

    try:
        response = requests.get("http://localhost:11434/api/tags")

        if response.status_code != 200:
            raise Exception()

        models = response.json().get("models", [])

        available_models = [
            m["name"] for m in models
        ]

        if not any(model in m for m in available_models):

            print("\n[ERROR] Ollama is running but the model was not found.\n")

            print(f"Missing model: {model}\n")

            print("Run:\n")
            print(f"ollama pull {model}\n")

            sys.exit()

    except:

        print("\n[ERROR] Ollama is not running.\n")

        print("Start Ollama before running the project.\n")

        print("Download:")
        print("https://ollama.com\n")

        sys.exit()
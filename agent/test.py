import requests
import os

api_key = os.getenv("OPENROUTER_API_KEY")
model_id = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-small-3.2-24b-instruct:free") # Or hardcode
openrouter_api_base = "https://openrouter.ai/api/v1"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    # Consider adding a Referer header if you're deploying, as OpenRouter recommends:
    # "HTTP-Referer": "YOUR_APP_DOMAIN_OR_NAME"
}

data = {
    "model": model_id,
    "messages": [
        {"role": "user", "content": "Hello, is this working?"}
    ],
    "temperature": 0.7,
    "max_tokens": 50,
}

try:
    response = requests.post(
        f"{openrouter_api_base}/chat/completions",
        headers=headers,
        json=data
    )
    response.raise_for_status() # Raise an exception for HTTP errors
    result = response.json()
    print("Direct API Test Success:")
    print(result['choices'][0]['message']['content'])
except requests.exceptions.RequestException as e:
    print(f"Direct API Test Failed: {e}")
    if response.status_code == 429: print("Rate limit hit.")
    elif response.status_code == 401: print("Authentication failed.")
    print(f"Response content: {response.text}")
except KeyError as e:
    print(f"Unexpected response structure: {e}")
    print(f"Full response: {result}")
# google_genai_to_ollama_service

## 1. What is the Service?
This service acts as a proxy, converting Google’s GenAI API (e.g., Gemini models) into an Ollama-compatible API interface. It bridges Google’s generative AI capabilities with Ollama’s local LLM ecosystem, supporting both streaming (real-time output) and non-streaming (full response) modes. This allows tools or clients expecting an Ollama API to leverage Google’s cloud-based models seamlessly.

## 2. How to Run
### Requirements
- Python 3.12+
- Google API key (from Google AI Studio or Vertex AI)
- Install dependencies:
  ```shell
  pip install langchain langchain-google-genai flask python-dotenv

### Setup
Set API Key: Create a .env file:
```
GOOGLE_API_KEY=your-google-api-key
```

### Run the Service:
```shell
python3 google_genai_to_ollama_service.py
```
Expose (Optional): Use a tunnel service like ngrok to make it accessible:
```shell
ngrok http 11434
```
Copy the ngrok URL (e.g., `https://something.ngrok-free.app`) for external access.

### Notes
- **Port**: Uses 11434 to match Ollama’s default port.
- **API Compatibility**: The script simplifies the Ollama API format; adjust endpoints or response structure if your Ollama client expects more fields (e.g., `model`, `created_at`).
- **Model**: Uses `gemini-2.0-flash`; update to your desired Gemini model.

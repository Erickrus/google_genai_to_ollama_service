import json
import os

from flask import Flask, request, jsonify, Response
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import google.generativeai as genai


from dotenv import load_dotenv
load_dotenv()


import logging
from logging.handlers import RotatingFileHandler


# Configure logging
def setup_logging():
    # Create a logger
    logger = logging.getLogger('google_genai_to_ollama_service')
    logger.setLevel(logging.DEBUG)  # Set the base log level

    # Define log format
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler (outputs to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Show INFO and above in console
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File handler (writes to a log file with rotation)
    file_handler = RotatingFileHandler(
        'google_genai_to_ollama_service.log',  # Log file name
        maxBytes=1024 * 1024,                  # 1 MB per file
        backupCount=5                          # Keep 5 backup files
    )
    file_handler.setLevel(logging.DEBUG)       # Show DEBUG and above in file
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger

# Initialize logging at the top of the file
logger = setup_logging()


# Initialize Flask app
app = Flask(__name__)

# Load Google API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Version of the service
SERVICE_VERSION = "0.6.3"

genai.configure(api_key=GOOGLE_API_KEY)

# Helper function to stream responses
def stream_response(model_name, generator):
    for chunk in generator:
        yield json.dumps({
            "model": model_name,
            "message": {"role": "assistant", "content": chunk if isinstance(chunk, str) else chunk.content},
            "done": False
        }) + "\n"
    yield json.dumps({
        "model": model_name,
        "message": {"role": "assistant", "content": ""},
        "done": True
    }) + "\n"

# 1. POST /api/generate
@app.route("/api/generate", methods=["POST"])
def handle_generate():
    # Force parsing of JSON even if Content-Type is wrong
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON data"}), 400

    model_name = data.get("model", "gemini-pro")
    prompt = data.get("prompt", "")
    stream = data.get("stream", False)
    options = data.get("options", {})

    llm = GoogleGenerativeAI(
        model=model_name,
        google_api_key=GOOGLE_API_KEY,
        temperature=options.get("temperature", 0.7) or 0.7
    )

    if not stream:
        response_text = llm.invoke(prompt)
        return jsonify({
            "model": model_name,
            "response": response_text,
            "done": True
        })
    else:
        response_stream = llm.stream(prompt)
        return Response(stream_response(model_name, response_stream), mimetype="application/json")

# 2. POST /api/chat
@app.route("/api/chat", methods=["POST"])
def handle_chat():
    data = request.get_json()
    model_name = data.get("model", "gemini-pro")
    messages = data.get("messages", [])
    stream = data.get("stream", False)

    # Convert Ollama messages to LangChain format for non-streaming
    langchain_messages = []
    # Convert Ollama messages to Google format for streaming
    google_contents = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
            google_contents.append({"role": "system", "parts": [{"text": content}]})
        elif role == "user":
            langchain_messages.append(HumanMessage(content=content))
            google_contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
            google_contents.append({"role": "model", "parts": [{"text": content}]})

    #logger.info(f"{google_contents}\n")
    if not stream:
        # Non-streaming: Use langchain_google_genai (works reliably)
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.environ['GOOGLE_API_KEY']
        )
        try:
            response = llm.invoke(langchain_messages)
            return jsonify({
                "model": model_name,
                "message": {"role": "assistant", "content": response.content},
                "done": True
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        # Streaming: Use google.generativeai directly to avoid langchain issues
        try:
            model = genai.GenerativeModel(model_name)
            response_stream = model.generate_content(
                contents=google_contents,
                stream=True
            )

            def google_stream_adapter(model_name, stream):
                for chunk in stream:
                    yield json.dumps({
                        "model": model_name,
                        "message": {"role": "assistant", "content": chunk.text},
                        "done": False
                    }) + "\n"
                yield json.dumps({
                    "model": model_name,
                    "message": {"role": "assistant", "content": ""},
                    "done": True
                }) + "\n"

            return Response(google_stream_adapter(model_name, response_stream), mimetype="application/json")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response(json.dumps({"error": str(e)}), mimetype="application/json", status=500)

# 3. POST /api/embeddings
@app.route("/api/embeddings", methods=["POST"])
def handle_embeddings():
    data = request.get_json()
    model_name = data.get("model", "gemini-pro")
    prompt = data.get("prompt", "")

    # Initialize GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=GOOGLE_API_KEY
    )

    # Generate embeddings
    embedding_vector = embeddings.embed_query(prompt)

    return jsonify({
        "embeddings": embedding_vector
    })

# 4. GET /api/tags
@app.route("/api/tags", methods=["GET"])
def handle_tags():
    # Fetch available models from google.generativeai
    models = []
    for model in genai.list_models():
        # Strip 'models/' prefix from name (e.g., 'models/gemini-pro' -> 'gemini-pro')
        model_name = model.name[7:]
        # Use a static modified_at date since list_models() doesn't provide this
        modified_at = model.updated_at.isoformat() if hasattr(model, 'updated_at') else datetime.now().isoformat()
        # Size is not provided by Google API, so set to 0 as placeholder
        models.append({
            "name": model_name,
            "modified_at": modified_at,
            "size": 0  # Placeholder, as Google API doesn't expose model size
        })
    return jsonify({
        "models": models
    })

# 5. GET /api/version
@app.route("/api/version", methods=["GET"])
def handle_version():
    return jsonify({
        "version": SERVICE_VERSION
    })

# Run the service
if __name__ == "__main__":
    port = 11434  # Default Ollama port
    print(f"Starting google_genai_to_ollama_service on port {port}...")
    app.run(host="0.0.0.0", port=port, threaded=True)

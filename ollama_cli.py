import argparse
import requests
import json
import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# Default Ollama server URL
DEFAULT_OLLAMA_URL = "http://localhost:11434"

def parse_args():
    parser = argparse.ArgumentParser(description="CLI REPL to interact with Ollama using langchain_ollama")
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Enable streaming mode (default: non-streaming)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-pro",  # Default to a known valid model
        help="Model name to use (default: gemini-pro)"
    )
    return parser.parse_args()

def stream_response(response):
    """Handle streaming response from Ollama API"""
    full_content = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "message" in data and "content" in data["message"]:
                content = data["message"]["content"]
                print(content, end="", flush=True)
                full_content += content
            if data.get("done"):
                print()  # Newline after streaming completes
                break
    return full_content

def chat_with_ollama(args):
    """Main REPL loop"""
    # Initialize ChatOllama with the specified model and base URL
    llm = ChatOllama(model=args.model, base_url=args.url)

    print(f"Connected to Ollama at {args.url} with model {args.model}")
    print(f"Streaming mode: {'ON' if args.stream else 'OFF'}")
    print("Type 'exit' to quit, 'clear' to reset conversation.")

    # Store conversation history as LangChain messages
    messages = []

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            elif user_input.lower() == "clear":
                messages = []
                print("Conversation cleared.")
                continue

            # Add user message to history
            messages.append(HumanMessage(content=user_input))

            if args.stream:
                # Streaming mode: Manually call /api/chat
                payload = {
                    "model": args.model,
                    "messages": [
                        {"role": "user" if m.type == "human" else "assistant" if m.type == "ai" else "system", "content": m.content}
                        for m in messages
                    ],
                    "stream": True
                }
                response = requests.post(
                    f"{args.url}/api/chat",
                    json=payload,
                    stream=True,
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code != 200:
                    print(f"Error: {response.status_code} - {response.text}")
                    continue
                assistant_content = stream_response(response)
                messages.append(AIMessage(content=assistant_content))
            else:
                # Non-streaming mode: Use langchain_ollama
                response = llm.invoke(messages)
                assistant_content = response.content
                print(assistant_content)
                messages.append(AIMessage(content=assistant_content))

        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' to quit or continue.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")

if __name__ == "__main__":
    args = parse_args()
    chat_with_ollama(args)

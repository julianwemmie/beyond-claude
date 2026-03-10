"""
Interactive three-way chat: talk to Llama 3.1 8B via Ollama, Modal, or Groq.

Demonstrates that all three providers expose the same OpenAI-compatible API —
switching between local, self-hosted cloud, and managed inference is just
changing base_url and model name.

Usage:
    # Make sure at least one provider is available:
    # - Ollama: ollama pull llama3.1:8b && ollama serve
    # - Modal:  modal deploy modal_llama.py && export MODAL_URL="https://..."
    # - Groq:   export GROQ_API_KEY="gsk_..."

    uv run main.py
"""

import os
import sys
import time

import openai


# --- Provider configuration ---
# Each provider is just a (base_url, api_key, model_name) tuple.
# The OpenAI client works identically with all of them.

PROVIDERS = {}


def setup_providers():
    """Detect which providers are available based on env vars and local services."""

    # 1. Local — Ollama (always try, fail gracefully)
    PROVIDERS["ollama"] = {
        "label": "Ollama (local)",
        "client": openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        ),
        "model": "llama3.1:8b",
    }

    # 2. Self-hosted cloud — Modal + vLLM
    modal_url = os.environ.get("MODAL_URL", "")
    if modal_url:
        PROVIDERS["modal"] = {
            "label": "Modal (vLLM on A10 GPU)",
            "client": openai.OpenAI(
                base_url=f"{modal_url.rstrip('/')}/v1",
                api_key="not-needed",
            ),
            "model": "meta-llama/Llama-3.1-8B-Instruct",
        }

    # 3. Managed provider — Groq
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        PROVIDERS["groq"] = {
            "label": "Groq (managed inference)",
            "client": openai.OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
            ),
            "model": "llama-3.1-8b-instant",
        }


def list_providers():
    """Print available providers."""
    print("\nAvailable providers:")
    for i, (key, p) in enumerate(PROVIDERS.items(), 1):
        print(f"  {i}. [{key}] {p['label']}  (model: {p['model']})")
    print()


def pick_provider() -> dict | None:
    """Let the user pick a provider."""
    keys = list(PROVIDERS.keys())
    if not keys:
        print("No providers available! Set up at least one:")
        print("  - Ollama: ollama pull llama3.1:8b")
        print("  - Modal:  export MODAL_URL=...")
        print("  - Groq:   export GROQ_API_KEY=...")
        return None

    if len(keys) == 1:
        p = PROVIDERS[keys[0]]
        print(f"Using only available provider: {p['label']}")
        return p

    list_providers()
    while True:
        choice = input("Pick a provider (number or name): ").strip().lower()

        # Try as number
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                return PROVIDERS[keys[idx]]
        except ValueError:
            pass

        # Try as name
        if choice in PROVIDERS:
            return PROVIDERS[choice]

        print(f"Invalid choice '{choice}'. Try again.")


def chat(provider: dict):
    """Run an interactive chat loop with the given provider."""
    client = provider["client"]
    model = provider["model"]
    messages = []

    print(f"\nChatting with {provider['label']} ({model})")
    print("Type 'quit' to exit, 'switch' to change provider, 'clear' to reset history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            return "quit"

        if not user_input:
            continue
        if user_input.lower() == "quit":
            return "quit"
        if user_input.lower() == "switch":
            return "switch"
        if user_input.lower() == "clear":
            messages.clear()
            print("  (history cleared)\n")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1024,
            )
            elapsed = time.time() - start

            reply = response.choices[0].message.content
            tokens = response.usage.completion_tokens
            tps = tokens / elapsed if elapsed > 0 else 0

            messages.append({"role": "assistant", "content": reply})

            print(f"\nAssistant: {reply}")
            print(f"  [{tokens} tokens, {elapsed:.2f}s, {tps:.0f} tok/s]\n")

        except Exception as e:
            print(f"\n  Error: {e}")
            print("  (removing last message from history)\n")
            messages.pop()  # remove the failed user message


def main():
    print("=" * 60)
    print("  Three-Way Llama Chat: Ollama vs Modal vs Groq")
    print("  Same model, same API, three different backends")
    print("=" * 60)

    setup_providers()

    if not PROVIDERS:
        print("\nNo providers detected. See usage instructions above.")
        sys.exit(1)

    provider = pick_provider()
    if not provider:
        sys.exit(1)

    while True:
        result = chat(provider)
        if result == "quit":
            break
        elif result == "switch":
            provider = pick_provider()
            if not provider:
                break


if __name__ == "__main__":
    main()

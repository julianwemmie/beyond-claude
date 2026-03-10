"""
Three-way inference comparison: Local (Ollama) vs Modal (vLLM) vs Groq.

All three use the same OpenAI-compatible API — switching providers is just
changing base_url and model name.

Usage:
    # Set your Modal URL after deploying modal_llama.py
    export MODAL_URL="https://your-user--llama-inference-serve.modal.run"

    # Set your Groq API key (sign up at console.groq.com)
    export GROQ_API_KEY="gsk_..."

    # Make sure Ollama is running with llama3.1:8b pulled
    ollama pull llama3.1:8b

    python benchmark.py
"""

import os
import sys
import time

import openai

PROMPT = "Explain what a hash table is, how it handles collisions, and give a Python example."


def benchmark(client: openai.OpenAI, model_name: str, label: str) -> dict | None:
    """Time a single completion and measure tokens/second."""
    try:
        start = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=512,
        )
        elapsed = time.time() - start

        output = response.choices[0].message.content
        tokens = response.usage.completion_tokens
        tps = tokens / elapsed if elapsed > 0 else 0

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  Tokens generated:  {tokens}")
        print(f"  Total time:        {elapsed:.2f}s")
        print(f"  Tokens/sec:        {tps:.1f}")
        print(f"  First 200 chars:   {output[:200]}...")
        return {"label": label, "tokens": tokens, "time": elapsed, "tps": tps}

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"  {label} — SKIPPED")
        print(f"{'='*60}")
        print(f"  Error: {e}")
        return None


def main():
    results = []

    # --- 1. Local (Ollama) ---
    print("\nTesting LOCAL (Ollama)...")
    local_client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    r = benchmark(local_client, "llama3.1:8b", "LOCAL (Ollama on your machine)")
    if r:
        results.append(r)

    # --- 2. Self-hosted cloud (Modal + vLLM) ---
    modal_url = os.environ.get("MODAL_URL", "")
    if modal_url:
        print("\nTesting MODAL (vLLM on A10G)...")
        modal_client = openai.OpenAI(
            base_url=f"{modal_url}/v1",
            api_key="not-needed",
        )
        r = benchmark(
            modal_client,
            "meta-llama/Llama-3.1-8B-Instruct",
            "MODAL (your vLLM on A10G GPU)",
        )
        if r:
            results.append(r)
    else:
        print("\n  MODAL — SKIPPED (set MODAL_URL env var after deploying)")

    # --- 3. Managed provider (Groq) ---
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        print("\nTesting GROQ (managed inference)...")
        groq_client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
        )
        r = benchmark(
            groq_client,
            "llama-3.1-8b-instant",
            "GROQ (managed inference provider)",
        )
        if r:
            results.append(r)
    else:
        print("\n  GROQ — SKIPPED (set GROQ_API_KEY env var)")

    # --- Comparison table ---
    if len(results) < 2:
        print(f"\nOnly {len(results)} provider(s) succeeded. Need at least 2 for comparison.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  THREE-WAY COMPARISON")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['label']:<40} {r['tps']:>8.1f} tok/s  {r['time']:>6.2f}s total")

    fastest = max(results, key=lambda r: r["tps"])
    slowest = min(results, key=lambda r: r["tps"])
    print(f"\n  Fastest: {fastest['label']}")
    print(f"  {fastest['tps'] / slowest['tps']:.1f}x faster than {slowest['label']}")


if __name__ == "__main__":
    main()

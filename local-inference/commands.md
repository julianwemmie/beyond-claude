
## Ollama

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Explore ollama
ollama help

# Pull and run a model
ollama pull llama3.2:3b
ollama run llama3.2:3b

# Use the OpenAI-compatible API
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## llama.cpp

```bash
# Build from source
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && make -j

# Run a GGUF model
./llama-cli -m model.gguf -p "Explain recursion" -n 256

# Start an OpenAI-compatible server
./llama-server -m model.gguf --port 8080
```
import subprocess

import modal

MINUTES = 60
VLLM_PORT = 8000
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
N_GPU = 1

# Build a container image with vLLM installed on a CUDA base.
# Uses .uv_pip_install() for faster dependency resolution.
# FlashInfer is pulled in automatically as a vLLM dependency.
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# Persistent volumes to cache model weights and vLLM compilation artifacts.
# These survive across cold starts so you only download the model once.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("llama-inference")


@app.function(
    image=vllm_image,
    gpu=f"A10:{N_GPU}",  # Modal uses "A10" (not "A10G") — 24GB VRAM
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # If Llama 3.1 8B is gated, you need a HuggingFace token:
    # secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """Spawn a vLLM OpenAI-compatible server serving Llama 3.1 8B."""
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--enforce-eager",
        "--tensor-parallel-size", str(N_GPU),
        "--max-model-len", "4096",
    ]
    print("Starting vLLM with:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)

"""Quick CUDA availability check and minimal train test."""
import sys, torch
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print("No GPU — will use CPU", flush=True)

# Local Run Guide (TPS Prob Demo)

This guide shows how to create a lightweight local environment and run a minimal demo that exercises the `tps_prob` path in our sampler. It does not require large models or GPUs.

Prerequisites
- Python 3.8+
- Internet access to download tiny HuggingFace models

1) Create A New Virtual Environment (VSCode)
- Open VSCode in the repo root
- Terminal → New Terminal
- Create and activate a venv
  - Linux/macOS
    - `python -m venv .venv`
    - `source .venv/bin/activate`
  - Windows (PowerShell)
    - `python -m venv .venv`
    - `.\.venv\Scripts\Activate.ps1`

2) Install Requirements (CPU)
- Upgrade pip and install minimal deps
```
pip install -U pip
pip install "transformers==4.38.2" torch numpy
```
Notes
- The full LLaDA models are too large for local CPU/consumer GPUs; this demo uses a tiny HF model only to validate the code path.

3) Run The Minimal Demo
- From the repo root
```
python demo_tps_prob.py
```
- You should see two samples printed (short strings). Quality is not meaningful here—the purpose is verifying that the sampler with `confidence_source="tps_prob"` runs end-to-end.

Sampler Constraints
- Ensure these divisibility constraints hold when you change arguments:
  - `gen_length % block_length == 0`
  - `steps % (gen_length / block_length) == 0`

Optional: AMD GPU Acceleration
- For AMD RX 6500 XT, CPU is recommended for this demo. If you want to try GPU:

Windows (DirectML)
- `pip install torch-directml`
- In your Python code
```
import torch_directml
device = torch_directml.device()
model.to(device)
# move input tensors to device as well
```
- Set `model.device = device` before calling `generate()`

Linux (ROCm)
- Install a ROCm-enabled PyTorch build following the official ROCm instructions.
- Consumer GPUs may not be fully supported; if issues arise, fallback to CPU.

Next Steps
- Use this to validate local code changes to the sampler and deps.
- For full LLaDA/RemeDi inference or training, switch to a server environment (AutoDL) with sufficient GPU memory; we will provide a separate server run guide during the training phase.

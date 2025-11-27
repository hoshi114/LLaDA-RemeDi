# Remask SFT Run Guide (Server)

This guide describes how to run the minimal Remask SFT training on a GPU server (e.g., AutoDL). It trains only the UPS head on top of a frozen LLaDA backbone.

Prerequisites
- A GPU with >=24GB recommended for LLaDA 8B; otherwise start with a smaller model for smoke tests.
- Python 3.8+ and recent PyTorch with CUDA/ROCm according to your server.
- Internet access to download HF models and datasets.

1) Environment
- Create a new virtual environment
```
python -m venv .venv && source .venv/bin/activate
pip install -U pip
```
- Install core deps
```
pip install "transformers==4.38.2" datasets torch numpy
```
- (Optional) bf16 support: use a CUDA build that supports bf16 for better speed.

2) Choose Model & Dataset
- Model (recommended): `GSAI-ML/LLaDA-8B-Base` (backbone frozen)
- Datasets (start small):
  - `simplescaling/s1K-1.1` (bring-up)
  - then `HuggingFaceH4/MATH-500` (math) / `OpenCoder-LLM/opc-sft-stage2` subsets

3) Run Minimal Training (UPS Head Only)
- Start with s1K as plain text (answer-only) to validate pipeline
```
python remedi/train_remask_sft.py \
  --model_name GSAI-ML/LLaDA-8B-Base \
  --dataset simplescaling/s1K-1.1 \
  --text_field text \
  --seq_len 1024 \
  --batch_size 1 \
  --epochs 1 \
  --lr 1e-4 \
  --lambda_ups 1.0 \
  --r_incorrect 0.1 \
  --mask_id 126336 \
  --save_path checkpoints/ups_head.pt
```
Notes
- For paired datasets, pass `--prompt_field` and `--answer_field` instead of `--text_field`.
- The script freezes the backbone and trains only `UPSHead`.
- Loss = diffusion CE (masked-only) + UPS BCE (all positions; masked positions use stop-grad `pθ(x0|xt)`).

4) Use The Trained UPS Head For Inference
- Load the head and pass it to the sampler
```python
from remedi.modeling_wrappers import load_ups_head
from generate import generate

ups_head, meta = load_ups_head('checkpoints/ups_head.pt')
ups_head = ups_head.to(model.device).eval()

out = generate(
    model=model,
    prompt=input_ids,
    attention_mask=attention_mask,
    steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    confidence_source='ups',
    ups_head=ups_head,
    mask_id=126336,
)
```

5) Tips
- Start with `batch_size=1` and short `seq_len` (512–1024) on limited GPUs; increase gradually.
- If hidden states are not returned, set `trust_remote_code=True` and ensure the model supports `output_hidden_states=True`. If still unavailable, adapt with forward hooks.
- To scale beyond UPS-only, enable LoRA (not included in this minimal script) on selected layers.

6) Metrics & Diagnostics
- The script prints step-wise losses: total, diffusion, UPS, plus mask/incorrect fractions.
- For reporting, also visualize:
 - per-step mask counts (monotonicity)
  - incorrect-token detection AUC (add later)
  - sample trajectories comparing `random / tps_prob / ups` confidences

```text
Be cautious with 8B models on a single GPU; if OOM, switch to gradient checkpointing, smaller sequence length, or a smaller backbone for smoke tests.
```

7) Continue Training On Another Dataset (e.g., MATH-500)
- Use the previously trained UPS head as initialization and continue training on a new dataset to adapt the unmasking policy.
```
python remedi/train_remask_sft.py \
  --model_name GSAI-ML/LLaDA-8B-Base \
  --dataset HuggingFaceH4/MATH-500 \
  --split test \
  --seq_len 1024 \
  --batch_size 1 \
  --epochs 1 \
  --lr 3e-5 \
  --lambda_ups 1.0 \
  --r_incorrect 0.1 \
  --mask_id 126336 \
  --load_ups_head checkpoints/ups_head_s1k_len1024_b1_r010_l1p0_linear.pt \
  --save_path checkpoints/ups_head_s1k_math_len1024_b1_r010_l1p0_linear.pt
```
Notes
- Field mapping is auto-detected when possible; if it fails, specify `--text_field` or `--prompt_field/--answer_field` explicitly.
- When `--load_ups_head` is provided, `--ups_width` is ignored in favor of checkpoint width. Hidden size mismatch triggers reinit.

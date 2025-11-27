# RemeDi Plan

This document summarises our plan to implement RemeDi on top of LLaDA, reflecting training scaffold (SMDM), datasets, pitfalls, and compute/experiment roadmap.


## 0) Quick Overview

- Goal: Add UPS (Unmasking Policy Stream) beside TPS to enable dynamic remasking during diffusion steps; validate via Remask SFT.
- Minimal change: Keep LLaDA backbone frozen initially; add a small UPS head on top of last hidden states; modify sampler to actively remask low-confidence tokens.
- Training scaffold: Reuse SMDM masked-diffusion trainer (`SMDM/pretrain/train_mdm.py`) as the shell; add incorrect-token noise and the UPS BCE loss.
- Datasets: Start with s1K for bring-up, then MATH-500/OPC for targeted eval; optional GSM8K/HumanEval for stronger evidence.


## 1) Architecture & Injection Points

- UPS head
  - Design: small per-token head (Linear/GELU/Linear) mapping last hidden state → scalar h^i. Use `sigmoid(h)` as confidence.
  - Attachment: prefer grabbing last-layer hidden states via `output_hidden_states=True`; otherwise hook the final block.
  - Backward-compat: if UPS absent, fall back to TPS-based confidence or random.

- Sampler changes (dynamic remask)
  - File: `generate.py:43` main loop; compute TPS logits as before.
  - Confidence source (pluggable):
    - `ups`: use `sigmoid(h)` from UPS head.
    - `tps_prob`: current behavior using `pθ(x0|xt)` (softmax gather at argmax).
    - `random`: uniform.
  - Selection & masking:
    - Per step, choose top-K positions among non-prompt tokens (K from `get_num_transfer_tokens()`), call them U_n.
    - For i ∈ U_n: if `[MASK]`, fill with TPS sample; if token already present, keep it.
    - For i ∉ U_n (answer region only): force `x[i] = [MASK]` (remask), ensuring low-confidence tokens are revisited later.
  - Monotonicity: The K schedule enforces decreasing noise (mask count) overall.
  - OpenCompass: Thread a `confidence_source` flag from `opencompass/opencompass/models/dllm.py:503` into `generate()` and keep EOS/EoT and CFG as-is.


## 2) Pitfalls

- Hidden-state accessibility: Some trust_remote_code models may not expose `hidden_states`. First enable `output_hidden_states=True`; if unavailable, register a forward hook on the final transformer block to capture the last hidden states. As a fallback for pipeline validation, derive heuristic confidence from TPS statistics (e.g., token probability, entropy, or top-1 margin).
- Monotonic-noise constraint: The diffusion process requires the mask count to decrease monotonically. The per-step top-K schedule must align with block sampling and the answer-span boundary. At each step unmask exactly K_n non-prompt positions and force all other answer positions to `[MASK]`; avoid additional masking or unmasking outside the schedule.
- UPS label construction: For masked positions, construct UPS labels using `pθ(x0|xt)` from TPS and apply stop-gradient (`detach`) to prevent gradient leakage. Label clean tokens with 1 and incorrect tokens with 0. Ensure the BCE target tensor is built with the correct indexing for all token types.
- Memory and stability: Freeze the backbone and train only the UPS head initially to reduce memory/compute and stabilize optimization. Introduce LoRA adapters later if necessary. Prefer shorter sequences, bfloat16, and gradient accumulation when constrained by memory.
- Evaluation comparability: When comparing confidence sources (`random`, `tps_prob`, `ups`), hold steps, `block_length`, EOS/EoT gating, CFG, and temperature fixed. Report the per-step mask-count curve to verify the monotonicity assumption.
- Backward compatibility: Maintain a no-UPS fallback path so CLI, Gradio, and OpenCompass flows remain functional. Gate the new behavior behind configuration flags and provide conservative defaults.


## 3) Training Plan Using SMDM

- Shell to reuse: `SMDM/pretrain/train_mdm.py` (masked diffusion). Key references:
  - Forward masking routine similar to GUIDELINES: `SMDM/pretrain/train_mdm.py:104` (forward_process), loss uses masked positions only (`:232–264`).
- Our Remask SFT deltas:
  1) Two noises at sampled t:
     - Mask noise: ρ_mask(t) = t.
     - Incorrect-token noise in answer region: ρ_incorrect(t) = 4 r t (1−t), r≈0.1.
  2) Losses:
     - Diffusion CE (masked-only), divided by p_mask as in SMDM/LLaDA.
     - UPS BCE over all positions with labels:
       - clean tokens: y=1; incorrect tokens: y=0; masked tokens: y=pθ(x0|xt) with stopgrad.
  3) SFT prompt rule: do not add any noise to the prompt tokens (follow `GUIDELINES.md`).
- Implementation sketch:
  - Wrap the HF LLaDA model with an UPS head module (e.g., `remedi/modeling_wrappers.py`) that returns `(logits, ups_scores)` from `forward(noisy_input)`.
  - Replace `forward_process` to output both masked indices and a second set of indices for incorrect tokens in the answer region.
  - Keep optimizer/training loop/ckpt/logging from SMDM; start with backbone frozen and train only the UPS head params; optionally enable LoRA later.


## 4) Datasets Strategy

- simplescaling/s1K-1.1 (general, small)
  - Role: bring-up dataset to validate end-to-end dynamic remask quickly; good for “first signal” on UPS.
- HuggingFaceH4/MATH-500 (math reasoning)
  - Role: stress-test remasking on multi-step reasoning; visualize “wrong→remask→correct” token trajectories; report small-sample accuracy.
- OpenCoder-LLM/opc-sft-stage2
  - Subsets:
    - `educational_instruct`: easier, stable for training.
    - `evol_instruct`: harder, tests robustness and iterative refinement.
    - `mceval_instruct`: closer to objective QA; useful for token-level remask evaluation.
  - Role: probe gains on code/instruction-style tasks beyond math.
- Optional (eval-only, to strengthen the report):
  - GSM8K (math word problems) and HumanEval/MBPP (code). We can sample a small subset if compute is tight.
- Order of use:
  1) s1K-1.1 for UPS-head SFT sanity.
  2) MATH-500 small eval and visualization.
  3) OPC subsets for instruction/code behavior; then GSM8K/HumanEval (subset) for headline comparisons if needed.


## 5) Compute & Experiments (AutoDL)

- When to move to server (AutoDL):
  - As soon as we start Remask SFT (training UPS head) or run OpenCompass/LM-Eval at scale. Local can do quick inference-only checks.
- Minimal experiment suite for report:
  1) Inference-only ablations (local ok): same prompts under `random / tps_prob / ups` confidences; log per-step top-K indices, remasked positions, and mask-count curve to verify monotonicity.
  2) Remask SFT on s1K-1.1 (server): freeze backbone, train UPS head.
     - Metrics: incorrect-token detection AUC, token-level accuracy on answer segment, qualitative trajectories.
     - Ablations: r (ρ_incorrect strength), steps vs block_length, confidence source.
  3) Targeted eval:
     - Math: MATH-500 small split; optionally GSM8K subset (20–50 items).
     - Code/Instr: OPC subsets and optionally HumanEval subset (10–20 tasks).
  4) Stability knobs: EOS/EoT gating, CFG, temperature; show a small grid or selected meaningful points.
- Artifacts to include:
  - Trajectory visualizations (before/after remask), mask-count per step, confusion matrices for UPS detection, and brief runtime/latency stats.


## 6) Step-by-Step Execution

- Phase A: Wire-up (local) [completed]
  - Added UPS head wrapper; added `confidence_source` to sampler; implemented dynamic remasking; preserved fallbacks.
- Phase B: Minimal SFT (server) [completed]
  - Implemented Remask SFT; trained UPS head on s1K-1.1 (seq_len=1024, batch=1); verified inference.
- Phase C: Targeted eval & continued SFT (server) [completed]
  - Continued training on MATH-500 (test) and GSM8K (train). Small-sample GSM8K (n=50): ups acc=0.360, tps_prob acc=0.260, random acc=0.120.
- Phase D: Optional scaling (server) [pending]
  - Enable LoRA for a couple of layers; ablations and larger-scale evaluation.


## 7) File Pointers

- `paper/RemeDi.tex`: UPS/TPS definitions; Remask SFT labels and schedule.
- `GUIDELINES.md`: forward_process; SFT prompt handling; masked-only CE weighting.
- `generate.py:8` add_gumbel_noise; `:22` get_num_transfer_tokens; `:43` main loop; `:100–107` confidence; `:111–118` token transfer (we’ll add remask for non-selected).
- `opencompass/opencompass/models/dllm.py:69` LLaDAModel; `:503` LLaDABaseModel.generate() call into sampler; `:518–529` sampler args.
- SMDM trainer shell: `SMDM/pretrain/train_mdm.py:104` forward_process; `:220–540` training loop & loss.


## 8) Defaults, Fallbacks, Risks

- Default to UPS head if available; otherwise use TPS probability; `random` only for baselines.
- If hidden states inaccessible, use hooks; if hooks fail, use TPS heuristics; later swap in proper UPS once training completes.
- Keep prompt clean during SFT; apply both noises only to the answer span.

## 9) Next Steps

- Ablations: sweep `r_incorrect` (0.05/0.1/0.2), `lambda_ups` (0.5–2.0), `ups_width` (0/512/1024), `seq_len` (1024/1536/2048).
- Broader evaluation: larger GSM8K/MATH subsets and OPC subsets; consider OpenCompass tasks.
- LoRA: optional partial backbone adaptation if further gains are desired.

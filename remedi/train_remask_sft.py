import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None  # defer import error until actually needed

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from remedi.modeling_wrappers import UPSHead


@dataclass
class SFTExample:
    input_ids: torch.Tensor   # (L,)
    prompt_length: int        # scalar


def collate_batch(batch, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate to rectangular tensors.

    Args:
        batch: list of SFTExample
        pad_id: tokenizer pad token id
    Returns:
        input_ids: (B, L)
        prompt_lengths: (B,)
    """
    max_len = max(x.input_ids.size(0) for x in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    prompt_lengths = torch.zeros((bsz,), dtype=torch.long)
    for i, ex in enumerate(batch):
        L = ex.input_ids.size(0)
        input_ids[i, :L] = ex.input_ids
        prompt_lengths[i] = ex.prompt_length
    return input_ids, prompt_lengths


def _detect_fields(sample: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Heuristically detect field mapping from a dataset sample.

    Returns (text_field, prompt_field, answer_field).
    If (prompt_field, answer_field) are resolved, text_field is None.
    If single-field found, (text_field, None, None) is returned.
    """
    keys = set(sample.keys())
    # Try paired first
    prompt_cands = ['prompt', 'instruction', 'question', 'input', 'query']
    answer_cands = ['answer', 'response', 'output', 'target', 'completion', 'solution']
    for p in prompt_cands:
        for a in answer_cands:
            if p in keys and a in keys:
                return None, p, a
    # Fallback single text field
    text_cands = ['text', 'content', 'response', 'output', 'answer', 'completion', 'target', 'solution', 'body']
    for f in text_cands:
        if f in keys:
            return f, None, None
    return None, None, None


def build_dataset(dataset: str,
                  subset: Optional[str],
                  tokenizer: AutoTokenizer,
                  text_field: Optional[str],
                  prompt_field: Optional[str],
                  answer_field: Optional[str],
                  max_length: int,
                  split: str = "train",
                  auto_fields: bool = True):
    """Load an HF dataset and tokenize into SFTExample items.

    This function supports three shapes:
    - Single text field (text_field): entire text treated as answer (prompt_length=0).
    - Two fields (prompt_field, answer_field): prompt+answer concatenation, prompt kept clean during SFT.
    - Fallback synthetic: if datasets not available, raises an error.
    """
    assert load_dataset is not None, "Please `pip install datasets` to use HF datasets."
    # Robust split loading: try the requested split, otherwise fallback to an available split
    try:
        ds = load_dataset(dataset, subset, split=split)
    except Exception:
        ds_all = load_dataset(dataset, subset)
        # choose a split by preference order
        pref = [split, 'train', 'train[:90%]', 'validation', 'test']
        chosen = None
        for k in pref:
            if k in ds_all:
                chosen = k
                break
        if chosen is None:
            # pick the first available key
            chosen = list(ds_all.keys())[0]
        print(f"[Info] Requested split '{split}' not found. Falling back to split '{chosen}'. Available: {list(ds_all.keys())}")
        ds = ds_all[chosen]

    # Inspect first sample to validate/resolve fields
    first = ds[0]
    keys = set(first.keys())
    resolved_tf, resolved_pf, resolved_af = text_field, prompt_field, answer_field
    def _has(field):
        return (field is not None) and (field in keys)
    # If user-provided fields don't exist, try auto-detect when allowed
    if prompt_field and answer_field:
        if not (_has(prompt_field) and _has(answer_field)):
            if auto_fields:
                resolved_tf, resolved_pf, resolved_af = _detect_fields(first)
            else:
                raise KeyError(f"Fields not found: {prompt_field}, {answer_field}. Available: {sorted(keys)}")
    elif text_field:
        if not _has(text_field):
            if auto_fields:
                resolved_tf, resolved_pf, resolved_af = _detect_fields(first)
            else:
                raise KeyError(f"Field not found: {text_field}. Available: {sorted(keys)}")
    else:
        if auto_fields:
            resolved_tf, resolved_pf, resolved_af = _detect_fields(first)
        else:
            raise ValueError("Please provide either (text_field) or (prompt_field, answer_field)")

    items = []
    for row in ds:
        if resolved_pf and resolved_af:
            prompt = str(row[resolved_pf])
            answer = str(row[resolved_af])
            text = prompt + answer
            enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length, return_tensors="pt")
            ids = enc["input_ids"][0]
            pl = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            items.append(SFTExample(ids, pl))
        elif resolved_tf:
            text = str(row[resolved_tf])
            enc = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length, return_tensors="pt")
            ids = enc["input_ids"][0]
            items.append(SFTExample(ids, 0))
        else:
            raise ValueError(f"Cannot resolve fields from dataset. Available keys: {sorted(keys)}")

    return items


def sample_noises(input_ids: torch.Tensor,
                  prompt_lengths: torch.Tensor,
                  mask_id: int,
                  vocab_size: int,
                  r_incorrect: float = 0.1,
                  eps: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct noisy inputs x_t with two noises: mask and incorrect tokens (answer span only).

    Returns:
        noisy: (B, L) x_t
        mask_indices: (B, L) boolean for x_t == [MASK]
        incorrect_indices: (B, L) boolean for incorrect tokens
        p_mask: (B, L) mask probability per token position (zeros on prompt)
    """
    B, L = input_ids.shape
    device = input_ids.device

    # Sample diffusion time per sample
    t = torch.rand((B,), device=device)
    p_mask_scalar = (1 - eps) * t + eps  # (B,)

    # Broadcast to positions but keep prompt clean (p=0 for prompt)
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    prompt_mask = pos < prompt_lengths.unsqueeze(1)

    p_mask = p_mask_scalar.unsqueeze(1).repeat(1, L)
    p_mask[prompt_mask] = 0.0

    # Bernoulli sample mask positions
    mask_indices = torch.rand((B, L), device=device) < p_mask
    noisy = torch.where(mask_indices, torch.tensor(mask_id, device=device), input_ids)

    # Incorrect token ratio per sample (paper: 4 r t (1-t))
    p_incorrect_scalar = 4.0 * r_incorrect * t * (1.0 - t)  # (B,)
    p_incorrect = p_incorrect_scalar.unsqueeze(1).repeat(1, L)
    # Only on answer span and only where not masked
    answer_mask = ~prompt_mask
    incorrect_candidates = answer_mask & (~mask_indices)
    incorrect_draw = torch.rand((B, L), device=device) < p_incorrect
    incorrect_indices = incorrect_candidates & incorrect_draw

    if incorrect_indices.any():
        # Replace with random alternative token (avoid original and mask_id)
        rand = torch.randint(low=0, high=vocab_size, size=input_ids.shape, device=device)
        # Avoid collisions: if equals original or mask id, add 1 modulo vocab_size
        collide = (rand == input_ids) | (rand == mask_id)
        rand = torch.where(collide, (rand + 1) % vocab_size, rand)
        noisy = torch.where(incorrect_indices, rand, noisy)

    return noisy, mask_indices, incorrect_indices, p_mask


def compute_losses(model,
                   ups_head: UPSHead,
                   noisy: torch.Tensor,
                   target: torch.Tensor,
                   mask_indices: torch.Tensor,
                   incorrect_indices: torch.Tensor,
                   p_mask: torch.Tensor,
                   lambda_ups: float = 1.0,
                   attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute diffusion CE on masked tokens and UPS BCE across all tokens.
    """
    # Backbone is frozen; wrap in no_grad to save memory
    with torch.no_grad():
        outputs = model(noisy, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        logits = outputs.logits  # (B, L, V)

    # Diffusion loss: masked positions only, weighted by 1/p_mask (following LLaDA/SMDM)
    if mask_indices.any():
        # Cast to float32 for numerical stability
        ce = F.cross_entropy(
            logits[mask_indices].float(),
            target[mask_indices],
            reduction='none'
        )
        denom = p_mask[mask_indices].float().clamp_min(1e-6)
        diff_loss = (ce / denom).sum() / (target.numel())
    else:
        diff_loss = torch.tensor(0.0, device=target.device)

    # UPS loss: build labels for all positions
    with torch.no_grad():
        p = F.softmax(logits.float(), dim=-1)
        gt_prob = torch.gather(p, dim=-1, index=target.unsqueeze(-1)).squeeze(-1).float()  # (B, L)
        y = torch.where(mask_indices, gt_prob, torch.ones_like(gt_prob))  # default 1
        y = torch.where(incorrect_indices, torch.zeros_like(y), y)        # incorrect -> 0
        y = y.detach()

    # Hidden states for UPS head
    hidden = outputs.hidden_states[-1] if outputs.hidden_states is not None else None
    assert hidden is not None, "Model must return hidden_states; see trust_remote_code or hooks."
    # Align dtypes to avoid matmul dtype mismatch
    head_dtype = next(ups_head.parameters()).dtype
    h = ups_head(hidden.to(head_dtype))  # (B, L), raw logits
    bce = F.binary_cross_entropy_with_logits(h.float(), y.float(), reduction='mean')

    loss = diff_loss + lambda_ups * bce
    metrics = {
        'loss': float(loss.detach().item()),
        'loss_diff': float(diff_loss.detach().item()),
        'loss_ups': float(bce.detach().item()),
        'mask_frac': float(mask_indices.float().mean().detach().item()),
        'incorrect_frac': float(incorrect_indices.float().mean().detach().item()),
    }
    return loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='HF model id, e.g., GSAI-ML/LLaDA-8B-Base')
    parser.add_argument('--dataset', type=str, default='simplescaling/s1K-1.1')
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--text_field', type=str, default=None)
    parser.add_argument('--prompt_field', type=str, default=None)
    parser.add_argument('--answer_field', type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_ups', type=float, default=1.0)
    parser.add_argument('--r_incorrect', type=float, default=0.1)
    parser.add_argument('--mask_id', type=int, default=126336)
    parser.add_argument('--ups_width', type=int, default=0, help='hidden MLP width; 0 for single linear')
    parser.add_argument('--split', type=str, default='train', help='dataset split name; auto-fallback if unavailable')
    parser.add_argument('--load_ups_head', type=str, default=None, help='path to an existing UPSHead checkpoint to continue training')
    parser.add_argument('--save_path', type=str, default='checkpoints/ups_head.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    # Ensure pad token id exists for batching
    if tokenizer.pad_token_id is None:
        # fall back to eos as pad
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError('Tokenizer must have pad_token_id or eos_token to form batches')

    # Load model (supports encoder-like LLaDA or causal LM)
    try:
        model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.eval()
    model.to(device)

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # Build dataset
    items = build_dataset(
        dataset=args.dataset,
        subset=args.subset,
        tokenizer=tokenizer,
        text_field=args.text_field,
        prompt_field=args.prompt_field,
        answer_field=args.answer_field,
        max_length=args.seq_len,
        split=args.split,
        auto_fields=True
    )
    def _collate(batch):
        return collate_batch(batch, pad_id=tokenizer.pad_token_id)
    loader = DataLoader(items, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)

    # Infer hidden size for UPS head: run a tiny forward to get hidden state shape
    with torch.no_grad():
        test_ids, test_pl = next(iter(loader))
        test_ids = test_ids.to(device)
        out = model(test_ids, output_hidden_states=True, return_dict=True)
        if out.hidden_states is None:
            raise RuntimeError('Model did not return hidden_states; set trust_remote_code=True or adapt with hooks.')
        hidden_size = out.hidden_states[-1].size(-1)
    
    # Build or load UPS head (for continued training)
    if args.load_ups_head:
        from remedi.modeling_wrappers import load_ups_head as _load_head
        print(f"Loading UPS head from {args.load_ups_head}")
        loaded_head, meta = _load_head(args.load_ups_head, hidden_size=hidden_size)
        ckpt_hs = int(meta.get('hidden_size', hidden_size))
        ckpt_width = int(meta.get('width', 0))
        if ckpt_hs != hidden_size:
            print(f"[Warn] checkpoint hidden_size={ckpt_hs} != runtime hidden_size={hidden_size}. Reinitializing a new head.")
            ups_head = UPSHead(hidden_size=hidden_size, width=args.ups_width).to(device)
        else:
            ups_head = loaded_head.to(device)
            # Note: if a different ups_width is passed, we keep the checkpoint width for compatibility
            if args.ups_width and args.ups_width != ckpt_width:
                print(f"[Info] Ignoring --ups_width={args.ups_width} due to loaded checkpoint width={ckpt_width}.")
    else:
        ups_head = UPSHead(hidden_size=hidden_size, width=args.ups_width).to(device)
    optimizer = torch.optim.AdamW(ups_head.parameters(), lr=args.lr)

    total_steps = 0
    for epoch in range(args.epochs):
        for input_ids, prompt_lengths in loader:
            input_ids = input_ids.to(device)
            prompt_lengths = prompt_lengths.to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.long)

            noisy, mask_idx, incorrect_idx, p_mask = sample_noises(
                input_ids, prompt_lengths, mask_id=args.mask_id, vocab_size=tokenizer.vocab_size, r_incorrect=args.r_incorrect
            )

            loss, metrics = compute_losses(
                model=model,
                ups_head=ups_head,
                noisy=noisy,
                target=input_ids,
                mask_indices=mask_idx,
                incorrect_indices=incorrect_idx,
                p_mask=p_mask,
                lambda_ups=args.lambda_ups,
                attention_mask=attention_mask
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_steps += 1
            if total_steps % 10 == 0:
                print(f"step={total_steps} loss={metrics['loss']:.4f} diff={metrics['loss_diff']:.4f} ups={metrics['loss_ups']:.4f} mask_frac={metrics['mask_frac']:.4f} incorrect_frac={metrics['incorrect_frac']:.4f}")

    # Save UPS head checkpoint
    ckpt = {
        'state_dict': ups_head.state_dict(),
        'hidden_size': hidden_size,
        'width': args.ups_width,
        'model_name': args.model_name,
        'tokenizer_name': args.model_name,
        'mask_id': args.mask_id,
    }
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(ckpt, args.save_path)
    print(f"Saved UPS head to {args.save_path}")


if __name__ == '__main__':
    import os
    main()

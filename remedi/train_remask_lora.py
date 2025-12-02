import argparse
import os
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from remedi.modeling_wrappers import UPSHead
from remedi.train_utils import (
    SFTExample,
    collate_batch,
    build_dataset,
    sample_noises,
)


def parse_lora_targets(arg: str) -> List[str]:
    if arg.lower() == 'auto':
        # Common attn/MLP module names across LLaMA/LLaDA-like models
        return [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'up_proj', 'down_proj', 'gate_proj',
            'fc1', 'fc2',
            'wq', 'wk', 'wv', 'wo',
        ]
    # comma-separated list
    return [x.strip() for x in arg.split(',') if x.strip()]


def create_lora_model(model: nn.Module,
                      target: List[str],
                      r: int,
                      alpha: int,
                      dropout: float,
                      is_causal_lm: bool):
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as e:
        raise RuntimeError("Please `pip install peft` to use LoRA training") from e

    # Freeze all backbone params first; LoRA layers will carry gradients
    for p in model.parameters():
        p.requires_grad = False

    task_type = TaskType.CAUSAL_LM if is_causal_lm else TaskType.FEATURE_EXTRACTION
    lconf = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target,
        task_type=task_type,
        bias='none',
    )
    lora_model = get_peft_model(model, lconf)
    return lora_model


def main():
    parser = argparse.ArgumentParser(description='RemeDi Remask SFT with LoRA (TPS + UPS)')

    # Base / data
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='simplescaling/s1K-1.1')
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--text_field', type=str, default=None)
    parser.add_argument('--prompt_field', type=str, default=None)
    parser.add_argument('--answer_field', type=str, default=None)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=1)

    # LoRA
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_target', type=str, default='auto', help="comma list or 'auto'")
    parser.add_argument('--load_lora_dir', type=str, default=None)
    parser.add_argument('--save_lora_dir', type=str, default='checkpoints/lora')

    # UPS head
    parser.add_argument('--ups_width', type=int, default=0)
    parser.add_argument('--load_ups_head', type=str, default=None)
    parser.add_argument('--save_ups_head', type=str, default='checkpoints/ups_head.pt')

    # Objectives / mask noise
    parser.add_argument('--lambda_ups', type=float, default=1.0)
    parser.add_argument('--r_incorrect', type=float, default=0.1)
    parser.add_argument('--mask_id', type=int, default=126336)

    # Optim
    parser.add_argument('--lr_lora', type=float, default=5e-5)
    parser.add_argument('--lr_ups', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--clip_grad', type=float, default=1.0)

    # Training mode
    parser.add_argument('--train_mode', type=str, default='joint', choices=['joint', 'ups-only', 'tps-only', 'alternating'])
    parser.add_argument('--alt_ratio', type=int, default=4, help='TPS:UPS steps ratio when train_mode=alternating')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=0, help='0 to save only at the end')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError('Tokenizer must have pad_token_id or eos_token to form batches')

    # Base model
    is_causal = False
    try:
        model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=dtype)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=dtype)
        is_causal = True
    model.to(device)
    model.eval()  # PEFT will attach trainable adapters; base stays eval to reduce dropout effects

    # Wrap with LoRA
    target_modules = parse_lora_targets(args.lora_target)
    model = create_lora_model(model, target_modules, args.lora_r, args.lora_alpha, args.lora_dropout, is_causal)
    model.print_trainable_parameters()

    # Optionally load existing LoRA adapters
    if args.load_lora_dir:
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError("Please `pip install peft` to load LoRA weights") from e
        model = PeftModel.from_pretrained(model, args.load_lora_dir)

    # Build dataset/loader
    items = build_dataset(
        dataset=args.dataset,
        subset=args.subset,
        tokenizer=tokenizer,
        text_field=args.text_field,
        prompt_field=args.prompt_field,
        answer_field=args.answer_field,
        max_length=args.seq_len,
        split=args.split,
        auto_fields=True,
    )
    def _collate(batch):
        return collate_batch(batch, pad_id=tokenizer.pad_token_id)
    loader = DataLoader(items, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)

    # Probe hidden size to build/load UPS head
    with torch.no_grad():
        test_ids, _ = next(iter(loader))
        test_ids = test_ids.to(device)
        out = model(test_ids, output_hidden_states=True, return_dict=True)
        if out.hidden_states is None:
            raise RuntimeError('Model did not return hidden_states; set trust_remote_code=True or adapt with hooks.')
        hidden_size = out.hidden_states[-1].size(-1)

    if args.load_ups_head:
        from remedi.modeling_wrappers import load_ups_head as _load_head
        print(f"Loading UPS head from {args.load_ups_head}")
        loaded_head, meta = _load_head(args.load_ups_head, hidden_size=hidden_size)
        ckpt_hs = int(meta.get('hidden_size', hidden_size))
        if ckpt_hs != hidden_size:
            print(f"[Warn] checkpoint hidden_size={ckpt_hs} != runtime hidden_size={hidden_size}. Reinitializing a new head.")
            ups_head = UPSHead(hidden_size=hidden_size, width=args.ups_width).to(device)
        else:
            ups_head = loaded_head.to(device)
    else:
        ups_head = UPSHead(hidden_size=hidden_size, width=args.ups_width).to(device)

    # Two optimizers
    lora_params = [p for p in model.parameters() if p.requires_grad]
    ups_params = list(ups_head.parameters())
    opt_lora = torch.optim.AdamW(lora_params, lr=args.lr_lora, weight_decay=args.weight_decay)
    opt_ups = torch.optim.AdamW(ups_params, lr=args.lr_ups, weight_decay=args.weight_decay)

    global_step = 0
    tps_steps_in_cycle = max(int(args.alt_ratio), 1)
    ups_steps_in_cycle = 1
    cycle_len = tps_steps_in_cycle + ups_steps_in_cycle

    def should_run_tps(step_idx: int) -> bool:
        if args.train_mode == 'tps-only':
            return True
        if args.train_mode == 'ups-only':
            return False
        if args.train_mode == 'joint':
            return True
        # alternating
        pos = step_idx % cycle_len
        return pos < tps_steps_in_cycle

    def should_run_ups(step_idx: int) -> bool:
        if args.train_mode == 'tps-only':
            return False
        if args.train_mode == 'ups-only':
            return True
        if args.train_mode == 'joint':
            return True
        # alternating
        pos = step_idx % cycle_len
        return pos >= tps_steps_in_cycle

    scaler = None  # bf16; AMP not required here

    for epoch in range(args.epochs):
        accum_i = 0
        opt_lora.zero_grad(set_to_none=True)
        opt_ups.zero_grad(set_to_none=True)
        for input_ids, prompt_lengths in loader:
            input_ids = input_ids.to(device)
            prompt_lengths = prompt_lengths.to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.long)

            noisy, mask_idx, incorrect_idx, p_mask = sample_noises(
                input_ids, prompt_lengths, mask_id=args.mask_id, vocab_size=tokenizer.vocab_size, r_incorrect=args.r_incorrect
            )

            loss_tps = torch.tensor(0.0, device=device)
            loss_ups = torch.tensor(0.0, device=device)

            # Forward
            outputs = model(noisy, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            logits = outputs.logits

            # Decide which objectives to run
            run_tps = (should_run_tps(global_step) or args.train_mode == 'joint') and bool(mask_idx.any())
            run_ups = (should_run_ups(global_step) or args.train_mode == 'joint')

            # TPS loss (CE on mask)
            if run_tps:
                ce = torch.nn.functional.cross_entropy(
                    logits[mask_idx].float(),
                    input_ids[mask_idx],
                    reduction='none',
                )
                denom = p_mask[mask_idx].float().clamp_min(1e-6)
                loss_tps = (ce / denom).sum() / (input_ids.numel())
                # Backprop only to LoRA
                (loss_tps / args.grad_accum).backward()
            else:
                loss_tps = torch.tensor(0.0, device=device)

            # UPS loss (BCE on all tokens); labels use stop-grad pθ(x0|xt)
            if run_ups:
                with torch.no_grad():
                    sel_logits = torch.gather(logits, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1).float()
                    log_denom = torch.logsumexp(logits.float(), dim=-1)
                    gt_prob = torch.exp(sel_logits - log_denom)
                    y = torch.where(mask_idx, gt_prob, torch.ones_like(gt_prob))
                    y = torch.where(incorrect_idx, torch.zeros_like(y), y)
                    y = y.detach()

                hidden = outputs.hidden_states[-1] if outputs.hidden_states is not None else None
                assert hidden is not None, 'Model must return hidden_states; set trust_remote_code=True or adapt with hooks.'
                # Detach hidden to prevent BCE from updating backbone
                head_dtype = next(ups_head.parameters()).dtype
                h = ups_head(hidden.detach().to(head_dtype))
                loss_ups = torch.nn.functional.binary_cross_entropy_with_logits(h.float(), y.float(), reduction='mean')
                (loss_ups / args.grad_accum).backward()
            
            accum_i += 1
            if accum_i % args.grad_accum == 0:
                if run_tps or args.train_mode in ('joint', 'tps-only'):
                    if args.clip_grad and args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], args.clip_grad)
                    opt_lora.step()
                    opt_lora.zero_grad(set_to_none=True)

                if run_ups or args.train_mode in ('joint', 'ups-only'):
                    if args.clip_grad and args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(ups_head.parameters(), args.clip_grad)
                    opt_ups.step()
                    opt_ups.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % args.log_every == 0:
                    total_loss = (loss_tps + args.lambda_ups * loss_ups).detach().item()
                    print(f"step={global_step} loss={total_loss:.4f} diff={loss_tps.detach().item():.4f} ups={loss_ups.detach().item():.4f} mask_frac={float(mask_idx.float().mean().item()):.4f} incorrect_frac={float(incorrect_idx.float().mean().item()):.4f}")

                if args.save_every and (global_step % args.save_every == 0):
                    os.makedirs(args.save_lora_dir, exist_ok=True)
                    model.save_pretrained(args.save_lora_dir)
                    ckpt = {
                        'state_dict': ups_head.state_dict(),
                        'hidden_size': hidden_size,
                        'width': args.ups_width,
                        'model_name': args.model_name,
                        'tokenizer_name': args.model_name,
                        'mask_id': args.mask_id,
                    }
                    os.makedirs(os.path.dirname(args.save_ups_head), exist_ok=True)
                    torch.save(ckpt, args.save_ups_head)
                    print(f"[ckpt] Saved LoRA -> {args.save_lora_dir}; UPS head -> {args.save_ups_head}")

    # Final save
    os.makedirs(args.save_lora_dir, exist_ok=True)
    model.save_pretrained(args.save_lora_dir)
    ckpt = {
        'state_dict': ups_head.state_dict(),
        'hidden_size': hidden_size,
        'width': args.ups_width,
        'model_name': args.model_name,
        'tokenizer_name': args.model_name,
        'mask_id': args.mask_id,
    }
    os.makedirs(os.path.dirname(args.save_ups_head), exist_ok=True)
    torch.save(ckpt, args.save_ups_head)
    print(f"Saved LoRA to {args.save_lora_dir} and UPS head to {args.save_ups_head}")


if __name__ == '__main__':
    main()

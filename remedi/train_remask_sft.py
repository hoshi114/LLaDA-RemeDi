import argparse
from typing import Optional

import os
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from remedi.modeling_wrappers import UPSHead
from remedi.train_utils import (
    SFTExample,
    collate_batch,
    build_dataset,
    sample_noises,
    compute_losses,
)


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
                attention_mask=attention_mask,
                freeze_backbone=True,
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

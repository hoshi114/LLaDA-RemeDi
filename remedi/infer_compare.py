import argparse
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from generate import generate
from remedi.modeling_wrappers import load_ups_head


def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Load HF model/tokenizer with trust_remote_code and move to device.

    Falls back between AutoModelForCausalLM and AutoModel to be compatible with
    encoder-style diffusion backbones.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Prefer non-causal if available; otherwise use CausalLM
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval().to(device)
    # Ensure downstream code can read model.device
    try:
        _ = model.device
    except AttributeError:
        model.device = device
    return model, tokenizer


def prepare_inputs(tokenizer, prompts: List[str], instruct: bool = False):
    """Tokenize prompts with left padding; optionally apply chat template.

    For Instruct models, we apply the chat template with generation prompt.
    """
    if instruct:
        messages = [{"role": "user", "content": p} for p in prompts]
        prompts = [tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False) for m in messages]

    # Prefer left padding to simplify block sampling implementation
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    enc = tokenizer(
        prompts,
        padding=True,
        add_special_tokens=False,
        return_tensors='pt'
    )
    return enc['input_ids'], enc['attention_mask']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--ups_head', type=str, default=None, help='Path to trained UPS head checkpoint (.pt)')
    parser.add_argument('--prompts_file', type=str, default=None, help='Text file with one prompt per line')
    parser.add_argument('--prompt', action='append', default=None, help='Prompt string; can be repeated')
    parser.add_argument('--instruct', action='store_true', help='Apply chat template for instruct models')
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--gen_length', type=int, default=128)
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--cfg_scale', type=float, default=0.0)
    parser.add_argument('--mask_id', type=int, default=126336)
    parser.add_argument('--logits_eos_inf', action='store_true')
    parser.add_argument('--confidence_eos_eot_inf', action='store_true')
    parser.add_argument('--max_prompts', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    # Compose prompts
    prompts: List[str] = []
    if args.prompts_file:
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    if args.prompt:
        prompts.extend(args.prompt)
    if not prompts:
        prompts = [
            "Give a brief plan for learning calculus.",
            "Write a Python function to reverse a string.",
            "A shop sells apples for $2 and bananas for $1. If I buy 3 apples and 4 bananas, how much do I pay?",
        ]
    prompts = prompts[: args.max_prompts]

    # Tokenize
    input_ids, attention_mask = prepare_inputs(tokenizer, prompts, instruct=args.instruct)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # Safety: pad_token and mask_id
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id != args.mask_id, "pad_token_id must differ from mask_id for correct masking behavior"

    # Load UPS head if provided; align to model dtype/device
    ups_head = None
    if args.ups_head:
        ups_head, meta = load_ups_head(args.ups_head, hidden_size=None)
        ups_head = ups_head.to(model.device).to(getattr(model, 'dtype', torch.float32)).eval()

    # Define confidence sources to compare
    sources = []
    if ups_head is not None:
        sources.append(('ups', ups_head))
    sources.append(('tps_prob', None))
    sources.append(('random', None))

    print(f"Prompts: {len(prompts)}; steps={args.steps} gen_length={args.gen_length} block_length={args.block_length}")
    for name, head in sources:
        print("=" * 80)
        print(f"Confidence source: {name}")
        out = generate(
            model=model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            confidence_source=name,
            ups_head=head,
            mask_id=args.mask_id,
            logits_eos_inf=args.logits_eos_inf,
            confidence_eos_eot_inf=args.confidence_eos_eot_inf,
        )
        decoded = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
        for i, s in enumerate(decoded):
            print(f"[Prompt {i}] {prompts[i]}")
            print(f"[Output] {s}")
            print("-" * 80)


if __name__ == '__main__':
    main()


import argparse
import json
import os
import random
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from generate import generate
from remedi.modeling_wrappers import load_ups_head
from remedi.eval_gsm_math import extract_answer_gsm8k, extract_answer_math500, is_equal


def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    except Exception:
        pass


def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32))
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32))
    model.eval().to(device)
    try:
        _ = model.device
    except AttributeError:
        model.device = device
    return model, tokenizer


def apply_template_if_needed(tokenizer, prompts: List[str], instruct: bool) -> List[str]:
    if not instruct:
        return prompts
    messages = [{"role": "user", "content": p} for p in prompts]
    return [tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False) for m in messages]


def prepare_inputs(tokenizer, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    enc = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='pt')
    return enc['input_ids'], enc['attention_mask']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--ups_head', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='openai/gsm8k')
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--format', type=str, default='gsm8k', choices=['gsm8k', 'math'])
    parser.add_argument('--max_samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--instruct', action='store_true')
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--gen_length', type=int, default=128)
    parser.add_argument('--block_length', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--cfg_scale', type=float, default=0.0)
    parser.add_argument('--mask_id', type=int, default=126336)
    parser.add_argument('--eos_gating', action='store_true')
    parser.add_argument('--out_jsonl', type=str, default='outputs/detailed_eval.jsonl')
    parser.add_argument('--trace_indices', action='store_true', help='store selected/newly-filled indices per step')
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()

    if args.deterministic:
        set_deterministic()
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    # Load dataset
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    total = min(args.max_samples, len(ds))

    # Load UPS head
    ups_head, meta = load_ups_head(args.ups_head, hidden_size=None)
    ups_head = ups_head.to(model.device).to(getattr(model, 'dtype', torch.float32)).eval()

    # Extractors
    extract_pred = extract_answer_gsm8k if args.format == 'gsm8k' else extract_answer_math500

    with open(args.out_jsonl, 'w', encoding='utf-8') as f:
        for i in range(total):
            row = ds[i]
            # Heuristic fields
            if args.format == 'gsm8k':
                prompt = str(row.get('question', row.get('prompt', '')))
                gold = str(row.get('answer', ''))
            else:
                prompt = str(row.get('problem', row.get('prompt', '')))
                gold = str(row.get('solution', ''))
            prompts = [prompt]
            templ = apply_template_if_needed(tokenizer, prompts, args.instruct)
            input_ids, attention_mask = prepare_inputs(tokenizer, templ)
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

            # UPS
            out_ups, trace_ups = generate(
                model=model,
                prompt=input_ids,
                attention_mask=attention_mask,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                confidence_source='ups',
                ups_head=ups_head,
                mask_id=args.mask_id,
                logits_eos_inf=args.eos_gating,
                confidence_eos_eot_inf=args.eos_gating,
                return_trace=True,
                trace_batch_index=0,
                trace_store_indices=args.trace_indices,
            )
            pred_ups = tokenizer.decode(out_ups[0, input_ids.size(1):], skip_special_tokens=True)

            # TPS
            out_tps, trace_tps = generate(
                model=model,
                prompt=input_ids,
                attention_mask=attention_mask,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                confidence_source='tps_prob',
                ups_head=None,
                mask_id=args.mask_id,
                logits_eos_inf=args.eos_gating,
                confidence_eos_eot_inf=args.eos_gating,
                return_trace=True,
                trace_batch_index=0,
                trace_store_indices=args.trace_indices,
            )
            pred_tps = tokenizer.decode(out_tps[0, input_ids.size(1):], skip_special_tokens=True)

            # Compare
            p_ups = extract_pred(pred_ups)
            p_tps = extract_pred(pred_tps)
            g_ans = extract_pred(gold)  # use same extractor for gold formatting
            cu = is_equal(p_ups, g_ans)
            ct = is_equal(p_tps, g_ans)
            label = 'tie_correct' if (cu and ct) else ('ups_win' if (cu and not ct) else ('tps_win' if (ct and not cu) else 'tie_wrong'))

            rec = {
                'id': int(i),
                'prompt': prompt,
                'gold': gold,
                'pred_ups': pred_ups,
                'pred_tps': pred_tps,
                'ans_ups': p_ups,
                'ans_tps': p_tps,
                'ans_gold': g_ans,
                'correct_ups': bool(cu),
                'correct_tps': bool(ct),
                'label': label,
                'trace_ups': trace_ups,
                'trace_tps': trace_tps,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

            # Free cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Saved detailed records to {args.out_jsonl}")


if __name__ == '__main__':
    main()


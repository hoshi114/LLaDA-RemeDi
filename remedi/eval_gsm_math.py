import argparse
import math
import re
from typing import List, Tuple, Optional, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from generate import generate
from remedi.modeling_wrappers import load_ups_head


def load_model_and_tokenizer(model_name: str, device: torch.device, lora_path: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32))
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32))
    model.eval().to(device)
    if lora_path:
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError("--lora_path specified but peft is not installed. Please `pip install peft`." ) from e
        model = PeftModel.from_pretrained(model, lora_path)
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


def extract_answer_gsm8k(text: str) -> Optional[str]:
    # GSM8K answers are usually after the last occurrence of '####'
    if '####' in text:
        last = text.rsplit('####', 1)[-1]
        last = last.strip()
        if last:
            # first non-empty line after ####
            for line in last.splitlines():
                if line.strip():
                    ans = re.sub(r'[,\s]+', '', line.strip())
                    return ans
        # fall through to numeric fallback if nothing valid after ####
    # fallback: last number in text
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else None


def extract_answer_math500(text: str) -> Optional[str]:
    # Common pattern: \boxed{...}
    m = re.findall(r"\\boxed\{([^\}]+)\}", text)
    if m:
        return re.sub(r'[,\s]+', '', m[-1].strip())
    # fallback: last number
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else None


def normalize_num(s: str) -> str:
    return re.sub(r'[,\s]+', '', s)


def is_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    a_n, b_n = normalize_num(a), normalize_num(b)
    if a_n == b_n:
        return True
    try:
        af = float(a_n)
        bf = float(b_n)
        return abs(af - bf) <= tol
    except Exception:
        return False


def build_prompts_from_dataset(dataset: str, subset: Optional[str], split: str,
                               prompt_field: Optional[str], answer_field: Optional[str],
                               max_samples: int) -> Tuple[List[str], List[str]]:
    ds = load_dataset(dataset, subset, split=split)
    prompts, answers = [], []
    # Auto-detect for common datasets
    if not prompt_field or not answer_field:
        keys = set(ds.column_names)
        if dataset.lower().endswith('gsm8k') or 'question' in keys:
            prompt_field = prompt_field or 'question'
            answer_field = answer_field or 'answer'
        elif 'problem' in keys and 'solution' in keys:
            prompt_field = prompt_field or 'problem'
            answer_field = answer_field or 'solution'
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        prompts.append(str(row[prompt_field]))
        answers.append(str(row[answer_field]))
    return prompts, answers


@torch.no_grad()
def run_source(model, tokenizer, prompts: List[str], instruct: bool,
               steps: int, gen_length: int, block_length: int, temperature: float,
               cfg_scale: float, mask_id: int, source: str, ups_head, eos_gating: bool,
               infer_bs: int = 1) -> List[str]:
    prompts_templ = apply_template_if_needed(tokenizer, prompts, instruct)
    outputs: List[str] = []
    for i in range(0, len(prompts_templ), infer_bs):
        chunk = prompts_templ[i:i + infer_bs]
        input_ids, attention_mask = prepare_inputs(tokenizer, chunk)
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        out = generate(
            model=model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            confidence_source=source,
            ups_head=ups_head,
            mask_id=mask_id,
            logits_eos_inf=eos_gating,
            confidence_eos_eot_inf=eos_gating,
        )
        decoded = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
        outputs.extend(decoded)
        # Free cache between micro-batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--ups_head', type=str, required=True)
    parser.add_argument('--lora_path', type=str, default=None, help='Optional LoRA adapter directory to load')
    parser.add_argument('--dataset', type=str, default='openai/gsm8k')
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--prompt_field', type=str, default=None)
    parser.add_argument('--answer_field', type=str, default=None)
    parser.add_argument('--format', type=str, default='gsm8k', choices=['gsm8k', 'math'])
    parser.add_argument('--max_samples', type=int, default=50)
    parser.add_argument('--instruct', action='store_true')
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--gen_length', type=int, default=128)
    parser.add_argument('--block_length', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--cfg_scale', type=float, default=0.0)
    parser.add_argument('--mask_id', type=int, default=126336)
    parser.add_argument('--eos_gating', action='store_true')
    parser.add_argument('--infer_bs', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer(args.model_name, device, lora_path=args.lora_path)

    # Load dataset prompts and answers
    prompts, answers = build_prompts_from_dataset(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        prompt_field=args.prompt_field,
        answer_field=args.answer_field,
        max_samples=args.max_samples,
    )

    # Load UPS head
    ups_head, meta = load_ups_head(args.ups_head, hidden_size=None)
    ups_head = ups_head.to(model.device).to(getattr(model, 'dtype', torch.float32)).eval()

    # Evaluate each source
    sources = [('ups', ups_head), ('tps_prob', None), ('random', None)]
    extract_pred = extract_answer_gsm8k if args.format == 'gsm8k' else extract_answer_math500
    extract_gold = extract_pred

    results: Dict[str, Dict[str, int]] = {}
    for name, head in sources:
        preds = run_source(
            model, tokenizer, prompts, args.instruct,
            args.steps, args.gen_length, args.block_length, args.temperature,
            args.cfg_scale, args.mask_id, name, head, args.eos_gating, args.infer_bs
        )
        correct = 0
        total = len(preds)
        extracted = 0
        for p, g in zip(preds, answers):
            p_ans = extract_pred(p)
            g_ans = extract_gold(g)
            if p_ans is not None:
                extracted += 1
            if is_equal(p_ans, g_ans):
                correct += 1
        results[name] = {
            'total': total,
            'extracted': extracted,
            'correct': correct,
        }

    print('=== Evaluation Summary ===')
    for name in ['ups', 'tps_prob', 'random']:
        if name in results:
            r = results[name]
            acc = r['correct'] / max(1, r['total'])
            cov = r['extracted'] / max(1, r['total'])
            print(f"{name:8s} acc={acc:.3f} extracted={cov:.3f} total={r['total']}")


if __name__ == '__main__':
    main()

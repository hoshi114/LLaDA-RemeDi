import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
             confidence_source=None, ups_head=None,
             return_trace=False, trace_batch_index=0, trace_store_indices=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Deprecated alias for confidence mode. 'low_confidence' equals 'tps_prob', or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        confidence_source: Confidence policy. One of {'ups', 'tps_prob', 'random'}. If None, falls back to `remasking`.
        ups_head: Optional UPSHead module that takes last hidden states (b,l,h) and returns per-token scores (b,l).
        return_trace: If True, also return a per-step trace for `trace_batch_index` sample.
        trace_batch_index: Which sample in the batch to trace (default 0).
        trace_store_indices: If True, store selected/newly-filled indices per step (may be large).
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    # Select confidence mode from either new flag or legacy 'remasking'
    if confidence_source is None:
        if remasking == 'low_confidence':
            confidence_source = 'tps_prob'
        elif remasking == 'random':
            confidence_source = 'random'
        else:
            raise NotImplementedError(remasking)

    trace = [] if return_trace else None
    for num_block in range(num_blocks):
        # Active answer slice for this block [s:e)
        s = prompt.shape[1] + num_block * block_length
        e = prompt.shape[1] + (num_block + 1) * block_length

        # Precompute the per-step absolute unmask counts K_abs by cumsum of transfer counts.
        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)  # (b, steps)
        K_abs = torch.cumsum(num_transfer_tokens, dim=1)  # (b, steps)
        for i in range(steps):
            # Mask index over full sequence and within active block
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                # Try to request hidden states for UPS when needed
                if confidence_source == 'ups' and ups_head is not None:
                    outputs = model(x_, attention_mask=attention_mask_, output_hidden_states=True, return_dict=True)
                    logits = outputs.logits
                    # Split outputs for CFG path to match logits split
                    # hidden states for CFG path are not required for confidence
                else:
                    logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                if confidence_source == 'ups' and ups_head is not None:
                    outputs = model(x, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
                    logits = outputs.logits
                else:
                    logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            # Compute per-token confidence over the active block [s:e)
            # Default: TPS probability baseline
            if confidence_source == 'tps_prob':
                # Memory-efficient probability of selected token via log-sum-exp
                target_tokens = torch.where(mask_index, x0, x)
                sel_logits = torch.gather(logits, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
                log_denom = torch.logsumexp(logits, dim=-1)
                confidence = torch.exp(sel_logits - log_denom)
            elif confidence_source == 'random':
                confidence = torch.rand((x.shape[0], x.shape[1]), device=x.device)
            elif confidence_source == 'ups' and ups_head is not None:
                # Lazily import to avoid cyclic
                from remedi.modeling_wrappers import get_last_hidden_states
                # If we didn't request outputs above (CFG case), fetch again with hidden states
                if 'outputs' not in locals() or outputs.logits is not logits:
                    outputs = model(x, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
                hidden = get_last_hidden_states(outputs)
                if hidden is None:
                    # Fallback to TPS prob if hidden unavailable (use log-sum-exp form)
                    target_tokens = torch.where(mask_index, x0, x)
                    sel_logits = torch.gather(logits, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
                    log_denom = torch.logsumexp(logits, dim=-1)
                    confidence = torch.exp(sel_logits - log_denom)
                else:
                    ups_scores = ups_head(hidden)  # (b, l)
                    confidence = torch.sigmoid(ups_scores)
            else:
                # Fallback: TPS probability via log-sum-exp
                target_tokens = torch.where(mask_index, x0, x)
                sel_logits = torch.gather(logits, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
                log_denom = torch.logsumexp(logits, dim=-1)
                confidence = torch.exp(sel_logits - log_denom)

            # Restrict to active block and non-prompt; others set to -inf
            conf_mask = torch.ones_like(confidence, dtype=torch.bool)
            conf_mask[:, :s] = False  # exclude prompt and earlier answer blocks by default
            conf_mask[:, e:] = False  # exclude future blocks
            confidence = torch.where(conf_mask, confidence, torch.full_like(confidence, -np.inf))

            # Compute absolute number to keep this step per sample
            num_keep = K_abs[:, i]  # (b,)

            # Select top-k per sample and build selection mask
            selected_index = torch.zeros_like(x, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = int(num_keep[j].item())
                if k <= 0:
                    continue
                # Safe topk: if all -inf, skip
                if torch.isneginf(confidence[j, s:e]).all():
                    continue
                k = min(k, e - s)
                vals, idx = torch.topk(confidence[j, s:e], k=k)
                # Filter out -inf selections (could happen early)
                valid = torch.isfinite(vals)
                if valid.any():
                    idx = idx[valid] + s
                    selected_index[j, idx] = True

            # Dynamic remask within active block: non-selected (answer positions) -> [MASK]
            in_block = torch.zeros_like(selected_index)
            in_block[:, s:e] = True
            answer_positions = in_block & (~prompt_index)
            # Trace: capture mask count before updates for the traced sample
            if return_trace and 0 <= trace_batch_index < x.size(0):
                tb = trace_batch_index
                mask_cnt_before = int((x[tb, s:e] == mask_id).sum().item())
            to_mask = answer_positions & (~selected_index)
            x[to_mask] = mask_id

            # For selected positions: if masked, write predicted tokens
            write_index = selected_index & mask_index
            # Newly filled positions (were mask before and selected now)
            if return_trace and 0 <= trace_batch_index < x.size(0):
                tb = trace_batch_index
                newly = (write_index[tb, s:e]).nonzero(as_tuple=False).squeeze(-1).tolist() if trace_store_indices else None
            x[write_index] = x0[write_index]

            # Append trace entry after updates
            if return_trace and 0 <= trace_batch_index < x.size(0):
                tb = trace_batch_index
                mask_cnt_after = int((x[tb, s:e] == mask_id).sum().item())
                sel_rel = (selected_index[tb, s:e]).nonzero(as_tuple=False).squeeze(-1).tolist() if trace_store_indices else None
                trace.append({
                    'block': int(num_block),
                    'step': int(i),
                    'keep_target': int(K_abs[tb, i].item()),
                    'mask_count_before': mask_cnt_before,
                    'mask_count_after': mask_cnt_after,
                    'num_selected': int(selected_index[tb, s:e].sum().item()),
                    'selected_rel_idx': sel_rel,
                    'newly_filled_rel_idx': newly,
                })

    if return_trace:
        return x, trace
    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
             "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
             "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()

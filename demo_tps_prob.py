import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from generate import generate


def main():
    # Minimal sanity check for the sampler tps_prob path on CPU using a tiny HF model.
    model_name = "sshleifer/tiny-gpt2"

    # Load tokenizer/model on CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token_id is None: 
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    # generate() expects model.device; patch it for generic HF models
    device = torch.device("cpu")
    model.to(device)

    # Inputs (two prompts, batch inference)
    prompts = [
        "Hello diffusion sampler,",
        "This is a tps_prob path test.",
    ]
    enc = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    # For this tiny AR model we use eos_token_id as a mask placeholder, only for this demo
    mask_id = tokenizer.eos_token_id
    assert mask_id is not None, "Tokenizer must have an eos_token_id for this demo."

    # Sampling params (ensure divisibility constraints)
    steps = 16
    gen_length = 16
    block_length = 16  # single block
    temperature = 0.7
    cfg_scale = 0.0

    # Run sampler with TPS probability as confidence source
    out = generate(
        model=model,
        prompt=input_ids,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        confidence_source="tps_prob",
        mask_id=mask_id,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
    )

    # Decode continuation (only the generated tail)
    decoded = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for i, s in enumerate(decoded):
        print(f"[Sample {i}] {s}")


if __name__ == "__main__":
    main()

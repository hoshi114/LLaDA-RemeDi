# LLaDA: Large Language Diffusion with mAsking

## Project Overview
LLaDA is an 8B-parameter language model that utilizes a **Masked Diffusion** approach instead of the traditional autoregressive (next-token prediction) method used by models like GPT/LLaMA. It demonstrates that diffusion models can scale to large language modeling tasks and supports both base generation and instruction tuning.

The project includes code for inference, a chat interface, a web demo with visualization of the diffusion process, and comprehensive evaluation scripts using `lm-eval` and `OpenCompass`.

## Key Technologies
*   **Python**
*   **PyTorch**: Core deep learning framework.
*   **Hugging Face Transformers**: Model loading and tokenization.
*   **Gradio**: Web interface for the interactive demo.
*   **OpenCompass & lm-eval**: Frameworks for evaluating model performance on benchmarks like MMLU, GSM8K, etc.

## Key Files & Scripts

### Inference & Interaction
*   **`generate.py`**: The core inference script. Contains the `generate()` function which implements the diffusion sampling loop (remasking, noise addition, token transfer).
*   **`chat.py`**: A command-line interface for multi-round conversations with `LLaDA-8B-Instruct`.
*   **`app.py`**: A Gradio web application that visualizes the denoising/generation process, allowing users to see tokens emerge from masks.

### Evaluation
*   **`eval_llada.py`**: The main entry point for running evaluations using the `lm-evaluation-harness`.
*   **`eval_llada_lm_eval.sh`**: Bash script documenting the exact commands and dependencies to run various benchmarks (MMLU, ARC-C, GSM8K, etc.).
*   **`eval_reverse.py`**: Script to test the "Reversal Curse" by generating text forwards and backwards (couplets).
*   **`get_log_likelihood.py`**: Calculates conditional likelihoods for evaluation.

### Configuration & Documentation
*   **`README.md`**: Main project documentation.
*   **`EVAL.md`**: Detailed instructions on benchmarking and evaluation results.
*   **`GUIDELINES.md`**: (Mentioned) Guidelines for pre-training and fine-tuning.

## Installation & Setup

1.  **Dependencies**:
    The project relies on specific versions of `transformers` and `lm_eval`.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust for your CUDA version
    pip install transformers==4.49.0 lm_eval==0.4.8 accelerate==0.34.2
    pip install gradio  # For the web demo
    pip install antlr4-python3-runtime==4.11 math_verify sympy hf_xet # For evaluation
    ```
    See `eval_llada_lm_eval.sh` for a complete list of eval-specific dependencies.

2.  **Model Weights**:
    Models are loaded from Hugging Face (`GSAI-ML/LLaDA-8B-Instruct` or `GSAI-ML/LLaDA-8B-Base`). Ensure you have internet access or cached weights.

## Usage Guide

### 1. Command-Line Chat
To chat with the Instruct model:
```bash
python chat.py
```

### 2. Web Demo (with Visualization)
To run the interactive web UI:
```bash
python app.py
```
Open the provided local URL (usually `http://127.0.0.1:7860`) in your browser.

### 3. Programmatic Generation
Use `generate.py` as a reference for integrating LLaDA into your own scripts. Key parameters in `generate()`:
*   `steps`: Number of diffusion steps (default: 128).
*   `gen_length`: Length of text to generate.
*   `block_length`: Size of semi-autoregressive blocks.
*   `remasking`: Strategy for masking tokens ('low_confidence' or 'random').

### 4. Evaluation
Run benchmarks using `accelerate`. Example for GSM8K:
```bash
accelerate launch eval_llada.py \
    --tasks gsm8k \
    --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024
```

## Architectural Notes
*   **Mask Token**: Uses a specific token ID (`126336`) for `[MASK]`.
*   **Sampling**: Implements a custom sampling loop that iteratively unmasks tokens based on confidence scores (`remasking='low_confidence'`).
*   **Performance**: Inference is generally slower than autoregressive models due to the iterative nature (requiring multiple passes/steps per sequence), though optimizations like "block diffusion" are explored.

# veRL-Zero

From-scratch GRPO reinforcement learning reasoning pipeline built on [veRL](https://github.com/volcengine/verl) with micro LLMs (0.5B).

## Overview

veRL-Zero is a minimal, reproducible project that trains small language models to develop mathematical reasoning capabilities using GRPO (Generalized Relative Policy Optimization). It demonstrates that even a 0.5B parameter model can begin learning structured reasoning through RL — starting from zero reasoning ability.

The pipeline is designed to run on a **single RTX 4090 (24GB)**, making it accessible for individual researchers and students.

## Architecture

```
veRL-Zero/
├── scripts/
│   ├── prepare_gsm8k.py     # GSM8K dataset preprocessing
│   ├── run_grpo.sh          # Training launch script
│   └── run_grpo_train.py    # GRPO training entry point
├── src/
│   ├── __init__.py
│   └── reward_function.py   # Custom reward function (format + correctness)
├── logs/
│   └── mvp_10_steps_analysis.md  # Training analysis report
├── .gitignore
└── README.md
```

## Environment & Dependencies

| Component | Version |
|-----------|---------|
| CUDA | 12.4 |
| PyTorch | 2.6 |
| vLLM | 0.8+ |
| veRL | verl-agent (fork) |
| Transformers | 4.x |
| Datasets | HuggingFace datasets |
| DeepSpeed | FSDP backend |

### Minimum Hardware

| Resource | Requirement |
|----------|-------------|
| GPU | 1x RTX 4090 (24GB VRAM) |
| Peak VRAM | ~20GB |
| Disk | ~20GB (model + data + checkpoints) |

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/clt123321/veRL-Zero.git
cd veRL-Zero

# Install veRL (on your training server)
pip install verl
pip install vllm
```

### 2. Download Model

```bash
# Download Qwen2.5-0.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./Qwen2.5-0.5B-Instruct
```

### 3. Prepare Data

```bash
python scripts/prepare_gsm8k.py
# Output: ./data/gsm8k/train.parquet, ./data/gsm8k/test.parquet
```

### 4. Run Training

```bash
bash scripts/run_grpo.sh
```

## Reward Function

The reward function (`src/reward_function.py`) enforces a structured output format:

| Condition | Reward |
|-----------|--------|
| No `<thinking>` or `<answer>` tags | -1.0 |
| Has tags but wrong answer | 0.0 |
| Has tags and correct answer | 1.0 |

This sparse reward design encourages the model to first learn output formatting before developing reasoning skills.

## Experimental Results (MVP - 10 Steps)

### Training Metrics

| Step | Score/Mean | PG Loss |
|------|-----------|---------|
| 1 | -0.913 | -0.035 |
| 5 | -0.900 | 0.003 |
| 8 | -0.738 | 0.041 |
| 10 | **-0.775** | -0.030 |

**Key observation**: Score improved from -0.913 to -0.775, indicating the model learned to produce structured `<thinking>`/`<answer>` output format. Accuracy gains require more training steps (100+).

### Inference Comparison

| Metric | Original | Trained (10 steps) |
|--------|----------|-------------------|
| Accuracy (8 questions) | 25.0% | 25.0% |
| Format compliance | Low | Improving |

Detailed analysis: [logs/mvp_10_steps_analysis.md](logs/mvp_10_steps_analysis.md)

## Roadmap

- [ ] **Scale training**: Run 100-500 steps for meaningful accuracy gains
- [ ] **WandB integration**: Real-time metric tracking and visualization
- [ ] **Softer reward shaping**: Partial credit for numerically close answers
- [ ] **Larger models**: Experiment with Qwen2.5-1.5B and 7B
- [ ] **Full GSM8K evaluation**: Test on complete 1,319-question test set
- [ ] **Multi-GPU support**: Scale to multi-GPU training with FSDP
- [ ] **Curriculum learning**: Progressive difficulty training strategy

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  GSM8K Data │ ──▶ │  GRPO Train  │ ──▶ │  Checkpoint  │
│  (7473 q's) │     │  (veRL+vLLM) │     │  (FSDP→HF)   │
└─────────────┘     └──────────────┘     └──────────────┘
                           │
                    ┌──────┴──────┐
                    │   Reward    │
                    │  Function   │
                    │ (format +   │
                    │ correctness)│
                    └─────────────┘
```

1. **Data Prep**: GSM8K questions are formatted with a system prompt requesting `<thinking>`/`<answer>` structured output
2. **GRPO Training**: The model generates N=5 responses per prompt; GRPO computes relative advantages to update the policy
3. **Reward Signal**: Custom reward function checks format compliance and answer correctness
4. **Checkpoint**: FSDP checkpoints are saved and can be converted to HuggingFace format for inference

## License

MIT

## Acknowledgments

- [veRL](https://github.com/volcengine/verl) - Volcano Engine RL framework
- [Qwen](https://qwen.readthedocs.io/) - Qwen2.5 model family
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) - Grade School Math dataset

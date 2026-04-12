# MVP 10-Step Training Analysis

## Hardware Configuration

| Item | Value |
|------|-------|
| GPU | 1x NVIDIA RTX 4090 (24GB VRAM) |
| Peak VRAM Usage | ~20GB |
| Platform | AutoDL Cloud GPU |
| CUDA | 12.4 |
| PyTorch | 2.6 |
| vLLM | 0.8+ |
| veRL | verl-agent (fork) |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-0.5B-Instruct |
| Algorithm | GRPO |
| Training Steps | 10 |
| Batch Size | 32 |
| Mini Batch | 16 |
| Micro Batch / GPU | 8 |
| Learning Rate | 5e-7 |
| Max Prompt Length | 512 |
| Max Response Length | 1024 |
| Rollout N (samples per prompt) | 5 |
| KL Coefficient | 0.001 |
| FSDP Param Offload | True |
| FSDP Optimizer Offload | True |
| GPU Memory Utilization (vLLM) | 0.4 |

## Training Metrics

| Step | Score/Mean | Rewards/Mean | PG Loss | Trend |
|------|-----------|-------------|---------|-------|
| 1 | -0.913 | -0.913 | -0.035 | Start |
| 2 | -0.900 | -0.900 | -0.009 | Up |
| 3 | -0.962 | -0.962 | -0.015 | Down (fluctuation) |
| 4 | -0.837 | -0.837 | -0.004 | Up |
| 5 | -0.900 | -0.900 | 0.003 | Down (fluctuation) |
| 6 | -0.887 | -0.887 | -0.027 | Flat |
| 7 | -0.812 | -0.812 | -0.025 | Up |
| 8 | -0.738 | -0.738 | 0.041 | Up Up |
| 9 | -0.850 | -0.850 | -0.004 | Down (fluctuation) |
| 10 | **-0.775** | **-0.775** | -0.030 | Up (overall improvement) |

**Overall**: Score/Mean improved from **-0.913** to **-0.775** (delta = +0.138)

## Inference Comparison (8 GSM8K Questions)

| Q# | Original | Trained | Ground Truth | Status |
|----|---------|---------|-------------|--------|
| 1 | 50 | 18 | 9 | Both wrong (trained closer) |
| 2 | 3 | 3 | 3 | Both correct |
| 3 | 000 | 40000 | 70000 | Both wrong (trained closer) |
| 4 | 180 | 180 | 540 | Both wrong |
| 5 | 0 | 16.5 | 1.5 | Both wrong |
| 6 | 64 | 64 | 64 | Both correct |
| 7 | 4 | 4 | 3 | Both wrong |
| 8 | -16.67 | -16.67 | 16.67 | Both wrong (sign error) |

**Accuracy**: Original 25.0% → Trained 25.0%

## Root Cause Analysis: Why Accuracy Didn't Improve in 10 Steps

### 1. The Model Was Learning Output Format, Not Reasoning

Our reward function (`src/reward_function.py`) uses a strict format check:
- No `<thinking>` and `<answer>` tags → **-1.0** (format penalty)
- Has tags but wrong answer → **0.0** (format correct, answer wrong)
- Has tags and correct answer → **1.0** (full reward)

The 0.5B model initially produced outputs without any structured tags, earning -1.0 on almost every sample. The training signal in the first 10 steps primarily taught the model to **produce outputs with `<thinking>` and `<answer>` tags**, which is a prerequisite for reasoning but not reasoning itself.

### 2. Insufficient Training Steps

- Only 10 steps × 32 batch size = 320 samples seen
- GSM8K training set has 7,473 samples
- The model saw less than 5% of the dataset
- GRPO typically needs 100-500+ steps to show meaningful accuracy gains

### 3. Model Capacity Limitation

- 0.5B parameters is extremely small for mathematical reasoning
- The model struggles with multi-step arithmetic even with correct formatting
- Evidence: Q1 (duck eggs) and Q3 (house flipping) predictions moved closer to the correct answer but didn't reach it

### 4. Reward Signal Sparsity

- The reward landscape is binary: -1.0, 0.0, or 1.0
- No partial credit for "close" answers (e.g., Q3: 40000 vs 70000)
- This makes the optimization landscape sparse and hard to navigate for a small model

## Key Takeaways

1. **Format learning is the first milestone** - The score improvement from -0.913 to -0.775 confirms the model is learning to structure its output
2. **Accuracy gains require more training** - At least 100+ steps needed for meaningful accuracy improvement
3. **Consider softer reward shaping** - Partial rewards for numerically close answers could accelerate learning
4. **The pipeline works end-to-end** - Data prep → Training → Checkpoint saving → Inference comparison is fully functional

## Next Steps

- [ ] Increase training to 100+ steps
- [ ] Add partial reward for close answers
- [ ] Integrate WandB for metric tracking
- [ ] Experiment with 1.5B model
- [ ] Add evaluation on full GSM8K test set

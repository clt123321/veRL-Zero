# veRL-Zero

从零开始构建的、基于 [veRL](https://github.com/volcengine/verl) 和微型大模型（0.5B）的纯血 GRPO 强化学习推理流水线。

## 项目简介

veRL-Zero 是一个极简、可复现的项目，使用 GRPO（广义相对策略优化）训练小型语言模型发展数学推理能力。它证明了即使是 0.5B 参数的模型，也能通过强化学习开始学习结构化推理——从零起步。

整个流水线设计为在 **单张 RTX 4090 (24GB)** 上运行，对个人研究者和学生友好。

## 项目结构

```
veRL-Zero/
├── scripts/
│   ├── prepare_gsm8k.py     # GSM8K 数据集预处理
│   ├── run_grpo.sh          # 训练启动脚本
│   └── run_grpo_train.py    # GRPO 训练入口
├── src/
│   ├── __init__.py
│   └── reward_function.py   # 自定义奖励函数（格式 + 正确性）
├── logs/
│   └── mvp_10_steps_analysis.md  # 训练分析报告
├── .gitignore
└── README.md
```

## 环境与依赖

| 组件 | 版本 |
|------|------|
| CUDA | 12.4 |
| PyTorch | 2.6 |
| vLLM | 0.8+ |
| veRL | verl-agent (fork) |
| Transformers | 4.x |
| Datasets | HuggingFace datasets |
| DeepSpeed | FSDP 后端 |

### 最低硬件要求

| 资源 | 要求 |
|------|------|
| GPU | 1x RTX 4090 (24GB 显存) |
| 峰值显存 | ~20GB |
| 磁盘 | ~20GB（模型 + 数据 + 检查点） |

## 快速开始

### 1. 环境搭建

```bash
# 克隆仓库
git clone https://github.com/clt123321/veRL-Zero.git
cd veRL-Zero

# 安装 veRL（在训练服务器上）
pip install verl
pip install vllm
```

### 2. 下载模型

```bash
# 下载 Qwen2.5-0.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./Qwen2.5-0.5B-Instruct
```

### 3. 准备数据

```bash
python scripts/prepare_gsm8k.py
# 输出: ./data/gsm8k/train.parquet, ./data/gsm8k/test.parquet
```

### 4. 启动训练

```bash
bash scripts/run_grpo.sh
```

## 奖励函数

奖励函数（`src/reward_function.py`）强制模型输出结构化格式：

| 条件 | 奖励值 |
|------|--------|
| 无 `<thinking>` 或 `<answer>` 标签 | -1.0 |
| 有标签但答案错误 | 0.0 |
| 有标签且答案正确 | 1.0 |

这种稀疏奖励设计促使模型先学会输出格式，再逐步发展推理能力。

## 实验结果（MVP - 10 步）

### 训练指标

| 步数 | Score/Mean | PG Loss |
|------|-----------|---------|
| 1 | -0.913 | -0.035 |
| 5 | -0.900 | 0.003 |
| 8 | -0.738 | 0.041 |
| 10 | **-0.775** | -0.030 |

**核心发现**：Score 从 -0.913 提升至 -0.775，说明模型学会了产生结构化的 `<thinking>`/`<answer>` 输出格式。准确率提升需要更多训练步数（100+）。

### 推理对比

| 指标 | 原始模型 | 训练后（10 步） |
|------|---------|---------------|
| 准确率（8 题） | 25.0% | 25.0% |
| 格式合规性 | 低 | 逐步改善 |

详细分析：[logs/project_1_10_steps_analysis.md](logs/project_1_10_steps_analysis.md)

完整实验报告：[EXPERIMENTS.md](EXPERIMENTS.md)

## 路线图

- [ ] **扩大训练规模**：运行 100-500 步以获得显著的准确率提升
- [ ] **接入 WandB**：实时指标追踪与可视化
- [ ] **更柔和的奖励塑形**：对数值接近的答案给予部分奖励
- [ ] **更大模型**：尝试 Qwen2.5-1.5B 和 7B
- [ ] **完整 GSM8K 评测**：在全部 1,319 题测试集上评估
- [ ] **多 GPU 支持**：使用 FSDP 扩展到多卡训练
- [ ] **课程学习**：渐进式难度训练策略

## 工作原理

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  GSM8K 数据 │ ──▶ │  GRPO 训练   │ ──▶ │   检查点     │
│  (7473 题)  │     │  (veRL+vLLM) │     │  (FSDP→HF)   │
└─────────────┘     └──────────────┘     └──────────────┘
                           │
                    ┌──────┴──────┐
                    │   奖励函数   │
                    │ (格式 +      │
                    │  正确性)     │
                    └─────────────┘
```

1. **数据准备**：GSM8K 题目被格式化为包含系统提示的结构化输入，要求模型使用 `<thinking>`/`<answer>` 标签输出
2. **GRPO 训练**：模型对每个提示生成 N=5 个回复，GRPO 计算相对优势来更新策略
3. **奖励信号**：自定义奖励函数检查格式合规性和答案正确性
4. **检查点保存**：FSDP 检查点保存后可转换为 HuggingFace 格式用于推理

## 许可证

MIT

## 致谢

- [veRL](https://github.com/volcengine/verl) - 火山引擎 RL 框架
- [Qwen](https://qwen.readthedocs.io/) - Qwen2.5 模型家族
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) - 小学数学数据集

# veRL-Zero 实验报告总览

本文档是 veRL-Zero 项目的综合实验记录，按「项目」维度组织。每个项目代表一次独立的训练实验，记录配置、结果、分析和后续计划。

---

## 项目列表

| 项目编号 | 名称 | 模型 | 步数 | 状态 | 关键结论 |
|---------|------|------|------|------|----------|
| [项目一](#项目一-qwen2505b-grpo-10-steps) | Qwen2.5-0.5B GRPO 10 Steps | Qwen2.5-0.5B-Instruct | 10 | ✅ 完成 | 模型学会输出格式，Score 从 -0.913 升至 -0.775 |
| [项目二](#项目二-wip) | （待定） | - | - | 🔄 进行中 | - |
| [项目三](#项目三-wip) | （待定） | - | - | 📋 计划中 | - |

---

## 如何添加新项目

当开始一个新实验时，按以下步骤操作：

1. 在对应的 `scripts/run_grpo_*.sh` 中创建新的训练脚本
2. 训练完成后，在本文档「项目列表」中添加条目
3. 在本文档末尾创建对应的「项目 N」章节
4. 在 `logs/` 下创建对应的分析文件 `project_n_analysis.md`
5. 提交到 GitHub

---

## 工作流程参考

### 完整训练流程

```
1. 数据准备
   python scripts/prepare_gsm8k.py
   # 输出: data/gsm8k/train.parquet, data/gsm8k/test.parquet

2. 启动训练
   bash scripts/run_grpo.sh

3. 监控训练（服务器上）
   tail -f /root/autodl-tmp/training.log

4. 推理对比
   # 转换检查点并运行对比（见各项目详情）

5. 记录实验
   # 更新本文档和 logs/ 下的分析文件
```

### 关键配置参数参考

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `total_training_steps` | 100-500 | MVP 用 10，正式实验用 100+ |
| `save_freq` | 20-50 | 不要设太小，避免 I/O 拖慢训练 |
| `test_freq` | 10-20 | 验证频率 |
| `train_batch_size` | 32 | 每步采样的 prompt 数量 |
| `rollout.n` | 5-8 | 每个 prompt 生成的 response 数量 |
| `actor.optim.lr` | 5e-7 到 1e-6 | 小模型用较小学习率 |
| `gpu_memory_utilization` | 0.4-0.5 | vLLM 显存占用比例 |

---

## 项目一：Qwen2.5-0.5B GRPO 10 Steps

**状态**：✅ 完成
**日期**：2026-04-12
**目的**：验证 veRL + GRPO 流水线可行性，测量 0.5B 模型在 GSM8K 上的基线表现

### 硬件配置

| 项目 | 值 |
|------|---|
| GPU | 1x NVIDIA RTX 4090 (24GB VRAM) |
| 峰值显存 | ~20GB |
| 平台 | AutoDL 云 GPU |
| CUDA | 12.4 |
| PyTorch | 2.6 |

### 训练配置

| 参数 | 值 |
|------|---|
| 模型 | Qwen2.5-0.5B-Instruct |
| 算法 | GRPO |
| 训练步数 | 10 |
| Batch Size | 32 |
| Mini Batch | 16 |
| Micro Batch / GPU | 8 |
| 学习率 | 5e-7 |
| Max Prompt Length | 512 |
| Max Response Length | 1024 |
| Rollout N (samples/prompt) | 5 |
| KL Coefficient | 0.001 |
| FSDP Param Offload | True |
| FSDP Optimizer Offload | True |
| vLLM GPU Memory Utilization | 0.4 |

### 训练指标

| 步数 | Score/Mean | Rewards/Mean | PG Loss | 趋势 |
|------|-----------|-------------|---------|------|
| 1 | -0.913 | -0.913 | -0.035 | 起始 |
| 2 | -0.900 | -0.900 | -0.009 | ↑ |
| 3 | -0.962 | -0.962 | -0.015 | ↓ 波动 |
| 4 | -0.837 | -0.837 | -0.004 | ↑ |
| 5 | -0.900 | -0.900 | 0.003 | ↓ 波动 |
| 6 | -0.887 | -0.887 | -0.027 | → |
| 7 | -0.812 | -0.812 | -0.025 | ↑ |
| 8 | -0.738 | -0.738 | 0.041 | ↑↑ |
| 9 | -0.850 | -0.850 | -0.004 | ↓ 波动 |
| 10 | **-0.775** | **-0.775** | -0.030 | ↑ 总体提升 |

**总体变化**：Score/Mean 从 **-0.913** 提升至 **-0.775**（delta = +0.138）

### 推理对比（8 道 GSM8K 题）

| 题号 | 题目 | 原始模型 | 训练后 | 正确答案 | 状态 |
|------|------|---------|--------|---------|------|
| 1 | 鸭蛋问题 | 50 | 18 | 9 | 均错误（训练后更接近） |
| 2 | 布料问题 | 3 | 3 | 3 | ✅ 均正确 |
| 3 | 翻房问题 | 000 | 40000 | 70000 | 均错误（训练后更接近） |
| 4 | 跑步问题 | 180 | 180 | 540 | 均错误 |
| 5 | 鸡饲料问题 | 0 | 16.5 | 1.5 | 均错误 |
| 6 | 玻璃杯问题 | 64 | 64 | 64 | ✅ 均正确 |
| 7 | 徒步问题 | 4 | 4 | 3 | 均错误 |
| 8 | 赛跑问题 | -16.67 | -16.67 | 16.67 | 均错误（符号错误） |

**准确率**：原始 25.0% → 训练后 25.0%（2/8 正确）

### 根因分析：为何 10 步准确率未提升

#### 1. 模型在学习输出格式，而非推理

奖励函数（`src/reward_function.py`）使用严格格式检查：
- 无 `<thinking>` 和 `<answer>` 标签 → **-1.0**（格式惩罚）
- 有标签但答案错误 → **0.0**（格式正确，答案错误）
- 有标签且答案正确 → **1.0**（满分）

0.5B 模型最初几乎所有输出都没有结构化标签，每次获得 -1.0。前 10 步的训练信号主要教会了模型**产生带 `<thinking>` 和 `<answer>` 标签的输出**，这是推理的前提但不是推理本身。

#### 2. 训练步数不足

- 仅 10 步 × 32 batch size = 320 个样本被看过
- GSM8K 训练集有 7,473 个样本
- 模型只看到了不到 5% 的数据集
- GRPO 通常需要 100-500+ 步才能看到显著的准确率提升

#### 3. 模型容量限制

- 0.5B 参数对数学推理来说非常小
- 即便格式正确，模型在多步算术上仍然吃力
- 证据：Q1（鸭蛋）和 Q3（翻房）的预测值更接近正确答案了，但仍未达到

#### 4. 奖励信号稀疏

- 奖励景观是二元的：-1.0、0.0 或 1.0
- 对"接近"的答案没有部分奖励（如 Q3：40000 vs 70000）
- 这使得优化景观稀疏，对小模型来说难以导航

### 核心结论

1. **格式学习是第一里程碑** - Score 从 -0.913 到 -0.775 的提升证实了模型在学习结构化输出
2. **准确率提升需要更多训练** - 至少 100+ 步才能看到有意义的效果
3. **考虑更柔和的奖励塑形** - 对接近答案给予部分奖励可加速学习
4. **流水线端到端可用** - 数据准备 → 训练 → 检查点保存 → 推理对比全部正常工作

### 后续改进建议

- [ ] **扩大训练规模**：运行 100-500 步以获得显著准确率提升
- [ ] **接入 WandB**：实时指标追踪与可视化（100+ 步必须）
- [ ] **更柔和的奖励塑形**：对数值接近的答案给予部分奖励（如误差 10% 以内给 0.5 分）
- [ ] **更大模型**：尝试 Qwen2.5-1.5B 或 7B
- [ ] **完整 GSM8K 评测**：在全部 1,319 题测试集上评估
- [ ] **多 GPU 支持**：使用 FSDP 扩展到多卡训练
- [ ] **课程学习**：渐进式难度训练策略

### 相关文件

- 训练脚本：`scripts/run_grpo.sh`
- 奖励函数：`src/reward_function.py`
- 数据预处理：`scripts/prepare_gsm8k.py`
- 详细分析：`logs/project_1_10_steps_analysis.md`

---

## 项目二（WIP）

（待定）

---

## 项目三（WIP）

（待定）

---

## 附录：训练流水线的艺术与工程技巧

### 工程技巧

#### 1. 用 tmux 保持训练会话

```bash
# 创建命名会话
tmux new -s verl_train

# 在会话中运行训练
bash scripts/run_grpo.sh

# 断开会话：按 Ctrl+B 然后 D
# 恢复会话
tmux attach -t verl_train
```

#### 2. 实时查看训练日志

```bash
tail -f /root/autodl-tmp/training.log
```

#### 3. 管理检查点磁盘空间

```bash
# 查看检查点大小
ls -lh /root/autodl-tmp/checkpoints/verl_grpo_gsm8k/*/global_step_*/actor/model_world_size_1_rank_0.pt

# 删除不需要的旧检查点（保留最新）
rm -rf /root/autodl-tmp/checkpoints/verl_grpo_gsm8k/*/global_step_{1,2,3,4,5}/
```

#### 4. save_freq 设置的艺术

- **不要设太小**（如 1-5）：频繁 I/O 拖慢训练
- **推荐**：20-50 步保存一次
- **检查点命名**：`global_step_10/actor/model_world_size_1_rank_0.pt`

#### 5. FSDP 检查点转换为 HuggingFace 格式

veRL 的 FSDP 格式不可直接用于 `AutoModelForCausalLM.from_pretrained()`，需要转换：

```python
import torch
from transformers import AutoModelForCausalLM

CKPT_DIR = "checkpoints/verl_grpo_gsm8k/qwen2.5_0.5b_grpo/global_step_10/actor"
ORIG_MODEL = "./Qwen2.5-0.5B-Instruct"
HF_OUTPUT = "./trained_model_hf"

state_dict = torch.load(f"{CKPT_DIR}/model_world_size_1_rank_0.pt", map_location="cpu")
model = AutoModelForCausalLM.from_pretrained(ORIG_MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
model.load_state_dict(state_dict, strict=False)
model.save_pretrained(HF_OUTPUT, safe_serialization=False)
```

### RL 训练技巧

#### 1. 学习率选择

- **0.5B 模型**：5e-7 到 1e-6
- **1.5B 模型**：1e-6 到 3e-6
- **7B 模型**：5e-7 到 1e-6（通常比小模型更低）
- 小模型需要更小的学习率防止梯度崩塌

#### 2. Batch Size 与显存

- 单卡 4090（24GB）推荐：`train_batch_size=32, ppo_mini_batch_size=16, gpu_mem_util=0.4`
- 如果 OOM：减小 `train_batch_size` 或 `gpu_mem_util`
- 如果想加快训练：增大 `rollout.n`（更多 baseline 样本）

#### 3. 奖励函数设计

稀疏奖励（-1/0/1）适合验证可行性，但正式实验推荐软奖励：

```python
# 数值接近给部分奖励
try:
    pred_num = float(predicted_answer)
    gt_num = float(ground_truth)
    ratio = min(pred_num, gt_num) / max(pred_num, gt_num) if max(pred_num, gt_num) != 0 else 0
    if ratio > 0.95:
        return 1.0
    elif ratio > 0.8:
        return 0.5
    elif ratio > 0.5:
        return 0.2
except:
    pass
```

#### 4. 何时知道训练在正常工作

- ✅ **好的信号**：Score/Mean 持续上升，格式合规率提升
- ⚠️ **注意**：PG Loss 波动大是正常的（PPO 是 on-policy，波动天生大）
- ❌ **危险信号**：Score 持续下降超过 20 步，或突然变成 -1.0（可能是模型彻底崩溃，开始乱生成）

#### 5. 早停策略

- 不要只看准确率——格式学习阶段（Score 从 -1 升到 -0.5）准确率不会变
- 建议至少跑 50-100 步再判断
- 如果 Score 连续 30 步没有改善，可以考虑调整学习率或奖励函数

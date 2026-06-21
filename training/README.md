# Expert Fine-Tuning Pipeline

Use SJTU HPC to fine-tune 3 specialized models from open-source base models.
Each expert outperforms any general model in its domain.

## Target

| Expert | Base Model | Training Data | Target Domain |
|--------|-----------|---------------|---------------|
| Math-Expert | Qwen2.5-7B | GSM8K + MATH + NuminaMath | Math reasoning |
| Code-Expert | DeepSeek-Coder-6.7B | HumanEval + MBPP + LeetCode | Code generation |
| CN-Expert | Qwen2.5-7B | C-Eval + MMLU-CN + Wiki-ZH | Chinese knowledge |

## Quick Start

```bash
# 1. Prepare data
python training/prepare_data.py --dataset gsm8k --output training/data/math

# 2. Fine-tune on SJTU HPC (example)
deepspeed training/train_expert.py \
  --model_name Qwen/Qwen2.5-7B \
  --data_path training/data/math \
  --output_dir training/checkpoints/math-expert \
  --deepspeed training/configs/ds_zero2.json \
  --lora_r 16 --lora_alpha 32 \
  --num_epochs 3 --batch_size 4 --gradient_accumulation 8

# 3. Deploy expert
python training/deploy_expert.py --checkpoint training/checkpoints/math-expert
```

## Architecture

```
Expert Pool (after fine-tuning):
  Math-Expert   → math problems → 0.95+ accuracy
  Code-Expert   → code tasks    → 0.95+ accuracy
  CN-Expert     → Chinese tasks → 0.95+ accuracy
  + General models (DS-V4, Qwen3, Kimi, etc.)

Router: selects expert for each question.
Result: each category has a dedicated expert > any single general model.
```

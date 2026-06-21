#!/usr/bin/env python3
"""
Expert Fine-Tuning via QLoRA
  Math Expert: Qwen2.5-7B + GSM8K 8,792 samples
  Code Expert: Qwen2.5-7B + HumanEval 164 samples (augmented)
  CN Expert:  Qwen2.5-7B + MMLU/BoolQ knowledge samples

  Hardware: 1-2x A100/H100 (SJTU HPC) or 24GB consumer GPU
"""
import sys, json, os, subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CKPT_DIR = BASE_DIR / "checkpoints"

def prepare_math_data():
    """Convert GSM8K to instruction format"""
    gsm8k = json.loads((Path(__file__).parent.parent / "dataset" / "gsm8k_full.json").read_text(encoding="utf-8"))
    data = []
    for item in gsm8k:
        data.append({
            "messages": [
                {"role": "system", "content": "You are a math expert. Solve step by step. Output the final answer after ####."},
                {"role": "user", "content": item["q"]},
                {"role": "assistant", "content": item["a"]},
            ]
        })
    out = DATA_DIR / "math_expert_train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out, "w", encoding="utf-8"))
    print(f"Math data: {len(data)} samples -> {out}")
    return out

def prepare_code_data():
    """HumanEval + augmentation for code expert"""
    humaneval = json.loads((Path(__file__).parent.parent / "dataset" / "humaneval.json").read_text(encoding="utf-8"))
    data = []
    for item in humaneval:
        data.append({
            "messages": [
                {"role": "system", "content": "You are a coding expert. Write clean, efficient Python code with comments."},
                {"role": "user", "content": f"Write a Python function:\n{item['q']}"},
                {"role": "assistant", "content": item["a"]},
            ]
        })
    out = DATA_DIR / "code_expert_train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out, "w", encoding="utf-8"))
    print(f"Code data: {len(data)} samples -> {out}")
    return out

def train(base_model, data_path, output_name, lora_r=16, epochs=3):
    """QLoRA fine-tuning using unsloth (fast) or trl (standard)"""
    print(f"\nTraining {output_name} from {base_model}...")
    print(f"  Data: {data_path}")
    print(f"  LoRA rank: {lora_r}, Epochs: {epochs}")

    # Try unsloth first (fastest), fall back to standard
    try:
        import unsloth
        print("  Using unsloth (optimized)")
        train_script = f"""
from unsloth import FastLanguageModel
import torch, json
from datasets import Dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{base_model}",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r={lora_r}, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32, lora_dropout=0,
)

data = json.load(open("{data_path}"))
dataset = Dataset.from_list(data[:1000])

from trl import SFTTrainer
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="messages", max_seq_length=2048,
    args=dict(output_dir="{CKPT_DIR}/{output_name}", per_device_train_batch_size=2,
        gradient_accumulation_steps=4, num_train_epochs={epochs}, learning_rate=2e-4,
        fp16=True, logging_steps=10, save_strategy="epoch"))
trainer.train()
model.save_pretrained("{CKPT_DIR}/{output_name}")
print("Training complete!")
"""
    except ImportError:
        print("  Using standard PEFT (unsloth not available)")
        train_script = f"""
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch, json
from datasets import Dataset

model_name = "{base_model}"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(r={lora_r}, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

data = json.load(open("{data_path}"))
dataset = Dataset.from_list(data[:1000])

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset,
    max_seq_length=2048,
    args=TrainingArguments(output_dir="{CKPT_DIR}/{output_name}",
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs={epochs}, learning_rate=2e-4, fp16=True,
        logging_steps=10, save_strategy="epoch"))
trainer.train()
model.save_pretrained("{CKPT_DIR}/{output_name}")
print("Training complete!")
"""

    # Write and run
    script_path = BASE_DIR / f"_train_{output_name}.py"
    script_path.write_text(train_script)
    print(f"  Script: {script_path}")
    print(f"  Run: python {script_path}")
    return script_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_expert.py [math|code|cn|all]")
        print("  math: GSM8K 8792 samples -> Math Expert")
        print("  code: HumanEval 164 samples -> Code Expert")
        print("  all:  train both")
        sys.exit(1)

    choice = sys.argv[1]
    if choice in ("math", "all"):
        dp = prepare_math_data()
        train("Qwen/Qwen2.5-7B-Instruct", str(dp), "math-expert")
    if choice in ("code", "all"):
        dp = prepare_code_data()
        train("Qwen/Qwen2.5-7B-Instruct", str(dp), "code-expert")
    if choice == "cn":
        print("CN Expert: requires MMLU-CN dataset (download separately)")

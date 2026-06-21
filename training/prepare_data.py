#!/usr/bin/env python3
"""Prepare fine-tuning data from public datasets"""
import sys, json, os
from pathlib import Path
from datasets import load_dataset

def prepare_gsm8k(output_dir):
    """Download GSM8K and convert to instruction format"""
    ds = load_dataset('gsm8k', 'main', split='train')
    data = []
    for item in ds:
        q = item['question']
        a = item['answer'].split('####')[-1].strip()
        data.append({
            "instruction": "Solve this math problem step by step. Output only the final number.",
            "input": q,
            "output": a
        })
    out = Path(output_dir) / 'gsm8k_train.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'GSM8K: {len(data)} samples -> {out}')

if __name__ == '__main__':
    prepare_gsm8k(sys.argv[2] if len(sys.argv) > 2 else 'training/data/math')

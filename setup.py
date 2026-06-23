#!/usr/bin/env python3
"""SynapseFlow Setup Wizard — 输入 API Key 即可使用"""
import json, os, getpass
from pathlib import Path

CONFIG_DIR = Path.home() / ".claude" / "tools"
CONFIG_FILE = CONFIG_DIR / "llm-config.json"
# Also support synapseflow config path
SYNAPSEFLOW_CONFIG = Path.home() / ".synapseflow" / "config.json"
TEMPLATE = Path(__file__).parent / "config" / "llm-config.template.json"

def setup():
    print("""
  ╔══════════════════════════════════════╗
  ║   SynapseFlow Setup Wizard          ║
  ║   输入 API Key，回车跳过不需要的    ║
  ╚══════════════════════════════════════╝
    """)

    config = json.loads(TEMPLATE.read_text(encoding="utf-8")) if TEMPLATE.exists() else {}

    keys = [
        ("deepseek_pro", "DeepSeek (免费)", "https://platform.deepseek.com"),
        ("qwen3",        "阿里百炼 Qwen3 (免费)", "https://dashscope.console.aliyun.com"),
        ("zhipu",        "智谱 GLM-4 (免费)", "https://bigmodel.cn"),
        ("kimi",         "Kimi 月之暗面 (免费)", "https://platform.moonshot.cn"),
        ("groq",         "Groq Llama (免费,需VPN)", "https://console.groq.com"),
        ("sjtu_zhiyuan", "SJTU 致远 HPC (校园)", "hpc@sjtu.edu.cn"),
    ]

    for key, name, url in keys:
        print(f"\n[{name}]")
        if key in config and config[key].get("api_key"):
            cur = config[key]["api_key"]
            masked = cur[:6] + "..." + cur[-4:] if len(cur) > 10 else "***"
            print(f"  当前: {masked}")
            new = input(f"  新 Key (回车跳过): ").strip()
            if new:
                config[key]["api_key"] = new
        else:
            print(f"  获取: {url}")
            new = input(f"  API Key (回车跳过): ").strip()
            if new:
                if key not in config:
                    config[key] = {}
                config[key]["api_key"] = new

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"\n✓ 配置已保存: {CONFIG_FILE}")
    print("  运行: python -m synapseflow '你的问题'")

if __name__ == "__main__":
    setup()

# -*- coding: utf-8 -*-
"""
Kimi Vision API bridge - read image and return text description
Usage: python kimi_vision.py <image_path> [question]
"""
import sys
import json
import base64
import os

# Read API Key
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm-config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

API_KEY = config.get("kimi_api_key")
if not API_KEY:
    print("[ERROR] kimi_api_key not found in llm-config.json")
    sys.exit(1)

# Read image
image_path = sys.argv[1] if len(sys.argv) > 1 else None
question = sys.argv[2] if len(sys.argv) > 2 else "Please describe this image in detail, including all text, window titles, buttons, error messages, etc. Reply in Chinese."

if not image_path or not os.path.exists(image_path):
    print("[ERROR] Image not found: {}".format(image_path or "(none)"))
    sys.exit(1)

with open(image_path, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

# Determine mime type
ext = os.path.splitext(image_path)[1].lower()
mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
mime_type = mime_map.get(ext, "image/png")

print("[Kimi Vision] Analyzing image: {} ...\n".format(os.path.basename(image_path)))

# Call Kimi API
import urllib.request
import urllib.error

url = "https://api.moonshot.cn/v1/chat/completions"
payload = json.dumps({
    "model": "moonshot-v1-8k-vision-preview",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:{};base64,{}".format(mime_type, image_data)}},
                {"type": "text", "text": question}
            ]
        }
    ],
    "temperature": 0.3
}).encode("utf-8")

req = urllib.request.Request(url, data=payload, headers={
    "Content-Type": "application/json",
    "Authorization": "Bearer {}".format(API_KEY)
})

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    answer = result["choices"][0]["message"]["content"]
    print(answer)
    print("\n[DONE] Analysis complete ---")
except urllib.error.HTTPError as e:
    err = e.read().decode("utf-8")
    print("[ERROR] Kimi API error ({}): {}".format(e.code, err))
    sys.exit(1)
except Exception as e:
    print("[ERROR] Request failed: {}".format(e))
    sys.exit(1)

#!/usr/bin/env python3
"""
monitor_server.py — SynapseFlow Real-Time Monitor HTTP Server

Lightweight HTTP server for the SynapseFlow dashboard.
Provides /api/status (polled by dashboard every 1-2s) and serves the dashboard UI.

Architecture:
  - PythonBrainstem (cached) for fast region classification (~2ms)
  - In-memory event log (ring buffer, last 200 events)
  - Active execution tracking per brain region
  - No external dependencies beyond Python stdlib

Usage:
  python engine/monitor_server.py [--port 8765] [--host localhost]
  Then open http://localhost:8765 in browser.
"""

import json
import time
import threading
import sys
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ─── Ensure project root in path ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.brainstem_wrapper import (
    PythonBrainstem, REGION_NAMES, N_REGIONS, N_DIMS, DIFF_WEIGHTS,
)

# ═══════════════════════════════════════════════════════════════
# Brain Region Configuration (matching dashboard mockup)
# ═══════════════════════════════════════════════════════════════

REGIONS = [
    {
        "id": 0, "name": "Motor Cortex", "icon": "<>",
        "desc": "Code / SQL / Algorithms",
        "primary": ["DS-PRO", "DS-Think"],
        "color": "#58a6ff",
    },
    {
        "id": 1, "name": "Parietal Cortex", "icon": "π",
        "desc": "Math / Numerical",
        "primary": ["DS-PRO", "QWEN3"],
        "color": "#d2a8ff",
    },
    {
        "id": 2, "name": "Prefrontal Cortex", "icon": "∟",
        "desc": "Logic / Reasoning",
        "primary": ["DS-PRO", "DS-Think"],
        "color": "#7ee787",
    },
    {
        "id": 3, "name": "Temporal Cortex", "icon": "📚",
        "desc": "Knowledge / Memory",
        "primary": ["GLM-4+", "QWEN3"],
        "color": "#f0883e",
    },
    {
        "id": 4, "name": "Language Area", "icon": "💬",
        "desc": "Writing / Chinese",
        "primary": ["GLM-4+", "KIMI"],
        "color": "#f85149",
    },
    {
        "id": 5, "name": "Visual Cortex", "icon": "👁",
        "desc": "Vision / Multimodal",
        "primary": ["KIMI", "QWEN3"],
        "color": "#bc8cff",
    },
]

MODELS = {
    "DS-PRO":    {"color": "#58a6ff", "provider": "DeepSeek"},
    "DS-Think":  {"color": "#d2a8ff", "provider": "SJTU HPC"},
    "GLM-4+":    {"color": "#7ee787", "provider": "Zhipu"},
    "QWEN3":     {"color": "#f0883e", "provider": "Alibaba"},
    "KIMI":      {"color": "#f85149", "provider": "Moonshot"},
}

# ═══════════════════════════════════════════════════════════════
# Monitor State
# ═══════════════════════════════════════════════════════════════

class MonitorState:
    """Thread-safe monitor state shared across HTTP requests."""

    def __init__(self):
        self.lock = threading.Lock()
        self.brainstem = PythonBrainstem(seed=42)

        # Region active status
        self.region_active = [False] * N_REGIONS
        self.region_last_active = [0.0] * N_REGIONS

        # Active executions {model: {region, status, start_time, output_len, elapsed}}
        self.executions = {}  # model_name -> dict

        # Event log (ring buffer)
        self.events = []  # list of {ts, msg, type}
        self.max_events = 200

        # Stats
        self.total_classifications = 0
        self.total_executions = 0
        self.start_time = time.time()
        self.demo_running = False

        # Brainstem standby: pre-load with balanced warming
        self._warm_brainstem()

    def _warm_brainstem(self):
        """Pre-warm the brainstem SDM with balanced prototype activations."""
        import numpy as np
        for d in range(N_DIMS):
            vec = np.zeros(N_DIMS, dtype=np.float64)
            vec[d] = 1.5
            self.brainstem.classify(vec)
        self.add_event("Brainstem SDM warmed up", "info")

    def add_event(self, msg: str, etype: str = "info"):
        ts = time.strftime("%p%I:%M:%S").lower().replace("pm", "下午").replace("am", "上午")
        if "下午" not in ts and "上午" not in ts:
            ts = time.strftime("%H:%M:%S")
        with self.lock:
            self.events.insert(0, {"ts": ts, "msg": msg, "type": etype})
            if len(self.events) > self.max_events:
                self.events.pop()

    def classify_question(self, question: str) -> dict:
        """Fast brainstem classification for standby monitoring."""
        import numpy as np
        features = self._extract_features(question)
        vec = np.array(features, dtype=np.float64)
        region_id, confidence, difficulty = self.brainstem.classify(vec)

        with self.lock:
            self.total_classifications += 1
            # Mark region as recently active
            self.region_active[region_id] = True
            self.region_last_active[region_id] = time.time()

        region_name = REGION_NAMES[region_id] if 0 <= region_id < N_REGIONS else "unknown"
        return {
            "region_id": int(region_id),
            "region_name": region_name,
            "confidence": round(float(confidence), 3),
            "difficulty": round(float(difficulty), 3),
        }

    def _extract_features(self, question: str) -> list:
        """Lightweight feature extraction (matches brain.py score_task dims)."""
        import re
        q = question.lower()

        dims = {
            "code": len(re.findall(r'(代码|sql|函数|算法|编程|python|写一个|实现|ddl|cte|class |def |import |SELECT|CREATE)', q)),
            "math": len(re.findall(r'(数学|概率|统计|方程|几何|代数|微积分|矩阵|向量|定理|证明|求导|积分|极限)', q)),
            "logic": len(re.findall(r'(说谎|谁说|真话|假话|悖论|逻辑|推理|判断|证明|充要|必要|充分)', q)),
            "knowledge": len(re.findall(r'(什么|如何|为什么|定义|概念|原理|历史|背景|综述)', q)),
            "writing": len(re.findall(r'(写作|论文|文档|说明|报告|总结|翻译|润色|解释|描述|写一篇)', q)),
            "arch": len(re.findall(r'(架构|设计|系统|方案|权衡|选型|策略|规划|框架)', q)),
            "trap_single": len(re.findall(r'(图遍历|递归cte|索引优化|sql调优)', q)),
            "trap_need_collab": len(re.findall(r'(多视角|对比分析|跨领域|综合评估|深度剖析)', q)),
            "group_theory": 0, "graph_theory": 0, "topology": 0,
            "linear_algebra": 0, "calculus": 0, "probability": 0,
            "number_theory": 0, "diff_eq": 0, "combinatorics": 0, "optimization": 0,
            "chinese": len(re.findall(r'[一-鿿]', q)) / max(1, len(q)) * 2,
            "safety": 0, "general": 0.3, "db": len(re.findall(r'(数据库|sql|mysql|postgresql|mongodb|redis)', q)),
        }

        feature_keys = [
            "code", "math", "logic", "knowledge", "writing",
            "arch", "trap_single", "trap_need_collab",
            "group_theory", "graph_theory", "topology", "linear_algebra",
            "calculus", "probability", "number_theory", "diff_eq",
            "combinatorics", "optimization",
            "chinese", "safety", "general", "db",
        ]
        return [float(dims.get(k, 0)) for k in feature_keys]

    def start_execution(self, model: str, region_name: str):
        with self.lock:
            self.executions[model] = {
                "model": model,
                "region": region_name,
                "status": "running",
                "start_time": time.time(),
                "output_len": 0,
                "elapsed": "0.0s",
            }
            self.total_executions += 1
        self.add_event(f"[START] {model} @ {region_name}", "info")

    def update_execution(self, model: str, status: str, output_len: int = 0, elapsed: str = ""):
        with self.lock:
            if model in self.executions:
                e = self.executions[model]
                e["status"] = status
                if output_len: e["output_len"] = output_len
                if elapsed: e["elapsed"] = elapsed
                if status in ("done", "failed"):
                    e["end_time"] = time.time()

        if status == "done":
            self.add_event(f"[DONE] {model}: {output_len}c in {elapsed}", "ok")
        elif status == "failed":
            self.add_event(f"[FAIL] {model}: {elapsed}", "err")

    def get_status(self) -> dict:
        """Return full state for /api/status polling."""
        with self.lock:
            # Decay region active status (auto-idle after 30s of inactivity)
            now = time.time()
            for i in range(N_REGIONS):
                if self.region_active[i] and (now - self.region_last_active[i]) > 30:
                    self.region_active[i] = False

            regions_state = []
            for r in REGIONS:
                rid = r["id"]
                regions_state.append({
                    "icon": r["icon"],
                    "name": r["name"],
                    "desc": r["desc"],
                    "primary": r["primary"],
                    "color": r["color"],
                    "active": self.region_active[rid],
                })

            exec_list = []
            for m, e in self.executions.items():
                info = MODELS.get(m, {"color": "#8b949e", "provider": "?"})
                elapsed = time.time() - e["start_time"] if e["status"] == "running" else 0
                exec_list.append({
                    "model": m,
                    "region": e["region"],
                    "status": e["status"],
                    "color": info["color"],
                    "output_len": e.get("output_len", 0),
                    "elapsed": f"{elapsed:.1f}s" if e["status"] == "running" else e.get("elapsed", "..."),
                })

            # Sort: running first, then pending, then done/failed
            order = {"running": 0, "pending": 1, "done": 2, "failed": 3}
            exec_list.sort(key=lambda x: order.get(x["status"], 4))

            uptime = int(now - self.start_time)

            return {
                "regions": regions_state,
                "executions": exec_list,
                "events": self.events[:50],  # last 50 events
                "stats": {
                    "num_regions": N_REGIONS,
                    "num_models": len(MODELS),
                    "num_plasticity": 11,
                    "num_papers": 20,
                    "num_langs": 4,
                    "avg_cost": "$0.20",
                },
                "brainstem": {
                    "reads": self.brainstem.read_count,
                    "writes": self.brainstem.write_count,
                    "avg_activation": round(self.brainstem.avg_activation, 3),
                },
                "uptime": uptime,
                "demo_running": self.demo_running,
            }

    def run_demo(self):
        """Run a demo parallel execution across brain regions."""
        if self.demo_running:
            return

        self.demo_running = True
        self.add_event("=== PARALLEL EXECUTION START ===", "info")
        self.add_event("Brainstem: classifying question...", "info")

        import random
        import numpy as np

        # Simulate brainstem activation
        vec = np.zeros(N_DIMS, dtype=np.float64)
        vec[1] = 2.0   # math
        vec[12] = 1.0  # calculus
        vec[13] = 0.5  # probability
        region_id, conf, diff = self.brainstem.classify(vec)

        with self.lock:
            self.region_active[1] = True  # Parietal
            self.region_active[2] = True  # Prefrontal
            self.region_last_active[1] = time.time()
            self.region_last_active[2] = time.time()

        region_name = REGIONS[region_id]["name"]
        self.add_event(f"Brainstem: {region_name} activated (conf={conf:.2f})", "ok")

        # Demo models to execute
        demo_models = [
            ("DS-PRO", "Parietal Cortex"),
            ("QWEN3", "Parietal Cortex"),
            ("DS-PRO", "Prefrontal Cortex"),
            ("DS-Think", "Prefrontal Cortex"),
            ("GLM-4+", "Temporal Cortex"),
        ]

        def run_model(model, region):
            self.start_execution(model, region)
            # Simulate work
            delay = 0.5 + random.random() * 2.5
            time.sleep(delay)

            if random.random() < 0.12:
                self.update_execution(model, "failed", 0, f"ERR: rate limit")
            else:
                chars = 400 + int(random.random() * 3500)
                elapsed_t = f"{delay:.1f}s"
                self.update_execution(model, "done", chars, elapsed_t)

        threads = []
        for model, region in demo_models:
            t = threading.Thread(target=run_model, args=(model, region))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.add_event("=== ALL MODELS COMPLETE: Synthesizing answer ===", "ok")
        self.add_event("[CORTEX] DS-PRO final validation: PASSED", "ok")
        self.add_event("[SYNAPSE] Pathway weights updated (STDP+LTP+LTD)", "info")

        # Decay active regions after a bit
        time.sleep(2)
        with self.lock:
            for i in range(N_REGIONS):
                self.region_active[i] = False

        self.demo_running = False
        self.add_event("Demo complete — ready for next query", "info")


# ═══════════════════════════════════════════════════════════════
# HTTP Server
# ═══════════════════════════════════════════════════════════════

DASHBOARD_PATH = Path(__file__).parent.parent / "ui" / "dashboard.html"

class MonitorHandler(BaseHTTPRequestHandler):
    """HTTP handler for SynapseFlow monitor."""

    # Share state across requests
    state: MonitorState = None

    def log_message(self, format, *args):
        """Suppress default logging to stderr."""
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/api/status":
            self._send_json(self.state.get_status())

        elif path == "/api/demo":
            # Run demo in background thread
            if not self.state.demo_running:
                t = threading.Thread(target=self.state.run_demo, daemon=True)
                t.start()
                self._send_json({"ok": True, "msg": "Demo started"})
            else:
                self._send_json({"ok": False, "msg": "Demo already running"})

        elif path == "/api/ping":
            self._send_json({"ok": True, "t": time.time()})

        elif path == "/" or path == "/index.html":
            self._serve_dashboard()

        elif path == "/health":
            self._send_json({"status": "ok", "uptime": int(time.time() - self.state.start_time)})

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "not found"}')

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/classify":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                data = json.loads(body)
                question = data.get("question", "")
                if question:
                    result = self.state.classify_question(question)
                    self._send_json({"ok": True, **result})
                else:
                    self._send_json({"ok": False, "error": "no question provided"})
            except json.JSONDecodeError:
                self._send_json({"ok": False, "error": "invalid JSON"})

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "not found"}')

    def _send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _serve_dashboard(self):
        if DASHBOARD_PATH.exists():
            html = DASHBOARD_PATH.read_text(encoding="utf-8")
            body = html.encode("utf-8")
        else:
            body = b"<h1>Dashboard not found</h1><p>Run: python engine/monitor_server.py from project root</p>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def start_server(host: str = "127.0.0.1", port: int = 8765):
    """Start the monitor HTTP server."""
    state = MonitorState()
    MonitorHandler.state = state

    server = HTTPServer((host, port), MonitorHandler)

    print(f"\n  {'='*60}")
    print(f"  [SynapseFlow] Monitor Server")
    print(f"  Dashboard: http://{host}:{port}")
    print(f"  API:       http://{host}:{port}/api/status")
    print(f"  Brainstem: PythonBrainstem ({state.brainstem.__class__.__name__})")
    print(f"  {'='*60}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down monitor server...")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SynapseFlow Monitor Server")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    args = parser.parse_args()

    # Ensure working from project root
    os.chdir(Path(__file__).parent.parent)
    start_server(args.host, args.port)

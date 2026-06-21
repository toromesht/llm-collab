#!/usr/bin/env python3
"""OpenAI-compatible API server for LMSYS submission"""
import json, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent import solve, POOL

# Minimal HTTP server (no Flask needed for testing)
from http.server import HTTPServer, BaseHTTPRequestHandler

class AgentAPI(BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_len))

        if self.path == '/v1/chat/completions':
            messages = body.get('messages', [])
            question = messages[-1]['content'] if messages else ''
            model = body.get('model', 'synapse-agent')

            try:
                answer = solve(question)
                resp = {
                    "id": "synapse-001",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": len(question), "completion_tokens": len(answer)}
                }
                self._json(200, resp)
            except Exception as e:
                self._json(500, {"error": str(e)})

        elif self.path == '/v1/models':
            self._json(200, {
                "object": "list",
                "data": [{"id": "synapse-agent", "object": "model"}]
            })

        else:
            self._json(404, {"error": "Not found"})

    def do_GET(self):
        if self.path == '/health':
            self._json(200, {"status": "ok", "models": list(POOL.keys()), "pool_size": len(POOL)})
        elif self.path == '/v1/models':
            self._json(200, {"object": "list", "data": [{"id": "synapse-agent", "object": "model", "owned_by": "synapseflow"}]})
        else:
            self._json(404, {"error": "Not found"})

    def _json(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def log_message(self, format, *args):
        print(f"[API] {args[0]}")

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"SynapseFlow API Server on http://localhost:{port}")
    print(f"Test: curl http://localhost:{port}/health")
    print(f"Chat: curl -X POST http://localhost:{port}/v1/chat/completions -d '{{\"model\":\"synapse-agent\",\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}]}}'")
    HTTPServer(('0.0.0.0', port), AgentAPI).serve_forever()

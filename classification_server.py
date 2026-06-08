# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import torch
from transformers import pipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one text classification model server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True, help="Hugging Face model ID for text classification.")
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device index to bind to when CUDA is available.",
    )
    return parser.parse_args()

def _json_response(
    handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200
) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)

class ClassificationInferenceServer(BaseHTTPRequestHandler):
    pipeline_instance: Any | None = None # transformers pipeline
    inference_lock = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(self, {"status": "ok"})
            return
        _json_response(self, {"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            _json_response(self, {"error": "not found"}, status=404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        texts = payload.get("texts", [])
        if not isinstance(texts, list):
            _json_response(self, {"error": "texts must be a list"}, status=400)
            return
        
        assert self.pipeline_instance is not None

        with self.inference_lock:
            # The pipeline handles batching internally when given a list
            raw_outputs = self.pipeline_instance(texts, top_k=None, batch_size=64)
        
        # Return raw outputs; client will handle remapping if needed
        _json_response(self, {"raw_outputs": raw_outputs})

    def log_message(self, format: str, *args: Any) -> None:
        return

def main() -> None:
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = args.device_id if torch.cuda.is_available() else -1
    print(f"Loading model {args.model} on device {device}...")
    
    # Initialize the pipeline
    # The pipeline will automatically move the model to the specified device
    pipeline_instance = pipeline(
        "text-classification",
        model=args.model,
        top_k=None, # Ensure all scores are returned
        device=device,
    )
    print(f"Model {args.model} loaded successfully.")

    ClassificationInferenceServer.pipeline_instance = pipeline_instance
    server = ThreadingHTTPServer((args.host, args.port), ClassificationInferenceServer)
    print(f"Serving classification model {args.model} on http://{args.host}:{args.port}")
    server.serve_forever()

if __name__ == "__main__":
    main()
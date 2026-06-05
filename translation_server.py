from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one translation server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
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


class TranslationServer(BaseHTTPRequestHandler):
    tokenizer = None
    model = None
    device = None
    translate_lock = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(self, {"status": "ok"})
            return
        _json_response(self, {"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/translate":
            _json_response(self, {"error": "not found"}, status=404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        texts = payload.get("texts", [])
        source_lang = payload.get("source_lang", "eng_Latn")
        target_lang = payload.get("target_lang")
        if not isinstance(texts, list):
            _json_response(self, {"error": "texts must be a list"}, status=400)
            return
        if not isinstance(target_lang, str) or not target_lang:
            _json_response(self, {"error": "target_lang is required"}, status=400)
            return
        assert self.tokenizer is not None
        assert self.model is not None
        assert self.device is not None
        with self.translate_lock:
            self.tokenizer.src_lang = source_lang
            inputs = self.tokenizer(
                [str(text) for text in texts],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_lang)
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                )
        translations = [
            self.tokenizer.decode(tokens, skip_special_tokens=True)
            for tokens in generated_tokens
        ]
        _json_response(self, {"translations": translations})

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
        model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)
    model.eval()

    TranslationServer.tokenizer = tokenizer
    TranslationServer.model = model
    TranslationServer.device = device
    server = ThreadingHTTPServer((args.host, args.port), TranslationServer)
    print(f"Serving translation on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

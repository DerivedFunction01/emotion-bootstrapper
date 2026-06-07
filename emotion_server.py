from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from emotion_bootstrapper import VerboseSemanticBootstrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one emotion bootstrap model server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=None)
    parser.add_argument("--multilingual", action="store_true", help="Use the multilingual model and hypotheses.")
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device index to bind to when CUDA is available.",
    )
    return parser.parse_args()


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class EmotionInferenceServer(BaseHTTPRequestHandler):
    bootstrapper: VerboseSemanticBootstrapper | None = None

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
        examples = payload.get("examples", [])
        if not isinstance(examples, list):
            _json_response(self, {"error": "examples must be a list"}, status=400)
            return
        assert self.bootstrapper is not None
        examples = list(examples)
        batch_examples: list[dict[str, Any]] = []
        for example in examples:
            item = {
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
            }
            if "token_type_ids" in example:
                item["token_type_ids"] = example["token_type_ids"]
            batch_examples.append(item)
        inputs = self.bootstrapper.tokenizer.pad(batch_examples, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.bootstrapper.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.bootstrapper.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, self.bootstrapper.entailment_id].detach().cpu().tolist()
        _json_response(self, {"entailment_scores": scores})

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    bootstrapper = VerboseSemanticBootstrapper(model=args.model, device_map="cpu", multilingual=args.multilingual)
    if torch.cuda.is_available():
        bootstrapper.device = torch.device(f"cuda:{args.device_id}")
        bootstrapper.model.to(bootstrapper.device)
    bootstrapper.model.eval()
    EmotionInferenceServer.bootstrapper = bootstrapper
    server = ThreadingHTTPServer((args.host, args.port), EmotionInferenceServer)
    print(f"Serving on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

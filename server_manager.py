#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

from emotion_cache import load_json, write_json_atomic


DEFAULT_PORT_BASE = 8000
DEFAULT_HOST = "127.0.0.1"
REGISTRY_PATH = Path("server_cluster.json")
CLASSIFICATION_REGISTRY_PATH = Path("classification_server_cluster.json")
TRANSLATION_REGISTRY_PATH = Path("translation_server_cluster.json")
PID_DIR = Path("server_pids")


@dataclass(frozen=True)
class ServerSpec:
    name: str
    host: str
    port: int
    device_id: int
    pid: int
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the model server cluster.")
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument(
        "--config",
        choices=["bootstrap", "translate", "classification"],
        default="bootstrap",
        help="Select which server configuration to manage.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--multilingual", action="store_true", help="Start the bootstrap server with multilingual hypotheses.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument(
        "--num-servers",
        type=int,
        default=None,
        help="Override the detected GPU count when starting servers.",
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=DEFAULT_PORT_BASE,
        help="First port to use; subsequent servers increment from here.",
    )
    parser.add_argument("--registry-path", default=None)
    parser.add_argument("--pid-dir", default=str(PID_DIR))
    return parser.parse_args()


def _server_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _write_registry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)


def _read_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing registry file: {path}")
    return load_json(path)


def _pid_file(pid_dir: Path, device_id: int) -> Path:
    return pid_dir / f"server_{device_id}.pid"


def _default_models_for_config(config: str, multilingual: bool = False) -> list[str] | str:
    if config == "bootstrap":
        if multilingual:
            return "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        return "facebook/bart-large-mnli"
    if config == "classification":
        return [
            "emotions-entailment/roberta-raw",
            "emotions-entailment/roberta-8-4-1.25-65-75",
            "tabularisai/multilingual-emotion-classification",
        ]
    # For translation, it's still a single model per server type
    # The server itself handles the different target languages
    # So, we return a list of one for consistency in iteration
    if config == "translate":
        return "facebook/nllb-200-distilled-600M"
    raise ValueError(f"Unsupported config: {config}")

def _default_registry_path_for_config(config: str) -> Path:
    if config == "bootstrap":
        return REGISTRY_PATH
    if config == "classification":
        return CLASSIFICATION_REGISTRY_PATH
    if config == "translate":
        return TRANSLATION_REGISTRY_PATH
    raise ValueError(f"Unsupported config: {config}")


def _server_script_for_config(config: str) -> str:
    if config == "bootstrap":
        return "emotion_server.py"
    if config == "classification":
        return "classification_server.py"
    if config == "translate":
        return "translation_server.py"
    raise ValueError(f"Unsupported config: {config}")


def _detect_server_count(args: argparse.Namespace) -> int:
    if args.num_servers is not None:
        if args.num_servers < 1:
            raise ValueError("--num-servers must be at least 1")
        return args.num_servers
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count < 1:
            raise ValueError("CUDA is available but no devices were detected")
        return count
    raise RuntimeError("No CUDA GPUs detected; refusing to start zero servers")


def start_servers(args: argparse.Namespace) -> None:
    pid_dir = Path(args.pid_dir)
    pid_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine models to start servers for
    if args.model: # If a specific model is provided, use it
        models_to_start = [args.model]
    else: # Otherwise, use defaults for the config
        models_to_start = _default_models_for_config(args.config, args.multilingual)
        if not isinstance(models_to_start, list): # Ensure it's always a list
            models_to_start = [models_to_start]

    registry_path = Path(args.registry_path) if args.registry_path else _default_registry_path_for_config(args.config)
    server_script = _server_script_for_config(args.config)

    specs: list[ServerSpec] = []
    
    # Detect available GPUs once
    available_gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [-1] # -1 for CPU
    
    # If num_servers is specified, use that many, cycling through available GPUs
    num_servers_to_start = args.num_servers if args.num_servers is not None else len(models_to_start)
    
    for i in range(num_servers_to_start):
        model_for_this_server = models_to_start[i % len(models_to_start)] # Cycle through models if more servers than models
        device_id = available_gpus[i % len(available_gpus)] # Cycle through available GPUs
        port = args.port_base + i
        pid_file = _pid_file(pid_dir, i) # Use generic index for pid file
        if pid_file.exists():
            raise FileExistsError(
                f"Refusing to start: pid file already exists for server {i}: {pid_file}"
            )

        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        cmd = [
            sys.executable,
            server_script,
            "--host",
            args.host,
            "--port",
            str(port),
            "--device-id",
            str(device_id),
            "--model",
            model_for_this_server,
        ]
        if args.config == "bootstrap" and args.multilingual:
            cmd.append("--multilingual")
        proc = subprocess.Popen(cmd, env=env)
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        specs.append(
            ServerSpec(
                name=f"server-{device_id}",
                host=args.host,
                port=port,
                device_id=device_id,
                pid=proc.pid,
                url=_server_url(args.host, port),
            )
        )
        print(f"Started {specs[-1].name} (model: {model_for_this_server}) at {specs[-1].url} (pid {proc.pid})")

    _write_registry(
        registry_path,
        {
            "config": args.config,
            "multilingual": getattr(args, "multilingual", False),
            "models": models_to_start, # Store all models started for this config
            "host": args.host,
            "port_base": args.port_base,
            "servers": [asdict(spec) for spec in specs],
        },
    )
    print(f"Wrote registry to {registry_path}")


def stop_servers(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry_path) if args.registry_path else _default_registry_path_for_config(args.config)
    pid_dir = Path(args.pid_dir)
    if registry_path.exists():
        registry = _read_registry(registry_path)
        servers = registry.get("servers", [])
    else:
        servers = []

    for server in servers:
        pid = int(server["pid"])
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Stopped {server['name']} (pid {pid})")
        except ProcessLookupError:
            print(f"{server['name']} already exited")

    if pid_dir.exists():
        for pid_file in pid_dir.glob("server_*.pid"):
            try:
                pid_file.unlink()
            except FileNotFoundError:
                pass

    if registry_path.exists():
        registry_path.unlink()
    print("Cluster stopped")


def status(args: argparse.Namespace) -> None:
    registry_path = Path(args.registry_path) if args.registry_path else _default_registry_path_for_config(args.config)
    if not registry_path.exists():
        print("Cluster is not running")
        return
    registry = _read_registry(registry_path)
    print(f"Config: {registry.get('config', args.config)}")
    print(f"Model: {registry.get('model')}")
    if registry.get("multilingual"):
        print("Multilingual: True")
    for server in registry.get("servers", []):
        print(f"{server['name']}: {server['url']} pid={server['pid']}")


def main() -> None:
    args = parse_args()
    if args.action == "start":
        start_servers(args)
    elif args.action == "stop":
        stop_servers(args)
    elif args.action == "status":
        status(args)


if __name__ == "__main__":
    main()

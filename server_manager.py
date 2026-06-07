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
        choices=["bootstrap", "translate"],
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


def _default_model_for_config(config: str, multilingual: bool = False) -> str:
    if config == "bootstrap":
        if multilingual:
            return "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        return "facebook/bart-large-mnli"
    if config == "translate":
        return "facebook/nllb-200-distilled-600M"
    raise ValueError(f"Unsupported config: {config}")


def _default_registry_path_for_config(config: str) -> Path:
    if config == "bootstrap":
        return REGISTRY_PATH
    if config == "translate":
        return TRANSLATION_REGISTRY_PATH
    raise ValueError(f"Unsupported config: {config}")


def _server_script_for_config(config: str) -> str:
    if config == "bootstrap":
        return "emotion_server.py"
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
    server_count = _detect_server_count(args)
    model = args.model or _default_model_for_config(args.config, args.multilingual)
    registry_path = Path(args.registry_path) if args.registry_path else _default_registry_path_for_config(args.config)
    server_script = _server_script_for_config(args.config)

    specs: list[ServerSpec] = []
    for device_id in range(server_count):
        port = args.port_base + device_id
        pid_file = _pid_file(pid_dir, device_id)
        if pid_file.exists():
            raise FileExistsError(
                f"Refusing to start: pid file already exists for server {device_id}: {pid_file}"
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
            model,
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
        print(f"Started {specs[-1].name} at {specs[-1].url} (pid {proc.pid})")

    _write_registry(
        registry_path,
        {
            "config": args.config,
            "multilingual": getattr(args, "multilingual", False),
            "model": model,
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

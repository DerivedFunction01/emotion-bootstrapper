#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import psutil
import multiprocessing as mp
import platform
import time
import json
import shutil
import torch

# =============================================================================
# CONFIGURATION
# =============================================================================

NGINX_PORT = 5000
GPU_SERVER_PORT = 5001
CPU_SERVER_PORT = 5002

PID_FILE = "server-roberta.pid"
NGINX_CONF_FILE = "nginx-roberta.conf"
SERVER_SCRIPT = "server:app"  # Adjust if your module name differs
CACHE_FILE = ".server_cache-roberta.json"
CACHE_DURATION = 60 * 60 * 24 * 7  # 7 days

GUNICORN_TIMEOUT = 120

# =============================================================================
# CACHING & DETECTION
# =============================================================================


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        if time.time() - cache.get("timestamp", 0) < CACHE_DURATION:
            return cache
    except Exception:
        pass
    return None


def save_cache(model_available, gpu_ram_gb):
    cache = {
        "timestamp": time.time(),
        "model_available": model_available,
        "gpu_ram_gb": gpu_ram_gb,
    }
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Warning: Could not write cache file: {e}")


def pre_download_model():
    cache = load_cache()
    if cache and cache.get("model_available"):
        print("Model is available (cached).")
        return True

    print("Checking for model availability (first run may take time)...")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from server import MODEL_PATH

        AutoTokenizer.from_pretrained(MODEL_PATH)
        AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"Could not preload model: {e}")
        print("   Ensure 'transformers' and 'torch' are installed.")
        return False


def get_gpu_ram():
    cache = load_cache()
    if cache and "gpu_ram_gb" in cache:
        ram = cache["gpu_ram_gb"]
        if ram > 0:
            print(f"GPU detected: {ram:.2f} GB (cached)")
        else:
            print("No GPU detected (cached)")
        return ram

    print("Detecting GPU memory...")
    if not torch.cuda.is_available():
        print("No CUDA GPU found.")
        return 0.0

    try:
        name = torch.cuda.get_device_name(0)
        ram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU detected: {name} – {ram_gb:.2f} GB")
        return ram_gb
    except Exception as e:
        print(f"GPU detection failed: {e}")
        return 0.0


def get_system_resources():
    cpu_cores = mp.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    return cpu_cores, ram_gb


# =============================================================================
# SCALING LOGIC
# =============================================================================


def calculate_gpu_workers(gpu_ram_gb: float) -> int:
    """Conservative but effective scaling based on real-world VRAM usage."""
    if gpu_ram_gb >= 40:
        return 5
    elif gpu_ram_gb >= 30:
        return 4
    elif gpu_ram_gb >= 22:
        return 3
    elif gpu_ram_gb >= 14:
        return 2
    else:
        return 1


# =============================================================================
# NGINX CONFIG GENERATION
# =============================================================================


def generate_nginx_config(gpu_weight: int, cpu_weight: int, cpu_enabled: bool):
    log_dir = os.path.abspath("logs")
    os.makedirs(log_dir, exist_ok=True)
    access_log = os.path.join(log_dir, "nginx_access.log").replace("\\", "/")
    error_log = os.path.join(log_dir, "nginx_error.log").replace("\\", "/")

    if cpu_enabled and cpu_weight > 0:
        upstream = f"""
        server 127.0.0.1:{GPU_SERVER_PORT} weight={gpu_weight} max_fails=3 fail_timeout=30s;
        server 127.0.0.1:{CPU_SERVER_PORT} weight={cpu_weight} backup;
        """
        print(f"Nginx: GPU primary (weight={gpu_weight}), CPU backup")
    else:
        upstream = f"""
        server 127.0.0.1:{GPU_SERVER_PORT};
        """
        print("Nginx: GPU-only mode")

    config = f"""worker_processes auto;
pid {os.path.abspath(PID_FILE).replace(os.sep, '/')};

events {{
    worker_connections 1024;
}}

http {{
    access_log {access_log};
    error_log  {error_log};

    map $binary_remote_addr $limit_key {{
        default $binary_remote_addr;
        ""      "";
    }}
    limit_conn_zone $limit_key zone=addr:10m;

    upstream model_servers {{{upstream}
    }}

    server {{
        listen {NGINX_PORT};
        server_name localhost;

        location / {{
            proxy_pass http://model_servers;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_connect_timeout {GUNICORN_TIMEOUT}s;
            proxy_send_timeout {GUNICORN_TIMEOUT}s;
            proxy_read_timeout {GUNICORN_TIMEOUT}s;
        }}

        limit_conn addr 50;
    }}
}}
"""
    with open(NGINX_CONF_FILE, "w") as f:
        f.write(config.strip() + "\n")


# =============================================================================
# SERVER MANAGEMENT
# =============================================================================


def check_nginx():
    if shutil.which("nginx"):
        print("Nginx is installed.")
        return True
    print(
        "Error: 'nginx' not found. Please install it (e.g., 'sudo apt install nginx')."
    )
    return False


def check_gunicorn():
    if shutil.which("gunicorn"):
        return True
    print("Warning: 'gunicorn' not found. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gunicorn"])
        return True
    except Exception:
        print("Failed to install gunicorn.")
        return False


def is_windows():
    return platform.system() == "Windows"


def check_waitress():
    """Checks if waitress is installed, which is needed for Windows."""
    if shutil.which("waitress-serve") is not None:
        return True
    print("⚠️  'waitress-serve' command not found.")
    install_prompt = (
        input(
            "   It's needed for Windows support. Install it now? (pip install waitress) [y/N]: "
        )
        .lower()
        .strip()
    )
    if install_prompt == "y":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "waitress"])
        return shutil.which("waitress-serve") is not None
    return False


# =============================================================================484
# MAIN FUNCTIONS
# =============================================================================


def start_servers(args):
    if is_windows():
        print(
            "ℹ️  Windows detected. Using 'waitress' server instead of Gunicorn/Nginx."
        )
        if not check_waitress():
            print("❌ Cannot start server on Windows without 'waitress'.")
            return

        model_available = pre_download_model()
        gpu_ram_gb = get_gpu_ram()
        save_cache(model_available, gpu_ram_gb)

        server_env = os.environ.copy()
        if gpu_ram_gb > 0:
            print(f"🚀 Starting server in GPU mode on http://127.0.0.1:{NGINX_PORT}")
            server_env["DEVICE_TYPE"] = "gpu"
        else:
            print(f"🚀 Starting server in CPU mode on http://127.0.0.1:{NGINX_PORT}")
            server_env["DEVICE_TYPE"] = "cpu"

        # Use waitress-serve on Windows. It runs in the foreground.
        waitress_cmd = (
            f"waitress-serve --host 127.0.0.1 --port {NGINX_PORT} {SERVER_SCRIPT}"
        )
        print("   To stop the server, press Ctrl+C in this window.")
        try:
            subprocess.run(waitress_cmd.split(), env=server_env)
        except KeyboardInterrupt:
            print("\n✅ Server stopped by user.")
        except FileNotFoundError:
            print("❌ 'waitress-serve' not found. Please run 'pip install waitress'.")
        except Exception as e:
            print(f"❌ An error occurred while running the server: {e}")
        return  # End of Windows-specific logic

    if not check_nginx() or not check_gunicorn():
        return

    model_ok = pre_download_model()
    cpu_cores, ram_gb = get_system_resources()
    gpu_ram_gb = get_gpu_ram()
    save_cache(model_ok, gpu_ram_gb)

    # GPU workers
    gpu_workers = (
        args.gpu_workers
        if args.gpu_workers is not None
        else calculate_gpu_workers(gpu_ram_gb)
    )
    gpu_threads_per_worker = 4

    # CPU server: only if explicitly requested
    start_cpu_server = args.cpu
    cpu_weight = 5 if start_cpu_server else 0
    gpu_weight = 30

    print("\n" + "=" * 70)
    print("FINAL CONFIGURATION")
    print(
        f"   GPU VRAM       : {gpu_ram_gb:.2f} GB → {gpu_workers} worker(s) × {gpu_threads_per_worker} threads each"
    )
    print(f"   CPU cores/RAM  : {cpu_cores} cores / {ram_gb:.1f} GB")
    if start_cpu_server:
        cpu_threads = min(12, max(4, cpu_cores // 4))
        print(f"   CPU Server     : Enabled (fallback, {cpu_threads} threads)")
    else:
        print(f"   CPU Server     : Disabled (use --cpu to enable)")
    print("=" * 70 + "\n")

    generate_nginx_config(gpu_weight, cpu_weight, start_cpu_server)

    # Launch GPU server
    gpu_cmd = (
        f"gunicorn --workers {gpu_workers} "
        f"--threads {gpu_threads_per_worker} "
        f"--worker-class gthread "
        f"--timeout {GUNICORN_TIMEOUT} "
        f"--bind 127.0.0.1:{GPU_SERVER_PORT} "
        f"--backlog 2048 "
        f"{SERVER_SCRIPT}"
    )
    env_gpu = os.environ.copy()
    env_gpu["DEVICE_TYPE"] = "gpu"
    subprocess.Popen(gpu_cmd.split(), env=env_gpu)
    print(f"Launched GPU server → {gpu_workers} workers on port {GPU_SERVER_PORT}")

    # Launch CPU server (optional)
    if start_cpu_server:
        cpu_threads = min(12, max(4, cpu_cores // 4))
        cpu_cmd = (
            f"gunicorn --workers 1 "
            f"--threads {cpu_threads} "
            f"--timeout {GUNICORN_TIMEOUT} "
            f"--bind 127.0.0.1:{CPU_SERVER_PORT} "
            f"--backlog 2048 "
            f"{SERVER_SCRIPT}"
        )
        env_cpu = os.environ.copy()
        env_cpu["DEVICE_TYPE"] = "cpu"
        subprocess.Popen(cpu_cmd.split(), env=env_cpu)
        print(f"Launched CPU fallback server on port {CPU_SERVER_PORT}")

    # Launch Nginx
    nginx_cmd = f"nginx -c {os.path.abspath(NGINX_CONF_FILE)}"
    subprocess.Popen(nginx_cmd.split())
    print(f"Nginx load balancer started on http://127.0.0.1:{NGINX_PORT}")
    print("\nAll services are running.")


def stop_servers():
    if is_windows():
        print("On Windows, stop the server with Ctrl+C in its terminal.")
        return

    print("Stopping services...")
    subprocess.run(
        f"nginx -s stop -c {os.path.abspath(NGINX_CONF_FILE)}", shell=True, check=False
    )
    subprocess.run("pkill -f 'gunicorn.*server'", shell=True, check=False)
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except OSError:
            pass
    print("All services stopped.")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manage multi-worker RoBERTa inference servers with smart GPU scaling",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "action",
        nargs="?",
        default=None,
        choices=["start", "stop", "restart", "status", "clear-cache"],
        help="Action to perform (default: auto start/stop based on PID)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Enable CPU fallback server (default: off when GPU is strong)",
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=None,
        help="Override number of GPU workers (auto-detected by default)",
    )

    args = parser.parse_args()
    action = args.action

    if action is None:
        action = "start" if not os.path.exists(PID_FILE) else "stop"

    if action == "start":
        start_servers(args)
    elif action == "stop":
        stop_servers()
    elif action == "restart":
        stop_servers()
        time.sleep(2)
        start_servers(args)
    elif action == "status":
        print("RUNNING" if os.path.exists(PID_FILE) else "STOPPED")
    elif action == "clear-cache":
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print("Cache cleared.")
        else:
            print("No cache file found.")

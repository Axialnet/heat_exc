"""
Main Application Launcher for CHO Heat Exchanger Monitoring Platform.
Starts uvicorn serving FastAPI backend and the static industrial dashboard.

Usage:
    python run_server.py [--host 0.0.0.0] [--port 8000] [--reload]
"""

import argparse
import os
import sys
from pathlib import Path

import uvicorn

# Ensure repository root is in python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="CHO Predictive Maintenance Platform Server")
    default_port = int(os.environ.get("PORT", 8000))
    parser.add_argument("--host", default="0.0.0.0", help="Binding host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=default_port, help="Port to listen on (default: $PORT or 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-file", default=None, help="File path to write logs to")
    args = parser.parse_args()

    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = log_f
        sys.stderr = log_f

    print("=" * 75)
    print("  CHO 24/7 HEAT EXCHANGER MONITORING & PREDICTIVE MAINTENANCE PLATFORM")
    print("=" * 75)
    print(f"  • Web Dashboard URL : http://localhost:{args.port}")
    print(f"  • REST API Docs     : http://localhost:{args.port}/docs")
    print(f"  • WebSocket Stream  : ws://localhost:{args.port}/ws")
    print("=" * 75)
    print("  Starting server...")

    uvicorn.run(
        "webapp.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()

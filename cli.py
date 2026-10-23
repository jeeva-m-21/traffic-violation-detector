#!/usr/bin/env python3
"""Unified CLI Entrypoint for Smart Traffic Violation Detection & ANPR Platform.

Provides unified commands for:
  - web: Launching the FastAPI enforcement dashboard.
  - local: Running local CPU inference & verification.
  - pipeline: Executing video detection pipelines.
  - orchestrate: Running Colab GPU cloud orchestrations.
  - test: Running the automated unit test suite.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def cmd_web(args):
    """Launch the FastAPI dashboard server."""
    host = args.host or os.getenv("HOST", "0.0.0.0")
    port = int(args.port or os.getenv("PORT", "8000"))
    reload = args.reload
    print(f"🚀 Launching Traffic Enforcement Web Dashboard on http://{host}:{port}")
    import uvicorn
    uvicorn.run("app:app", host=host, port=port, reload=reload)


def cmd_local(args):
    """Run local CPU hardware inference."""
    script = ROOT_DIR / "scripts" / "pipelines" / "run_local_inference.py"
    cmd = [sys.executable, str(script)]
    if args.video:
        cmd.append(args.video)
    print(f"🔬 Running local CPU inference: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


def cmd_pipeline(args):
    """Execute a dedicated video processing pipeline."""
    mapping = {
        "integrated": "run_integrated_pipeline.py",
        "helmet": "run_helmet_pipeline.py",
        "anpr": "run_anpr_pipeline.py",
        "wrongway": "run_violation_pipeline.py",
    }
    script_name = mapping.get(args.type)
    if not script_name:
        print(f"Unknown pipeline type: {args.type}. Available: {list(mapping.keys())}")
        sys.exit(1)
    script = ROOT_DIR / "scripts" / "pipelines" / script_name
    cmd = [sys.executable, str(script)]
    print(f"🎥 Executing pipeline [{args.type}]: {script}")
    sys.exit(subprocess.call(cmd))


def cmd_orchestrate(args):
    """Run remote Colab GPU orchestrator."""
    mapping = {
        "integrated": "orchestrate_integrated.py",
        "helmet": "orchestrate_helmet.py",
        "anpr": "orchestrate_anpr.py",
        "wrongway": "orchestrate_violation.py",
        "plate-training": "orchestrate_plate_training.py",
    }
    script_name = mapping.get(args.target)
    if not script_name:
        print(f"Unknown orchestrator target: {args.target}. Available: {list(mapping.keys())}")
        sys.exit(1)
    script = ROOT_DIR / "scripts" / "orchestration" / script_name
    cmd = [sys.executable, str(script)]
    print(f"☁️ Running Colab orchestration [{args.target}]: {script}")
    sys.exit(subprocess.call(cmd))


def cmd_test(args):
    """Run automated unit tests."""
    print("🧪 Running automated unit test suite...")
    cmd = [sys.executable, "-m", "unittest", "discover", "tests"]
    sys.exit(subprocess.call(cmd, cwd=str(ROOT_DIR)))


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detection & ANPR Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # web
    web_p = subparsers.add_parser("web", help="Launch FastAPI web dashboard")
    web_p.add_argument("--host", default="0.0.0.0", help="Binding host (default: 0.0.0.0)")
    web_p.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    web_p.add_argument("--reload", action="store_true", help="Enable live auto-reloading")
    web_p.set_defaults(func=cmd_web)

    # local
    loc_p = subparsers.add_parser("local", help="Run local hardware CPU inference")
    loc_p.add_argument("--video", default=None, help="Path to input video file")
    loc_p.set_defaults(func=cmd_local)

    # pipeline
    pip_p = subparsers.add_parser("pipeline", help="Execute video detection pipeline")
    pip_p.add_argument(
        "type",
        choices=["integrated", "helmet", "anpr", "wrongway"],
        help="Type of pipeline to run",
    )
    pip_p.set_defaults(func=cmd_pipeline)

    # orchestrate
    orc_p = subparsers.add_parser("orchestrate", help="Run Colab GPU cloud orchestrator")
    orc_p.add_argument(
        "target",
        choices=["integrated", "helmet", "anpr", "wrongway", "plate-training"],
        help="Target orchestrator to run",
    )
    orc_p.set_defaults(func=cmd_orchestrate)

    # test
    tst_p = subparsers.add_parser("test", help="Run unit test suite")
    tst_p.set_defaults(func=cmd_test)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A dependency-free, stdio MCP server for safe Isaac Lab job control.

The server intentionally exposes a small whitelist of project commands instead
of accepting arbitrary shell input.  Isaac Sim jobs run as child processes and
write their output under ``.mcp/isaaclab/jobs``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any


SERVER_NAME = "tancho-isaaclab"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2025-03-26"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_ROOT = PROJECT_ROOT / ".mcp" / "isaaclab" / "jobs"
DEFAULT_TASK = "TanchoV3-Flat-v0"
ALLOWED_TASKS = {DEFAULT_TASK}
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


TOOLS = [
    {
        "name": "project_info",
        "description": "Return the Tancho Isaac Lab project and runtime configuration.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_tasks",
        "description": "Run the project's Isaac Lab task discovery script.",
        "inputSchema": {
            "type": "object",
            "properties": {"timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120, "default": 60}},
            "additionalProperties": False,
        },
    },
    {
        "name": "start_job",
        "description": "Start an approved validation, diagnostic, training, or policy-play job and return a job id.",
        "inputSchema": {
            "type": "object",
            "required": ["profile"],
            "properties": {
                "profile": {
                    "type": "string",
                    "enum": ["zero_agent", "pose_scan", "wheel_pd_scan", "train", "play"],
                },
                "task": {"type": "string", "enum": [DEFAULT_TASK], "default": DEFAULT_TASK},
                "num_envs": {"type": "integer", "minimum": 1, "maximum": 4096},
                "headless": {"type": "boolean", "default": True},
                "max_iterations": {"type": "integer", "minimum": 1, "maximum": 100000},
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 10000},
                "constant_scan": {"type": "boolean", "default": False},
                "checkpoint": {"type": "string", "description": "Project-relative checkpoint path; only valid for play."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "job_status",
        "description": "Return status and recent log output for a job started by this MCP server.",
        "inputSchema": {
            "type": "object",
            "required": ["job_id"],
            "properties": {
                "job_id": {"type": "string"},
                "tail_lines": {"type": "integer", "minimum": 1, "maximum": 500, "default": 80},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "stop_job",
        "description": "Gracefully stop a job started by this MCP server.",
        "inputSchema": {
            "type": "object",
            "required": ["job_id"],
            "properties": {"job_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "list_checkpoints",
        "description": "List recent RSL-RL checkpoint files under this project.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}},
            "additionalProperties": False,
        },
    },
]


def _runtime_prefix() -> list[str]:
    configured_python = os.environ.get("ISAACLAB_MCP_PYTHON")
    if configured_python:
        python_path = Path(configured_python).expanduser().resolve()
        if not python_path.is_file():
            raise ValueError(f"ISAACLAB_MCP_PYTHON does not exist: {python_path}")
        return [str(python_path)]
    conda = os.environ.get("ISAACLAB_MCP_CONDA", "conda")
    env_name = os.environ.get("ISAACLAB_MCP_CONDA_ENV", "env_isaaclab")
    return [conda, "run", "--no-capture-output", "-n", env_name, "python"]


def _safe_checkpoint(value: str) -> Path:
    candidate = (PROJECT_ROOT / value).resolve()
    logs_root = (PROJECT_ROOT / "logs" / "rsl_rl").resolve()
    if candidate.suffix != ".pt" or not candidate.is_relative_to(logs_root):
        raise ValueError("checkpoint must be a .pt file below logs/rsl_rl")
    if not candidate.is_file():
        raise ValueError(f"checkpoint does not exist: {value}")
    return candidate


def _build_command(args: dict[str, Any]) -> list[str]:
    profile = args["profile"]
    task = args.get("task", DEFAULT_TASK)
    if task not in ALLOWED_TASKS:
        raise ValueError(f"task is not allowed: {task}")

    scripts = {
        "zero_agent": PROJECT_ROOT / "scripts" / "zero_agent.py",
        "pose_scan": PROJECT_ROOT / "scripts" / "diagnostics" / "pose_scan.py",
        "wheel_pd_scan": PROJECT_ROOT / "scripts" / "diagnostics" / "wheel_pd_scan.py",
        "train": PROJECT_ROOT / "scripts" / "rsl_rl" / "train.py",
        "play": PROJECT_ROOT / "scripts" / "rsl_rl" / "play.py",
    }
    script = scripts[profile]
    command = _runtime_prefix() + [str(script), "--task", task]
    defaults = {"zero_agent": 8, "pose_scan": 8, "wheel_pd_scan": 8, "train": 4096, "play": 16}
    if profile in {"pose_scan", "wheel_pd_scan"}:
        if "num_envs" in args:
            raise ValueError("num_envs is fixed at 8 for diagnostic profiles")
    else:
        command += ["--num_envs", str(args.get("num_envs", defaults[profile]))]
    if args.get("headless", True):
        command.append("--headless")
    if "max_iterations" in args:
        if profile != "train":
            raise ValueError("max_iterations is only valid for train")
        command += ["--max_iterations", str(args["max_iterations"])]
    if "max_steps" in args:
        if profile not in {"zero_agent", "pose_scan", "wheel_pd_scan"}:
            raise ValueError("max_steps is only valid for zero_agent and diagnostic profiles")
        max_steps = int(args["max_steps"])
        if not 1 <= max_steps <= 10000:
            raise ValueError("max_steps must be between 1 and 10000")
        command += ["--max_steps", str(max_steps)]
    if args.get("constant_scan", False):
        if profile != "wheel_pd_scan":
            raise ValueError("constant_scan is only valid for wheel_pd_scan")
        command.append("--constant_scan")
    if "checkpoint" in args:
        if profile != "play":
            raise ValueError("checkpoint is only valid for play")
        command += ["--checkpoint", str(_safe_checkpoint(args["checkpoint"]))]
    return command


def _tail(path: Path, count: int) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return "".join(lines[-count:])


def _job_snapshot(job: dict[str, Any], tail_lines: int = 80) -> dict[str, Any]:
    process: subprocess.Popen[str] = job["process"]
    return_code = process.poll()
    status = "running" if return_code is None else ("completed" if return_code == 0 else "failed")
    return {
        "job_id": job["job_id"],
        "profile": job["profile"],
        "status": status,
        "pid": process.pid,
        "return_code": return_code,
        "started_at": job["started_at"],
        "log_path": str(job["log_path"]),
        "log_tail": _tail(job["log_path"], tail_lines),
    }


def _call_tool(name: str, args: dict[str, Any]) -> Any:
    if name == "project_info":
        return {
            "project_root": str(PROJECT_ROOT),
            "default_task": DEFAULT_TASK,
            "runtime_prefix": _runtime_prefix(),
            "job_state_root": str(STATE_ROOT),
        }
    if name == "list_tasks":
        timeout = int(args.get("timeout_seconds", 60))
        command = _runtime_prefix() + [str(PROJECT_ROOT / "scripts" / "list_envs.py")]
        result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        return {"return_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if name == "start_job":
        command = _build_command(args)
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        log_path = STATE_ROOT / f"{job_id}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        finally:
            log_handle.close()
        job = {
            "job_id": job_id,
            "profile": args["profile"],
            "process": process,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "log_path": log_path,
        }
        with JOBS_LOCK:
            JOBS[job_id] = job
        return _job_snapshot(job)
    if name == "job_status":
        with JOBS_LOCK:
            job = JOBS.get(args["job_id"])
        if job is None:
            raise ValueError("unknown job_id (job ids are scoped to the current MCP server process)")
        return _job_snapshot(job, int(args.get("tail_lines", 80)))
    if name == "stop_job":
        with JOBS_LOCK:
            job = JOBS.get(args["job_id"])
        if job is None:
            raise ValueError("unknown job_id")
        process: subprocess.Popen[str] = job["process"]
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        return _job_snapshot(job)
    if name == "list_checkpoints":
        limit = int(args.get("limit", 20))
        root = PROJECT_ROOT / "logs" / "rsl_rl"
        paths = sorted(root.rglob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)[:limit] if root.exists() else []
        return [{"path": str(path.relative_to(PROJECT_ROOT)), "size_bytes": path.stat().st_size} for path in paths]
    raise ValueError(f"unknown tool: {name}")


def _response(request_id: Any, result: Any = None, error: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result
    return message


def _handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        return None
    if method == "initialize":
        requested = message.get("params", {}).get("protocolVersion", PROTOCOL_VERSION)
        return _response(request_id, {
            "protocolVersion": requested,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
    if method == "ping":
        return _response(request_id, {})
    if method == "tools/list":
        return _response(request_id, {"tools": TOOLS})
    if method == "tools/call":
        params = message.get("params", {})
        try:
            value = _call_tool(params.get("name", ""), params.get("arguments") or {})
            return _response(request_id, {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, indent=2)}]})
        except Exception as exc:
            return _response(request_id, {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            })
    return _response(request_id, error={"code": -32601, "message": f"Method not found: {method}"})


def main() -> None:
    for raw_line in sys.stdin:
        try:
            message = json.loads(raw_line)
            reply = _handle(message)
        except Exception as exc:
            reply = _response(None, error={"code": -32700, "message": str(exc)})
        if reply is not None:
            sys.stdout.write(json.dumps(reply, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

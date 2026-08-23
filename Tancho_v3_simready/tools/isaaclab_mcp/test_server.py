from __future__ import annotations

import unittest
from unittest import mock

from tools.isaaclab_mcp import server


class ServerTests(unittest.TestCase):
    def test_initialize(self) -> None:
        response = server._handle({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        })
        self.assertEqual(response["result"]["serverInfo"]["name"], "tancho-isaaclab")

    def test_lists_expected_tools(self) -> None:
        response = server._handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {item["name"] for item in response["result"]["tools"]}
        self.assertEqual(names, {"project_info", "list_tasks", "start_job", "job_status", "stop_job", "list_checkpoints"})

    def test_train_command_is_whitelisted(self) -> None:
        with mock.patch.dict(server.os.environ, {"ISAACLAB_MCP_PYTHON": server.sys.executable}):
            command = server._build_command({"profile": "train", "max_iterations": 10})
        self.assertIn("scripts/rsl_rl/train.py", command[1])
        self.assertEqual(command[-2:], ["--max_iterations", "10"])

    def test_rejects_checkpoint_outside_logs(self) -> None:
        with self.assertRaises(ValueError):
            server._safe_checkpoint("README.zh.md")


    def test_pose_scan_command_is_whitelisted(self) -> None:
        with mock.patch.dict(server.os.environ, {"ISAACLAB_MCP_PYTHON": server.sys.executable}):
            command = server._build_command({"profile": "pose_scan", "max_steps": 20})
        self.assertIn("scripts/diagnostics/pose_scan.py", command[1])
        self.assertEqual(command[-2:], ["--max_steps", "20"])

    def test_wheel_pd_constant_scan_is_whitelisted(self) -> None:
        with mock.patch.dict(server.os.environ, {"ISAACLAB_MCP_PYTHON": server.sys.executable}):
            command = server._build_command({"profile": "wheel_pd_scan", "max_steps": 20, "constant_scan": True})
        self.assertIn("scripts/diagnostics/wheel_pd_scan.py", command[1])
        self.assertEqual(command[-3:], ["--max_steps", "20", "--constant_scan"])

    def test_rejects_diagnostic_argument_for_train(self) -> None:
        with self.assertRaises(ValueError):
            server._build_command({"profile": "train", "max_steps": 20})

    def test_rejects_num_envs_for_diagnostics(self) -> None:
        with self.assertRaises(ValueError):
            server._build_command({"profile": "pose_scan", "num_envs": 4})


if __name__ == "__main__":
    unittest.main()

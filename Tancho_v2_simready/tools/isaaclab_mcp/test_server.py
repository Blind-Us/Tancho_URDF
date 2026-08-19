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


if __name__ == "__main__":
    unittest.main()

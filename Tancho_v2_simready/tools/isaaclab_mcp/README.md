# Tancho Isaac Lab MCP adapter

This local stdio MCP server exposes a deliberately small command whitelist for
the Tancho Isaac Lab project. It does not accept arbitrary shell commands.

## Tools

- `project_info`
- `list_tasks`
- `start_job` (`zero_agent`, `train`, or `play`)
- `job_status`
- `stop_job`
- `list_checkpoints`

Job output is stored in `.mcp/isaaclab/jobs/`. Job IDs live for the lifetime of
the MCP server process.

## Codex configuration

Add this to the local Codex MCP configuration, using the absolute project path:

```toml
[mcp_servers.tanchoIsaacLab]
command = "python3"
args = ["/media/azul/861896C11896B023/Tancho/Tancho_v2_simready/tools/isaaclab_mcp/server.py"]

[mcp_servers.tanchoIsaacLab.env]
ISAACLAB_MCP_CONDA_ENV = "env_isaaclab"
```

If Conda is not available to the MCP process, point directly at the environment
Python instead:

```toml
[mcp_servers.tanchoIsaacLab.env]
ISAACLAB_MCP_PYTHON = "/absolute/path/to/env_isaaclab/bin/python"
```

Restart or reload the local Codex client after changing its MCP configuration.

## Protocol smoke test

```bash
python3 -m unittest tools.isaaclab_mcp.test_server
```


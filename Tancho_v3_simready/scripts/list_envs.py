import argparse

parser = argparse.ArgumentParser(description="List Tancho V3 Isaac Lab environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
args_cli = parser.parse_args()

import gymnasium as gym
from prettytable import PrettyTable

import tancho_v3_lab.tasks  # noqa: F401


def main():
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Tancho V3 Isaac Lab Environments"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if "TanchoV3-" in task_spec.id and (args_cli.keyword is None or args_cli.keyword in task_spec.id):
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            index += 1
    print(table)


if __name__ == "__main__":
    main()

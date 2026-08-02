"""Public command-line interface for report reproduction."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TextIO

from buffalo_weight.report_environment import SetupServices, setup_official_environment
from buffalo_weight.system_setup import default_setup_services


def main(
    argv: Sequence[str] | None = None,
    services: SetupServices | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the public CLI; for example, ``main(["setup"])`` prepares the environment."""
    arguments = _build_parser().parse_args(argv)
    if arguments.command != "setup":
        raise ValueError(f"command was {arguments.command!r}; expected 'setup'")
    try:
        return _run_setup(services or default_setup_services(), stdout)
    except ValueError as error:
        print(f"rejected: {error}", file=stderr)
        return 1


def _run_setup(services: SetupServices, stdout: TextIO) -> int:
    messages = setup_official_environment(services)
    for message in messages:
        print(message, file=stdout)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce buffalo-weight report evidence")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("setup", help="prepare and audit the official environment")
    return parser

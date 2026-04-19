"""Helpers to print all runtime configs and validate fps consistency before running.

Designed to be called once at the top of every entrypoint so a human always sees
exactly which values (defaults + overrides + dataclass field values) are about
to drive the run.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields, is_dataclass
from typing import Iterable


def format_dataclass_block(title: str, obj) -> str:
    """Render one dataclass instance (recursively) as a labeled text block."""
    lines = [f"[{title}]"]
    lines.extend(_format_dataclass_lines(obj, indent=2))
    return "\n".join(lines)


def _format_dataclass_lines(obj, indent: int) -> list[str]:
    """Return the indented lines for one dataclass instance, recursing into nested ones."""
    pad = " " * indent
    out: list[str] = []
    if not is_dataclass(obj):
        out.append(f"{pad}{obj!r}")
        return out
    for f in fields(obj):
        value = getattr(obj, f.name)
        if is_dataclass(value):
            out.append(f"{pad}{f.name}:")
            out.extend(_format_dataclass_lines(value, indent + 2))
        elif isinstance(value, dict) and value and all(is_dataclass(v) for v in value.values()):
            out.append(f"{pad}{f.name}:")
            for k, v in value.items():
                out.append(f"{pad}  {k}:")
                out.extend(_format_dataclass_lines(v, indent + 4))
        else:
            out.append(f"{pad}{f.name:<28} = {value!r}")
    return out


def format_args_block(title: str, args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """Render argparse args as a labeled text block, marking each as [user] or [default]."""
    defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    lines = [f"[{title}]"]
    for key, value in sorted(vars(args).items()):
        marker = "[user]   " if value != defaults.get(key) else "[default]"
        lines.append(f"  {marker} {key:<28} = {value!r}")
    return "\n".join(lines)


def print_runtime_banner(*blocks: str, require_confirm: bool = True) -> None:
    """Print a visible config banner; if require_confirm, ask user to confirm before continuing.

    The confirmation prompt is auto-skipped when stdin is not a TTY (e.g. piped/scripted
    runs) or when the LEROBOT_NO_CONFIRM environment variable is set to a truthy value.
    """
    rule = "=" * 78
    print()
    print(rule)
    print("  RUNTIME CONFIG  --  verify before continuing")
    print(rule)
    for block in blocks:
        print(block)
        print()
    print(rule)
    env_skip = os.environ.get("LEROBOT_NO_CONFIRM", "").lower() in ("1", "true", "yes")
    if require_confirm and sys.stdin.isatty() and not env_skip:
        try:
            answer = input("Continue with these settings? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Aborted by user.")
            sys.exit(0)
    print()


def validate_loop_freq_vs_cameras(loop_freq_hz: int, camera_fps_values: Iterable[int]) -> None:
    """Raise ValueError if the host loop frequency is below any camera's configured fps."""
    fps_list = list(camera_fps_values)
    if not fps_list:
        return
    max_cam_fps = max(fps_list)
    if loop_freq_hz < max_cam_fps:
        raise ValueError(
            f"Host max_loop_freq_hz ({loop_freq_hz}) is below max camera fps ({max_cam_fps}). "
            f"The host loop will starve cameras and produce stale observations. "
            f"Either raise max_loop_freq_hz to >= {max_cam_fps} or lower camera fps."
        )


def validate_client_fps_vs_cameras(client_fps: int, camera_fps_values: Iterable[int]) -> None:
    """Raise ValueError if the client recording fps does not match the cameras' configured fps."""
    fps_list = list(camera_fps_values)
    if not fps_list:
        return
    max_cam_fps = max(fps_list)
    if client_fps != max_cam_fps:
        raise ValueError(
            f"Client --fps ({client_fps}) does not match camera fps ({max_cam_fps}). "
            f"Recording at a different rate from the cameras yields duplicate or dropped frames. "
            f"Set --fps {max_cam_fps} (or change camera fps to {client_fps})."
        )

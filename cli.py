"""Local command: hermes cutout call. It prints JSON and does not write."""

from __future__ import annotations

import argparse
import json
import sys

if __package__:
    from .errors import BAD_ARGUMENT, fail
    from .service import cli_preview
else:
    from errors import BAD_ARGUMENT, fail
    from service import cli_preview


def setup_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="command", required=True)
    call = sub.add_parser("call", help="Print the cutout plan as JSON")
    call.add_argument("--image", required=True)
    call.add_argument("--folder", required=True)
    call.add_argument("--name", default=None)


def handle(args: argparse.Namespace) -> int:
    if getattr(args, "command", None) != "call":
        payload = fail(
            BAD_ARGUMENT,
            "The command is not call, so nothing was written.",
            "Run hermes cutout call with --image and --folder.",
        )
    else:
        payload = cli_preview(args.image, args.folder, args.name)
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    if payload.get("ok") is True:
        return 0
    return 2

"""argparse + dispatch for ``mdview``."""
from __future__ import annotations

import argparse
from typing import List, Optional, cast

from .app import ViewArgs, ViewerApp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mdview",
        description="Modal viewer for streamed markdown (file, fifo, or stdin).",
    )
    p.add_argument(
        "path",
        help="File to display. Use '-' for stdin (e.g. `bridge | view -`). "
             "If the path is a fifo or '-', the viewer auto-follows and "
             "exits on EOF unless overridden.",
    )
    p.add_argument(
        "--title", default="thinking",
        help="Panel title (default: 'thinking').",
    )
    p.add_argument(
        "--subtitle", default="",
        help="Panel subtitle (shown in the border on the top-right).",
    )
    follow = p.add_mutually_exclusive_group()
    follow.add_argument(
        "--follow", action="store_true", dest="follow", default=None,
        help="Tail-follow the input as it grows. Default: on for stdin "
             "and fifos, off for regular files.",
    )
    follow.add_argument(
        "--no-follow", action="store_false", dest="follow",
        help="Open the file as-is, no follow.",
    )
    eof = p.add_mutually_exclusive_group()
    eof.add_argument(
        "--exit-on-eof", action="store_true", dest="exit_on_eof", default=None,
        help="Exit when the input source EOFs. Default: on for stdin and "
             "fifos (writer closing = done); off for regular files (no "
             "natural EOF signal — user quits with q).",
    )
    eof.add_argument(
        "--no-exit-on-eof", action="store_false", dest="exit_on_eof",
        help="Stay open after EOF; user quits with q.",
    )
    # ── inline-mode sizing ─────────────────────────────────────────────
    p.add_argument(
        "--inline", action="store_true",
        help="Render inline below the cursor instead of taking over the "
             "screen with alt-screen. Default: alt-screen.",
    )
    p.add_argument(
        "--height", type=int, default=20,
        help="Rows for the inline area (default 20). Ignored unless "
             "--inline is set.",
    )
    p.add_argument(
        "--width", type=int, default=0,
        help="Columns for the inline area (default 0 = full terminal "
             "width). Ignored unless --inline is set.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # Defaults that depend on the input source kind happen in the App
    # (it stats the path); leaving --follow / --exit-on-eof as None
    # here signals "auto" to the App.
    return ViewerApp(cast(ViewArgs, vars(args))).run_viewer()

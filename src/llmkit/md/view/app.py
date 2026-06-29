"""Textual-based modal viewer for streamed markdown output.

Input sources: a regular file, a fifo, or stdin (path "-"). The viewer
auto-detects the kind and picks sensible defaults for ``follow`` and
``exit_on_eof``:

  regular file : follow=off, exit_on_eof=off  → static view, q to quit
  fifo         : follow=on,  exit_on_eof=on   → live stream, exits on
                                                 writer close
  stdin ("-")  : follow=on,  exit_on_eof=on   → pipeline shape;
                                                 keyboard via /dev/tty

The defaults can be overridden via CLI flags (``--follow``,
``--exit-on-eof`` and their ``--no-...`` counterparts).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Optional, TypedDict

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Markdown


class ViewArgs(TypedDict):
    """Resolved CLI args for the viewer. Lives here (not in cli.py)
    so ViewerApp can type its constructor against it without circular
    imports. Mirror of the argparse Namespace."""
    path: str
    title: str
    subtitle: str
    follow: Optional[bool]         # None = auto-detect
    exit_on_eof: Optional[bool]    # None = auto-detect
    inline: bool
    height: int
    width: int


class ViewerApp(App[int]):
    BINDINGS = [
        Binding("q,escape", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("f", "toggle_follow", "Toggle follow"),
    ]

    CSS = """
    #scroller {
        /* Rounded chrome around the scrollable area. The title /
           subtitle are placed in the border (top-left / top-right)
           via border_title / border_subtitle set in on_mount. */
        border: round $primary;
        padding: 0 1;
        /* 1fr = "consume the remaining vertical space after the
           fixed-height Footer sibling". Required because the inner
           Markdown widget defaults to height:auto, which is 0 when
           content is empty at mount; without 1fr on the scroller
           the layout collapses regardless of --height. */
        height: 1fr;
    }
    """

    def __init__(self, config: ViewArgs) -> None:
        super().__init__()
        # Use ANSI default colors instead of the theme's RGB palette
        # so the background is the terminal's own bg, not textual's
        # dark fill. Set on the instance (not as a class attr) to
        # play nice with pyright's view of the Reactive descriptor.
        self.ansi_color = True
        self.config: ViewArgs = config
        # Classify the input source and pick defaults.
        self._source = _classify(config["path"])
        self._following = (
            config["follow"]
            if config["follow"] is not None
            else self._source != "file"
        )
        self._exit_on_eof = (
            config["exit_on_eof"]
            if config["exit_on_eof"] is not None
            else self._source != "file"
        )
        # File-mode state.
        self._file_offset = 0
        # Stdin/fifo state.
        self._read_fd: Optional[int] = None
        if config["title"]:
            self.title = config["title"]
        if config["subtitle"]:
            self.sub_title = config["subtitle"]

    # ── compose ────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        with VerticalScroll(id="scroller"):
            yield Markdown(id="md")
        yield Footer()

    # ── lifecycle ──────────────────────────────────────────────────────
    def on_mount(self) -> None:
        scroller = self.query_one("#scroller", VerticalScroll)
        if self.config["title"]:
            scroller.border_title = self.config["title"]
        if self.config["subtitle"]:
            scroller.border_subtitle = self.config["subtitle"]
        if self.config["inline"]:
            # In inline mode, Screen.styles.height drives the rendered
            # area; the `size=(W, H)` arg only caps that value.
            self.screen.styles.height = self.config["height"]
        if self._following:
            scroller.anchor()
        # Open the read fd for stdin / fifo here so we can attach it
        # to the running event loop in the worker.
        if self._source != "file":
            self._read_fd = self._open_read_fd()
        self.run_worker(self._stream_markdown(), exclusive=True)

    # ── streaming worker ───────────────────────────────────────────────
    async def _stream_markdown(self) -> None:
        md = self.query_one("#md", Markdown)
        scroller = self.query_one("#scroller", VerticalScroll)
        stream = Markdown.get_stream(md)
        # Status indicator: subtitle starts as "waiting…" so a slow
        # TTFT doesn't leave the user staring at empty chrome with no
        # feedback. Flips to "streaming" on first content, "complete"
        # on EOF when we're not auto-exiting.
        base_sub = self.config["subtitle"]
        scroller.border_subtitle = self._subtitle("waiting…", base_sub)
        seen_first_chunk = False
        try:
            while True:
                chunk = await self._next_chunk()
                if chunk is None:
                    # EOF.
                    if self._exit_on_eof:
                        self.exit()
                        return
                    scroller.border_subtitle = self._subtitle("complete", base_sub)
                    return
                if not seen_first_chunk:
                    scroller.border_subtitle = self._subtitle("streaming", base_sub)
                    seen_first_chunk = True
                await stream.write(chunk)
        finally:
            await stream.stop()
            if self._read_fd is not None and self._read_fd != 0:
                # Don't close fd 0 (stdin) — Python owns it.
                try:
                    os.close(self._read_fd)
                except OSError:
                    pass

    @staticmethod
    def _subtitle(status: str, base: str) -> str:
        """Compose the border subtitle: "<base> · <status>" or just <status>."""
        return f"{base} · {status}" if base else status

    async def _next_chunk(self) -> Optional[str]:
        """Block until new bytes are available, or EOF. Returns the
        chunk, or None to signal the worker should terminate.

        Cooperative: when there's no data, we `await asyncio.sleep`
        instead of blocking the thread. select.select() would freeze
        textual's event loop and starve input handling — a 50ms select
        per iteration is enough to make keystrokes feel sluggish."""
        while True:
            if self._source == "file":
                chunk = self._read_new_bytes_from_file()
                if chunk:
                    return chunk
                if not self._following:
                    # Static view: read what's there, then idle until q.
                    return None
                await asyncio.sleep(0.05)
            else:
                # stdin or fifo: non-blocking read; EAGAIN/EWOULDBLOCK
                # → yield to the event loop, try again. Never blocks
                # the thread synchronously.
                assert self._read_fd is not None
                try:
                    data = os.read(self._read_fd, 4096)
                except BlockingIOError:
                    await asyncio.sleep(0.05)
                    continue
                except OSError:
                    return None
                if not data:
                    # EOF — writer closed.
                    return None
                return data.decode("utf-8", errors="replace")

    def _read_new_bytes_from_file(self) -> str:
        try:
            size = os.path.getsize(self.config["path"])
        except OSError:
            return ""
        if size <= self._file_offset:
            return ""
        try:
            with open(self.config["path"], "rb") as f:
                f.seek(self._file_offset)
                chunk = f.read(size - self._file_offset)
        except OSError:
            return ""
        if not chunk:
            return ""
        self._file_offset += len(chunk)
        return chunk.decode("utf-8", errors="replace")

    def _open_read_fd(self) -> int:
        """Open the input fd for stdin or fifo sources."""
        if self._source == "stdin":
            return sys.stdin.fileno()
        # fifo: open RW so we don't block waiting for a writer; once a
        # writer connects and closes, our reads will see EOF.
        return os.open(self.config["path"], os.O_RDWR | os.O_NONBLOCK)

    # ── actions ────────────────────────────────────────────────────────
    def action_toggle_follow(self) -> None:
        self._following = not self._following
        if self._following:
            self.query_one("#scroller", VerticalScroll).anchor()

    # ── entry point ────────────────────────────────────────────────────
    def run_viewer(self) -> int:
        if self._source == "file" and not Path(self.config["path"]).is_file():
            print(
                f"mdview: '{self.config['path']}' not found",
                file=sys.stderr,
            )
            return 2
        try:
            if self.config["inline"]:
                width = self.config["width"] or shutil.get_terminal_size().columns
                self.run(inline=True, size=(width, self.config["height"]))
            else:
                self.run()
        except KeyboardInterrupt:
            return 130
        return 0


def _classify(path: str) -> str:
    """Return 'stdin', 'fifo', or 'file' for the input source."""
    if path == "-":
        return "stdin"
    try:
        st = os.stat(path)
    except OSError:
        return "file"   # may not exist yet; caller validates
    if stat.S_ISFIFO(st.st_mode):
        return "fifo"
    return "file"

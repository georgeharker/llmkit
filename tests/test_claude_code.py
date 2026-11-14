"""The claude_code adapter's CLI: started with no MCP servers, and gone after a
timeout even when it hangs and ignores SIGTERM."""

from __future__ import annotations

import os
import signal
import stat
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sdk = pytest.importorskip("claude_agent_sdk")

import llmkit.bridge._spawn as _spawn  # noqa: E402
import llmkit.bridge.adapters.claude_code as cc  # noqa: E402
from llmkit.bridge import ChatRequest, Provider, chat_to_str  # noqa: E402


@pytest.fixture
def no_managed_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cc, "_MANAGED_MCP_PATHS", {})


def _options() -> Any:
    return cc._build_options(sdk, Provider(model="haiku"), ChatRequest(user="hi"))


def test_no_mcp_servers(no_managed_mcp: None, tmp_path: Path) -> None:
    opts = _options()
    assert opts.strict_mcp_config is True
    assert opts.env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    assert opts.setting_sources == []
    assert not opts.mcp_servers
    # And it reaches the CLI's argv: strict, with no --mcp-config at all.
    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\n")
    opts.cli_path = str(cli)
    cmd = _spawn.cli_transport("hi", opts)._build_command()
    assert "--strict-mcp-config" in cmd
    assert "--mcp-config" not in cmd


def test_managed_mcp_disables_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With managed-mcp.json deployed the CLI exits on --strict-mcp-config, so
    strict mode is skipped, with one warning per process."""
    managed = tmp_path / "managed-mcp.json"
    managed.write_text('{"mcpServers": {}}')
    monkeypatch.setattr(cc, "_MANAGED_MCP_PATHS", {sys.platform: str(managed)})
    monkeypatch.setattr(cc, "_warned_managed_mcp", False)
    assert _options().strict_mcp_config is False
    assert _options().strict_mcp_config is False
    err = capsys.readouterr().err
    assert err.count("managed-mcp.json is deployed") == 1
    # The connector switch still applies.
    assert _options().env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"


def test_timeout_kills_a_hung_cli(
    no_managed_mcp: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A CLI stuck in startup that ignores SIGTERM: the call times out, and
    close() escalates to SIGKILL instead of leaving it running. The timeout
    must outlast the SDK's own startup (~1 s on newer SDKs), so the CLI is
    actually running when it fires."""
    grace = 0.3
    timeout = 3.0
    monkeypatch.setattr(_spawn, "_CLOSE_GRACE", grace)
    pidfile = tmp_path / "pid"
    cli = tmp_path / "claude"
    cli.write_text(
        '#!/bin/sh\nif [ "$1" = "-v" ]; then echo "9.9.9 (Claude Code)"; exit 0; fi\n'
        f"echo $$ > {pidfile}\ntrap '' TERM\nexec sleep 300\n"
    )
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    build = cc._build_options

    def with_fake_cli(s: Any, p: Provider, r: ChatRequest) -> Any:
        opts = build(s, p, r)
        opts.cli_path = str(cli)
        return opts

    monkeypatch.setattr(cc, "_build_options", with_fake_cli)

    t0 = time.monotonic()
    code, _ = chat_to_str(
        Provider(adapter="claude_code", model="haiku", timeout=timeout),
        ChatRequest(user="hi"),
    )
    elapsed = time.monotonic() - t0
    pid = int(pidfile.read_text())
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    if alive:
        os.kill(pid, signal.SIGKILL)
    assert code == 1
    assert f"no complete response within {timeout:g}s" in capsys.readouterr().err
    assert not alive
    # SIGTERM after one grace, SIGKILL after two: not the SDK's own 5 s + 5 s.
    assert elapsed < timeout + 2 * grace + 1.5

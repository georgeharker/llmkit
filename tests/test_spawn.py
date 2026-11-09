"""llmkit must start child processes with posix_spawn, never fork() — see
``llmkit.bridge._spawn``. These spy on CPython's two launch paths and fail if
the fork one (``_posixsubprocess.fork_exec``) is ever taken."""

from __future__ import annotations

import _posixsubprocess
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import anyio
import pytest

import llmkit.bridge._spawn as _spawn


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    seen = {"spawn": 0, "fork": 0}
    real_spawn = subprocess.Popen._posix_spawn  # type: ignore[attr-defined]
    real_fork = _posixsubprocess.fork_exec

    def spy_spawn(self: Any, *a: Any, **kw: Any) -> Any:
        seen["spawn"] += 1
        return real_spawn(self, *a, **kw)

    def spy_fork(*a: Any, **kw: Any) -> Any:
        seen["fork"] += 1
        return real_fork(*a, **kw)

    monkeypatch.setattr(subprocess.Popen, "_posix_spawn", spy_spawn)
    monkeypatch.setattr(_posixsubprocess, "fork_exec", spy_fork)
    return seen


def _fake_cli(tmp_path: Path) -> Path:
    """A stand-in `claude`: answers the `-v` probe, otherwise waits on stdin."""
    cli = tmp_path / "claude"
    cli.write_text(
        '#!/bin/sh\nif [ "$1" = "-v" ]; then echo "9.9.9 (Claude Code)"; exit 0; fi\n'
        "cat >/dev/null\n"
    )
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    return cli


def test_open_process_spawns(launches: dict[str, int], tmp_path: Path) -> None:
    async def go() -> bytes:
        async with await _spawn.open_process(
            ["pwd"], stdin=subprocess.DEVNULL, cwd=tmp_path
        ) as p:
            assert p.stdout is not None
            out = await p.stdout.receive()
            await p.wait()
            return out

    assert os.path.realpath(anyio.run(go).decode().strip()) == os.path.realpath(
        tmp_path
    )
    assert launches == {"spawn": 1, "fork": 0}


def test_open_process_refuses_fork_options() -> None:
    with pytest.raises(ValueError, match="start_new_session"):
        anyio.run(
            lambda: _spawn.open_process(["true"], start_new_session=True)
        )


def test_missing_program() -> None:
    with pytest.raises(FileNotFoundError):
        _spawn.spawn_argv(["llmkit-no-such-program"])


def test_claude_code_transport_spawns(
    launches: dict[str, int], tmp_path: Path
) -> None:
    """The claude_code adapter's transport: the version probe and the CLI
    launch both posix_spawn."""
    sdk = pytest.importorskip("claude_agent_sdk")
    options = sdk.ClaudeAgentOptions(cli_path=str(_fake_cli(tmp_path)))

    async def go() -> None:
        t = _spawn.cli_transport("hi", options)
        await t.connect()
        await t.close()

    anyio.run(go)
    assert launches["fork"] == 0
    assert launches["spawn"] == 2  # `claude -v`, then the CLI itself


def test_other_sdk_users_untouched(tmp_path: Path) -> None:
    """Outside a spawn transport's connect, the SDK's anyio is the real one."""
    from claude_agent_sdk._internal.transport import subprocess_cli as sc

    _spawn._spawn_transport_cls()
    proxy = getattr(sc, _spawn._SDK_ANYIO)
    assert isinstance(proxy, _spawn._AnyioProxy)
    assert proxy.Lock is anyio.Lock
    assert not _spawn._SPAWNING.get()

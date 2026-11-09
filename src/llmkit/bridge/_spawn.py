"""Start child processes WITHOUT fork().

Why: on macOS, once a process has used Apple's Network.framework (a host's
DuckDB Quack client does, through its HTTP stack), the framework's fork handler
(``nw_settings_child_has_forked``) segfaults the child of every later fork()
before it can exec. llmkit is embedded in such hosts (cribsheet's daemon), so
an adapter that launches a program — ``claude_code`` starting the ``claude``
CLI — must never fork. posix_spawn runs no fork handlers.

CPython takes the posix_spawn path only when the program is an explicit path,
there is no ``cwd``, file descriptors are inherited (macOS has no posix_spawn
closefrom), and none of preexec_fn / start_new_session / pass_fds / umask /
process_group / user / group are set. :func:`open_process` arranges all of it:

- the program is resolved to an absolute path (a missing one raises
  FileNotFoundError, exactly as ``subprocess`` would);
- ``close_fds=False``: safe, because Python's own fds are non-inheritable
  (PEP 446);
- a ``cwd`` is applied by ``/bin/sh`` (``cd``, then ``exec`` the program);
- options that would force fork() are refused, not silently forked.

The Claude Agent SDK spawns the CLI itself, through ``anyio.open_process``,
which has no ``close_fds`` parameter and so always forks. :func:`cli_transport`
returns an SDK transport whose ``connect`` routes that call (and the CLI version
probe) through :func:`open_process` instead; any other user of the SDK in the
same process is untouched.
"""

from __future__ import annotations

import contextvars
import errno
import os
import shutil
from collections.abc import Sequence
from subprocess import PIPE
from typing import Any

_SH = "/bin/sh"
# $1 is the directory; the rest is the program and its arguments.
_CD_EXEC = 'cd -- "$1" || exit 127; shift; exec "$@"'
# Options that force CPython back onto fork(), with the default that doesn't
# (anyio.open_process forwards several of these at their defaults).
_FORK_FREE_DEFAULTS: dict[str, Any] = {
    "preexec_fn": None,
    "start_new_session": False,
    "pass_fds": (),
    "process_group": None,
    "user": None,
    "group": None,
    "extra_groups": None,
    "umask": -1,
    "executable": None,
    "startupinfo": None,
    "creationflags": 0,
}

# Set while a :func:`cli_transport` is connecting: only then does the SDK's
# ``anyio.open_process`` go through :func:`open_process`.
_SPAWNING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "llmkit_spawning", default=False
)


def spawn_argv(
    cmd: Sequence[str | os.PathLike[str]], cwd: str | os.PathLike[str] | None = None
) -> list[str]:
    """``cmd`` rewritten so CPython can posix_spawn it: an absolute program
    path, and ``cwd`` (if any) applied by ``/bin/sh`` rather than by Popen."""
    if not cmd:
        raise ValueError("empty command")
    args = [os.fspath(a) for a in cmd]
    prog = args[0]
    exe = prog if os.path.dirname(prog) else shutil.which(prog)
    if exe is None:
        raise FileNotFoundError(
            errno.ENOENT, f"No such file or directory: {prog!r}", prog
        )
    exe = os.path.abspath(exe)
    if cwd is None:
        return [exe, *args[1:]]
    d = os.fspath(cwd)
    if not os.path.isdir(d):
        raise FileNotFoundError(errno.ENOENT, f"No such directory: {d!r}", d)
    return [_SH, "-c", _CD_EXEC, "llmkit", d, exe, *args[1:]]


async def open_process(
    command: Sequence[str | os.PathLike[str]],
    *,
    stdin: Any = PIPE,
    stdout: Any = PIPE,
    stderr: Any = PIPE,
    cwd: str | os.PathLike[str] | None = None,
    env: Any = None,
    **kwargs: Any,
) -> Any:
    """``anyio.open_process``, spawned without fork(). Returns an
    ``anyio.abc.Process``."""
    if isinstance(command, (str, bytes)):
        raise ValueError("llmkit spawn takes an argv sequence, not a shell string")
    forking = sorted(
        k
        for k, v in kwargs.items()
        if k not in _FORK_FREE_DEFAULTS or v != _FORK_FREE_DEFAULTS[k]
    )
    if forking:
        raise ValueError(f"llmkit spawn refuses options that force fork(): {forking}")
    # anyio.open_process can't pass close_fds; its backend forwards any kwarg
    # to Popen (asyncio and trio alike). Not re-exported publicly by anyio.
    from anyio._core._eventloop import get_async_backend

    return await get_async_backend().open_process(
        spawn_argv(command, cwd),
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        env=env,
        close_fds=False,
    )


class _AnyioProxy:
    """Stands in for ``anyio`` inside the SDK's transport module: everything
    passes through, except ``open_process`` while a spawn transport connects."""

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def open_process(self, command: Any, **kwargs: Any) -> Any:
        if _SPAWNING.get():
            return await open_process(command, **kwargs)
        return await self._real.open_process(command, **kwargs)


_transport_cls: Any = None
# The SDK transport module's `import anyio` binding.
_SDK_ANYIO = "anyio"


def _spawn_transport_cls() -> Any:
    global _transport_cls
    if _transport_cls is not None:
        return _transport_cls
    from claude_agent_sdk._internal.transport import subprocess_cli as sc

    if not hasattr(sc, _SDK_ANYIO) or not hasattr(sc, "SubprocessCLITransport"):
        raise RuntimeError(
            "llmkit: claude_agent_sdk's subprocess transport changed shape; "
            "can't guarantee a fork-free CLI launch"
        )
    # The module's own `import anyio` binding, swapped for the proxy (attribute
    # access by name: it isn't a re-export).
    real = getattr(sc, _SDK_ANYIO)
    if not isinstance(real, _AnyioProxy):
        setattr(sc, _SDK_ANYIO, _AnyioProxy(real))

    class SpawnCLITransport(sc.SubprocessCLITransport):  # type: ignore[misc, name-defined]
        """The SDK's CLI transport, launching the CLI with posix_spawn."""

        async def connect(self) -> None:
            token = _SPAWNING.set(True)
            try:
                await super().connect()
            finally:
                _SPAWNING.reset(token)

    _transport_cls = SpawnCLITransport
    return _transport_cls


def cli_transport(prompt: Any, options: Any) -> Any:
    """A Claude Agent SDK transport for ``sdk.query(..., transport=...)`` that
    starts the ``claude`` CLI without fork()."""
    return _spawn_transport_cls()(prompt=prompt, options=options)

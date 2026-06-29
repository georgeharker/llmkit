"""Typed, extensible provider/profile config — the generic half of what
zsh-ai's ``models.py`` used to do inline.

The model has three tables, all generic:

  ``[defaults]``         fields merged into every provider (a partial
                         :class:`Provider`).
  ``[providers.<name>]`` a named backend config: adapter (transport),
                         model, endpoint, key, tuning. Resolve a usable
                         one with :meth:`Config.resolve` (overlays defaults).
  ``[profiles.<name>]``  a named map of *opaque* selector-key → provider
                         name. The library never interprets the keys — a
                         consumer (e.g. zsh-ai's "widgets": ask/modify/
                         question/fim) gives them meaning via
                         :meth:`Config.select`.

Extending the schema is by *subclassing*, not by stuffing untyped values
into a bag: parameterize :class:`ConfigParser` with a richer
:class:`Provider` subclass and/or override :meth:`ConfigParser.build` to
return a :class:`Config` subclass carrying extra typed sections. The
parser's return type tracks your subclass — no ``Any``, no post-hoc
mutation. The only escape hatch for genuinely-unknown *keys* is
:attr:`Provider.extra`, typed as :data:`TomlValue` (TOML's closed value
grammar), used purely for forward-compatible passthrough.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Generic, Mapping, Optional, Self, TypeAlias, TypeVar, cast

# Exactly what ``tomllib.load`` can produce — a closed, recursive set.
# Used for unknown passthrough keys and as the parser's input value type.
TomlValue: TypeAlias = (
    "str | int | float | bool | datetime | date | time "
    "| list[TomlValue] | dict[str, TomlValue]"
)

# Known provider fields (anything else in a [providers.*] table → .extra).
_PROVIDER_FIELDS = frozenset(
    {
        "model",
        "adapter",
        "endpoint",
        "api_key",
        "api_key_env",
        "max_tokens",
        "temperature",
        "enable_thinking",
        "stop",
    }
)


def _normalise_thinking(value: object) -> Optional[str]:
    """Normalise an ``enable_thinking`` value to the bridge's tri-state
    ``auto`` | ``true`` | ``false`` (or ``None`` if unspecified)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    s = str(value).strip().lower()
    if s in ("yes", "true", "1", "on"):
        return "true"
    if s in ("no", "false", "0", "off"):
        return "false"
    return "auto"


def _as_stop(value: object) -> tuple[str, ...]:
    """Coerce a ``stop`` field (scalar or list) into a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _as_table(value: object) -> dict[str, object]:
    """A TOML table → dict, or {} for anything else (incl. missing)."""
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class Provider:
    """A backend config. Every field is optional — ``None`` means
    "unspecified", so partial providers (and ``[defaults]``) compose via
    :meth:`merged`. Hard defaults (adapter, endpoint, …) are applied by
    the bridge at use time, NOT here, so a ``Provider`` only ever holds
    what was actually configured.
    """

    model: Optional[str] = None
    adapter: Optional[str] = None  # "openai-compatible" | "claude_code"
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    enable_thinking: Optional[str] = None  # auto|true|false
    stop: tuple[str, ...] = ()
    # Forward-compat passthrough for unknown [providers.*] keys.
    extra: Mapping[str, "TomlValue"] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, raw: Mapping[str, object]) -> Self:
        """Build from a raw TOML table. Known keys are coerced into typed
        fields; unrecognised keys are preserved verbatim in :attr:`extra`.
        Subclasses that add fields override this to claim their own keys."""
        extra = {k: v for k, v in raw.items() if k not in _PROVIDER_FIELDS}
        return cls(
            model=_opt_str(raw.get("model")),
            adapter=_opt_str(raw.get("adapter")),
            endpoint=_opt_str(raw.get("endpoint")),
            api_key=_opt_str(raw.get("api_key")),
            api_key_env=_opt_str(raw.get("api_key_env")),
            max_tokens=_opt_int(raw.get("max_tokens")),
            temperature=_opt_float(raw.get("temperature")),
            enable_thinking=_normalise_thinking(raw.get("enable_thinking")),
            stop=_as_stop(raw.get("stop")),
            extra=cast("Mapping[str, TomlValue]", extra),
        )

    def merged(self, base: Self) -> Self:
        """Overlay ``self`` onto ``base`` (``self`` wins). Generic over
        the concrete type — works field-by-field for any subclass, so a
        richer provider's extra fields compose too. Mappings merge,
        tuples fall back when empty, scalars pick non-``None``."""
        out: dict[str, object] = {}
        for f in dataclasses.fields(self):
            out[f.name] = self._merge_field(
                getattr(self, f.name), getattr(base, f.name)
            )
        return type(self)(**out)  # type: ignore[arg-type]  # preserves subclass type

    @staticmethod
    def _merge_field(self_val: object, base_val: object) -> object:
        """Default per-field merge. Override in a subclass for fields that
        need bespoke composition."""
        if isinstance(self_val, Mapping) and isinstance(base_val, Mapping):
            return {**base_val, **self_val}
        if isinstance(self_val, tuple):
            return self_val if self_val else base_val
        return self_val if self_val is not None else base_val


def _opt_str(v: object) -> Optional[str]:
    return None if v is None else str(v)


def _opt_int(v: object) -> Optional[int]:
    return None if v is None else int(v)  # type: ignore[call-overload]


def _opt_float(v: object) -> Optional[float]:
    return None if v is None else float(v)  # type: ignore[arg-type]


P = TypeVar("P", bound=Provider)


@dataclass(frozen=True)
class Config(Generic[P]):
    """Parsed config. Parameterised by the provider type so subclasses can
    carry a richer ``Provider`` (and add their own typed sections by
    subclassing this and overriding :meth:`ConfigParser.build`)."""

    defaults: P
    providers: Mapping[str, P]
    profiles: Mapping[str, Mapping[str, str]]  # name → (opaque key → provider)

    def resolve(self, name: str) -> P:
        """The named provider with ``[defaults]`` overlaid beneath it."""
        return self.providers[name].merged(self.defaults)

    def select(self, profile: str, key: str) -> Optional[str]:
        """Provider name for ``key`` in ``profile`` (key opaque), or None."""
        return self.profiles.get(profile, {}).get(key)


class ConfigParser(Generic[P]):
    """Parses the three known tables into a :class:`Config`. Extend by:

      * setting :attr:`provider_cls` to a :class:`Provider` subclass, and/or
      * overriding :meth:`build` to return a :class:`Config` subclass with
        extra typed sections read from ``data``.

    The base parser (``ConfigParser()``) yields ``Config[Provider]``.
    """

    provider_cls: type[P] = cast("type[P]", Provider)

    def provider(self, raw: Mapping[str, object]) -> P:
        return self.provider_cls.from_toml(raw)

    def parse(self, data: Mapping[str, object]) -> Config[P]:
        defaults = self.provider(_as_table(data.get("defaults")))
        providers = {
            name: self.provider(_as_table(v))
            for name, v in _as_table(data.get("providers")).items()
        }
        profiles = {
            name: {str(k): str(pv) for k, pv in _as_table(v).items()}
            for name, v in _as_table(data.get("profiles")).items()
        }
        return self.build(defaults, providers, profiles, data)

    def build(
        self,
        defaults: P,
        providers: Mapping[str, P],
        profiles: Mapping[str, Mapping[str, str]],
        data: Mapping[str, object],
    ) -> Config[P]:
        """Assemble the final Config. Override to return a subclass that
        reads extra tables from ``data`` (e.g. a typed ``widgets`` field)."""
        return Config(defaults=defaults, providers=providers, profiles=profiles)


def load(path: str) -> Config[Provider]:
    """Convenience: parse a TOML file with the base parser."""
    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)
    return ConfigParser().parse(data)


__all__ = [
    "TomlValue",
    "Provider",
    "Config",
    "ConfigParser",
    "load",
]

"""Registry used to resolve EOS plug-ins by stable names."""

from __future__ import annotations

from collections.abc import Iterator

from .base import EOSPlugin


class EOSRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, EOSPlugin] = {}

    def register(self, plugin: EOSPlugin, *, replace: bool = False) -> EOSPlugin:
        if not isinstance(plugin, EOSPlugin):
            raise TypeError("registry entries must be EOSPlugin instances")
        if plugin.key in self._plugins and not replace:
            raise KeyError(f"EOS plug-in {plugin.key!r} is already registered")
        self._plugins[plugin.key] = plugin
        return plugin

    def get(self, key: str) -> EOSPlugin:
        try:
            return self._plugins[key]
        except KeyError as error:
            available = ", ".join(self.keys()) or "none"
            raise KeyError(
                f"unknown EOS plug-in {key!r}; available: {available}"
            ) from error

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def __iter__(self) -> Iterator[EOSPlugin]:
        for key in self.keys():
            yield self._plugins[key]


EOS_MODELS = EOSRegistry()


def register_eos(plugin: EOSPlugin, *, replace: bool = False) -> EOSPlugin:
    return EOS_MODELS.register(plugin, replace=replace)


def get_eos(key: str) -> EOSPlugin:
    return EOS_MODELS.get(key)


def available_eos() -> tuple[str, ...]:
    return EOS_MODELS.keys()

"""Helpers for naming sensors and capturing figures for reports."""
from __future__ import annotations

from typing import Any, List, MutableSequence, Optional, Sequence, Tuple


def friendly_sensor_name(sensor: str, mapping: Optional[Any]) -> str:
    if isinstance(mapping, dict):
        return mapping.get(sensor, sensor)
    return sensor


def ref_display(ref_idx: int, ref_name: Optional[str]) -> str:
    if isinstance(ref_name, str) and ref_name:
        return ref_name
    return f'T{ref_idx}'


def sensor_context(sensor: str, mapping: Optional[Any], ref_idx: int, ref_name: Optional[str]) -> Tuple[str, str]:
    return friendly_sensor_name(sensor, mapping), ref_display(ref_idx, ref_name)


def ensure_registry(registry: Optional[MutableSequence[Tuple[str, str, Any]]]) -> MutableSequence[Tuple[str, str, Any]]:
    return registry if registry is not None else []


def capture_figure(registry: MutableSequence[Tuple[str, str, Any]], category: str, title: str, fig: Any) -> None:
    registry.append((category, title, fig))


def print_errors(errors: Sequence[Tuple[str, str]]) -> None:
    for name, message in errors:
        print(f"[warn] {name}: {message}")

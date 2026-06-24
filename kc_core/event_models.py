"""Shared event and command objects for KC Agent.

This module is intentionally dependency-light so decoders, receivers, the
state-machine, and tests can import the same objects without importing UI,
network, or Windows-only packages.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

JsonDict = dict[str, Any]

SCHEMA_VERSION = "kc.agent.battle_event.v1"


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class AgentEvent:
    """Normalized event consumed by the KC Agent state machine."""

    type: str
    payload: JsonDict = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: new_id("evt"))
    ts_ms: int = field(default_factory=now_ms)
    source: str = "unknown"
    correlation: JsonDict = field(default_factory=dict)

    def to_packet(self) -> JsonDict:
        return asdict(self)


@dataclass
class SceneRequirement:
    """A scene/target condition that must be observed before proceeding."""

    scene: str
    required_targets: list[str] = field(default_factory=list)
    timeout_ms: int = 8000

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class WaitSpec:
    wait_id: str
    expected_scene: str
    required_targets: list[str] = field(default_factory=list)
    trigger_event_id: Optional[str] = None
    timeout_ms: int = 8000
    stable_frames: int = 2
    reason: str = ""
    correlation: JsonDict = field(default_factory=dict)


@dataclass
class SceneObservation:
    """Latest screen-side scene/target observation produced by scene identify."""

    scene: str
    targets: JsonDict = field(default_factory=dict)
    wait_id: Optional[str] = None
    source_event_id: Optional[str] = None
    correlation: JsonDict = field(default_factory=dict)

    def target(self, name: str) -> Optional[JsonDict]:
        target = self.targets.get(name)
        return target if isinstance(target, dict) else None


@dataclass
class Command:
    command_id: str
    command: str
    target: str
    reason: str
    trigger_event_id: Optional[str] = None
    wait_id: Optional[str] = None
    requires_scene: Optional[str] = None
    safety: JsonDict = field(default_factory=dict)
    correlation: JsonDict = field(default_factory=dict)
    created_at_ms: int = field(default_factory=now_ms)
    valid_ms: int = 5000
    target_observation: JsonDict = field(default_factory=dict)

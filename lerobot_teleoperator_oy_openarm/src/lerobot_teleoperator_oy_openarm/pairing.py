"""Process-local pairing state for OY OpenArm LeRobot plugins."""

from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .oy_openarm_leader import OyOpenArmLeader

_lock = threading.RLock()
_active_leader_ref: weakref.ReferenceType[OyOpenArmLeader] | None = None


def register_active_leader(leader: OyOpenArmLeader) -> None:
    """Register the one connected OY OpenArm leader in this process."""

    global _active_leader_ref
    with _lock:
        active = _active_leader_ref() if _active_leader_ref is not None else None
        if active is not None and active is not leader and active.is_connected:
            raise RuntimeError("另一个 OyOpenArmLeader 已在当前进程中连接。")
        _active_leader_ref = weakref.ref(leader)


def unregister_active_leader(leader: OyOpenArmLeader) -> None:
    """Clear the process-local leader registration if it belongs to ``leader``."""

    global _active_leader_ref
    with _lock:
        active = _active_leader_ref() if _active_leader_ref is not None else None
        if active is leader or active is None:
            _active_leader_ref = None


def get_active_leader() -> OyOpenArmLeader | None:
    """Return the currently connected OY OpenArm leader, if one exists."""

    with _lock:
        active = _active_leader_ref() if _active_leader_ref is not None else None
        if active is None or not active.is_connected:
            return None
        return active

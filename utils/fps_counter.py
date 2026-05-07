"""
utils/fps_counter.py — Rolling-average FPS counter.

Uses a deque of per-frame timestamps so the reading stays smooth even
when individual frames spike.  Thread-safe (GIL is sufficient here).
"""

from collections import deque
import time
from typing import Deque


class FPSCounter:
    """Compute frames-per-second from a sliding window of frame timestamps."""

    def __init__(self, window: int = 30) -> None:
        """
        Args:
            window: Number of most-recent frames to include in the average.
        """
        self._timestamps: Deque[float] = deque(maxlen=window)

    # ── Public API ────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """Record the arrival of a new frame.  Call once per captured frame."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Return the smoothed FPS value (0.0 if fewer than 2 frames recorded)."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def fps_str(self) -> str:
        """Formatted string suitable for on-screen display, e.g. '59.8 FPS'."""
        return f"{self.fps:.1f} FPS"

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        self._timestamps.clear()

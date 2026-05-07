"""
core/mouse_controller.py — Virtual mouse controller driven by hand gestures.

Gesture → action
-----------------
  POINT     (1 finger)  →  move cursor
  OPEN_HAND (5 fingers) →  left click  (after stability + cooldown guards)
  all others            →  idle

Movement pipeline
-----------------
  Centroid (full-frame px)
    → clip to active ROI region
    → normalise [0, 1]
    → scale to screen
    → EMA smooth (adaptive α)
    → dead-zone filter
    → clamp to screen bounds
    → pyautogui.moveTo

EMA smoothing
-------------
  α is adaptive:
    • Normal frame   → smoothing_alpha  (default 0.20)
    • Large jump     → emergency_alpha  (default 0.04)  — absorbs teleports

Dead-zone
---------
  Cursor is only moved when the Euclidean screen-space delta from the
  previous committed position exceeds dead_zone_px.  Eliminates the
  micro-jitter visible when the hand is stationary.

Click guards
------------
  Fire only when:
    1.  OPEN_HAND streak ≥ click_stable_frames
    2.  time since last click ≥ click_cooldown_s
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE    = 0.0
    _PAG = True
except ImportError:
    logger.warning("pyautogui not found — mouse control disabled.")
    _PAG = False

_GESTURE_MOVE  = "POINT"
_GESTURE_CLICK = "OPEN_HAND"


class MouseController:
    """Per-frame virtual mouse driver.

    Args:
        cfg: :class:`config.MouseConfig` instance; if omitted the global
             ``MOUSE`` singleton from ``config.py`` is used.
    """

    def __init__(self, cfg=None) -> None:
        if cfg is None:
            from config import MOUSE
            cfg = MOUSE
        self._cfg     = cfg
        self._enabled = _PAG and cfg.enabled

        self._screen_w: Optional[int]   = None
        self._screen_h: Optional[int]   = None

        # EMA-smoothed cursor position
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None

        # Last committed cursor position (for dead-zone comparison)
        self._committed_x: Optional[float] = None
        self._committed_y: Optional[float] = None

        self._click_streak:    int   = 0
        self._last_click_time: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        gesture:     Optional[str],
        centroid:    Optional[Tuple[int, int]],
        frame_shape: Tuple[int, int],
    ) -> str:
        """Drive the OS cursor from one frame's detection result.

        Args:
            gesture:     Stable gesture label or None.
            centroid:    Hand centroid in full-frame pixels or None.
            frame_shape: (height, width) of the camera frame.

        Returns:
            Action string: "MOVING" | "CLICKED" | "COOLDOWN" | "IDLE"
        """
        if not self._enabled or gesture is None or centroid is None:
            self._click_streak = 0
            return "IDLE"

        self._ensure_screen_size()

        if gesture == _GESTURE_MOVE:
            self._click_streak = 0
            self._move(centroid, frame_shape)
            return "MOVING"

        if gesture == _GESTURE_CLICK:
            self._click_streak += 1
            action = self._try_click()
            self._move(centroid, frame_shape)
            return action

        self._click_streak = 0
        return "IDLE"

    def reset(self) -> None:
        """Clear smoothing state (e.g. on scene cuts)."""
        self._smooth_x     = None
        self._smooth_y     = None
        self._committed_x  = None
        self._committed_y  = None
        self._click_streak = 0

    # ── Screen size ───────────────────────────────────────────────────────────

    def _ensure_screen_size(self) -> None:
        if self._screen_w is None:
            self._screen_w, self._screen_h = pyautogui.size()
            logger.info("Screen: %d×%d", self._screen_w, self._screen_h)

    # ── Coordinate mapping ────────────────────────────────────────────────────

    def _map_to_screen(
        self,
        centroid:    Tuple[int, int],
        frame_shape: Tuple[int, int],
    ) -> Tuple[float, float]:
        fh, fw = frame_shape
        cfg    = self._cfg

        margin_x = fw * cfg.map_margin_frac
        margin_y = fh * cfg.map_margin_frac
        x_lo, x_hi = margin_x, fw - margin_x
        y_lo, y_hi = margin_y, fh - margin_y

        cx, cy = centroid
        norm_x = max(0.0, min(1.0, (cx - x_lo) / (x_hi - x_lo)))
        norm_y = max(0.0, min(1.0, (cy - y_lo) / (y_hi - y_lo)))

        return norm_x * self._screen_w, norm_y * self._screen_h

    # ── Cursor movement ───────────────────────────────────────────────────────

    def _move(
        self,
        centroid:    Tuple[int, int],
        frame_shape: Tuple[int, int],
    ) -> None:
        """EMA smooth → dead-zone check → clamp → moveTo."""
        cfg          = self._cfg
        raw_x, raw_y = self._map_to_screen(centroid, frame_shape)

        cold_start = self._smooth_x is None

        if cold_start:
            # Jump directly to first position; no smoothing, no dead-zone
            self._smooth_x = raw_x
            self._smooth_y = raw_y
        else:
            dx   = raw_x - self._smooth_x
            dy   = raw_y - self._smooth_y
            dist = (dx * dx + dy * dy) ** 0.5

            alpha = cfg.emergency_alpha if dist > cfg.max_jump_px else cfg.smoothing_alpha
            self._smooth_x += alpha * dx
            self._smooth_y += alpha * dy

        # Dead-zone: skip moveTo if delta from last committed position is tiny.
        # Cold-start bypasses the dead-zone so the cursor jumps immediately
        # to the first detected position.
        if not cold_start:
            ddx = self._smooth_x - (self._committed_x or 0.0)
            ddy = self._smooth_y - (self._committed_y or 0.0)
            if (ddx * ddx + ddy * ddy) ** 0.5 < cfg.dead_zone_px:
                return

        m  = cfg.screen_margin_px
        sx = int(max(m, min(self._screen_w - m, self._smooth_x)))
        sy = int(max(m, min(self._screen_h - m, self._smooth_y)))

        pyautogui.moveTo(sx, sy)
        self._committed_x = self._smooth_x
        self._committed_y = self._smooth_y

    # ── Click ─────────────────────────────────────────────────────────────────

    def _try_click(self) -> str:
        cfg = self._cfg
        now = time.monotonic()

        if self._click_streak < cfg.click_stable_frames:
            return "COOLDOWN"
        if (now - self._last_click_time) < cfg.click_cooldown_s:
            return "COOLDOWN"

        pyautogui.click()
        self._last_click_time = now
        self._click_streak    = 0
        logger.debug("Left click fired")
        return "CLICKED"
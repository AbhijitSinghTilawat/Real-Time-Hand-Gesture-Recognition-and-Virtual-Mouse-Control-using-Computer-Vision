"""
core/camera.py — Thin wrapper around cv2.VideoCapture.

Responsibilities
----------------
* Open / release the capture device.
* Apply resolution & FPS hints to the driver.
* Expose a clean iterator so callers never touch raw VideoCapture.

Usage
-----
    with Camera() as cam:
        for frame in cam:
            process(frame)
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Generator, Optional, Type

import cv2
import numpy as np

from config import CAMERA, CameraConfig

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the capture device cannot be opened or a frame read fails."""


class Camera:
    """Managed webcam capture device.

    Supports use as a context manager and as an iterator:

        with Camera() as cam:
            for frame in cam:          # yields BGR ndarray
                ...

    Args:
        cfg: Camera configuration dataclass (defaults to global ``CAMERA``).
    """

    def __init__(self, cfg: CameraConfig = CAMERA) -> None:
        self._cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> "Camera":
        """Open the capture device and apply resolution / FPS settings."""
        self._cap = cv2.VideoCapture(self._cfg.device_index)
        if not self._cap.isOpened():
            raise CameraError(
                f"Cannot open camera at index {self._cfg.device_index}. "
                "Check that no other application holds the device."
            )

        # Apply preferred settings (hints only — the driver may ignore them)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._cfg.target_fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info("Camera opened: %dx%d @ %.0f fps", actual_w, actual_h, actual_fps)
        return self

    def release(self) -> None:
        """Release the underlying VideoCapture resource."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None
            logger.info("Camera released.")

    # ── Context-manager protocol ──────────────────────────────────────────────

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val:  Optional[BaseException],
        exc_tb:   Optional[TracebackType],
    ) -> None:
        self.release()

    # ── Iterator protocol ─────────────────────────────────────────────────────

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Yield frames indefinitely until the capture is released or read fails."""
        if self._cap is None or not self._cap.isOpened():
            raise CameraError("Camera is not open. Call open() or use 'with' statement.")

        while True:
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("Frame grab failed — stream may have ended.")
                break

            if self._cfg.flip_horizontal:
                frame = cv2.flip(frame, 1)

            yield frame

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def resolution(self) -> tuple[int, int]:
        """Return (width, height) as reported by the driver."""
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

"""
core/pipeline.py — Main orchestration pipeline.

Frame loop
----------
  Camera
    → GestureDetector.process()
    → MouseController.update()
    → Overlay.draw_debug()   (contour / hull / tips / centroid / stats)
    → Overlay.draw_hud()     (info panel + gesture label)
    → cv2.imshow (main)
    → cv2.imshow (mask)   [optional]
    → cv2.imshow (edges)  [optional]
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2

from config import PIPELINE
from core.camera           import Camera, CameraError
from core.overlay          import Overlay
from core.detector         import GestureDetector
from core.mouse_controller import MouseController
from utils.fps_counter     import FPSCounter

logger = logging.getLogger(__name__)

_WIN_MAIN  = "Gesture Recognition — press Q to quit"
_WIN_MASK  = "Mask"
_WIN_EDGES = "Edges"


class Pipeline:
    """Orchestrates capture → detect → mouse → render → display.

    Args:
        camera:      Open :class:`Camera` instance.
        overlay:     :class:`Overlay` renderer.
        fps_counter: :class:`FPSCounter` instance.
        detector:    Optional :class:`GestureDetector`; defaults created if None.
        mouse:       Optional :class:`MouseController`; defaults created if None.
    """

    def __init__(
        self,
        camera:      Camera,
        overlay:     Overlay,
        fps_counter: FPSCounter,
        detector:    Optional[GestureDetector]  = None,
        mouse:       Optional[MouseController]  = None,
    ) -> None:
        self._camera      = camera
        self._overlay     = overlay
        self._fps_counter = fps_counter
        self._detector    = detector or GestureDetector()
        self._mouse       = mouse    or MouseController()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Block until Q / ESC or stream ends."""
        self._open_windows()
        logger.info("Pipeline running — press Q to quit.")

        try:
            for frame in self._camera:
                self._fps_counter.tick()
                self._process_frame(frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    logger.info("Quit key received.")
                    break
        except CameraError as exc:
            logger.error("Camera error: %s", exc)
        finally:
            cv2.destroyAllWindows()
            logger.info("Pipeline stopped.")

    # ── Frame processing ───────────────────────────────────────────────────────

    def _process_frame(self, frame: "np.ndarray") -> None:  # noqa: F821
        result = self._detector.process(frame)

        mouse_status = "IDLE"
        gesture:  Optional[str] = None
        fingers:  Optional[int] = None

        if result is not None:
            gesture  = result["gesture"]
            fingers  = result["fingers"]
            centroid = result["centroid"]

            # Virtual mouse (non-blocking; returns status string)
            mouse_status = self._mouse.update(
                gesture, centroid, frame.shape[:2]
            )

            # Debug overlay — contour, hull, tips, centroid, stats panel
            frame = self._overlay.draw_debug(
                frame,
                contour       = result["contour"],
                hull_points   = result["hull_points"],
                fingertips    = result["fingertips"],
                centroid      = centroid,
                contour_area  = result["contour_area"],
                defects_count = result["defects_count"],
                gesture       = gesture,
                fingers       = fingers,
            )

            # Side windows
            if PIPELINE.show_debug_windows:
                cv2.imshow(_WIN_MASK,  result["mask"])
                cv2.imshow(_WIN_EDGES, result["edges"])

        # HUD (FPS, status, gesture label)
        frame = self._overlay.draw_hud(
            frame,
            fps     = self._fps_counter.fps_str,
            status  = mouse_status,
            gesture = gesture,
            fingers = fingers,
        )

        cv2.imshow(_WIN_MAIN, frame)

    # ── Window setup ──────────────────────────────────────────────────────────

    def _open_windows(self) -> None:
        w, h = self._camera.resolution
        cv2.namedWindow(_WIN_MAIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WIN_MAIN, w, h)

        if PIPELINE.show_debug_windows:
            for title in (_WIN_MASK, _WIN_EDGES):
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(title, w // 2, h // 2)
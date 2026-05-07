"""
main.py — Entry point for the hand gesture recognition system.

Run:
    python main.py

Keyboard shortcuts (in the video window):
    Q / ESC  →  quit
"""

import logging
import sys

from config import CAMERA, PIPELINE
from core   import Camera, CameraError, Overlay, Pipeline
from utils  import FPSCounter


# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    """Bootstrap all components and start the capture loop.

    Returns:
        Exit code (0 = clean exit, 1 = fatal error).
    """
    fps_counter = FPSCounter(window=PIPELINE.fps_smoothing_window)
    overlay     = Overlay()

    try:
        with Camera(CAMERA) as camera:
            pipeline = Pipeline(
                camera=camera,
                overlay=overlay,
                fps_counter=fps_counter,
            )
            pipeline.run()

    except CameraError as exc:
        logger.error("Fatal camera error: %s", exc)
        return 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

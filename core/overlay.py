"""
core/overlay.py — HUD renderer + debug drawing.

Public methods
--------------
  draw_hud(frame, ...)     — info panel (FPS, status, fingers, gesture)
                             + large gesture label at bottom
  draw_debug(frame, result)— contour outline, convex hull, fingertip circles,
                             centroid dot, and a right-side stats panel
  draw_crosshair(frame)    — subtle centre crosshair

All methods mutate *frame* in-place and return it.  The pipeline never
calls cv2.draw* directly — every visual element lives in this module.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np

from config import OVERLAY, PIPELINE, Palette, OverlayConfig


class Overlay:
    """Create once; call per-frame."""

    def __init__(self, cfg: OverlayConfig = OVERLAY) -> None:
        self._cfg = cfg

    # ── Public API ─────────────────────────────────────────────────────────────

    def draw_hud(
        self,
        frame:   np.ndarray,
        fps:     str = "",
        status:  str = "READY",
        gesture: Optional[str] = None,
        fingers: Optional[int] = None,
    ) -> np.ndarray:
        """Top-left info panel + bottom-centre gesture label."""
        if not PIPELINE.show_hud:
            return frame
        frame = self._info_panel(frame, fps, status, gesture, fingers)
        if gesture and gesture not in ("FIST", "UNKNOWN"):
            frame = self._gesture_label(frame, gesture)
        return frame

    def draw_debug(
        self,
        frame:         np.ndarray,
        contour:       np.ndarray,
        hull_points:   np.ndarray,
        fingertips:    List[Tuple[int, int]],
        centroid:      Tuple[int, int],
        contour_area:  float,
        defects_count: int,
        gesture:       str,
        fingers:       int,
    ) -> np.ndarray:
        """Draw all debug visual elements on *frame*.

        Elements
        --------
        • Contour outline  — bright green, 2 px
        • Convex hull      — gold, 1 px dashed (drawn as thin lines)
        • Fingertip circles— orange-red filled dots
        • Centroid dot     — cyan cross + filled circle
        • Right-side stats panel (area, defects, raw gesture)
        """
        if not PIPELINE.draw_debug_overlay:
            return frame

        cfg = self._cfg

        # Contour
        cv2.drawContours(frame, [contour], -1, Palette.CONTOUR_CLR,
                         cfg.contour_thickness)

        # Convex hull
        cv2.drawContours(frame, [hull_points], -1, Palette.HULL_CLR,
                         cfg.hull_thickness)

        # Fingertip circles
        for (tx, ty) in fingertips:
            cv2.circle(frame, (tx, ty), cfg.fingertip_radius,
                       Palette.FINGERTIP_CLR, -1)
            cv2.circle(frame, (tx, ty), cfg.fingertip_radius + 2,
                       Palette.WHITE, 1)

        # Centroid
        cx, cy = centroid
        r = cfg.centroid_radius
        cv2.circle(frame, (cx, cy), r, Palette.CENTROID_CLR, -1)
        cv2.line(frame, (cx - r - 4, cy), (cx + r + 4, cy),
                 Palette.CENTROID_CLR, 1)
        cv2.line(frame, (cx, cy - r - 4), (cx, cy + r + 4),
                 Palette.CENTROID_CLR, 1)

        # Stats panel (bottom-right)
        frame = self._stats_panel(
            frame, contour_area=contour_area,
            defects_count=defects_count,
            gesture=gesture, fingers=fingers,
        )
        return frame

    def draw_crosshair(self, frame: np.ndarray) -> np.ndarray:
        h, w   = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size, gap = 14, 5
        for dx, dy, ex, ey in [
            (-size, 0, -gap, 0), (gap, 0, size, 0),
            (0, -size, 0, -gap), (0, gap, 0, size),
        ]:
            cv2.line(frame, (cx+dx, cy+dy), (cx+ex, cy+ey),
                     Palette.ACCENT_DIM, 1)
        return frame

    # ── Info panel (top-left) ──────────────────────────────────────────────────

    def _info_panel(
        self,
        frame:   np.ndarray,
        fps:     str,
        status:  str,
        gesture: Optional[str],
        fingers: Optional[int],
    ) -> np.ndarray:
        cfg = self._cfg
        pad = cfg.panel_padding
        ff  = cfg.font_face

        lines = [
            (fps,                                          cfg.font_scale_lg, cfg.thickness),
            (f"STATUS   {status}",                        cfg.font_scale_sm, 1),
            (f"FINGERS  {fingers if fingers is not None else chr(8212)}",
                                                           cfg.font_scale_sm, 1),
            (f"GESTURE  {gesture or chr(8212)}",          cfg.font_scale_sm, 1),
        ]

        sizes   = [cv2.getTextSize(t, ff, s, th)[0] for t, s, th in lines]
        panel_w = max(w for w, _ in sizes) + pad * 2
        panel_h = sum(h for _, h in sizes) + pad * (len(lines) + 1)
        x0, y0  = 10, 10

        frame = _alpha_rect(frame, (x0, y0), (x0 + panel_w, y0 + panel_h),
                            Palette.OVERLAY_BG, 0.68)
        cv2.line(frame, (x0, y0), (x0 + panel_w, y0), Palette.ACCENT, 2)

        colours = [
            Palette.ACCENT,
            _status_colour(status),
            Palette.ACCENT_WARM,
            Palette.ACCENT_DIM,
        ]

        y_cur = y0 + pad
        for (text, scale, thick), (_, th), colour in zip(lines, sizes, colours):
            y_cur += th
            cv2.putText(frame, text, (x0 + pad, y_cur),
                        ff, scale, colour, thick, cv2.LINE_AA)
            y_cur += pad

        return frame

    # ── Gesture label (bottom-centre) ─────────────────────────────────────────

    def _gesture_label(self, frame: np.ndarray, gesture: str) -> np.ndarray:
        h, w  = frame.shape[:2]
        label = gesture.upper()
        cfg   = self._cfg

        (lw, lh), _ = cv2.getTextSize(label, cfg.font_face, 1.4, cfg.thickness + 1)
        x = (w - lw) // 2
        y = h - 36

        cv2.putText(frame, label, (x + 2, y + 2), cfg.font_face, 1.4,
                    Palette.BLACK, cfg.thickness + 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x, y), cfg.font_face, 1.4,
                    Palette.ACCENT, cfg.thickness + 1, cv2.LINE_AA)
        return frame

    # ── Stats panel (bottom-right) ────────────────────────────────────────────

    def _stats_panel(
        self,
        frame:         np.ndarray,
        contour_area:  float,
        defects_count: int,
        gesture:       str,
        fingers:       int,
    ) -> np.ndarray:
        """Semi-transparent panel in the bottom-right corner."""
        cfg = self._cfg
        pad = cfg.panel_padding
        ff  = cfg.font_face
        h_f, w_f = frame.shape[:2]

        lines = [
            f"AREA     {contour_area:,.0f} px",
            f"DEFECTS  {defects_count}",
            f"RAW G    {gesture}",
            f"RAW F    {fingers}",
        ]

        sizes   = [cv2.getTextSize(t, ff, cfg.font_scale_sm, 1)[0] for t in lines]
        panel_w = max(w for w, _ in sizes) + pad * 2
        panel_h = sum(h for _, h in sizes) + pad * (len(lines) + 1)

        x0 = w_f - panel_w - 10
        y0 = h_f - panel_h - 10

        frame = _alpha_rect(frame, (x0, y0), (x0 + panel_w, y0 + panel_h),
                            Palette.OVERLAY_BG, 0.65)
        cv2.line(frame, (x0, y0), (x0 + panel_w, y0), Palette.ACCENT_DIM, 1)

        y_cur = y0 + pad
        for text, (_, th) in zip(lines, sizes):
            y_cur += th
            cv2.putText(frame, text, (x0 + pad, y_cur),
                        ff, cfg.font_scale_sm, Palette.ACCENT_DIM, 1,
                        cv2.LINE_AA)
            y_cur += pad

        return frame


# ── Helpers ────────────────────────────────────────────────────────────────────

def _alpha_rect(
    frame: np.ndarray,
    pt1:   tuple,
    pt2:   tuple,
    colour: tuple,
    alpha:  float,
) -> np.ndarray:
    ov = frame.copy()
    cv2.rectangle(ov, pt1, pt2, colour, -1)
    cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0, frame)
    return frame


def _status_colour(status: str) -> tuple:
    if status == "CLICKED":
        return Palette.SUCCESS
    if status in ("COOLDOWN", "MOVING"):
        return Palette.ACCENT_WARM
    return Palette.ACCENT_DIM
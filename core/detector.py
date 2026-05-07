"""
core/detector.py — Real-time hand gesture detector (v4).

Key improvements over v3
-------------------------
  CLAHE          Adaptive histogram equalisation on LAB L-channel before
                 segmentation.  Normalises local contrast under overexposed
                 and dim-room conditions without relaxing the chromatic A/B
                 skin window — the single most impactful fix for bright-light
                 instability.

  Tighter thresholds
                 LAB A [133–173] and B [130–173] reduce warm-object false
                 positives.  HSV H max 22, S min 35, V min 60 similarly tightened.

  Contour smoothing
                 Largest contour is approximated with approxPolyDP
                 (ε = 0.2 % of arc length) before convexity analysis.
                 Smoother contours produce cleaner hull vertices and fewer
                 spurious shallow defects.

  Higher area threshold
                 min_contour_area raised to 10 000 px² (was 6 000).

  POINT (1-finger) special-case  ← primary fix for one-finger failure
                 When zero valid defects are found (which includes ALL
                 single-finger poses since a lone finger creates no inter-
                 finger gap), the shape is tested:
                   aspect (h/w) > point_aspect_min  AND
                   solidity ∈ (point_solidity_lo, point_solidity_hi)
                 → classified as POINT (1 finger) instead of FIST.
                 This is the definitive fix; the defect path simply cannot
                 detect a single raised finger.

  Richer return dict
                 Added: fingertips, defects_count, contour_area
                 (all consumed by pipeline for debug drawing and overlay).

  Increased temporal smoothing
                 smoothing_window 10→15, debounce_frames 5→7.

Return contract (None when no hand detected):
    {
        "mask"         : np.ndarray  (H×W, uint8)     binary skin mask
        "edges"        : np.ndarray  (H×W, uint8)     Canny silhouette
        "contour"      : np.ndarray  (N×1×2, int32)   smoothed hand contour
        "hull_points"  : np.ndarray  (M×1×2, int32)   convex hull vertices
        "fingertips"   : list[(x,y)]                   validated fingertip coords
        "centroid"     : (int, int)                    (cx, cy) full-frame px
        "fingers"      : int         [0–5]             STABLE (smoothed + debounced)
        "gesture"      : str         FIST|POINT|PEACE|THREE|FOUR|OPEN_HAND|UNKNOWN
        "raw_fingers"  : int                           per-frame count (debug)
        "contour_area" : float                         px²  (debug / overlay)
        "defects_count": int                           valid defects (debug / overlay)
    }
"""

from __future__ import annotations

import logging
import math
from collections import Counter, deque
from typing import Deque, List, Optional, Tuple, TypedDict

import cv2
import numpy as np

from config import DETECTOR, DetectorConfig

logger = logging.getLogger(__name__)


# ── Return type ────────────────────────────────────────────────────────────────

class DetectionResult(TypedDict):
    mask:          np.ndarray
    edges:         np.ndarray
    contour:       np.ndarray
    hull_points:   np.ndarray
    fingertips:    List[Tuple[int, int]]
    centroid:      Tuple[int, int]
    fingers:       int
    gesture:       str
    raw_fingers:   int
    contour_area:  float
    defects_count: int


_GESTURE_MAP: dict[int, str] = {
    0: "FIST",
    1: "POINT",
    2: "PEACE",
    3: "THREE",
    4: "FOUR",
    5: "OPEN_HAND",
}


# ── Temporal filter ────────────────────────────────────────────────────────────

class TemporalFilter:
    """Rolling-mode smoothing + streak-based debounce for finger counts.

    Debounce tracks the *raw* input value (not the mode) so tie-breaking
    in Counter.most_common cannot silently reset a valid streak.  The mode
    is computed at commit time to smooth over single-frame outliers.
    """

    def __init__(self, window: int, debounce: int, no_hand_reset: int) -> None:
        self._window        = window
        self._debounce      = debounce
        self._no_hand_reset = no_hand_reset
        self._history:       Deque[int]    = deque(maxlen=window)
        self._candidate:     Optional[int] = None
        self._streak:        int           = 0
        self._stable:        Optional[int] = None
        self._no_hand_count: int           = 0

    def update(self, raw: int) -> int:
        self._no_hand_count = 0
        self._history.append(raw)

        if raw == self._candidate:
            self._streak += 1
        else:
            self._candidate = raw
            self._streak    = 1

        if self._stable is None or self._streak >= self._debounce:
            self._stable = Counter(self._history).most_common(1)[0][0]

        return self._stable  # type: ignore[return-value]

    def on_no_hand(self) -> None:
        self._no_hand_count += 1
        if self._no_hand_count >= self._no_hand_reset:
            self.reset()

    def reset(self) -> None:
        self._history.clear()
        self._candidate     = None
        self._streak        = 0
        self._stable        = None
        self._no_hand_count = 0

    @property
    def stable(self) -> Optional[int]:
        return self._stable


# ── Detector ───────────────────────────────────────────────────────────────────

class GestureDetector:
    """Lighting-robust, temporally stable hand gesture detector.

    **Stateful** — do not share across threads.  Create one instance per
    capture loop.

    Usage::

        det = GestureDetector()
        result = det.process(bgr_frame)
        if result:
            print(result["gesture"], result["fingers"])
    """

    def __init__(self, cfg: DetectorConfig = DETECTOR) -> None:
        self._cfg = cfg

        self._close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_close_ksize, cfg.morph_close_ksize)
        )
        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_open_ksize, cfg.morph_open_ksize)
        )

        self._lab_lower = np.array(
            [cfg.lab_l_min, cfg.lab_a_min, cfg.lab_b_min], dtype=np.uint8
        )
        self._lab_upper = np.array(
            [cfg.lab_l_max, cfg.lab_a_max, cfg.lab_b_max], dtype=np.uint8
        )
        self._hsv_lower = np.array(
            [cfg.hsv_h_min, cfg.hsv_s_min, cfg.hsv_v_min], dtype=np.uint8
        )
        self._hsv_upper = np.array(
            [cfg.hsv_h_max, cfg.hsv_s_max, cfg.hsv_v_max], dtype=np.uint8
        )

        # CLAHE object created once; reused every frame (no per-frame alloc)
        self._clahe = cv2.createCLAHE(
            clipLimit    = cfg.clahe_clip_limit,
            tileGridSize = (cfg.clahe_tile_grid, cfg.clahe_tile_grid),
        )

        self._filter = TemporalFilter(
            window        = cfg.smoothing_window,
            debounce      = cfg.debounce_frames,
            no_hand_reset = cfg.no_hand_reset_frames,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, bgr: np.ndarray) -> Optional[DetectionResult]:
        """Run the full pipeline; return a temporally stable result or None."""
        raw = self._raw_detect(bgr)

        if raw is None:
            self._filter.on_no_hand()
            return None

        stable_n = self._filter.update(raw["raw_fingers"])
        gesture  = _GESTURE_MAP.get(stable_n, "UNKNOWN")

        return DetectionResult(
            mask          = raw["mask"],
            edges         = raw["edges"],
            contour       = raw["contour"],
            hull_points   = raw["hull_points"],
            fingertips    = raw["fingertips"],
            centroid      = raw["centroid"],
            fingers       = stable_n,
            gesture       = gesture,
            raw_fingers   = raw["raw_fingers"],
            contour_area  = raw["contour_area"],
            defects_count = raw["defects_count"],
        )

    def reset_filter(self) -> None:
        self._filter.reset()

    # ── Raw per-frame detection ────────────────────────────────────────────────

    def _raw_detect(self, bgr: np.ndarray) -> Optional[dict]:
        h_full, w_full = bgr.shape[:2]

        roi_slices, (ox, oy) = self._roi_bounds(h_full, w_full)
        working = bgr[roi_slices]                   # zero-copy NumPy view

        # Step 1 — Gaussian blur (suppresses chroma noise before conversion)
        blurred = self._preprocess(working)

        # Step 2 — CLAHE contrast normalisation (brightness robustness)
        enhanced = self._apply_clahe(blurred)

        # Step 3 — Dual skin segmentation + morphology
        mask_roi = self._build_skin_mask(enhanced)

        # Step 4 — Contour extraction
        contour_roi = self._largest_contour(mask_roi)
        if contour_roi is None:
            return None

        # Step 5 — Smooth contour (cleaner hull / fewer spurious defects)
        contour_roi = _smooth_contour(contour_roi, self._cfg.approx_epsilon_frac)

        # Step 6 — Translate to full-frame coordinates
        contour = contour_roi + np.array([[[ox, oy]]], dtype=np.int32)

        mask_full = np.zeros((h_full, w_full), dtype=np.uint8)
        mask_full[roi_slices] = mask_roi

        # Step 7 — Finger counting + metadata for debug drawing
        raw_n, hull_pts, tips, n_defects = self._count_fingers(
            contour, (h_full, w_full)
        )
        area  = float(cv2.contourArea(contour))
        edges = self._compute_edges(mask_full, contour)

        return {
            "mask":          mask_full,
            "edges":         edges,
            "contour":       contour,
            "hull_points":   hull_pts,
            "fingertips":    tips,
            "centroid":      _centroid(contour),
            "raw_fingers":   raw_n,
            "contour_area":  area,
            "defects_count": n_defects,
        }

    # ── ROI ────────────────────────────────────────────────────────────────────

    def _roi_bounds(
        self, h: int, w: int
    ) -> Tuple[Tuple[slice, slice], Tuple[int, int]]:
        if not self._cfg.use_roi:
            return (slice(None), slice(None)), (0, 0)
        f  = self._cfg.roi_center_frac
        my = int(h * (1.0 - f) / 2.0)
        mx = int(w * (1.0 - f) / 2.0)
        return (slice(my, h - my), slice(mx, w - mx)), (mx, my)

    # ── Pre-processing ─────────────────────────────────────────────────────────

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        k = self._cfg.preprocess_blur_ksize
        return cv2.GaussianBlur(bgr, (k, k), sigmaX=0) if k > 1 else bgr

    def _apply_clahe(self, bgr: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the LAB L-channel; return contrast-normalised BGR.

        Only the luminance channel is modified — the chromatic A/B skin
        signature is untouched.  This makes the skin window effective under
        both dim and overexposed conditions.
        """
        lab        = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        l_eq       = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    # ── Skin segmentation ──────────────────────────────────────────────────────

    def _build_skin_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Dual LAB + HSV with adaptive fallback merge + morphological clean."""
        lab_mask = cv2.inRange(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB),
            self._lab_lower, self._lab_upper,
        )

        if int(np.count_nonzero(lab_mask)) >= self._cfg.lab_fallback_area_px:
            combined = lab_mask
        else:
            logger.debug("LAB area weak — activating HSV fallback")
            hsv_mask = cv2.inRange(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
                self._hsv_lower, self._hsv_upper,
            )
            combined = cv2.bitwise_or(lab_mask, hsv_mask)

        cfg     = self._cfg
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, self._close_kernel,
                                    iterations=cfg.morph_close_iters)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  self._open_kernel,
                                    iterations=cfg.morph_open_iters)
        return cv2.medianBlur(combined, cfg.median_blur_ksize)

    # ── Contour extraction ─────────────────────────────────────────────────────

    def _largest_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        viable = [c for c in contours
                  if cv2.contourArea(c) >= self._cfg.min_contour_area]
        if not viable:
            return None

        candidate = max(viable, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(candidate))
        if hull_area == 0:
            return None
        if cv2.contourArea(candidate) / hull_area < self._cfg.min_solidity:
            return None

        return candidate

    # ── Finger counting ────────────────────────────────────────────────────────

    def _count_fingers(
        self,
        contour:     np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> Tuple[int, np.ndarray, List[Tuple[int, int]], int]:
        """Return (finger_count, hull_pts, fingertip_pts, valid_defect_count).

        Four-gate defect filter
        -----------------------
        G0  fingertip separation   ≥ min_fingertip_dist_px
        G1  valley depth           ≥ min_defect_depth_px
        G2  wrist exclusion        far-point in upper N % of bbox
        G3  inter-finger angle     ≤ max_defect_angle_deg

        Zero-defect special case (POINT detection)
        ------------------------------------------
        When zero valid defects survive the gates — which is always the case
        for a single raised finger since it creates no inter-finger gap — the
        contour shape is evaluated:

            aspect (h/w) > point_aspect_min
            AND solidity ∈ (point_solidity_lo, point_solidity_hi)
            → POINT (1)   otherwise → FIST (0)
        """
        cfg      = self._cfg
        hull_pts = cv2.convexHull(contour)               # for drawing
        hull_idx = cv2.convexHull(contour, returnPoints=False)

        if hull_idx is None or len(hull_idx) < 3 or len(contour) < 4:
            return self._zero_defect_result(contour, hull_pts)

        try:
            defects = cv2.convexityDefects(contour, hull_idx)
        except cv2.error as exc:
            logger.debug("convexityDefects: %s", exc)
            return self._zero_defect_result(contour, hull_pts)

        if defects is None:
            return self._zero_defect_result(contour, hull_pts)

        _, y_box, _, h_box = cv2.boundingRect(contour)
        wrist_y = y_box + h_box * cfg.defect_bottom_cutoff_pct

        valid    = 0
        tip_set: set               = set()
        tips:    List[Tuple[int, int]] = []

        for row in defects:
            s_idx, e_idx, f_idx, raw_depth = row[0]

            start = contour[s_idx][0]
            end   = contour[e_idx][0]
            far   = contour[f_idx][0]

            # G0 — fingertip separation
            if _dist(start, end) < cfg.min_fingertip_dist_px:
                continue
            # G1 — depth
            if (raw_depth / 256.0) < cfg.min_defect_depth_px:
                continue
            # G2 — wrist exclusion
            if far[1] > wrist_y:
                continue
            # G3 — inter-finger angle
            if _angle_deg(start, far, end) > cfg.max_defect_angle_deg:
                continue

            valid += 1
            for pt in (tuple(start), tuple(end)):
                if pt not in tip_set:
                    tip_set.add(pt)
                    tips.append(pt)  # type: ignore[arg-type]

        if valid == 0:
            return self._zero_defect_result(contour, hull_pts)

        return min(valid + 1, 5), hull_pts, tips, valid

    def _zero_defect_result(
        self,
        contour:  np.ndarray,
        hull_pts: np.ndarray,
    ) -> Tuple[int, np.ndarray, List[Tuple[int, int]], int]:
        """Classify a zero-valid-defect contour as POINT or FIST.

        A single raised finger produces:
          • Tall, narrow bounding rect  →  h/w > point_aspect_min
          • Moderate solidity           →  ∈ (point_solidity_lo, hi)

        A closed fist produces:
          • Compact bounding rect (h ≈ w)
          • High solidity (solid blob)

        The solidity upper bound prevents a partially-open palm from being
        mistaken for a point.
        """
        cfg = self._cfg
        _, _, w, h = cv2.boundingRect(contour)
        aspect = h / w if w > 0 else 0.0

        area      = cv2.contourArea(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity  = area / hull_area if hull_area > 0 else 0.0

        is_point = (
            aspect   > cfg.point_aspect_min   and
            solidity > cfg.point_solidity_lo   and
            solidity < cfg.point_solidity_hi
        )
        n = 1 if is_point else 0
        return n, hull_pts, [], 0

    # ── Edge computation ───────────────────────────────────────────────────────

    def _compute_edges(
        self, mask: np.ndarray, contour: np.ndarray
    ) -> np.ndarray:
        """Canny on binary mask with 12 px boundary-context padding."""
        cfg      = self._cfg
        h_f, w_f = mask.shape[:2]
        edges    = np.zeros_like(mask)
        pad      = 12

        x, y, w, h = cv2.boundingRect(contour)
        x1 = max(x - pad, 0);        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, w_f);  y2 = min(y + h + pad, h_f)

        if x2 > x1 and y2 > y1:
            edges[y1:y2, x1:x2] = cv2.Canny(
                mask[y1:y2, x1:x2],
                cfg.canny_threshold1,
                cfg.canny_threshold2,
            )
        return edges


# ── Module-level helpers ───────────────────────────────────────────────────────

def _smooth_contour(contour: np.ndarray, epsilon_frac: float) -> np.ndarray:
    """Approximate contour with fewer vertices (cleaner hull / defects)."""
    arc    = cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, epsilon_frac * arc, closed=True)
    return approx if len(approx) >= 4 else contour


def _centroid(contour: np.ndarray) -> Tuple[int, int]:
    """Centroid via image moments; fallback to bbox centre."""
    m = cv2.moments(contour)
    if m["m00"] != 0.0:
        return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
    x, y, w, h = cv2.boundingRect(contour)
    return (x + w // 2, y + h // 2)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.linalg.norm(d))


def _angle_deg(
    a: np.ndarray, vertex: np.ndarray, b: np.ndarray
) -> float:
    """Return ∠a–vertex–b in [0°, 180°].  Safe on degenerate input (→ 180°)."""
    va = a.astype(np.float64) - vertex.astype(np.float64)
    vb = b.astype(np.float64) - vertex.astype(np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 180.0
    cos = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos))
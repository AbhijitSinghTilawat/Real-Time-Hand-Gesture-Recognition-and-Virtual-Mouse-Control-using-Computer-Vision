"""
config.py — Central configuration for the gesture recognition pipeline.
All tunable constants live here; no magic numbers scattered in modules.
"""

from dataclasses import dataclass


# ── Colour palette (BGR) ──────────────────────────────────────────────────────
class Palette:
    ACCENT        = (0, 255, 180)   # Neon teal     — primary HUD
    ACCENT_DIM    = (0, 160, 110)   # Dimmed teal   — secondary text
    ACCENT_WARM   = (0, 200, 255)   # Amber         — finger count / moving
    DANGER        = (0, 80,  255)   # Red-orange    — warnings
    SUCCESS       = (100, 220, 100) # Soft green    — click confirmed
    HULL_CLR      = (255, 200, 0)   # Gold          — convex hull
    CONTOUR_CLR   = (0, 255, 80)    # Bright green  — contour outline
    FINGERTIP_CLR = (0, 100, 255)   # Orange-red    — fingertip circles
    CENTROID_CLR  = (255, 255, 0)   # Cyan          — centroid dot
    OVERLAY_BG    = (10,  10,  10)  # Near-black    — panel background
    WHITE         = (255, 255, 255)
    BLACK         = (0,   0,   0)


@dataclass(frozen=True)
class CameraConfig:
    device_index:    int  = 0
    frame_width:     int  = 1280
    frame_height:    int  = 720
    target_fps:      int  = 30
    flip_horizontal: bool = True


@dataclass(frozen=True)
class OverlayConfig:
    font_face:     int   = 0       # cv2.FONT_HERSHEY_SIMPLEX
    font_scale_lg: float = 0.72
    font_scale_md: float = 0.58
    font_scale_sm: float = 0.46
    thickness:     int   = 2
    panel_padding: int   = 12

    # Debug drawing dimensions
    contour_thickness:  int = 2
    hull_thickness:     int = 1
    fingertip_radius:   int = 9
    centroid_radius:    int = 7


@dataclass(frozen=True)
class PipelineConfig:
    fps_smoothing_window: int  = 30
    show_hud:             bool = True
    show_fps:             bool = True
    show_debug_windows:   bool = True   # separate Mask + Edges side windows
    draw_debug_overlay:   bool = True   # contour / hull / tips on main frame


@dataclass(frozen=True)
class DetectorConfig:
    # ── Pre-processing ────────────────────────────────────────────────────────
    preprocess_blur_ksize: int   = 5      # Gaussian kernel, odd; 0/1 = off

    # CLAHE applied to the LAB L-channel before colour thresholding.
    # Normalises local contrast so skin stays detectable under both
    # overexposed (blown-out) and dim-room conditions without relaxing
    # the chromatic A/B skin window.
    clahe_clip_limit: float = 2.0
    clahe_tile_grid:  int   = 8

    # ── LAB skin thresholds ───────────────────────────────────────────────────
    # OpenCV 8-bit LAB:  L = L* × 2.55 ;  A/B = a*/b* + 128
    # Tightened vs v3 — reduces warm-object false positives under bright light.
    #   L  : 20–230  (cap 230 rejects blown-out highlights)
    #   A  : 133–173 (a* +5…+45  reddish-pink skin; v3 was 130–185)
    #   B  : 130–173 (b* +2…+45  warm/yellowish;    v3 was 125–185)
    lab_l_min: int = 20;  lab_l_max: int = 230
    lab_a_min: int = 133; lab_a_max: int = 173
    lab_b_min: int = 130; lab_b_max: int = 173
    lab_fallback_area_px: int = 4_000    # px² below which HSV fallback is OR'd in

    # ── HSV skin thresholds (fallback) ────────────────────────────────────────
    # OpenCV 8-bit HSV: H ∈ [0,179], S/V ∈ [0,255]
    # H max 22 (v3: 25) — fewer orange objects pass
    # S min 35 (v3: 25) — rejects near-achromatic patches
    # V min 60 (v3: 50) — rejects dim regions
    hsv_h_min: int = 0;  hsv_h_max: int = 22
    hsv_s_min: int = 35; hsv_s_max: int = 230
    hsv_v_min: int = 60; hsv_v_max: int = 255

    # ── Morphological cleaning (closing → opening → median) ───────────────────
    morph_close_ksize: int = 9; morph_close_iters: int = 2   # fill holes
    morph_open_ksize:  int = 5; morph_open_iters:  int = 1   # remove blobs
    median_blur_ksize: int = 7                                 # salt & pepper

    # ── ROI ───────────────────────────────────────────────────────────────────
    use_roi:         bool  = True
    roi_center_frac: float = 0.60

    # ── Contour validity ──────────────────────────────────────────────────────
    min_contour_area:    float = 10_000.0  # px² (raised from 6 000)
    min_solidity:        float = 0.50
    approx_epsilon_frac: float = 0.002     # contour smoothing ε = frac × arc

    # ── POINT (1-finger) special-case ─────────────────────────────────────────
    # A single raised finger produces ZERO convexity defects, so the generic
    # defect counter always returns 0 (FIST) for it.  We fix this by checking
    # the bounding-box shape and solidity when defect count is zero.
    #
    # Classify as POINT when:
    #   aspect (h/w) > point_aspect_min          — finger is taller than wide
    #   solidity ∈ (point_solidity_lo, hi)        — not sparse noise, not a fist
    point_aspect_min:  float = 1.6
    point_solidity_lo: float = 0.58
    point_solidity_hi: float = 0.93

    # ── Convexity-defect gates ────────────────────────────────────────────────
    # Gate 0 — minimum start↔end hull-point distance (px)
    min_fingertip_dist_px:    float = 30.0
    # Gate 1 — minimum valley depth in actual pixels (cv2 stores in 256ths)
    min_defect_depth_px:      float = 25.0
    # Gate 2 — wrist exclusion: valley far-point must be in upper N% of bbox
    defect_bottom_cutoff_pct: float = 0.78
    # Gate 3 — maximum angle at valley (°); genuine finger gaps are acute
    max_defect_angle_deg:     float = 80.0

    # ── Canny edges ───────────────────────────────────────────────────────────
    canny_threshold1: int = 30
    canny_threshold2: int = 90

    # ── Temporal filter ───────────────────────────────────────────────────────
    smoothing_window:     int = 15   # rolling-mode window (was 10)
    debounce_frames:      int = 7    # consecutive ticks to commit (was 5)
    no_hand_reset_frames: int = 6


@dataclass(frozen=True)
class MouseConfig:
    enabled: bool = True

    # EMA smoothing: α ∈ (0,1] — higher = faster cursor, more jitter
    smoothing_alpha: float = 0.20
    emergency_alpha: float = 0.04   # applied when jump > max_jump_px
    max_jump_px:     int   = 200

    # Dead-zone: cursor moves only when screen-space delta > this value
    dead_zone_px: int = 6

    # Active mapping region matches the detector's ROI:
    #   margin = (1 - roi_center_frac) / 2 = 0.20
    map_margin_frac: float = 0.20

    # Click guards
    click_stable_frames: int   = 5
    click_cooldown_s:    float = 0.50

    screen_margin_px: int = 4


# ── Singletons ────────────────────────────────────────────────────────────────
CAMERA   = CameraConfig()
OVERLAY  = OverlayConfig()
PIPELINE = PipelineConfig()
DETECTOR = DetectorConfig()
MOUSE    = MouseConfig()
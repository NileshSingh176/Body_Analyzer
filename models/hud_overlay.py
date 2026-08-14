"""
hud_overlay.py — PCL shared HUD helpers
  • draw_footer_hud  : navy #001f5b footer strip at bottom of frame  (legacy)
  • draw_lhs_panel   : navy #001f5b panel on LEFT side — canvas is EXPANDED so
                       video content is never hidden or overlapped
  • expand_canvas_for_lhs : widen frame to make room for the LHS panel
  • draw_pcl_logo    : PCL logo top-right corner overlay
"""

import cv2
import numpy as np
import os
from PIL import Image

# ── Navy blue in BGR ─────────────────────────────────────────────
NAVY_BGR   = (0x5b, 0x1f, 0x00)   # #001f5b → BGR
WHITE      = (255, 255, 255)
GREY_LBL   = (180, 180, 180)
STRIP_H    = 58                    # footer height in pixels — compact

# ── LHS panel ────────────────────────────────────────────────────
PANEL_W         = 80      # width of the left-side panel (pixels)
PANEL_ITEM_H    = 72       # height per metric cell in the panel
PANEL_LOGO_H    = 60       # height reserved for logo at top of panel
PANEL_DIVIDER   = (80, 50, 15)   # subtle separator line colour

# ── Logo: load once at module import ────────────────────────────
_LOGO_H    = 68                    # height in pixels

def _load_logo():
    _this_dir  = os.path.dirname(os.path.abspath(__file__))
    _parent    = os.path.dirname(_this_dir)

    logo_names = [
        "PCL_Logo.png",
        "pcl_logo_transparent.png",
        "pcl_logo.png",
    ]
    search_dirs = [
        _parent,
        _this_dir,
        os.path.join(_this_dir, "static"),
        os.path.join(_parent,   "static"),
        os.getcwd(),
        os.path.join(os.getcwd(), ".."),
    ]
    candidates = [os.path.join(d, n) for d in search_dirs for n in logo_names]

    for p in candidates:
        if os.path.exists(p):
            try:
                logo_pil = Image.open(p).convert("RGBA")
                aspect   = logo_pil.width / logo_pil.height
                target_w = max(1, int(_LOGO_H * aspect))
                logo_pil = logo_pil.resize((target_w, _LOGO_H), Image.LANCZOS)
                return np.array(logo_pil)          # H×W×4 RGBA uint8
            except Exception:
                pass
    return None

_PCL_LOGO = _load_logo()


# ─────────────────────────────────────────────────────────────────
def draw_footer_hud(frame, items):
    """
    Draw navy #001f5b footer strip at bottom of frame.

    items: list of (label_str, value_str)
      e.g. [("REPS","5"), ("CORRECT","4"), ("KNEE","95°")]
    """
    h, w = frame.shape[:2]
    y0   = h - STRIP_H

    # Navy background
    cv2.rectangle(frame, (0, y0), (w, h), NAVY_BGR, -1)
    # Thin top separator line
    cv2.line(frame, (0, y0), (w, y0), (100, 60, 20), 1)

    n = len(items)
    if n == 0:
        return

    col_w = w // n
    for i, (lbl, val) in enumerate(items):
        cx = i * col_w + col_w // 2

        # Value — white, bold-ish (upper half of strip)
        val_s  = str(val)
        vscale = 0.62
        vthick = 2
        (vw, vh), _ = cv2.getTextSize(val_s, cv2.FONT_HERSHEY_SIMPLEX, vscale, vthick)
        val_y = y0 + int(STRIP_H * 0.50)   # 50% down — centred in upper half
        cv2.putText(frame, val_s,
                    (cx - vw // 2, val_y),
                    cv2.FONT_HERSHEY_SIMPLEX, vscale, WHITE, vthick, cv2.LINE_AA)

        # Label — small grey, just below value
        lbl_s  = str(lbl).upper()
        lscale = 0.34
        (lw2, lh2), _ = cv2.getTextSize(lbl_s, cv2.FONT_HERSHEY_SIMPLEX, lscale, 1)
        lbl_y = y0 + STRIP_H - 8           # 8px from bottom edge
        cv2.putText(frame, lbl_s,
                    (cx - lw2 // 2, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, lscale, GREY_LBL, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
def expand_canvas_for_lhs(frame):
    """
    Return a NEW, wider frame with PANEL_W extra pixels on the LEFT
    filled with the navy colour.  The original video sits unchanged
    in the right portion — nothing is cropped or overwritten.

    Call this ONCE per frame BEFORE drawing anything, then pass the
    returned canvas to draw_lhs_panel() and the rest of your drawing
    code (skeleton, distance lines, etc.) using x-offset = PANEL_W.

    Usage pattern in a module
    ─────────────────────────
        canvas = expand_canvas_for_lhs(frame)
        draw_lhs_panel(canvas, items)
        draw_pcl_logo(canvas)
        # … all other cv2.putText / cv2.line calls on canvas …
        out.write(canvas)
    """
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w + PANEL_W, 3), dtype=np.uint8)
    # Fill panel column with navy
    canvas[:, :PANEL_W] = NAVY_BGR
    # Copy original frame to the right
    canvas[:, PANEL_W:] = frame
    return canvas


# ─────────────────────────────────────────────────────────────────
def draw_lhs_panel(canvas, items):
    """
    Draw a vertical metrics strip in the navy panel on the LEFT side
    of a canvas that was previously widened by expand_canvas_for_lhs().

    items : list of (label_str, value_str)
      e.g. [("REPS","5"), ("CORRECT","4"), ("KNEE","95°")]

    Layout (top-to-bottom inside PANEL_W pixels):
      • Logo area  (PANEL_LOGO_H px)
      • One cell per item  (PANEL_ITEM_H px each)
      • Thin horizontal dividers between cells
    """
    h = canvas.shape[0]

    # ── Vertical right-edge separator ────────────────────────────
    cv2.line(canvas, (PANEL_W - 1, 0), (PANEL_W - 1, h),
             PANEL_DIVIDER, 2)

    if not items:
        return

    # ── Logo sits at top ─────────────────────────────────────────
    # (draw_pcl_logo will handle it; we just reserve the space)
    logo_bottom = PANEL_LOGO_H

    # ── Metric cells ─────────────────────────────────────────────
    cx = PANEL_W // 2      # horizontal centre of panel

    for idx, (lbl, val) in enumerate(items):
        y_top = logo_bottom + idx * PANEL_ITEM_H
        y_bot = y_top + PANEL_ITEM_H

        # Horizontal divider between cells
        if idx > 0:
            cv2.line(canvas, (8, y_top), (PANEL_W - 8, y_top),
                     PANEL_DIVIDER, 1)

        # ── Value (large, white) ──────────────────────────────
        val_s  = str(val)
        vscale = 0.60
        vthick = 2
        (vw, vh), vbase = cv2.getTextSize(
            val_s, cv2.FONT_HERSHEY_SIMPLEX, vscale, vthick)

        # Shrink if too wide for the panel
        while vw > PANEL_W - 10 and vscale > 0.30:
            vscale -= 0.05
            (vw, vh), vbase = cv2.getTextSize(
                val_s, cv2.FONT_HERSHEY_SIMPLEX, vscale, vthick)

        cell_mid = (y_top + y_bot) // 2
        val_y    = cell_mid - 4          # nudge value up slightly

        cv2.putText(canvas, val_s,
                    (cx - vw // 2, val_y),
                    cv2.FONT_HERSHEY_SIMPLEX, vscale,
                    WHITE, vthick, cv2.LINE_AA)

        # ── Label (small, grey) ───────────────────────────────
        lbl_s  = str(lbl).upper()
        lscale = 0.32
        (lw2, _), _ = cv2.getTextSize(
            lbl_s, cv2.FONT_HERSHEY_SIMPLEX, lscale, 1)
        lbl_y = val_y + vh + 10

        cv2.putText(canvas, lbl_s,
                    (cx - lw2 // 2, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, lscale,
                    GREY_LBL, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
def draw_pcl_logo(frame, margin=10, use_lhs_panel=False):
    """
    Overlay PCL logo.
    • use_lhs_panel=False (default) : top-right corner (legacy behaviour)
    • use_lhs_panel=True            : centred inside the LHS navy panel
    """
    if _PCL_LOGO is None:
        return

    logo = _PCL_LOGO
    lh, lw = logo.shape[:2]
    fh, fw = frame.shape[:2]

    if use_lhs_panel:
        # ── Fit logo inside the LHS panel width ──────────────────
        max_w = PANEL_W - 2 * margin
        max_h = PANEL_LOGO_H - 2 * margin
        scale = min(max_w / lw, max_h / lh)
        new_w = max(1, int(lw * scale))
        new_h = max(1, int(lh * scale))
        logo_pil = Image.fromarray(logo).resize((new_w, new_h), Image.LANCZOS)
        logo     = np.array(logo_pil)
        lh, lw   = logo.shape[:2]
        # Centre horizontally in panel, top-aligned with margin
        x1 = (PANEL_W - lw) // 2
        y1 = margin
    else:
        # ── Legacy: top-right corner ──────────────────────────────
        max_w = fw // 3
        if lw > max_w:
            scale    = max_w / lw
            new_w    = max(1, int(lw * scale))
            new_h    = max(1, int(lh * scale))
            logo_pil = Image.fromarray(logo).resize((new_w, new_h), Image.LANCZOS)
            logo     = np.array(logo_pil)
            lh, lw   = logo.shape[:2]
        x1 = fw - lw - margin
        y1 = margin

    x2 = x1 + lw
    y2 = y1 + lh

    if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh:
        return

    roi   = frame[y1:y2, x1:x2].astype(np.float32)
    alpha = logo[:, :, 3:4].astype(np.float32) / 255.0
    rgb   = logo[:, :, :3][:, :, ::-1].astype(np.float32)  # RGBA→BGR

    blended = roi * (1.0 - alpha) + rgb * alpha
    frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
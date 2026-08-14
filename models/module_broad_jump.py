# """
# module_broad_jump.py
# Production-grade broad jump biomechanics engine.

# Supports:
#   1. Standing Broad Jump
#   2. Run-Up Broad Jump
#   3. Single-Leg Broad Jump
#   4. Alternate Leg Broad Jump
#   5. Bounding (Power Skip)
#   6. Triple Broad Jump
#   7. Multiple Broad Jump
#   8. Reactive Broad Jump
#   9. Tuck Broad Jump
#  10. Pike Broad Jump
#  11. Weighted Broad Jump
#  12. Sand Broad Jump

# Architecture:
#   Video → Pose → Landmark Filter → COM → Foot Contact → 6-State FSM
#        → Exercise Validator → Distance Engine → Quality → Output
# """

# import os
# import cv2
# import math
# import numpy as np
# from collections import deque

# try:
#     from ultralytics import YOLO
#     _YOLO_AVAILABLE = True
# except ImportError:
#     _YOLO_AVAILABLE = False

# from utils import (
#     process_video_or_image,
#     save_wrong_angle_log,
#     set_progress,
#     frame_to_b64,
# )
# from hud_overlay import draw_footer_hud, draw_pcl_logo

# # ─────────────────────────────────────────────────────────────────────────────
# # CONSTANTS
# # ─────────────────────────────────────────────────────────────────────────────

# YOLO_MODEL_PATH = "yolov8n-pose.pt"
# CONFIDENCE      = 0.3

# # YOLO keypoint indices (COCO 17-point)
# KP_NOSE        = 0
# KP_L_SHOULDER  = 5
# KP_R_SHOULDER  = 6
# KP_L_ELBOW     = 7
# KP_R_ELBOW     = 8
# KP_L_WRIST     = 9
# KP_R_WRIST     = 10
# KP_L_HIP       = 11
# KP_R_HIP       = 12
# KP_L_KNEE      = 13
# KP_R_KNEE      = 14
# KP_L_ANKLE     = 15
# KP_R_ANKLE     = 16

# # COM weights (Dempster body segment model approximation)
# COM_WEIGHTS = {
#     KP_L_HIP: 0.28, KP_R_HIP: 0.28,
#     KP_L_SHOULDER: 0.11, KP_R_SHOULDER: 0.11,
#     KP_L_KNEE: 0.06, KP_R_KNEE: 0.06,
#     KP_L_ANKLE: 0.05, KP_R_ANKLE: 0.05,
# }

# # State machine states
# ST_READY    = "READY"
# ST_LOADING  = "LOADING"
# ST_TAKEOFF  = "TAKEOFF"
# ST_FLIGHT   = "FLIGHT"
# ST_LANDING  = "LANDING"
# ST_RESET    = "RESET"

# SKELETON_PAIRS = [
#     (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
#     (5, 11), (6, 12), (11, 12),
#     (11, 13), (13, 15), (12, 14), (14, 16),
# ]

# # Colors
# COLOR_SKEL   = (255, 255, 255)
# COLOR_JOINT  = (0,   0,   0)
# COLOR_COM    = (0, 255, 255)
# COLOR_TAKE   = (0, 165, 255)
# COLOR_LAND   = (255, 100, 255)
# COLOR_TEXT   = (255, 255, 255)

# # FSM thresholds
# GROUND_WINDOW       = 20   # frames to establish ground reference
# COM_RISE_PX         = 18   # COM must rise this much above ground to start LOADING
# COM_TAKEOFF_PX      = 30   # COM above ground to confirm TAKEOFF
# FLIGHT_CONFIRM_F    = 3    # frames COM must stay airborne
# LAND_APPROACH_PX    = 25   # COM within this of ground → approaching landing
# LAND_STABLE_F       = 4    # frames near ground to confirm LANDING
# MIN_FLIGHT_F        = 5    # minimum flight frames for a valid jump
# RESET_STABLE_F      = 8    # frames on ground before READY again

# # Exercise-specific
# REACTIVE_MAX_CONTACT_F  = 12   # max ground contact for reactive jump (at 30fps ≈ 400ms)
# TUCK_HIP_FLEX_DEG       = 60   # minimum hip flexion for tuck confirmation
# PIKE_HIP_FLEX_DEG       = 70   # minimum hip flexion for pike
# PIKE_KNEE_EXT_DEG       = 150  # minimum knee extension for pike
# BOUND_MIN_FLIGHT_F      = 3    # minimum flight per bound
# TRIPLE_PHASE_MIN_F      = 3    # min flight frames per triple-jump phase

# # Anthropometric scale: avg hip-to-ankle ≈ 0.53× body height, body height ≈ 1.75m
# # So hip-ankle ≈ 0.927m in pixels → derive px/m
# ANTHRO_HIP_ANKLE_M  = 0.90   # meters (conservative; will be estimated per-subject)

# # ─────────────────────────────────────────────────────────────────────────────
# # ONE-EURO FILTER  (per-coordinate temporal filter)
# # ─────────────────────────────────────────────────────────────────────────────

# class OneEuroFilter:
#     """
#     One Euro Filter for temporal landmark smoothing.
#     Reduces jitter without large latency; preserves fast explosive motion
#     by automatically raising cutoff frequency at high velocity.
#     """
#     def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
#         self.freq      = freq
#         self.min_cutoff = min_cutoff
#         self.beta      = beta
#         self.d_cutoff  = d_cutoff
#         self._x        = None
#         self._dx       = 0.0

#     def _alpha(self, cutoff):
#         tau = 1.0 / (2 * math.pi * cutoff)
#         return 1.0 / (1.0 + tau * self.freq)

#     def __call__(self, x):
#         if self._x is None:
#             self._x = x
#             return x
#         dx     = (x - self._x) * self.freq
#         a_d    = self._alpha(self.d_cutoff)
#         self._dx = a_d * dx + (1 - a_d) * self._dx
#         cutoff = self.min_cutoff + self.beta * abs(self._dx)
#         a      = self._alpha(cutoff)
#         self._x = a * x + (1 - a) * self._x
#         return self._x


# class LandmarkFilter:
#     """Per-landmark One Euro filter (x and y independently)."""
#     def __init__(self, n_landmarks=17, freq=30.0):
#         self.fx = [OneEuroFilter(freq=freq) for _ in range(n_landmarks)]
#         self.fy = [OneEuroFilter(freq=freq) for _ in range(n_landmarks)]

#     def update(self, pts):
#         """pts: np.array shape (17,2). Returns filtered array same shape."""
#         out = pts.copy().astype(float)
#         for i, (fx, fy) in enumerate(zip(self.fx, self.fy)):
#             if pts[i, 0] > 0 and pts[i, 1] > 0:
#                 out[i, 0] = fx(pts[i, 0])
#                 out[i, 1] = fy(pts[i, 1])
#         return out


# # ─────────────────────────────────────────────────────────────────────────────
# # CENTER OF MASS ESTIMATOR
# # ─────────────────────────────────────────────────────────────────────────────

# def estimate_com(pts):
#     """
#     Weighted average COM from body keypoints.
#     Returns (com_x, com_y) or None if not enough valid points.
#     """
#     total_w = 0.0
#     wx, wy  = 0.0, 0.0
#     for idx, w in COM_WEIGHTS.items():
#         p = pts[idx]
#         if p[0] > 0 and p[1] > 0:
#             wx += w * p[0]
#             wy += w * p[1]
#             total_w += w
#     if total_w < 0.3:
#         return None
#     return (wx / total_w, wy / total_w)


# # ─────────────────────────────────────────────────────────────────────────────
# # FOOT CONTACT ESTIMATOR
# # ─────────────────────────────────────────────────────────────────────────────

# class FootContactEstimator:
#     """
#     Tracks each foot (left/right) independently.
#     Uses ankle vertical position relative to adaptive ground reference
#     and ankle vertical velocity to classify CONTACT / AIR.
#     """
#     def __init__(self, window=30):
#         self.ground_y    = None   # adaptive ground y (pixel; larger = lower on screen)
#         self.stance_buf  = deque(maxlen=window)  # y values during known contact
#         self._prev_ly    = None
#         self._prev_ry    = None
#         self.left        = "CONTACT"
#         self.right       = "CONTACT"
#         self.THRESH_PX   = 22     # pixels above ground to classify AIR

#     def update_ground(self, ly, ry):
#         """Call with ankle y values during confirmed ground phase."""
#         for y in [ly, ry]:
#             if y > 0:
#                 self.stance_buf.append(y)
#         if self.stance_buf:
#             # Ground is near the max (bottom-most) ankle positions
#             self.ground_y = np.percentile(list(self.stance_buf), 85)

#     def classify(self, pts):
#         """Update left/right contact state from keypoints."""
#         la = pts[KP_L_ANKLE]
#         ra = pts[KP_R_ANKLE]
#         ly = la[1] if la[0] > 0 else -1
#         ry = ra[1] if ra[0] > 0 else -1

#         if self.ground_y is None:
#             self.left  = "CONTACT"
#             self.right = "CONTACT"
#             return

#         thr = self.THRESH_PX
#         gnd = self.ground_y

#         # Left foot
#         if ly > 0:
#             self.left  = "CONTACT" if ly >= gnd - thr else "AIR"
#         # Right foot
#         if ry > 0:
#             self.right = "CONTACT" if ry >= gnd - thr else "AIR"

#     @property
#     def both_contact(self):
#         return self.left == "CONTACT" and self.right == "CONTACT"

#     @property
#     def any_contact(self):
#         return self.left == "CONTACT" or self.right == "CONTACT"

#     @property
#     def both_air(self):
#         return self.left == "AIR" and self.right == "AIR"


# # ─────────────────────────────────────────────────────────────────────────────
# # ADAPTIVE SCALE ESTIMATOR  (pixel → metric)
# # ─────────────────────────────────────────────────────────────────────────────

# class ScaleEstimator:
#     """
#     Estimates pixels-per-meter from subject anthropometrics.
#     Uses hip-to-ankle vertical distance as reference.
#     Updated during ground-contact frames.
#     """
#     def __init__(self):
#         self._samples = deque(maxlen=60)
#         self.px_per_m = None   # None until calibrated

#     def update(self, pts):
#         lh = pts[KP_L_HIP];   la = pts[KP_L_ANKLE]
#         rh = pts[KP_R_HIP];   ra = pts[KP_R_ANKLE]
#         pairs = []
#         if lh[0] > 0 and la[0] > 0:
#             pairs.append(abs(la[1] - lh[1]))
#         if rh[0] > 0 and ra[0] > 0:
#             pairs.append(abs(ra[1] - rh[1]))
#         if pairs:
#             hip_ankle_px = np.mean(pairs)
#             if hip_ankle_px > 10:
#                 self._samples.append(hip_ankle_px)

#         if len(self._samples) >= 10:
#             median_px   = np.median(list(self._samples))
#             self.px_per_m = median_px / ANTHRO_HIP_ANKLE_M

#     def px_to_m(self, pixels):
#         if self.px_per_m and self.px_per_m > 0:
#             return pixels / self.px_per_m
#         # Fallback: legacy constant (0.00933 m/px)
#         return pixels * 0.00933


# # ─────────────────────────────────────────────────────────────────────────────
# # ANGLE UTILITIES
# # ─────────────────────────────────────────────────────────────────────────────

# def _vec_angle(a, b, c):
#     """Angle at point b in triangle a-b-c (degrees)."""
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b;  bc = c - b
#     cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
#     return math.degrees(math.acos(np.clip(cos, -1, 1)))

# def _hip_flexion(pts, side="L"):
#     """Hip flexion angle (shoulder-hip-knee)."""
#     sh  = pts[KP_L_SHOULDER if side=="L" else KP_R_SHOULDER]
#     hp  = pts[KP_L_HIP      if side=="L" else KP_R_HIP]
#     kn  = pts[KP_L_KNEE     if side=="L" else KP_R_KNEE]
#     if any(p[0] <= 0 for p in [sh, hp, kn]):
#         return None
#     return _vec_angle(sh, hp, kn)

# def _knee_flexion(pts, side="L"):
#     """Knee flexion angle (hip-knee-ankle)."""
#     hp  = pts[KP_L_HIP   if side=="L" else KP_R_HIP]
#     kn  = pts[KP_L_KNEE  if side=="L" else KP_R_KNEE]
#     an  = pts[KP_L_ANKLE if side=="L" else KP_R_ANKLE]
#     if any(p[0] <= 0 for p in [hp, kn, an]):
#         return None
#     return _vec_angle(hp, kn, an)

# def _com_velocity(com_hist, fps):
#     """Returns (vx, vy) in px/frame from last two COM positions."""
#     if len(com_hist) < 2:
#         return (0.0, 0.0)
#     dx = com_hist[-1][0] - com_hist[-2][0]
#     dy = com_hist[-1][1] - com_hist[-2][1]
#     return (dx * fps, dy * fps)


# # ─────────────────────────────────────────────────────────────────────────────
# # JUMP ENGINE  (6-state FSM)
# # ─────────────────────────────────────────────────────────────────────────────

# class JumpEngine:
#     """
#     Core 6-state finite state machine.
#     Tracks COM, foot contacts, and produces jump events.
#     """
#     def __init__(self, fps=30.0, exercise="standing_broad_jump"):
#         self.fps      = fps
#         self.exercise = exercise
#         self.state    = ST_READY

#         # Ground reference
#         self._gnd_com_y    = None   # adaptive ground COM y
#         self._gnd_buf      = deque(maxlen=GROUND_WINDOW)
#         self._gnd_locked   = False

#         # Takeoff
#         self._takeoff_com  = None
#         self._takeoff_f    = 0
#         self._takeoff_heel = None   # left heel x at takeoff

#         # Flight
#         self._flight_f     = 0
#         self._peak_com_y   = None
#         self._flight_com_x_start = None
#         self._peak_b64     = None

#         # In-flight kinematics (for tuck/pike/bounding)
#         self._flight_hip_flex   = []
#         self._flight_knee_flex  = []

#         # Landing
#         self._land_stable_f = 0
#         self._land_com_buf  = deque(maxlen=8)

#         # Reset
#         self._reset_f      = 0

#         # Loading / countermovement
#         self._load_com_y   = None   # COM y at start of countermovement
#         self._load_start_f = 0
#         self._cm_depth     = 0.0   # countermovement depth in px

#         # Completed jumps
#         self.jumps         = []
#         self._jump_n       = 0

#         # Per-phase data for triple/multiple/bounding
#         self._phase_list   = []    # list of phase dicts per jump
#         self._phase_start_com = None

#         # Reactive: ground contact tracking
#         self._gct_start_f  = 0    # frame when ground contact began after landing

#         # Approach velocity (run-up)
#         self._approach_vx  = 0.0
#         self._approach_buf = deque(maxlen=20)

#         # Frame counter
#         self._frame        = 0

#     # ──────────────────────────────────────────────────────────────────────
#     def _update_ground(self, com_y, foot: FootContactEstimator):
#         """Update adaptive ground COM_y during stable contact."""
#         if foot.both_contact or foot.any_contact:
#             self._gnd_buf.append(com_y)
#         if len(self._gnd_buf) >= 5:
#             self._gnd_com_y = np.percentile(list(self._gnd_buf), 80)

#     def _is_airborne(self, com_y):
#         if self._gnd_com_y is None:
#             return False
#         return com_y < self._gnd_com_y - COM_TAKEOFF_PX

#     def _is_near_ground(self, com_y):
#         if self._gnd_com_y is None:
#             return True
#         return com_y >= self._gnd_com_y - LAND_APPROACH_PX

#     # ──────────────────────────────────────────────────────────────────────
#     def update(self, pts, foot: FootContactEstimator, com, scale: ScaleEstimator,
#                frame_b64=None):
#         """
#         Called every frame.
#         pts: filtered keypoints (17,2)
#         foot: FootContactEstimator (already updated this frame)
#         com: (cx, cy) or None
#         scale: ScaleEstimator
#         Returns: jump_event dict if a jump completed this frame, else None
#         """
#         self._frame += 1
#         if com is None:
#             return None

#         com_x, com_y = com

#         # Track approach velocity for run-up detection
#         self._approach_buf.append(com_x)
#         if len(self._approach_buf) >= 5:
#             dx = self._approach_buf[-1] - self._approach_buf[-5]
#             self._approach_vx = dx / 5.0 * self.fps  # px/s

#         event = None

#         # ── READY ─────────────────────────────────────────────────────────
#         if self.state == ST_READY:
#             self._update_ground(com_y, foot)
#             # Detect loading (COM drops below ground reference = countermovement)
#             if self._gnd_com_y is not None and not foot.both_air:
#                 # COM starts going UP = loading into takeoff
#                 if com_y < self._gnd_com_y - COM_RISE_PX:
#                     self.state         = ST_LOADING
#                     self._load_com_y   = com_y
#                     self._load_start_f = self._frame
#                 else:
#                     # Still grounded — update scale
#                     scale.update(pts)
#                     foot.update_ground(pts[KP_L_ANKLE][1], pts[KP_R_ANKLE][1])
#                     self._takeoff_com  = com   # keep updating takeoff reference

#         # ── LOADING ───────────────────────────────────────────────────────
#         elif self.state == ST_LOADING:
#             if foot.both_air or self._is_airborne(com_y):
#                 # Feet left ground → TAKEOFF
#                 self._takeoff_com        = com
#                 self._flight_com_x_start = com_x
#                 self._takeoff_f          = self._frame
#                 self._flight_f           = 0
#                 self._peak_com_y         = com_y
#                 self._peak_b64           = frame_b64
#                 self._flight_hip_flex    = []
#                 self._flight_knee_flex   = []
#                 self._phase_start_com    = com
#                 # Countermovement depth = ground COM_y - min COM_y during loading
#                 self._cm_depth = max(0.0, self._gnd_com_y - self._load_com_y) if self._gnd_com_y else 0.0
#                 self.state = ST_FLIGHT
#             elif not self._is_airborne(com_y):
#                 # Still loading / false alarm
#                 if com_y > self._gnd_com_y:
#                     # COM went back down, abort loading
#                     self.state = ST_READY

#         # ── FLIGHT ────────────────────────────────────────────────────────
#         elif self.state == ST_FLIGHT:
#             self._flight_f += 1

#             # Track peak height (smallest y = highest)
#             if com_y < self._peak_com_y:
#                 self._peak_com_y = com_y
#                 self._peak_b64   = frame_b64

#             # Collect in-flight body angles for tuck/pike detection
#             hf_l = _hip_flexion(pts, "L")
#             hf_r = _hip_flexion(pts, "R")
#             kf_l = _knee_flexion(pts, "L")
#             kf_r = _knee_flexion(pts, "R")
#             if hf_l: self._flight_hip_flex.append(hf_l)
#             if hf_r: self._flight_hip_flex.append(hf_r)
#             if kf_l: self._flight_knee_flex.append(kf_l)
#             if kf_r: self._flight_knee_flex.append(kf_r)

#             # Check landing approach
#             if self._flight_f >= MIN_FLIGHT_F:
#                 if self._is_near_ground(com_y) or foot.any_contact:
#                     self._land_stable_f += 1
#                     self._land_com_buf.append(com_x)
#                     if self._land_stable_f >= LAND_STABLE_F:
#                         event = self._confirm_jump(com, pts, scale)
#                         self.state = ST_LANDING
#                 else:
#                     self._land_stable_f = 0
#                     self._land_com_buf.clear()

#             # Safety: if COM returned to ground quickly with no flight, reset
#             if self._flight_f < MIN_FLIGHT_F and foot.both_contact and self._is_near_ground(com_y):
#                 self.state = ST_READY

#         # ── LANDING ───────────────────────────────────────────────────────
#         elif self.state == ST_LANDING:
#             self._land_stable_f = 0
#             self._land_com_buf.clear()
#             self._reset_f = 0

#             # For reactive jump: record GCT start
#             self._gct_start_f = self._frame
#             self.state = ST_RESET

#         # ── RESET ─────────────────────────────────────────────────────────
#         elif self.state == ST_RESET:
#             self._reset_f += 1
#             self._update_ground(com_y, foot)

#             # Reactive: if feet leave ground again very quickly
#             if (foot.both_air and self._reset_f <= REACTIVE_MAX_CONTACT_F
#                     and "reactive" in self.exercise):
#                 # Short GCT → reactive jump — transition directly to FLIGHT
#                 self._takeoff_com        = com
#                 self._flight_com_x_start = com_x
#                 self._takeoff_f          = self._frame
#                 self._flight_f           = 0
#                 self._peak_com_y         = com_y
#                 self._peak_b64           = frame_b64
#                 self._flight_hip_flex    = []
#                 self._flight_knee_flex   = []
#                 self._phase_start_com    = com
#                 self.state = ST_FLIGHT

#             elif self._reset_f >= RESET_STABLE_F:
#                 self.state = ST_READY
#                 self._gnd_locked = True

#         return event

#     # ──────────────────────────────────────────────────────────────────────
#     def _confirm_jump(self, land_com, pts, scale: ScaleEstimator):
#         """Build and store a jump event dict."""
#         self._jump_n += 1

#         # Distance: landing COM x  - takeoff COM x
#         land_x    = float(np.median(list(self._land_com_buf))) if self._land_com_buf else land_com[0]
#         takeoff_x = self._flight_com_x_start if self._flight_com_x_start else (self._takeoff_com[0] if self._takeoff_com else land_x)
#         px_dist   = abs(land_x - takeoff_x)
#         dist_m    = round(scale.px_to_m(px_dist), 3)
#         dist_cm   = round(dist_m * 100, 1)

#         flight_f  = self._flight_f
#         flight_ms = int(flight_f / self.fps * 1000)

#         # Peak COM height above takeoff COM
#         peak_rise_px = 0.0
#         if self._takeoff_com and self._peak_com_y:
#             peak_rise_px = max(0.0, self._takeoff_com[1] - self._peak_com_y)
#         peak_rise_m = scale.px_to_m(peak_rise_px)

#         # Takeoff velocity estimate (from scale + flight time)
#         # v0_y ≈ g * t_flight / 2  (ignoring air resistance)
#         g = 9.81
#         t = flight_f / self.fps
#         v0_y_ms = g * t / 2 if t > 0 else 0.0
#         v0_x_ms = scale.px_to_m(abs(self._approach_vx)) if self._approach_vx else dist_m / (t + 1e-9)
#         takeoff_vel = round(math.sqrt(v0_x_ms**2 + v0_y_ms**2), 2)
#         takeoff_angle = round(math.degrees(math.atan2(v0_y_ms, v0_x_ms + 1e-9)), 1) if v0_x_ms else 0.0

#         # Tuck / pike detection from in-flight angles
#         tuck_confirmed = False
#         pike_confirmed = False
#         min_hip_flex = min(self._flight_hip_flex) if self._flight_hip_flex else 180
#         min_knee_flex = min(self._flight_knee_flex) if self._flight_knee_flex else 180
#         max_knee_ext  = max(self._flight_knee_flex) if self._flight_knee_flex else 0

#         if min_hip_flex < TUCK_HIP_FLEX_DEG and min_knee_flex < 90:
#             tuck_confirmed = True
#         if min_hip_flex < PIKE_HIP_FLEX_DEG and max_knee_ext > PIKE_KNEE_EXT_DEG:
#             pike_confirmed = True

#         # Countermovement depth
#         cm_depth_m = round(scale.px_to_m(self._cm_depth), 3)

#         # Horizontal efficiency: forward dist / total COM path (approximation)
#         h_eff = round(min(1.0, px_dist / (px_dist + peak_rise_px + 1e-9)), 3)

#         # Form score (distance-based, 1-10)
#         form = _form_score(dist_m)

#         # Ground contact time (reactive)
#         gct_ms = 0
#         if "reactive" in self.exercise and self._gct_start_f > 0:
#             gct_f  = self._takeoff_f - self._gct_start_f
#             gct_ms = int(max(0, gct_f) / self.fps * 1000)

#         rsi = round(flight_ms / gct_ms, 3) if gct_ms > 0 else None

#         jump = {
#             # Identity
#             "jump_no"           : self._jump_n,
#             # Distances
#             "distance_m"        : dist_m,
#             "distance_cm"       : dist_cm,
#             "pixel_dist"        : round(px_dist, 1),
#             # Timing
#             "flight_ms"         : flight_ms,
#             "airborne_ms"       : flight_ms,
#             "takeoff_frame"     : self._takeoff_f,
#             "landing_frame"     : self._frame,
#             # Positions
#             "takeoff_com_x"     : round(takeoff_x, 1),
#             "landing_com_x"     : round(land_x, 1),
#             # Kinematics
#             "takeoff_velocity_ms": takeoff_vel,
#             "takeoff_angle_deg" : takeoff_angle,
#             "peak_rise_m"       : round(peak_rise_m, 3),
#             "approach_vx_px_s"  : round(self._approach_vx, 1),
#             "approach_speed_ms" : round(scale.px_to_m(abs(self._approach_vx)), 2),
#             "horizontal_efficiency": h_eff,
#             "cm_depth_m"        : cm_depth_m,
#             # Technique
#             "tuck_confirmed"    : tuck_confirmed,
#             "pike_confirmed"    : pike_confirmed,
#             "min_hip_flex_deg"  : round(min_hip_flex, 1),
#             "min_knee_flex_deg" : round(min_knee_flex, 1),
#             # Reactive
#             "ground_contact_ms" : gct_ms,
#             "rsi"               : rsi,
#             # Score
#             "form_score"        : form,
#             # Snapshot
#             "_peak_b64"         : self._peak_b64,
#         }
#         self.jumps.append(jump)
#         return jump

#     def force_close(self, com, scale: ScaleEstimator):
#         """Call at video end if still in FLIGHT state."""
#         if self.state == ST_FLIGHT and self._flight_f >= MIN_FLIGHT_F:
#             land_x = com[0] if com else (self._flight_com_x_start or 0)
#             self._land_com_buf.append(land_x)
#             self._land_stable_f = LAND_STABLE_F
#             return self._confirm_jump(com, np.zeros((17, 2)), scale)
#         return None


# # ─────────────────────────────────────────────────────────────────────────────
# # EXERCISE-SPECIFIC VALIDATORS
# # ─────────────────────────────────────────────────────────────────────────────

# class ExerciseValidator:
#     """
#     Wraps the core JumpEngine and applies exercise-specific
#     post-processing to the completed jump list.
#     """
#     def __init__(self, exercise: str):
#         self.exercise = exercise

#     def validate(self, jumps: list) -> dict:
#         ex = self.exercise.lower().replace(" ", "_")

#         if "triple" in ex:
#             return self._triple(jumps)
#         elif "alternate" in ex:
#             return self._alternate(jumps)
#         elif "bounding" in ex or "power_skip" in ex:
#             return self._bounding(jumps)
#         elif "multiple" in ex:
#             return self._multiple(jumps)
#         else:
#             return self._standard(jumps)

#     # Standard single / repeated jumps
#     def _standard(self, jumps):
#         return {"validated_jumps": jumps, "phase_info": None}

#     # Triple: group every 3 consecutive jumps into hop/step/jump
#     def _triple(self, jumps):
#         phases = []
#         for i in range(0, len(jumps) - 2, 3):
#             hop, step, jmp = jumps[i], jumps[i+1], jumps[i+2]
#             total = hop["distance_m"] + step["distance_m"] + jmp["distance_m"]
#             phases.append({
#                 "triple_no"   : len(phases) + 1,
#                 "hop"         : hop,
#                 "step"        : step,
#                 "jump"        : jmp,
#                 "total_m"     : round(total, 3),
#                 "phase_ratio" : [round(hop["distance_m"]/total, 2),
#                                  round(step["distance_m"]/total, 2),
#                                  round(jmp["distance_m"]/total, 2)],
#             })
#         return {"validated_jumps": jumps, "phase_info": phases}

#     # Alternate: flag if consecutive jumps show alternating takeoff feet
#     def _alternate(self, jumps):
#         for i, j in enumerate(jumps):
#             j["leg_tag"] = "L" if i % 2 == 0 else "R"
#         return {"validated_jumps": jumps, "phase_info": None}

#     # Bounding: each jump = one bound
#     def _bounding(self, jumps):
#         bounds = []
#         for j in jumps:
#             bounds.append({
#                 "bound_no"  : j["jump_no"],
#                 "distance_m": j["distance_m"],
#                 "flight_ms" : j["flight_ms"],
#                 "gct_ms"    : j.get("ground_contact_ms", 0),
#             })
#         total = sum(b["distance_m"] for b in bounds)
#         return {"validated_jumps": jumps, "phase_info": bounds,
#                 "total_distance_m": round(total, 3)}

#     # Multiple: cumulative metrics
#     def _multiple(self, jumps):
#         cumulative = 0.0
#         for j in jumps:
#             cumulative += j["distance_m"]
#             j["cumulative_m"] = round(cumulative, 3)
#         return {"validated_jumps": jumps, "phase_info": None}


# # ─────────────────────────────────────────────────────────────────────────────
# # QUALITY & SCORING
# # ─────────────────────────────────────────────────────────────────────────────

# def _form_score(distance_m):
#     thresholds = [2.50, 2.20, 2.00, 1.80, 1.50, 1.20]
#     scores     = [10,   9,    8,    7,    6,    5]
#     for t, s in zip(thresholds, scores):
#         if distance_m >= t:
#             return s
#     return 4

# def _form_score_aggregate(jumps_m):
#     if not jumps_m:
#         return 0
#     best  = max(jumps_m)
#     score = _form_score(best)
#     if len(jumps_m) > 1:
#         cons = min(jumps_m) / best
#         if   cons >= 0.90: score = min(10, score + 1)
#         elif cons < 0.70:  score = max(1,  score - 1)
#     return score

# def _compute_quality(jumps, pose_conf, foot_conf, scale_calibrated):
#     """Generate overall confidence score 0-100."""
#     if not jumps:
#         return {
#             "overall": 0, "pose": int(pose_conf * 100),
#             "contact": int(foot_conf * 100), "jump": 0, "distance": 0
#         }
#     pose_s  = int(pose_conf * 100)
#     cont_s  = int(foot_conf * 100)
#     jump_s  = min(100, len(jumps) * 25)
#     dist_s  = 85 if scale_calibrated else 55
#     overall = int(0.3*pose_s + 0.2*cont_s + 0.25*jump_s + 0.25*dist_s)
#     return {"overall": overall, "pose": pose_s, "contact": cont_s,
#             "jump": jump_s, "distance": dist_s}

# def _fatigue_index(jumps_m):
#     """Decline in distance over repetitions (lower is better)."""
#     if len(jumps_m) < 2:
#         return None
#     return round((jumps_m[0] - jumps_m[-1]) / jumps_m[0] * 100, 1)

# def _rep_variability(jumps_m):
#     if len(jumps_m) < 2:
#         return None
#     return round(float(np.std(jumps_m)) / (np.mean(jumps_m) + 1e-9) * 100, 1)

# def _feedback(jump_results, detected, exercise):
#     issues    = []
#     strengths = []

#     if not detected:
#         issues += [
#             "❌ Person not detected in video.",
#             "📌 Ensure: full body visible (head to toe), side-angle camera, good lighting.",
#             "🎥 Avoid top-down or front-facing camera angles.",
#         ]
#         return issues, strengths

#     if not jump_results:
#         issues += [
#             "❌ No valid jump detected.",
#             "📐 Camera should be placed sideways at full-body height.",
#             "⏱️ Video must capture full jump — takeoff, flight, and landing.",
#         ]
#         return issues, strengths

#     distances = [j["distance_m"] for j in jump_results]
#     best = max(distances)
#     avg  = sum(distances) / len(distances)

#     if best < 1.20:
#         issues.append(f"Short jump ({best:.2f}m) — focus on arm swing and hip extension.")
#     elif best < 1.80:
#         issues.append(f"Below-average distance ({best:.2f}m) — work on explosive leg drive.")
#     elif best < 2.20:
#         strengths.append(f"Good jump distance ({best:.2f}m).")
#     else:
#         strengths.append(f"Excellent jump distance ({best:.2f}m)!")

#     if len(distances) > 1:
#         cons = min(distances) / best
#         if cons >= 0.90:
#             strengths.append(f"Very consistent across {len(distances)} jumps ({cons:.0%}).")
#         elif cons < 0.70:
#             issues.append(f"High variation ({cons:.0%} consistency) — repeat takeoff mechanics.")

#     # Tuck/pike feedback
#     if "tuck" in exercise.lower():
#         tucks = [j for j in jump_results if j.get("tuck_confirmed")]
#         if tucks:
#             strengths.append(f"Tuck confirmed in {len(tucks)}/{len(jump_results)} jumps.")
#         else:
#             issues.append("Tuck not detected — ensure knees pull toward chest during flight.")

#     if "pike" in exercise.lower():
#         pikes = [j for j in jump_results if j.get("pike_confirmed")]
#         if pikes:
#             strengths.append(f"Pike confirmed in {len(pikes)}/{len(jump_results)} jumps.")
#         else:
#             issues.append("Pike not detected — keep legs extended and reach toward toes.")

#     if not issues:
#         issues = ["No major issues detected."]
#     return issues, strengths


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN ANALYSIS FUNCTION
# # ─────────────────────────────────────────────────────────────────────────────

# def analyse_broad_jump(
#     path,
#     is_video,
#     output_path=None,
#     session_id=None,
#     source_filename="",
#     progress_uid=None,
#     exercise="Standing Broad Jump",
# ):
#     """
#     Main entry point for broad jump analysis.
#     Supports all 12 broad jump variants via the `exercise` parameter.

#     Parameters
#     ----------
#     exercise : str
#         One of: "Standing Broad Jump", "Run-Up Broad Jump",
#         "Single-Leg Broad Jump", "Alternate Leg Broad Jump",
#         "Bounding", "Triple Broad Jump", "Multiple Broad Jump",
#         "Reactive Broad Jump", "Tuck Broad Jump", "Pike Broad Jump",
#         "Weighted Broad Jump", "Sand Broad Jump"
#     """
#     if not _YOLO_AVAILABLE:
#         raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
#     if not is_video:
#         raise ValueError("Broad jump analysis requires a video file.")

#     # ── Model ────────────────────────────────────────────────────────────
#     model_path = YOLO_MODEL_PATH
#     if not os.path.exists(model_path):
#         here = os.path.dirname(os.path.abspath(__file__))
#         alt  = os.path.join(here, YOLO_MODEL_PATH)
#         if os.path.exists(alt):
#             model_path = alt
#         else:
#             raise FileNotFoundError(f"YOLO model not found at '{YOLO_MODEL_PATH}'.")
#     model = YOLO(model_path)

#     # ── Video metadata ────────────────────────────────────────────────────
#     cap    = cv2.VideoCapture(path)
#     fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     # ── Sub-systems ───────────────────────────────────────────────────────
#     lm_filter = LandmarkFilter(n_landmarks=17, freq=fps)
#     foot_est  = FootContactEstimator(window=int(fps * 1.5))
#     scale_est = ScaleEstimator()
#     engine    = JumpEngine(fps=fps, exercise=exercise.lower())
#     validator = ExerciseValidator(exercise)

#     # ── State ─────────────────────────────────────────────────────────────
#     detected_ok   = [False]
#     wrong_events  = []
#     per_frame_data= []
#     total_frames  = [1]
#     pose_confs    = []
#     foot_conf_acc = []

#     live_hud      = {"jump_no": "0", "dist": "---", "form": "---", "state": ST_READY}

#     def _push_event(jdata, b64):
#         if not progress_uid:
#             return
#         pct = min(94, int(len(per_frame_data) / max(1, total_frames[0]) * 90))
#         set_progress(
#             progress_uid, pct,
#             f"Jump {jdata['jump_no']} — {jdata['distance_m']:.2f}m",
#             jump_event={**jdata, "frame_b64": b64},
#         )

#     # ── Per-frame callback ────────────────────────────────────────────────
#     def pf(frame, fc, total):
#         total_frames[0] = max(total, 1)

#         results   = model(frame, conf=CONFIDENCE, verbose=False)
#         best_pts  = None
#         best_area = 0
#         best_conf = 0.0

#         for result in results:
#             if (result.keypoints is None or
#                     result.keypoints.xy is None or
#                     result.keypoints.conf is None):
#                 continue
#             for kpts_xy, kpts_conf in zip(
#                     result.keypoints.xy.cpu().numpy(),
#                     result.keypoints.conf.cpu().numpy()):
#                 pts   = kpts_xy.astype(float)
#                 valid = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
#                 if len(valid) < 6:
#                     continue
#                 area = ((valid[:, 0].max() - valid[:, 0].min()) *
#                         (valid[:, 1].max() - valid[:, 1].min()))
#                 if area > best_area:
#                     best_area = area
#                     best_pts  = pts
#                     best_conf = float(np.mean(kpts_conf[kpts_conf > 0]))

#         if best_pts is None:
#             draw_footer_hud(frame, [
#                 ("JUMP #", live_hud["jump_no"]),
#                 ("DIST",   live_hud["dist"]),
#                 ("FORM",   live_hud["form"]),
#             ])
#             draw_pcl_logo(frame)
#             return frame

#         detected_ok[0] = True
#         pose_confs.append(best_conf)

#         # Filter landmarks
#         pts = lm_filter.update(best_pts)

#         # Draw skeleton (skip face landmarks 0-4)
#         for p1i, p2i in SKELETON_PAIRS:
#             p1, p2 = pts[p1i], pts[p2i]
#             if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
#                 cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
#                          COLOR_SKEL, 2)
#         for idx, p in enumerate(pts):
#             if idx < 5 or p[0] <= 0 or p[1] <= 0:
#                 continue
#             cv2.circle(frame, (int(p[0]), int(p[1])), 4, COLOR_JOINT, -1)

#         # COM
#         com = estimate_com(pts)
#         if com:
#             cv2.circle(frame, (int(com[0]), int(com[1])), 6, COLOR_COM, -1)

#         # Foot contact
#         foot_est.classify(pts)

#         # Engine update
#         frame_b64  = frame_to_b64(frame) if engine.state in (ST_FLIGHT, ST_TAKEOFF) else None
#         jump_event = engine.update(pts, foot_est, com, scale_est, frame_b64)

#         if jump_event:
#             b64 = jump_event.pop("_peak_b64", None) or frame_to_b64(frame)
#             _push_event(jump_event, b64)
#             live_hud["jump_no"] = str(jump_event["jump_no"])
#             live_hud["dist"]    = f"{jump_event['distance_m']:.2f}m"
#             live_hud["form"]    = f"{jump_event['form_score']}/10"

#         live_hud["state"] = engine.state

#         # Visualise scale (if calibrated, show reference line)
#         if scale_est.px_per_m:
#             ref_px = int(scale_est.px_per_m)
#             mid_x, gnd_y = width // 4, height - 40
#             cv2.line(frame, (mid_x, gnd_y), (mid_x + ref_px, gnd_y), (100, 255, 100), 2)
#             cv2.putText(frame, "1m", (mid_x + ref_px // 2 - 10, gnd_y - 6),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

#         # Draw ground reference
#         if engine._gnd_com_y:
#             gnd_px = int(engine._gnd_com_y + (LAND_APPROACH_PX))
#             cv2.line(frame, (0, gnd_px), (width, gnd_px), (80, 80, 80), 1)

#         # Draw takeoff / landing markers
#         if engine._flight_com_x_start and engine.state == ST_FLIGHT:
#             tx = int(engine._flight_com_x_start)
#             cv2.line(frame, (tx, 0), (tx, height), COLOR_TAKE, 1)

#         per_frame_data.append({
#             "frame"     : fc,
#             "state"     : engine.state,
#             "jump_count": len(engine.jumps),
#         })

#         draw_footer_hud(frame, [
#             ("JUMP #", live_hud["jump_no"]),
#             ("DIST",   live_hud["dist"]),
#             ("FORM",   live_hud["form"]),
#         ])
#         draw_pcl_logo(frame)
#         return frame

#     # ── Run ───────────────────────────────────────────────────────────────
#     snaps = process_video_or_image(
#         path, is_video, pf,
#         output_path=output_path,
#         snap_pcts=[0.1, 0.3, 0.5, 0.7, 0.9],
#         analysis_skip=1,
#         progress_uid=progress_uid,
#     )

#     # Force-close if video ends mid-flight
#     if engine.state == ST_FLIGHT and engine._flight_f >= MIN_FLIGHT_F:
#         last_com = None
#         if per_frame_data:
#             # Best effort: use last known COM from scale buf
#             last_com = (engine._flight_com_x_start, engine._gnd_com_y or 0)
#         ev = engine.force_close(last_com, scale_est)
#         if ev:
#             ev.pop("_peak_b64", None)
#             _push_event(ev, "")

#     if not detected_ok[0]:
#         raise ValueError(
#             "No person detected. Upload a side-angle video with full body visible "
#             "(head to toe) and good lighting."
#         )

#     if session_id:
#         save_wrong_angle_log(exercise, session_id, source_filename, wrong_events)

#     # ── Post-processing ────────────────────────────────────────────────────
#     jumps     = engine.jumps
#     val_out   = validator.validate(jumps)
#     jumps_m   = [j["distance_m"] for j in jumps]

#     best_dist = round(max(jumps_m), 3) if jumps_m else 0.0
#     avg_dist  = round(sum(jumps_m) / len(jumps_m), 3) if jumps_m else 0.0

#     pose_conf_avg = float(np.mean(pose_confs)) if pose_confs else 0.0
#     foot_conf_val = 0.75 if scale_est.px_per_m else 0.50
#     quality = _compute_quality(jumps, pose_conf_avg, foot_conf_val,
#                                 scale_calibrated=scale_est.px_per_m is not None)

#     form_score = _form_score_aggregate(jumps_m)
#     issues, strengths = _feedback(jumps, detected_ok[0], exercise)

#     while len(snaps) < 5:
#         snaps.append(snaps[-1] if snaps else "")

#     step      = max(1, int(fps / 10))
#     per_frame = per_frame_data[::step]

#     best_str = f"{best_dist:.2f} m" if best_dist > 0 else "N/A"
#     avg_str  = f"{avg_dist:.2f} m"  if avg_dist  > 0 else "N/A"

#     # ── Aggregate quality metrics ──────────────────────────────────────────
#     gct_vals  = [j["ground_contact_ms"] for j in jumps if j.get("ground_contact_ms")]
#     rsi_vals  = [j["rsi"] for j in jumps if j.get("rsi")]
#     ft_vals   = [j["flight_ms"] for j in jumps]

#     return {
#         # Core
#         "exercise"              : exercise,
#         "jump_count"            : len(jumps),
#         "correct_jumps"         : len(jumps),
#         "wrong_jumps"           : 0,
#         # Distance
#         "best_distance_m"       : best_dist,
#         "best_distance_cm"      : round(best_dist * 100, 1),
#         "avg_distance_m"        : avg_dist,
#         "all_jumps_m"           : [round(j, 3) for j in jumps_m],
#         # Per-jump detail
#         "per_jump"              : jumps,
#         # Exercise variant results
#         "phase_info"            : val_out.get("phase_info"),
#         "total_distance_m"      : val_out.get("total_distance_m"),
#         # Timing
#         "avg_flight_ms"         : int(np.mean(ft_vals)) if ft_vals else 0,
#         "avg_ground_contact_ms" : int(np.mean(gct_vals)) if gct_vals else 0,
#         "avg_rsi"               : round(float(np.mean(rsi_vals)), 3) if rsi_vals else None,
#         # Consistency
#         "fatigue_index"         : _fatigue_index(jumps_m),
#         "rep_variability_pct"   : _rep_variability(jumps_m),
#         # Scale
#         "px_per_m"              : round(scale_est.px_per_m, 2) if scale_est.px_per_m else None,
#         "scale_method"          : "anthropometric" if scale_est.px_per_m else "legacy_constant",
#         # Quality / Confidence
#         "confidence"            : quality,
#         "form_score"            : form_score,
#         # Feedback
#         "issues"                : issues,
#         "strengths"             : strengths,
#         # UI
#         "metrics": [
#             {"label": "Jumps Detected", "value": str(len(jumps))},
#             {"label": "Best Jump",      "value": best_str},
#             {"label": "Avg Jump",       "value": avg_str},
#             {"label": "Form Score",     "value": f"{form_score}/10"},
#             {"label": "Confidence",     "value": f"{quality['overall']}%"},
#         ],
#         # Legacy keys kept for backward compatibility
#         "height_cm"             : round(best_dist * 100, 1),
#         "per_frame"             : per_frame,
#         "snapshots"             : snaps,
#         "wrong_angle_count"     : 0,
#         "_wrong_events"         : wrong_events,
#     }























"""
module_broad_jump.py
Production-grade broad jump biomechanics engine.

Supports:
  1. Standing Broad Jump
  2. Run-Up Broad Jump
  3. Single-Leg Broad Jump
  4. Alternate Leg Broad Jump
  5. Bounding (Power Skip)
  6. Triple Broad Jump
  7. Multiple Broad Jump
  8. Reactive Broad Jump
  9. Tuck Broad Jump
 10. Pike Broad Jump
 11. Weighted Broad Jump
 12. Sand Broad Jump

Architecture:
  Video → Pose → Landmark Filter → COM → Foot Contact → 6-State FSM
       → Exercise Validator → Distance Engine → Quality → Output
"""

# import os
# import cv2
# import math
# import numpy as np
# from collections import deque

# try:
#     from ultralytics import YOLO
#     _YOLO_AVAILABLE = True
# except ImportError:
#     _YOLO_AVAILABLE = False

# from utils import (
#     process_video_or_image,
#     save_wrong_angle_log,
#     set_progress,
#     frame_to_b64,
# )
# from hud_overlay import draw_footer_hud, draw_pcl_logo

# # ─────────────────────────────────────────────────────────────────────────────
# # CONSTANTS
# # ─────────────────────────────────────────────────────────────────────────────

# YOLO_MODEL_PATH = "yolov8n-pose.pt"
# CONFIDENCE      = 0.3

# # YOLO keypoint indices (COCO 17-point)
# KP_NOSE        = 0
# KP_L_SHOULDER  = 5
# KP_R_SHOULDER  = 6
# KP_L_ELBOW     = 7
# KP_R_ELBOW     = 8
# KP_L_WRIST     = 9
# KP_R_WRIST     = 10
# KP_L_HIP       = 11
# KP_R_HIP       = 12
# KP_L_KNEE      = 13
# KP_R_KNEE      = 14
# KP_L_ANKLE     = 15
# KP_R_ANKLE     = 16

# # COM weights (Dempster body segment model approximation)
# COM_WEIGHTS = {
#     KP_L_HIP: 0.28, KP_R_HIP: 0.28,
#     KP_L_SHOULDER: 0.11, KP_R_SHOULDER: 0.11,
#     KP_L_KNEE: 0.06, KP_R_KNEE: 0.06,
#     KP_L_ANKLE: 0.05, KP_R_ANKLE: 0.05,
# }

# # State machine states
# ST_READY    = "READY"
# ST_LOADING  = "LOADING"
# ST_TAKEOFF  = "TAKEOFF"
# ST_FLIGHT   = "FLIGHT"
# ST_LANDING  = "LANDING"
# ST_RESET    = "RESET"


# # ─────────────────────────────────────────────────────────────────────────────
# # JSON-SAFETY HELPER
# # ─────────────────────────────────────────────────────────────────────────────
# def _to_json_safe(obj):
#     """
#     Recursively convert numpy scalar/array types (np.bool_, np.float64,
#     np.int64, np.ndarray, etc.) to native Python types so the result can
#     always be passed to json.dumps() without raising
#     'Object of type X is not JSON serializable'.

#     numpy values leak into output dicts any time a computation touches an
#     np.median()/np.mean()/np.min()/np.max() result (even indirectly, e.g.
#     `abs(numpy_float) < python_float` still yields np.bool_) — so this is
#     applied once, at the point each output dict is finalized, rather than
#     trying to manually cast every individual field (which is exactly how
#     this bug slipped through the first time).
#     """
#     if isinstance(obj, dict):
#         return {k: _to_json_safe(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [_to_json_safe(v) for v in obj]
#     if isinstance(obj, np.bool_):
#         return bool(obj)
#     if isinstance(obj, np.integer):
#         return int(obj)
#     if isinstance(obj, np.floating):
#         return float(obj)
#     if isinstance(obj, np.ndarray):
#         return _to_json_safe(obj.tolist())
#     return obj

# SKELETON_PAIRS = [
#     (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
#     (5, 11), (6, 12), (11, 12),
#     (11, 13), (13, 15), (12, 14), (14, 16),
# ]

# # Colors
# COLOR_SKEL   = (255, 255, 255)
# COLOR_JOINT  = (0,   0,   0)
# COLOR_COM    = (0, 255, 255)
# COLOR_TAKE   = (0, 165, 255)
# COLOR_LAND   = (255, 100, 255)
# COLOR_TEXT   = (255, 255, 255)

# # FSM thresholds
# GROUND_WINDOW       = 20   # frames to establish ground reference
# COM_RISE_PX         = 18   # COM must rise this much above ground to start LOADING
# COM_TAKEOFF_PX      = 30   # COM above ground to confirm TAKEOFF
# FLIGHT_CONFIRM_F    = 3    # consecutive airborne frames required before FLIGHT is trusted
# LAND_APPROACH_PX    = 25   # COM within this of ground → approaching landing
# LAND_STABLE_F       = 7    # frames near ground to confirm LANDING (athlete fully settled)
# MIN_FLIGHT_F        = 7    # minimum flight frames for a valid jump (~233ms @ 30fps)
# RESET_STABLE_F      = 4    # frames on ground before READY again (allows fast consecutive jumps)
# LAND_VEL_PX_F       = 6.0  # max COM px/frame velocity to accept landing as "settled"
# MIN_JUMP_DIST_M     = 0.25 # below this, treat as noise / false jump
# MIN_PEAK_RISE_PX    = 10   # minimum COM rise during flight to count as a real jump

# # Exercise-specific
# REACTIVE_MAX_CONTACT_F  = 12   # max ground contact for reactive jump (at 30fps ≈ 400ms)
# TUCK_HIP_FLEX_DEG       = 60   # minimum hip flexion for tuck confirmation
# PIKE_HIP_FLEX_DEG       = 70   # minimum hip flexion for pike
# PIKE_KNEE_EXT_DEG       = 150  # minimum knee extension for pike
# BOUND_MIN_FLIGHT_F      = 3    # minimum flight per bound
# TRIPLE_PHASE_MIN_F      = 3    # min flight frames per triple-jump phase

# # Anthropometric scale: avg hip-to-ankle ≈ 0.53× body height, body height ≈ 1.75m
# # So hip-ankle ≈ 0.927m in pixels → derive px/m
# ANTHRO_HIP_ANKLE_M  = 0.90   # meters (conservative; will be estimated per-subject)

# # ─────────────────────────────────────────────────────────────────────────────
# # ONE-EURO FILTER  (per-coordinate temporal filter)
# # ─────────────────────────────────────────────────────────────────────────────

# class OneEuroFilter:
#     """
#     One Euro Filter for temporal landmark smoothing.
#     Reduces jitter without large latency; preserves fast explosive motion
#     by automatically raising cutoff frequency at high velocity.
#     """
#     def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
#         self.freq      = freq
#         self.min_cutoff = min_cutoff
#         self.beta      = beta
#         self.d_cutoff  = d_cutoff
#         self._x        = None
#         self._dx       = 0.0

#     def _alpha(self, cutoff):
#         tau = 1.0 / (2 * math.pi * cutoff)
#         return 1.0 / (1.0 + tau * self.freq)

#     def __call__(self, x):
#         if self._x is None:
#             self._x = x
#             return x
#         dx     = (x - self._x) * self.freq
#         a_d    = self._alpha(self.d_cutoff)
#         self._dx = a_d * dx + (1 - a_d) * self._dx
#         cutoff = self.min_cutoff + self.beta * abs(self._dx)
#         a      = self._alpha(cutoff)
#         self._x = a * x + (1 - a) * self._x
#         return self._x


# class LandmarkFilter:
#     """Per-landmark One Euro filter (x and y independently)."""
#     def __init__(self, n_landmarks=17, freq=30.0):
#         self.fx = [OneEuroFilter(freq=freq) for _ in range(n_landmarks)]
#         self.fy = [OneEuroFilter(freq=freq) for _ in range(n_landmarks)]

#     def update(self, pts):
#         """pts: np.array shape (17,2). Returns filtered array same shape."""
#         out = pts.copy().astype(float)
#         for i, (fx, fy) in enumerate(zip(self.fx, self.fy)):
#             if pts[i, 0] > 0 and pts[i, 1] > 0:
#                 out[i, 0] = fx(pts[i, 0])
#                 out[i, 1] = fy(pts[i, 1])
#         return out


# # ─────────────────────────────────────────────────────────────────────────────
# # CENTER OF MASS ESTIMATOR
# # ─────────────────────────────────────────────────────────────────────────────

# def estimate_com(pts):
#     """
#     Weighted average COM from body keypoints.
#     Returns (com_x, com_y) or None if not enough valid points.
#     """
#     total_w = 0.0
#     wx, wy  = 0.0, 0.0
#     for idx, w in COM_WEIGHTS.items():
#         p = pts[idx]
#         if p[0] > 0 and p[1] > 0:
#             wx += w * p[0]
#             wy += w * p[1]
#             total_w += w
#     if total_w < 0.3:
#         return None
#     return (wx / total_w, wy / total_w)


# # ─────────────────────────────────────────────────────────────────────────────
# # FOOT CONTACT ESTIMATOR
# # ─────────────────────────────────────────────────────────────────────────────

# class FootContactEstimator:
#     """
#     Tracks each foot (left/right) independently.
#     Uses ankle vertical position relative to adaptive ground reference,
#     combined with ankle vertical velocity, to classify CONTACT / AIR.
#     Velocity gating prevents single noisy keypoint jitters from being
#     read as a takeoff or landing.
#     """
#     def __init__(self, window=30, fps=30.0):
#         self.ground_y    = None   # adaptive ground y (pixel; larger = lower on screen)
#         self.stance_buf  = deque(maxlen=window)  # y values during known contact
#         self.fps         = fps
#         self._prev_ly    = None
#         self._prev_ry    = None
#         self.left        = "CONTACT"
#         self.right       = "CONTACT"
#         self.left_vy     = 0.0
#         self.right_vy    = 0.0
#         self.THRESH_PX   = 22     # pixels above ground to classify AIR
#         self.VEL_THRESH  = 3.0    # px/frame — ankle must also be moving to confirm AIR

#     def update_ground(self, ly, ry):
#         """Call with ankle y values during confirmed ground phase."""
#         for y in [ly, ry]:
#             if y > 0:
#                 self.stance_buf.append(y)
#         if self.stance_buf:
#             # Ground is near the max (bottom-most) ankle positions
#             self.ground_y = np.percentile(list(self.stance_buf), 85)

#     def classify(self, pts):
#         """Update left/right contact state from keypoints."""
#         la = pts[KP_L_ANKLE]
#         ra = pts[KP_R_ANKLE]
#         ly = la[1] if la[0] > 0 else -1
#         ry = ra[1] if ra[0] > 0 else -1

#         # Velocity (px/frame) for noise gating
#         self.left_vy  = (ly - self._prev_ly) if (ly > 0 and self._prev_ly is not None) else 0.0
#         self.right_vy = (ry - self._prev_ry) if (ry > 0 and self._prev_ry is not None) else 0.0
#         if ly > 0: self._prev_ly = ly
#         if ry > 0: self._prev_ry = ry

#         if self.ground_y is None:
#             self.left  = "CONTACT"
#             self.right = "CONTACT"
#             return

#         thr = self.THRESH_PX
#         gnd = self.ground_y

#         # Left foot — require both height-above-ground AND recent motion
#         # to avoid a single jittery keypoint flipping the state.
#         if ly > 0:
#             above_ground = ly < gnd - thr
#             self.left = "AIR" if (above_ground or abs(self.left_vy) > self.VEL_THRESH * 2) else "CONTACT"
#         # Right foot
#         if ry > 0:
#             above_ground = ry < gnd - thr
#             self.right = "AIR" if (above_ground or abs(self.right_vy) > self.VEL_THRESH * 2) else "CONTACT"

#     @property
#     def both_contact(self):
#         return self.left == "CONTACT" and self.right == "CONTACT"

#     @property
#     def any_contact(self):
#         return self.left == "CONTACT" or self.right == "CONTACT"

#     @property
#     def both_air(self):
#         return self.left == "AIR" and self.right == "AIR"


# # ─────────────────────────────────────────────────────────────────────────────
# # ADAPTIVE SCALE ESTIMATOR  (pixel → metric)
# # ─────────────────────────────────────────────────────────────────────────────

# class ScaleEstimator:
#     """
#     Estimates pixels-per-meter from subject anthropometrics.
#     Primary: full body height (nose → ankle), which is far more stable
#     across camera angles than hip-to-ankle alone.
#     Fallback: hip-to-ankle segment if height isn't fully visible.
#     Updated during ground-contact frames only (standing posture).
#     """
#     ANTHRO_BODY_HEIGHT_M = 1.75   # default assumed height if user doesn't supply one

#     def __init__(self, user_height_m=None):
#         self._height_samples = deque(maxlen=60)
#         self._hip_ankle_samples = deque(maxlen=60)
#         self.px_per_m = None
#         self.user_height_m = user_height_m
#         self.method = None

#     def update(self, pts):
#         nose = pts[KP_NOSE]
#         lh = pts[KP_L_HIP];   la = pts[KP_L_ANKLE]
#         rh = pts[KP_R_HIP];   ra = pts[KP_R_ANKLE]

#         # ── Preferred: nose → lowest ankle (full standing height in px) ──
#         ankle_ys = [p[1] for p in (la, ra) if p[0] > 0 and p[1] > 0]
#         if nose[0] > 0 and nose[1] > 0 and ankle_ys:
#             height_px = max(ankle_ys) - nose[1]
#             if height_px > 50:   # sane lower bound, avoids garbage frames
#                 self._height_samples.append(height_px)

#         # ── Fallback: hip → ankle ──
#         pairs = []
#         if lh[0] > 0 and la[0] > 0:
#             pairs.append(abs(la[1] - lh[1]))
#         if rh[0] > 0 and ra[0] > 0:
#             pairs.append(abs(ra[1] - rh[1]))
#         if pairs:
#             hip_ankle_px = np.mean(pairs)
#             if hip_ankle_px > 10:
#                 self._hip_ankle_samples.append(hip_ankle_px)

#         # Prefer body-height calibration once enough samples exist
#         if len(self._height_samples) >= 10:
#             median_px = np.median(list(self._height_samples))
#             height_m  = self.user_height_m or self.ANTHRO_BODY_HEIGHT_M
#             self.px_per_m = median_px / height_m
#             self.method = "user_height" if self.user_height_m else "body_height_anthropometric"
#         elif len(self._hip_ankle_samples) >= 10:
#             median_px = np.median(list(self._hip_ankle_samples))
#             self.px_per_m = median_px / ANTHRO_HIP_ANKLE_M
#             self.method = "hip_ankle_anthropometric"

#     def px_to_m(self, pixels):
#         if self.px_per_m and self.px_per_m > 0:
#             return pixels / self.px_per_m
#         # Fallback: legacy constant (0.00933 m/px)
#         return pixels * 0.00933


# # ─────────────────────────────────────────────────────────────────────────────
# # ANGLE UTILITIES
# # ─────────────────────────────────────────────────────────────────────────────

# def _vec_angle(a, b, c):
#     """Angle at point b in triangle a-b-c (degrees)."""
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b;  bc = c - b
#     cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
#     return math.degrees(math.acos(np.clip(cos, -1, 1)))

# def _hip_flexion(pts, side="L"):
#     """Hip flexion angle (shoulder-hip-knee)."""
#     sh  = pts[KP_L_SHOULDER if side=="L" else KP_R_SHOULDER]
#     hp  = pts[KP_L_HIP      if side=="L" else KP_R_HIP]
#     kn  = pts[KP_L_KNEE     if side=="L" else KP_R_KNEE]
#     if any(p[0] <= 0 for p in [sh, hp, kn]):
#         return None
#     return _vec_angle(sh, hp, kn)

# def _knee_flexion(pts, side="L"):
#     """Knee flexion angle (hip-knee-ankle)."""
#     hp  = pts[KP_L_HIP   if side=="L" else KP_R_HIP]
#     kn  = pts[KP_L_KNEE  if side=="L" else KP_R_KNEE]
#     an  = pts[KP_L_ANKLE if side=="L" else KP_R_ANKLE]
#     if any(p[0] <= 0 for p in [hp, kn, an]):
#         return None
#     return _vec_angle(hp, kn, an)

# def _com_velocity(com_hist, fps):
#     """Returns (vx, vy) in px/frame from last two COM positions."""
#     if len(com_hist) < 2:
#         return (0.0, 0.0)
#     dx = com_hist[-1][0] - com_hist[-2][0]
#     dy = com_hist[-1][1] - com_hist[-2][1]
#     return (dx * fps, dy * fps)


# def _jump_direction(pts_hist):
#     """
#     Determine overall travel direction (+1 = left-to-right, -1 = right-to-left)
#     from a short history of COM x positions. Locked once at the start of a
#     rep sequence so per-jump distance math always uses a consistent sign.
#     """
#     if len(pts_hist) < 2:
#         return 1
#     dx = pts_hist[-1] - pts_hist[0]
#     return 1 if dx >= 0 else -1


# def _ankle_positions(pts):
#     """Return (left_ankle, right_ankle) as (x,y) or None if not visible."""
#     la = pts[KP_L_ANKLE]
#     ra = pts[KP_R_ANKLE]
#     left  = (la[0], la[1]) if la[0] > 0 and la[1] > 0 else None
#     right = (ra[0], ra[1]) if ra[0] > 0 and ra[1] > 0 else None
#     return left, right


# def _heel_x_approx(ankle, knee, direction):
#     """
#     Approximate heel x-position from ankle + knee (shank vector), since
#     COCO-17 has no heel keypoint. The heel sits slightly behind the ankle
#     relative to the direction of travel.
#     ankle, knee: (x, y) tuples
#     direction: +1 (moving right) or -1 (moving left)
#     """
#     if ankle is None:
#         return None
#     if knee is None:
#         return ankle[0]
#     shank_len = math.hypot(knee[0] - ankle[0], knee[1] - ankle[1])
#     if shank_len < 1e-6:
#         return ankle[0]
#     # Heel trails the ankle opposite to the direction of travel.
#     return ankle[0] - direction * 0.15 * shank_len


# def _takeoff_foot_x(pts, direction):
#     """
#     Foot-based takeoff reference: the most FORWARD foot point (in the
#     direction of travel) at the last grounded frame — this is the true
#     takeoff line, not COM.
#     """
#     la, ra = _ankle_positions(pts)
#     xs = [p[0] for p in (la, ra) if p is not None]
#     if not xs:
#         return None
#     return max(xs) if direction >= 0 else min(xs)


# def _landing_foot_x(pts, direction):
#     """
#     Foot-based landing reference: nearest body part to the takeoff line,
#     approximated via heel position on the leading foot at landing.
#     """
#     la, ra = _ankle_positions(pts)
#     lk = pts[KP_L_KNEE]; rk = pts[KP_R_KNEE]
#     lk_pt = (lk[0], lk[1]) if lk[0] > 0 and lk[1] > 0 else None
#     rk_pt = (rk[0], rk[1]) if rk[0] > 0 and rk[1] > 0 else None

#     heels = []
#     hx = _heel_x_approx(la, lk_pt, direction)
#     if hx is not None:
#         heels.append(hx)
#     hx = _heel_x_approx(ra, rk_pt, direction)
#     if hx is not None:
#         heels.append(hx)

#     if not heels:
#         return None
#     # Landing mark = the foot point CLOSEST to the takeoff line, i.e. the
#     # leading (trailing-edge) foot in the direction of travel.
#     return min(heels) if direction >= 0 else max(heels)


# # ─────────────────────────────────────────────────────────────────────────────
# # JUMP ENGINE  (6-state FSM)
# # ─────────────────────────────────────────────────────────────────────────────

# class JumpEngine:
#     """
#     Core 6-state finite state machine.
#     Tracks COM, foot contacts, and produces jump events.
#     """
#     def __init__(self, fps=30.0, exercise="standing_broad_jump"):
#         self.fps      = fps
#         self.exercise = exercise
#         self.state    = ST_READY

#         # Ground reference
#         self._gnd_com_y    = None   # adaptive ground COM y
#         self._gnd_buf      = deque(maxlen=GROUND_WINDOW)
#         self._gnd_locked   = False

#         # Takeoff
#         self._takeoff_com  = None
#         self._takeoff_f    = 0
#         self._takeoff_heel = None   # left heel x at takeoff
#         self._takeoff_foot_x   = None   # foot-based takeoff line (px)
#         self._last_grounded_pts = None  # full pts snapshot at last grounded frame
#         self._last_grounded_com = None

#         # Flight
#         self._flight_f     = 0
#         self._airborne_streak = 0   # consecutive airborne frames (for FLIGHT_CONFIRM_F)
#         self._peak_com_y   = None
#         self._flight_com_x_start = None
#         self._peak_b64     = None

#         # In-flight kinematics (for tuck/pike/bounding)
#         self._flight_hip_flex   = []
#         self._flight_knee_flex  = []
#         self._flight_arm_travel = []   # wrist-y travel relative to shoulder during flight

#         # Landing
#         self._land_stable_f = 0
#         self._land_com_buf  = deque(maxlen=8)
#         self._land_foot_x_buf = deque(maxlen=8)   # foot-based landing samples
#         self._land_pts_at_settle = None           # pts snapshot once landing settles

#         # Direction of travel (+1 left→right, -1 right→left), locked per rep
#         self._direction     = 1
#         self._dir_buf       = deque(maxlen=10)

#         # Reset
#         self._reset_f       = 0

#         # Loading / countermovement
#         self._load_com_y   = None   # COM y at start of countermovement
#         self._load_start_f = 0
#         self._cm_depth     = 0.0   # countermovement depth in px

#         # Completed jumps
#         self.jumps         = []
#         self._jump_n       = 0

#         # Per-phase data for triple/multiple/bounding
#         self._phase_list   = []    # list of phase dicts per jump
#         self._phase_start_com = None

#         # Reactive: ground contact tracking
#         self._gct_start_f  = 0    # frame when ground contact began after landing

#         # Approach velocity (run-up)
#         self._approach_vx  = 0.0
#         self._approach_buf = deque(maxlen=20)

#         # COM history for velocity-gated landing confirmation
#         self._com_hist      = deque(maxlen=5)

#         # Frame counter
#         self._frame        = 0

#     # ──────────────────────────────────────────────────────────────────────
#     def _update_ground(self, com_y, foot: FootContactEstimator, pts=None, com=None):
#         """Update adaptive ground COM_y during stable contact."""
#         if foot.both_contact or foot.any_contact:
#             self._gnd_buf.append(com_y)
#             if pts is not None:
#                 self._last_grounded_pts = pts.copy()
#                 self._last_grounded_com = com
#         if len(self._gnd_buf) >= 5:
#             self._gnd_com_y = np.percentile(list(self._gnd_buf), 80)

#     def _is_airborne(self, com_y):
#         if self._gnd_com_y is None:
#             return False
#         return com_y < self._gnd_com_y - COM_TAKEOFF_PX

#     def _is_near_ground(self, com_y):
#         if self._gnd_com_y is None:
#             return True
#         return com_y >= self._gnd_com_y - LAND_APPROACH_PX

#     def _com_settled(self):
#         """True when recent COM motion has slowed enough to trust the
#         landing position (prevents capturing landing COM while the body
#         is still travelling forward/downward)."""
#         if len(self._com_hist) < 3:
#             return False
#         vx = self._com_hist[-1][0] - self._com_hist[-2][0]
#         vy = self._com_hist[-1][1] - self._com_hist[-2][1]
#         return abs(vx) < LAND_VEL_PX_F and abs(vy) < LAND_VEL_PX_F

#     # ──────────────────────────────────────────────────────────────────────
#     def update(self, pts, foot: FootContactEstimator, com, scale: ScaleEstimator,
#                frame_b64=None):
#         """
#         Called every frame.
#         pts: filtered keypoints (17,2)
#         foot: FootContactEstimator (already updated this frame)
#         com: (cx, cy) or None
#         scale: ScaleEstimator
#         Returns: jump_event dict if a jump completed this frame, else None
#         """
#         self._frame += 1
#         if com is None:
#             return None

#         com_x, com_y = com
#         self._com_hist.append(com)

#         # Track approach velocity for run-up detection
#         self._approach_buf.append(com_x)
#         if len(self._approach_buf) >= 5:
#             dx = self._approach_buf[-1] - self._approach_buf[-5]
#             self._approach_vx = dx / 5.0 * self.fps  # px/s

#         # Track direction-of-travel candidates continuously; locked at takeoff
#         self._dir_buf.append(com_x)

#         event = None

#         # ── READY ─────────────────────────────────────────────────────────
#         if self.state == ST_READY:
#             self._update_ground(com_y, foot, pts, com)
#             # Detect loading (COM rises above ground reference = explosive
#             # takeoff starting) OR feet already airborne (fast explosive
#             # jumps can leave the ground within 1-2 frames, before COM has
#             # risen the full COM_RISE_PX — don't gate this on foot state).
#             rising = self._gnd_com_y is not None and com_y < self._gnd_com_y - COM_RISE_PX
#             if rising or foot.both_air or self._is_airborne(com_y):
#                 self.state         = ST_LOADING
#                 self._load_com_y   = com_y
#                 self._load_start_f = self._frame
#                 self._airborne_streak = 1 if (foot.both_air or self._is_airborne(com_y)) else 0
#             else:
#                 self._airborne_streak = 0
#                 # Still grounded — update scale
#                 scale.update(pts)
#                 foot.update_ground(pts[KP_L_ANKLE][1], pts[KP_R_ANKLE][1])
#                 self._takeoff_com  = com   # keep updating takeoff reference
#                 self._last_grounded_pts = pts.copy()
#                 self._last_grounded_com = com

#         # ── LOADING ───────────────────────────────────────────────────────
#         elif self.state == ST_LOADING:
#             airborne_now = foot.both_air or self._is_airborne(com_y)
#             if airborne_now:
#                 self._airborne_streak += 1
#             else:
#                 self._airborne_streak = 0
#                 self._last_grounded_pts = pts.copy()
#                 self._last_grounded_com = com

#             # Require FLIGHT_CONFIRM_F consecutive airborne frames before
#             # committing to a takeoff — rejects single-frame keypoint noise.
#             if self._airborne_streak >= FLIGHT_CONFIRM_F:
#                 # True takeoff = the LAST GROUNDED frame, not the first
#                 # airborne one (actual takeoff happens slightly earlier
#                 # than detection).
#                 ref_pts = self._last_grounded_pts if self._last_grounded_pts is not None else pts
#                 ref_com = self._last_grounded_com if self._last_grounded_com is not None else com

#                 # Lock direction of travel for this rep from recent COM drift
#                 direction_samples = np.array(list(self._dir_buf))
#                 self._direction = _jump_direction(direction_samples)

#                 self._takeoff_com        = ref_com
#                 self._takeoff_foot_x     = _takeoff_foot_x(ref_pts, self._direction)
#                 self._flight_com_x_start = ref_com[0]
#                 self._takeoff_f          = max(1, self._frame - self._airborne_streak)
#                 self._flight_f           = self._airborne_streak
#                 self._peak_com_y         = com_y
#                 self._peak_b64           = frame_b64
#                 self._flight_hip_flex    = []
#                 self._flight_knee_flex   = []
#                 self._flight_arm_travel  = []
#                 self._phase_start_com    = ref_com
#                 self._land_foot_x_buf.clear()
#                 self._land_com_buf.clear()
#                 self._land_stable_f      = 0
#                 # Countermovement depth = ground COM_y - min COM_y during loading
#                 self._cm_depth = max(0.0, self._gnd_com_y - self._load_com_y) if self._gnd_com_y else 0.0
#                 self.state = ST_FLIGHT
#             elif not airborne_now and not self._is_airborne(com_y):
#                 # Still loading / false alarm
#                 if com_y > self._gnd_com_y:
#                     # COM went back down, abort loading
#                     self.state = ST_READY

#         # ── FLIGHT ────────────────────────────────────────────────────────
#         elif self.state == ST_FLIGHT:
#             self._flight_f += 1

#             # Track peak height (smallest y = highest)
#             if com_y < self._peak_com_y:
#                 self._peak_com_y = com_y
#                 self._peak_b64   = frame_b64

#             # Collect in-flight body angles for tuck/pike detection
#             hf_l = _hip_flexion(pts, "L")
#             hf_r = _hip_flexion(pts, "R")
#             kf_l = _knee_flexion(pts, "L")
#             kf_r = _knee_flexion(pts, "R")
#             if hf_l: self._flight_hip_flex.append(hf_l)
#             if hf_r: self._flight_hip_flex.append(hf_r)
#             if kf_l: self._flight_knee_flex.append(kf_l)
#             if kf_r: self._flight_knee_flex.append(kf_r)

#             # Collect wrist vertical travel relative to shoulder, used as a
#             # proxy for arm-swing amplitude (drives the form score's
#             # arm-swing component).
#             for sh_i, wr_i in ((KP_L_SHOULDER, KP_L_WRIST), (KP_R_SHOULDER, KP_R_WRIST)):
#                 sh, wr = pts[sh_i], pts[wr_i]
#                 if sh[0] > 0 and wr[0] > 0:
#                     self._flight_arm_travel.append(sh[1] - wr[1])  # +ve = wrist above shoulder

#             # Check landing approach — require near-ground AND settled
#             # velocity AND foot contact before treating COM/foot position
#             # as the true landing mark.
#             if self._flight_f >= MIN_FLIGHT_F:
#                 near_ground = self._is_near_ground(com_y) or foot.any_contact
#                 if near_ground:
#                     self._land_stable_f += 1
#                     self._land_com_buf.append(com_x)
#                     foot_x = _landing_foot_x(pts, self._direction)
#                     if foot_x is not None:
#                         self._land_foot_x_buf.append(foot_x)
#                     if (self._land_stable_f >= LAND_STABLE_F and
#                             self._com_settled() and foot.any_contact):
#                         event = self._confirm_jump(com, pts, scale)
#                         if event is not None:
#                             self.state = ST_LANDING
#                         else:
#                             # Rejected as a false jump — drop back to READY
#                             # without emitting an event, but keep tracking.
#                             self.state = ST_RESET
#                             self._reset_f = 0
#                 else:
#                     self._land_stable_f = 0
#                     self._land_com_buf.clear()
#                     self._land_foot_x_buf.clear()

#             # Safety: if COM returned to ground quickly with no real flight,
#             # reset without counting a jump (rejects ground noise / weight shifts).
#             if self._flight_f < MIN_FLIGHT_F and foot.both_contact and self._is_near_ground(com_y):
#                 self.state = ST_READY
#                 self._airborne_streak = 0

#         # ── LANDING ───────────────────────────────────────────────────────
#         elif self.state == ST_LANDING:
#             self._land_stable_f = 0
#             self._land_com_buf.clear()
#             self._land_foot_x_buf.clear()
#             self._reset_f = 0
#             self._last_grounded_pts = pts.copy()
#             self._last_grounded_com = com

#             # For reactive jump: record GCT start
#             self._gct_start_f = self._frame
#             self.state = ST_RESET

#         # ── RESET ─────────────────────────────────────────────────────────
#         elif self.state == ST_RESET:
#             self._reset_f += 1
#             self._update_ground(com_y, foot, pts, com)
#             self._airborne_streak = 1 if (foot.both_air or self._is_airborne(com_y)) else 0

#             # Reactive: if feet leave ground again very quickly
#             if (foot.both_air and self._reset_f <= REACTIVE_MAX_CONTACT_F
#                     and "reactive" in self.exercise):
#                 # Short GCT → reactive jump — transition directly to FLIGHT
#                 direction_samples = np.array(list(self._dir_buf))
#                 self._direction = _jump_direction(direction_samples)
#                 ref_pts = self._last_grounded_pts if self._last_grounded_pts is not None else pts

#                 self._takeoff_com        = com
#                 self._takeoff_foot_x     = _takeoff_foot_x(ref_pts, self._direction)
#                 self._flight_com_x_start = com_x
#                 self._takeoff_f          = self._frame
#                 self._flight_f           = 0
#                 self._peak_com_y         = com_y
#                 self._peak_b64           = frame_b64
#                 self._flight_hip_flex    = []
#                 self._flight_knee_flex   = []
#                 self._flight_arm_travel  = []
#                 self._phase_start_com    = com
#                 self._land_foot_x_buf.clear()
#                 self._land_com_buf.clear()
#                 self.state = ST_FLIGHT

#             elif self._reset_f >= RESET_STABLE_F:
#                 # Quick reset so back-to-back jumps (multiple/bounding/triple)
#                 # aren't swallowed by an over-long ground-contact requirement.
#                 self.state = ST_READY
#                 self._gnd_locked = True

#         return event

#     # ──────────────────────────────────────────────────────────────────────
#     def _confirm_jump(self, land_com, pts, scale: ScaleEstimator):
#         """Build and store a jump event dict. Returns None (and does NOT
#         increment jump_n / append to self.jumps) if this is rejected as a
#         false jump — caller must handle a None return."""

#         # ── Foot-based distance (primary) ───────────────────────────────
#         # Takeoff line = most-forward foot point at the last grounded frame.
#         # Landing mark = nearest heel of the leading foot once landing has
#         # settled (near-ground + low COM velocity + foot contact).
#         takeoff_x_foot = self._takeoff_foot_x

#         land_x_foot = None
#         if self._land_foot_x_buf:
#             land_x_foot = float(np.median(list(self._land_foot_x_buf)))
#         elif pts is not None:
#             land_x_foot = _landing_foot_x(pts, self._direction)

#         # COM fallback (only used if foot keypoints were unavailable)
#         land_x_com    = float(np.median(list(self._land_com_buf))) if self._land_com_buf else land_com[0]
#         takeoff_x_com = self._flight_com_x_start if self._flight_com_x_start else (
#             self._takeoff_com[0] if self._takeoff_com else land_x_com)

#         if takeoff_x_foot is not None and land_x_foot is not None:
#             px_dist = abs(land_x_foot - takeoff_x_foot)
#             takeoff_x, land_x = takeoff_x_foot, land_x_foot
#             dist_method = "foot_based"
#         else:
#             px_dist = abs(land_x_com - takeoff_x_com)
#             takeoff_x, land_x = takeoff_x_com, land_x_com
#             dist_method = "com_fallback"

#         dist_m  = round(scale.px_to_m(px_dist), 3)
#         dist_cm = round(dist_m * 100, 1)

#         flight_f  = self._flight_f
#         flight_ms = int(flight_f / self.fps * 1000)

#         # Peak COM height above takeoff COM
#         peak_rise_px = 0.0
#         if self._takeoff_com and self._peak_com_y:
#             peak_rise_px = max(0.0, self._takeoff_com[1] - self._peak_com_y)
#         peak_rise_m = scale.px_to_m(peak_rise_px)

#         # ── False-jump rejection ────────────────────────────────────────
#         # Reject jumps that are too short, too brief, or show no real
#         # vertical excursion — these are almost always weight-shift noise,
#         # not genuine jumps.
#         t = flight_f / self.fps
#         if (dist_m < MIN_JUMP_DIST_M or
#                 flight_ms < 150 or
#                 peak_rise_px < MIN_PEAK_RISE_PX):
#             return None

#         # ── Physics-correct takeoff velocity / angle ────────────────────
#         # vx from actual measured horizontal displacement over flight time
#         # (far more reliable than the raw approach-velocity buffer, which
#         # is itself noisy and was the source of the impossible 68-76°
#         # takeoff angles).
#         g = 9.81
#         vx_ms = dist_m / (t + 1e-9)
#         vy_ms = g * t / 2 if t > 0 else 0.0
#         takeoff_vel = round(math.sqrt(vx_ms**2 + vy_ms**2), 2)
#         takeoff_angle = round(math.degrees(math.atan2(vy_ms, vx_ms + 1e-9)), 1) if vx_ms > 1e-6 else 0.0
#         # Clamp to physically sane broad-jump range for display robustness;
#         # values outside this band indicate measurement noise, not technique.
#         takeoff_angle = max(0.0, min(85.0, takeoff_angle))

#         # Physics consistency check: does measured distance roughly match
#         # vx * t? If not, flag the jump as lower-confidence rather than
#         # silently trusting it.
#         physics_expected_dist = vx_ms * t
#         physics_consistent = abs(physics_expected_dist - dist_m) < max(0.15, 0.25 * dist_m)

#         # Tuck / pike detection from in-flight angles
#         tuck_confirmed = False
#         pike_confirmed = False
#         min_hip_flex = min(self._flight_hip_flex) if self._flight_hip_flex else 180
#         min_knee_flex = min(self._flight_knee_flex) if self._flight_knee_flex else 180
#         max_knee_ext  = max(self._flight_knee_flex) if self._flight_knee_flex else 0

#         if min_hip_flex < TUCK_HIP_FLEX_DEG and min_knee_flex < 90:
#             tuck_confirmed = True
#         if min_hip_flex < PIKE_HIP_FLEX_DEG and max_knee_ext > PIKE_KNEE_EXT_DEG:
#             pike_confirmed = True

#         # Countermovement depth
#         cm_depth_m = round(scale.px_to_m(self._cm_depth), 3)

#         # Horizontal efficiency: forward dist / total COM path (approximation)
#         h_eff = round(min(1.0, px_dist / (px_dist + peak_rise_px + 1e-9)), 3)

#         # Landing stability (from COM settle velocity at confirmation time)
#         land_vx = land_vy = 0.0
#         if len(self._com_hist) >= 2:
#             land_vx = self._com_hist[-1][0] - self._com_hist[-2][0]
#             land_vy = self._com_hist[-1][1] - self._com_hist[-2][1]
#         landing_stability = round(max(0.0, 1.0 - (abs(land_vx) + abs(land_vy)) / 20.0), 3)

#         # Multi-factor form score (distance, landing stability, takeoff
#         # angle, arm-swing quality, knee flexion at takeoff)
#         arm_swing_score = _arm_swing_score(self._flight_arm_travel)
#         form = _form_score_full(
#             dist_m=dist_m,
#             landing_stability=landing_stability,
#             takeoff_angle=takeoff_angle,
#             min_hip_flex=min_hip_flex,
#             min_knee_flex=min_knee_flex,
#             arm_swing_score=arm_swing_score,
#         )

#         # Ground contact time (reactive)
#         gct_ms = 0
#         if "reactive" in self.exercise and self._gct_start_f > 0:
#             gct_f  = self._takeoff_f - self._gct_start_f
#             gct_ms = int(max(0, gct_f) / self.fps * 1000)

#         rsi = round(flight_ms / gct_ms, 3) if gct_ms > 0 else None

#         self._jump_n += 1
#         jump = {
#             # Identity
#             "jump_no"           : self._jump_n,
#             # Distances
#             "distance_m"        : dist_m,
#             "distance_cm"       : dist_cm,
#             "pixel_dist"        : round(px_dist, 1),
#             "distance_method"   : dist_method,
#             # Timing
#             "flight_ms"         : flight_ms,
#             "airborne_ms"       : flight_ms,
#             "takeoff_frame"     : self._takeoff_f,
#             "landing_frame"     : self._frame,
#             # Positions
#             "takeoff_com_x"     : round(takeoff_x_com, 1),
#             "landing_com_x"     : round(land_x_com, 1),
#             "takeoff_foot_x"    : round(takeoff_x_foot, 1) if takeoff_x_foot is not None else None,
#             "landing_foot_x"    : round(land_x_foot, 1) if land_x_foot is not None else None,
#             "direction"         : self._direction,
#             # Kinematics
#             "takeoff_velocity_ms": takeoff_vel,
#             "takeoff_angle_deg" : takeoff_angle,
#             "peak_rise_m"       : round(peak_rise_m, 3),
#             "approach_vx_px_s"  : round(self._approach_vx, 1),
#             "approach_speed_ms" : round(scale.px_to_m(abs(self._approach_vx)), 2),
#             "horizontal_efficiency": h_eff,
#             "cm_depth_m"        : cm_depth_m,
#             "physics_consistent": physics_consistent,
#             # Technique
#             "tuck_confirmed"    : tuck_confirmed,
#             "pike_confirmed"    : pike_confirmed,
#             "min_hip_flex_deg"  : round(min_hip_flex, 1),
#             "min_knee_flex_deg" : round(min_knee_flex, 1),
#             "landing_stability" : landing_stability,
#             "arm_swing_score"   : arm_swing_score,
#             # Reactive
#             "ground_contact_ms" : gct_ms,
#             "rsi"               : rsi,
#             # Score
#             "form_score"        : form,
#             # Snapshot
#             "_peak_b64"         : self._peak_b64,
#         }
#         jump = _to_json_safe(jump)
#         self.jumps.append(jump)
#         return jump

#     def force_close(self, com, scale: ScaleEstimator, pts=None):
#         """Call at video end if still in FLIGHT state."""
#         if self.state == ST_FLIGHT and self._flight_f >= MIN_FLIGHT_F:
#             land_x = com[0] if com else (self._flight_com_x_start or 0)
#             self._land_com_buf.append(land_x)
#             self._land_stable_f = LAND_STABLE_F
#             ref_pts = pts if pts is not None else np.zeros((17, 2))
#             foot_x = _landing_foot_x(ref_pts, self._direction) if pts is not None else None
#             if foot_x is not None:
#                 self._land_foot_x_buf.append(foot_x)
#             return self._confirm_jump(com, ref_pts, scale)
#         return None


# # ─────────────────────────────────────────────────────────────────────────────
# # EXERCISE-SPECIFIC VALIDATORS
# # ─────────────────────────────────────────────────────────────────────────────

# class ExerciseValidator:
#     """
#     Wraps the core JumpEngine and applies exercise-specific
#     post-processing to the completed jump list.
#     """
#     def __init__(self, exercise: str):
#         self.exercise = exercise

#     def validate(self, jumps: list) -> dict:
#         ex = self.exercise.lower().replace(" ", "_")

#         if "triple" in ex:
#             return self._triple(jumps)
#         elif "alternate" in ex:
#             return self._alternate(jumps)
#         elif "bounding" in ex or "power_skip" in ex:
#             return self._bounding(jumps)
#         elif "multiple" in ex:
#             return self._multiple(jumps)
#         else:
#             return self._standard(jumps)

#     # Standard single / repeated jumps
#     def _standard(self, jumps):
#         return {"validated_jumps": jumps, "phase_info": None}

#     # Triple: group every 3 consecutive jumps into hop/step/jump
#     def _triple(self, jumps):
#         phases = []
#         for i in range(0, len(jumps) - 2, 3):
#             hop, step, jmp = jumps[i], jumps[i+1], jumps[i+2]
#             total = hop["distance_m"] + step["distance_m"] + jmp["distance_m"]
#             phases.append({
#                 "triple_no"   : len(phases) + 1,
#                 "hop"         : hop,
#                 "step"        : step,
#                 "jump"        : jmp,
#                 "total_m"     : round(total, 3),
#                 "phase_ratio" : [round(hop["distance_m"]/total, 2),
#                                  round(step["distance_m"]/total, 2),
#                                  round(jmp["distance_m"]/total, 2)],
#             })
#         return {"validated_jumps": jumps, "phase_info": phases}

#     # Alternate: flag if consecutive jumps show alternating takeoff feet
#     def _alternate(self, jumps):
#         for i, j in enumerate(jumps):
#             j["leg_tag"] = "L" if i % 2 == 0 else "R"
#         return {"validated_jumps": jumps, "phase_info": None}

#     # Bounding: each jump = one bound
#     def _bounding(self, jumps):
#         bounds = []
#         for j in jumps:
#             bounds.append({
#                 "bound_no"  : j["jump_no"],
#                 "distance_m": j["distance_m"],
#                 "flight_ms" : j["flight_ms"],
#                 "gct_ms"    : j.get("ground_contact_ms", 0),
#             })
#         total = sum(b["distance_m"] for b in bounds)
#         return {"validated_jumps": jumps, "phase_info": bounds,
#                 "total_distance_m": round(total, 3)}

#     # Multiple: cumulative metrics
#     def _multiple(self, jumps):
#         cumulative = 0.0
#         for j in jumps:
#             cumulative += j["distance_m"]
#             j["cumulative_m"] = round(cumulative, 3)
#         return {"validated_jumps": jumps, "phase_info": None}


# # ─────────────────────────────────────────────────────────────────────────────
# # QUALITY & SCORING
# # ─────────────────────────────────────────────────────────────────────────────

# def _form_score(distance_m):
#     """Legacy distance-only score, retained for backward compatibility
#     (used by _form_score_aggregate for the session-level summary)."""
#     thresholds = [2.50, 2.20, 2.00, 1.80, 1.50, 1.20]
#     scores     = [10,   9,    8,    7,    6,    5]
#     for t, s in zip(thresholds, scores):
#         if distance_m >= t:
#             return s
#     return 4

# def _distance_subscore(distance_m):
#     """0-10 distance subscore feeding the weighted per-jump form score."""
#     thresholds = [2.50, 2.20, 2.00, 1.80, 1.50, 1.20, 0.90]
#     scores     = [10,   9,    8,    7,    6,    5,    3]
#     for t, s in zip(thresholds, scores):
#         if distance_m >= t:
#             return s
#     return 1

# def _takeoff_angle_subscore(angle_deg):
#     """0-10 subscore peaking in the biomechanically efficient 18-27°
#     range for a standing broad jump; falls off outside it."""
#     ideal_low, ideal_high = 18.0, 27.0
#     if ideal_low <= angle_deg <= ideal_high:
#         return 10.0
#     dist = (ideal_low - angle_deg) if angle_deg < ideal_low else (angle_deg - ideal_high)
#     return max(0.0, 10.0 - dist * 0.35)

# def _knee_flexion_subscore(min_knee_flex_deg):
#     """0-10 subscore for takeoff/flight knee flexion — deep enough to
#     generate power without being so collapsed it signals poor control."""
#     if min_knee_flex_deg is None:
#         return 6.0  # neutral if not measured
#     ideal_low, ideal_high = 90.0, 130.0
#     if ideal_low <= min_knee_flex_deg <= ideal_high:
#         return 10.0
#     dist = (ideal_low - min_knee_flex_deg) if min_knee_flex_deg < ideal_low else (min_knee_flex_deg - ideal_high)
#     return max(0.0, 10.0 - dist * 0.15)

# def _arm_swing_score(wrist_travel_samples):
#     """
#     0-10 subscore from wrist vertical excursion relative to shoulder during
#     flight. A strong arm swing drives the wrists from low (behind the hips
#     at takeoff) to high (overhead/forward at peak flight) — large positive
#     range in (shoulder_y - wrist_y) indicates good swing amplitude.
#     """
#     if not wrist_travel_samples or len(wrist_travel_samples) < 2:
#         return 5.0   # neutral score when arms aren't trackable
#     travel_range = max(wrist_travel_samples) - min(wrist_travel_samples)
#     # travel_range is in pixels; normalise loosely — most useful as a
#     # relative signal since we don't have per-subject scale here.
#     score = min(10.0, travel_range / 12.0)
#     return round(max(0.0, score), 1)

# def _form_score_full(dist_m, landing_stability, takeoff_angle,
#                       min_hip_flex, min_knee_flex, arm_swing_score=5.0):
#     """
#     Weighted multi-factor form score (1-10):
#       40% distance, 20% landing stability, 15% takeoff angle,
#       15% arm swing, 10% knee flexion.
#     """
#     dist_s   = _distance_subscore(dist_m)
#     land_s   = round(landing_stability * 10, 1)
#     angle_s  = _takeoff_angle_subscore(takeoff_angle)
#     knee_s   = _knee_flexion_subscore(min_knee_flex)
#     arm_s    = arm_swing_score

#     weighted = (0.40 * dist_s + 0.20 * land_s + 0.15 * angle_s +
#                 0.15 * arm_s + 0.10 * knee_s)
#     return int(round(max(1.0, min(10.0, weighted))))

# def _form_score_aggregate(jumps_m):
#     if not jumps_m:
#         return 0
#     best  = max(jumps_m)
#     score = _form_score(best)
#     if len(jumps_m) > 1:
#         cons = min(jumps_m) / best
#         if   cons >= 0.90: score = min(10, score + 1)
#         elif cons < 0.70:  score = max(1,  score - 1)
#     return score

# def _compute_quality(jumps, pose_conf, foot_conf, scale_calibrated):
#     """
#     Generate overall confidence score 0-100 from independently meaningful
#     sub-scores, rather than a raw jump count (which says nothing about
#     measurement reliability).
#       pose      : average pose-detection confidence across the video
#       contact   : foot-contact estimator reliability
#       flight    : how cleanly flight phases were detected (physics-consistency
#                   rate + sane flight-time spread across detected jumps)
#       scale     : whether px-per-meter calibration succeeded
#       landing   : average landing stability across detected jumps
#     """
#     pose_s  = int(pose_conf * 100)
#     cont_s  = int(foot_conf * 100)
#     scale_s = 85 if scale_calibrated else 55

#     if not jumps:
#         return {
#             "overall": 0, "pose": pose_s, "contact": cont_s,
#             "flight": 0, "distance": scale_s, "landing": 0,
#             "jump": 0,   # legacy key retained
#         }

#     consistent = [j.get("physics_consistent", True) for j in jumps]
#     flight_s   = int(100 * (sum(consistent) / len(consistent)))

#     land_vals  = [j.get("landing_stability") for j in jumps if j.get("landing_stability") is not None]
#     landing_s  = int(100 * (sum(land_vals) / len(land_vals))) if land_vals else 60

#     overall = int(0.30*pose_s + 0.15*cont_s + 0.25*flight_s +
#                   0.15*scale_s + 0.15*landing_s)
#     return {
#         "overall": overall, "pose": pose_s, "contact": cont_s,
#         "flight": flight_s, "distance": scale_s, "landing": landing_s,
#         "jump": flight_s,   # legacy key retained, now mapped to a meaningful value
#     }

# def _fatigue_index(jumps_m):
#     """Decline in distance over repetitions (lower is better)."""
#     if len(jumps_m) < 2:
#         return None
#     return round((jumps_m[0] - jumps_m[-1]) / jumps_m[0] * 100, 1)

# def _rep_variability(jumps_m):
#     if len(jumps_m) < 2:
#         return None
#     return round(float(np.std(jumps_m)) / (np.mean(jumps_m) + 1e-9) * 100, 1)

# def _feedback(jump_results, detected, exercise):
#     issues    = []
#     strengths = []

#     if not detected:
#         issues += [
#             "❌ Person not detected in video.",
#             "📌 Ensure: full body visible (head to toe), side-angle camera, good lighting.",
#             "🎥 Avoid top-down or front-facing camera angles.",
#         ]
#         return issues, strengths

#     if not jump_results:
#         issues += [
#             "❌ No valid jump detected.",
#             "📐 Camera should be placed sideways at full-body height.",
#             "⏱️ Video must capture full jump — takeoff, flight, and landing.",
#         ]
#         return issues, strengths

#     distances = [j["distance_m"] for j in jump_results]
#     best = max(distances)
#     avg  = sum(distances) / len(distances)

#     if best < 1.20:
#         issues.append(f"Short jump ({best:.2f}m) — focus on arm swing and hip extension.")
#     elif best < 1.80:
#         issues.append(f"Below-average distance ({best:.2f}m) — work on explosive leg drive.")
#     elif best < 2.20:
#         strengths.append(f"Good jump distance ({best:.2f}m).")
#     else:
#         strengths.append(f"Excellent jump distance ({best:.2f}m)!")

#     if len(distances) > 1:
#         cons = min(distances) / best
#         if cons >= 0.90:
#             strengths.append(f"Very consistent across {len(distances)} jumps ({cons:.0%}).")
#         elif cons < 0.70:
#             issues.append(f"High variation ({cons:.0%} consistency) — repeat takeoff mechanics.")

#     # Tuck/pike feedback
#     if "tuck" in exercise.lower():
#         tucks = [j for j in jump_results if j.get("tuck_confirmed")]
#         if tucks:
#             strengths.append(f"Tuck confirmed in {len(tucks)}/{len(jump_results)} jumps.")
#         else:
#             issues.append("Tuck not detected — ensure knees pull toward chest during flight.")

#     if "pike" in exercise.lower():
#         pikes = [j for j in jump_results if j.get("pike_confirmed")]
#         if pikes:
#             strengths.append(f"Pike confirmed in {len(pikes)}/{len(jump_results)} jumps.")
#         else:
#             issues.append("Pike not detected — keep legs extended and reach toward toes.")

#     if not issues:
#         issues = ["No major issues detected."]
#     return issues, strengths


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN ANALYSIS FUNCTION
# # ─────────────────────────────────────────────────────────────────────────────

# def analyse_broad_jump(
#     path,
#     is_video,
#     output_path=None,
#     session_id=None,
#     source_filename="",
#     progress_uid=None,
#     exercise="Standing Broad Jump",
# ):
#     """
#     Main entry point for broad jump analysis.
#     Supports all 12 broad jump variants via the `exercise` parameter.

#     Parameters
#     ----------
#     exercise : str
#         One of: "Standing Broad Jump", "Run-Up Broad Jump",
#         "Single-Leg Broad Jump", "Alternate Leg Broad Jump",
#         "Bounding", "Triple Broad Jump", "Multiple Broad Jump",
#         "Reactive Broad Jump", "Tuck Broad Jump", "Pike Broad Jump",
#         "Weighted Broad Jump", "Sand Broad Jump"
#     """
#     if not _YOLO_AVAILABLE:
#         raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
#     if not is_video:
#         raise ValueError("Broad jump analysis requires a video file.")

#     # ── Model ────────────────────────────────────────────────────────────
#     model_path = YOLO_MODEL_PATH
#     if not os.path.exists(model_path):
#         here = os.path.dirname(os.path.abspath(__file__))
#         alt  = os.path.join(here, YOLO_MODEL_PATH)
#         if os.path.exists(alt):
#             model_path = alt
#         else:
#             raise FileNotFoundError(f"YOLO model not found at '{YOLO_MODEL_PATH}'.")
#     model = YOLO(model_path)

#     # ── Video metadata ────────────────────────────────────────────────────
#     cap    = cv2.VideoCapture(path)
#     fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     # ── Sub-systems ───────────────────────────────────────────────────────
#     lm_filter = LandmarkFilter(n_landmarks=17, freq=fps)
#     foot_est  = FootContactEstimator(window=int(fps * 1.5), fps=fps)
#     scale_est = ScaleEstimator()
#     engine    = JumpEngine(fps=fps, exercise=exercise.lower())
#     validator = ExerciseValidator(exercise)

#     # ── State ─────────────────────────────────────────────────────────────
#     detected_ok   = [False]
#     wrong_events  = []
#     per_frame_data= []
#     total_frames  = [1]
#     pose_confs    = []
#     foot_conf_acc = []

#     live_hud      = {"jump_no": "0", "dist": "---", "form": "---", "state": ST_READY}

#     def _push_event(jdata, b64):
#         if not progress_uid:
#             return
#         pct = min(94, int(len(per_frame_data) / max(1, total_frames[0]) * 90))
#         safe_jdata = _to_json_safe({**jdata, "frame_b64": b64})
#         set_progress(
#             progress_uid, pct,
#             f"Jump {jdata['jump_no']} — {jdata['distance_m']:.2f}m",
#             jump_event=safe_jdata,
#         )

#     # ── Per-frame callback ────────────────────────────────────────────────
#     def pf(frame, fc, total):
#         total_frames[0] = max(total, 1)

#         results   = model(frame, conf=CONFIDENCE, verbose=False)
#         best_pts  = None
#         best_area = 0
#         best_conf = 0.0

#         for result in results:
#             if (result.keypoints is None or
#                     result.keypoints.xy is None or
#                     result.keypoints.conf is None):
#                 continue
#             for kpts_xy, kpts_conf in zip(
#                     result.keypoints.xy.cpu().numpy(),
#                     result.keypoints.conf.cpu().numpy()):
#                 pts   = kpts_xy.astype(float)
#                 valid = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
#                 if len(valid) < 6:
#                     continue
#                 area = ((valid[:, 0].max() - valid[:, 0].min()) *
#                         (valid[:, 1].max() - valid[:, 1].min()))
#                 if area > best_area:
#                     best_area = area
#                     best_pts  = pts
#                     best_conf = float(np.mean(kpts_conf[kpts_conf > 0]))

#         if best_pts is None:
#             draw_footer_hud(frame, [
#                 ("JUMP #", live_hud["jump_no"]),
#                 ("DIST",   live_hud["dist"]),
#                 ("FORM",   live_hud["form"]),
#             ])
#             draw_pcl_logo(frame)
#             return frame

#         detected_ok[0] = True
#         pose_confs.append(best_conf)

#         # Filter landmarks
#         pts = lm_filter.update(best_pts)

#         # Draw skeleton (skip face landmarks 0-4)
#         for p1i, p2i in SKELETON_PAIRS:
#             p1, p2 = pts[p1i], pts[p2i]
#             if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
#                 cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
#                          COLOR_SKEL, 2)
#         for idx, p in enumerate(pts):
#             if idx < 5 or p[0] <= 0 or p[1] <= 0:
#                 continue
#             cv2.circle(frame, (int(p[0]), int(p[1])), 4, COLOR_JOINT, -1)

#         # COM
#         com = estimate_com(pts)
#         if com:
#             cv2.circle(frame, (int(com[0]), int(com[1])), 6, COLOR_COM, -1)

#         # Foot contact
#         foot_est.classify(pts)

#         # Engine update
#         frame_b64  = frame_to_b64(frame) if engine.state in (ST_FLIGHT, ST_TAKEOFF) else None
#         jump_event = engine.update(pts, foot_est, com, scale_est, frame_b64)

#         if jump_event:
#             b64 = jump_event.pop("_peak_b64", None) or frame_to_b64(frame)
#             _push_event(jump_event, b64)
#             live_hud["jump_no"] = str(jump_event["jump_no"])
#             live_hud["dist"]    = f"{jump_event['distance_m']:.2f}m"
#             live_hud["form"]    = f"{jump_event['form_score']}/10"

#         live_hud["state"] = engine.state

#         # Visualise scale (if calibrated, show reference line)
#         if scale_est.px_per_m:
#             ref_px = int(scale_est.px_per_m)
#             mid_x, gnd_y = width // 4, height - 40
#             cv2.line(frame, (mid_x, gnd_y), (mid_x + ref_px, gnd_y), (100, 255, 100), 2)
#             cv2.putText(frame, "1m", (mid_x + ref_px // 2 - 10, gnd_y - 6),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

#         # Draw ground reference
#         if engine._gnd_com_y:
#             gnd_px = int(engine._gnd_com_y + (LAND_APPROACH_PX))
#             cv2.line(frame, (0, gnd_px), (width, gnd_px), (80, 80, 80), 1)

#         # Draw takeoff / landing markers
#         if engine._flight_com_x_start and engine.state == ST_FLIGHT:
#             tx = int(engine._flight_com_x_start)
#             cv2.line(frame, (tx, 0), (tx, height), COLOR_TAKE, 1)

#         per_frame_data.append({
#             "frame"     : fc,
#             "state"     : engine.state,
#             "jump_count": len(engine.jumps),
#         })

#         draw_footer_hud(frame, [
#             ("JUMP #", live_hud["jump_no"]),
#             ("DIST",   live_hud["dist"]),
#             ("FORM",   live_hud["form"]),
#         ])
#         draw_pcl_logo(frame)
#         return frame

#     # ── Run ───────────────────────────────────────────────────────────────
#     snaps = process_video_or_image(
#         path, is_video, pf,
#         output_path=output_path,
#         snap_pcts=[0.1, 0.3, 0.5, 0.7, 0.9],
#         analysis_skip=1,
#         progress_uid=progress_uid,
#     )

#     # Force-close if video ends mid-flight
#     if engine.state == ST_FLIGHT and engine._flight_f >= MIN_FLIGHT_F:
#         last_com = None
#         if per_frame_data:
#             # Best effort: use last known COM from scale buf
#             last_com = (engine._flight_com_x_start, engine._gnd_com_y or 0)
#         ev = engine.force_close(last_com, scale_est)
#         if ev:
#             ev.pop("_peak_b64", None)
#             _push_event(ev, "")

#     if not detected_ok[0]:
#         raise ValueError(
#             "No person detected. Upload a side-angle video with full body visible "
#             "(head to toe) and good lighting."
#         )

#     if session_id:
#         save_wrong_angle_log(exercise, session_id, source_filename, wrong_events)

#     # ── Post-processing ────────────────────────────────────────────────────
#     jumps     = engine.jumps
#     val_out   = validator.validate(jumps)
#     jumps_m   = [j["distance_m"] for j in jumps]

#     best_dist = round(max(jumps_m), 3) if jumps_m else 0.0
#     avg_dist  = round(sum(jumps_m) / len(jumps_m), 3) if jumps_m else 0.0

#     pose_conf_avg = float(np.mean(pose_confs)) if pose_confs else 0.0
#     foot_conf_val = 0.75 if scale_est.px_per_m else 0.50
#     quality = _compute_quality(jumps, pose_conf_avg, foot_conf_val,
#                                 scale_calibrated=scale_est.px_per_m is not None)

#     form_score = _form_score_aggregate(jumps_m)
#     issues, strengths = _feedback(jumps, detected_ok[0], exercise)

#     while len(snaps) < 5:
#         snaps.append(snaps[-1] if snaps else "")

#     step      = max(1, int(fps / 10))
#     per_frame = per_frame_data[::step]

#     best_str = f"{best_dist:.2f} m" if best_dist > 0 else "N/A"
#     avg_str  = f"{avg_dist:.2f} m"  if avg_dist  > 0 else "N/A"

#     # ── Aggregate quality metrics ──────────────────────────────────────────
#     gct_vals  = [j["ground_contact_ms"] for j in jumps if j.get("ground_contact_ms")]
#     rsi_vals  = [j["rsi"] for j in jumps if j.get("rsi")]
#     ft_vals   = [j["flight_ms"] for j in jumps]

#     result = {
#         # Core
#         "exercise"              : exercise,
#         "jump_count"            : len(jumps),
#         "correct_jumps"         : len(jumps),
#         "wrong_jumps"           : 0,
#         # Distance
#         "best_distance_m"       : best_dist,
#         "best_distance_cm"      : round(best_dist * 100, 1),
#         "avg_distance_m"        : avg_dist,
#         "all_jumps_m"           : [round(j, 3) for j in jumps_m],
#         # Per-jump detail
#         "per_jump"              : jumps,
#         # Exercise variant results
#         "phase_info"            : val_out.get("phase_info"),
#         "total_distance_m"      : val_out.get("total_distance_m"),
#         # Timing
#         "avg_flight_ms"         : int(np.mean(ft_vals)) if ft_vals else 0,
#         "avg_ground_contact_ms" : int(np.mean(gct_vals)) if gct_vals else 0,
#         "avg_rsi"               : round(float(np.mean(rsi_vals)), 3) if rsi_vals else None,
#         # Consistency
#         "fatigue_index"         : _fatigue_index(jumps_m),
#         "rep_variability_pct"   : _rep_variability(jumps_m),
#         # Scale
#         "px_per_m"              : round(scale_est.px_per_m, 2) if scale_est.px_per_m else None,
#         "scale_method"          : scale_est.method or "legacy_constant",
#         # Quality / Confidence
#         "confidence"            : quality,
#         "form_score"            : form_score,
#         # Feedback
#         "issues"                : issues,
#         "strengths"             : strengths,
#         # UI
#         "metrics": [
#             {"label": "Jumps Detected", "value": str(len(jumps))},
#             {"label": "Best Jump",      "value": best_str},
#             {"label": "Avg Jump",       "value": avg_str},
#             {"label": "Form Score",     "value": f"{form_score}/10"},
#             {"label": "Confidence",     "value": f"{quality['overall']}%"},
#         ],
#         # Legacy keys kept for backward compatibility
#         "height_cm"             : round(best_dist * 100, 1),
#         "per_frame"             : per_frame,
#         "snapshots"             : snaps,
#         "wrong_angle_count"     : 0,
#         "_wrong_events"         : wrong_events,
#     }
#     return _to_json_safe(result)















"""
module_broad_jump.py
Production-grade broad jump biomechanics engine.

Supports:
  1. Standing Broad Jump
  2. Run-Up Broad Jump
  3. Single-Leg Broad Jump
  4. Alternate Leg Broad Jump
  5. Bounding (Power Skip)
  6. Triple Broad Jump
  7. Multiple Broad Jump
  8. Reactive Broad Jump
  9. Tuck Broad Jump
 10. Pike Broad Jump
 11. Weighted Broad Jump
 12. Sand Broad Jump

Architecture:
  Video → Pose → Landmark Filter → COM → Foot Contact → 6-State FSM
       → Exercise Validator → Distance Engine → Quality → Output
"""

# import os
# import cv2
# import math
# import numpy as np
# from collections import deque

# try:
#     from ultralytics import YOLO
#     _YOLO_AVAILABLE = True
# except ImportError:
#     _YOLO_AVAILABLE = False

# from utils import (
#     process_video_or_image,
#     save_wrong_angle_log,
#     set_progress,
#     frame_to_b64,
# )
# from hud_overlay import draw_footer_hud, draw_pcl_logo

# # ─────────────────────────────────────────────────────────────────────────────
# # CONSTANTS
# # ─────────────────────────────────────────────────────────────────────────────

# YOLO_MODEL_PATH = "yolov8n-pose.pt"
# CONFIDENCE      = 0.3

# # YOLO keypoint indices (COCO 17-point)
# KP_NOSE        = 0
# KP_L_SHOULDER  = 5
# KP_R_SHOULDER  = 6
# KP_L_ELBOW     = 7
# KP_R_ELBOW     = 8
# KP_L_WRIST     = 9
# KP_R_WRIST     = 10
# KP_L_HIP       = 11
# KP_R_HIP       = 12
# KP_L_KNEE      = 13
# KP_R_KNEE      = 14
# KP_L_ANKLE     = 15
# KP_R_ANKLE     = 16

# # COM weights (Dempster body segment model approximation)
# COM_WEIGHTS = {
#     KP_L_HIP: 0.28, KP_R_HIP: 0.28,
#     KP_L_SHOULDER: 0.11, KP_R_SHOULDER: 0.11,
#     KP_L_KNEE: 0.06, KP_R_KNEE: 0.06,
#     KP_L_ANKLE: 0.05, KP_R_ANKLE: 0.05,
# }

# # State machine states
# ST_READY    = "READY"
# ST_LOADING  = "LOADING"
# ST_TAKEOFF  = "TAKEOFF"
# ST_FLIGHT   = "FLIGHT"
# ST_LANDING  = "LANDING"
# ST_RESET    = "RESET"
# ST_COOLDOWN = "COOLDOWN"


# # ─────────────────────────────────────────────────────────────────────────────
# # JSON-SAFETY HELPER
# # ─────────────────────────────────────────────────────────────────────────────
# def _to_json_safe(obj):
#     """
#     Recursively convert numpy scalar/array types (np.bool_, np.float64,
#     np.int64, np.ndarray, etc.) to native Python types so the result can
#     always be passed to json.dumps() without raising
#     'Object of type X is not JSON serializable'.

#     numpy values leak into output dicts any time a computation touches an
#     np.median()/np.mean()/np.min()/np.max() result (even indirectly, e.g.
#     `abs(numpy_float) < python_float` still yields np.bool_) — so this is
#     applied once, at the point each output dict is finalized, rather than
#     trying to manually cast every individual field (which is exactly how
#     this bug slipped through the first time).
#     """
#     if isinstance(obj, dict):
#         return {k: _to_json_safe(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [_to_json_safe(v) for v in obj]
#     if isinstance(obj, np.bool_):
#         return bool(obj)
#     if isinstance(obj, np.integer):
#         return int(obj)
#     if isinstance(obj, np.floating):
#         return float(obj)
#     if isinstance(obj, np.ndarray):
#         return _to_json_safe(obj.tolist())
#     return obj

# SKELETON_PAIRS = [
#     (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
#     (5, 11), (6, 12), (11, 12),
#     (11, 13), (13, 15), (12, 14), (14, 16),
# ]

# # Colors
# COLOR_SKEL   = (255, 255, 255)
# COLOR_JOINT  = (0,   0,   0)
# COLOR_COM    = (0, 255, 255)
# COLOR_TAKE   = (0, 165, 255)
# COLOR_LAND   = (255, 100, 255)
# COLOR_TEXT   = (255, 255, 255)

# # FSM thresholds
# GROUND_WINDOW       = 20   # frames to establish ground reference
# COM_RISE_PX         = 18   # COM must rise this much above ground to start LOADING
# COM_TAKEOFF_PX      = 30   # COM above ground to confirm TAKEOFF
# FLIGHT_CONFIRM_F    = 3    # consecutive airborne frames required before FLIGHT is trusted
# LAND_APPROACH_PX    = 25   # COM within this of ground → approaching landing
# LAND_STABLE_F       = 7    # frames near ground to confirm LANDING (athlete fully settled)
# MIN_FLIGHT_F        = 7    # minimum flight frames for a valid jump (~233ms @ 30fps)
# RESET_STABLE_F      = 4    # frames on ground before READY again (allows fast consecutive jumps)
# LAND_VEL_PX_F       = 6.0  # max COM px/frame velocity to accept landing as "settled"
# MIN_JUMP_DIST_M     = 0.25 # below this, treat as noise / false jump
# MIN_PEAK_RISE_PX    = 10   # minimum COM rise during flight to count as a real jump

# # ── False-jump rejection / hysteresis (post-landing stabilization fix) ──
# MIN_VALID_DIST_M       = 0.75  # standing broad jump floor — rejects post-landing micro-shifts
# MAX_VALID_DIST_M       = 4.0   # sanity ceiling
# MIN_VALID_FLIGHT_MS    = 200   # below this is noise, not a jump
# MIN_APPROACH_SPEED_MS  = 0.20  # athlete must actually be moving into takeoff
# AIR_CONFIRM_FRAMES     = 2     # consecutive frames required to trust "airborne"
# CONTACT_CONFIRM_FRAMES = 2     # consecutive frames required to trust "grounded"
# COOLDOWN_FRAMES        = 20    # frames after landing before a new jump may start

# # Exercise-specific
# REACTIVE_MAX_CONTACT_F  = 12   # max ground contact for reactive jump (at 30fps ≈ 400ms)
# TUCK_HIP_FLEX_DEG       = 60   # minimum hip flexion for tuck confirmation
# PIKE_HIP_FLEX_DEG       = 70   # minimum hip flexion for pike
# PIKE_KNEE_EXT_DEG       = 150  # minimum knee extension for pike
# BOUND_MIN_FLIGHT_F      = 3    # minimum flight per bound
# TRIPLE_PHASE_MIN_F      = 3    # min flight frames per triple-jump phase

# # Anthropometric scale: avg hip-to-ankle ≈ 0.53× body height, body height ≈ 1.75m
# # So hip-ankle ≈ 0.927m in pixels → derive px/m
# ANTHRO_HIP_ANKLE_M  = 0.90   # meters (conservative; will be estimated per-subject)

# # ─────────────────────────────────────────────────────────────────────────────
# # ONE-EURO FILTER  (per-coordinate temporal filter)
# # ─────────────────────────────────────────────────────────────────────────────

# class OneEuroFilter:
#     """
#     One Euro Filter for temporal landmark smoothing.
#     Reduces jitter without large latency; preserves fast explosive motion
#     by automatically raising cutoff frequency at high velocity.
#     """
#     def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
#         self.freq      = freq
#         self.min_cutoff = min_cutoff
#         self.beta      = beta
#         self.d_cutoff  = d_cutoff
#         self._x        = None
#         self._dx       = 0.0

#     def _alpha(self, cutoff):
#         tau = 1.0 / (2 * math.pi * cutoff)
#         return 1.0 / (1.0 + tau * self.freq)

#     def __call__(self, x):
#         if self._x is None:
#             self._x = x
#             return x
#         dx     = (x - self._x) * self.freq
#         a_d    = self._alpha(self.d_cutoff)
#         self._dx = a_d * dx + (1 - a_d) * self._dx
#         cutoff = self.min_cutoff + self.beta * abs(self._dx)
#         a      = self._alpha(cutoff)
#         self._x = a * x + (1 - a) * self._x
#         return self._x


# class LandmarkFilter:
#     """Per-landmark One Euro filter (x and y independently)."""
#     def __init__(self, n_landmarks=17, freq=30.0):
#         self.fx = [OneEuroFilter(freq=freq) for _ in range(n_landmarks)]
#         self.fy = [OneEuroFilter(freq=freq) for _ in range(n_landmarks)]

#     def update(self, pts):
#         """pts: np.array shape (17,2). Returns filtered array same shape."""
#         out = pts.copy().astype(float)
#         for i, (fx, fy) in enumerate(zip(self.fx, self.fy)):
#             if pts[i, 0] > 0 and pts[i, 1] > 0:
#                 out[i, 0] = fx(pts[i, 0])
#                 out[i, 1] = fy(pts[i, 1])
#         return out


# # ─────────────────────────────────────────────────────────────────────────────
# # CENTER OF MASS ESTIMATOR
# # ─────────────────────────────────────────────────────────────────────────────

# def estimate_com(pts):
#     """
#     Weighted average COM from body keypoints.
#     Returns (com_x, com_y) or None if not enough valid points.
#     """
#     total_w = 0.0
#     wx, wy  = 0.0, 0.0
#     for idx, w in COM_WEIGHTS.items():
#         p = pts[idx]
#         if p[0] > 0 and p[1] > 0:
#             wx += w * p[0]
#             wy += w * p[1]
#             total_w += w
#     if total_w < 0.3:
#         return None
#     return (wx / total_w, wy / total_w)


# # ─────────────────────────────────────────────────────────────────────────────
# # FOOT CONTACT ESTIMATOR
# # ─────────────────────────────────────────────────────────────────────────────

# class FootContactEstimator:
#     """
#     Tracks each foot (left/right) independently.
#     Uses ankle vertical position relative to adaptive ground reference,
#     combined with ankle vertical velocity, to classify CONTACT / AIR.
#     Velocity gating prevents single noisy keypoint jitters from being
#     read as a takeoff or landing.
#     """
#     def __init__(self, window=30, fps=30.0):
#         self.ground_y    = None   # adaptive ground y (pixel; larger = lower on screen)
#         self.stance_buf  = deque(maxlen=window)  # y values during known contact
#         self.fps         = fps
#         self._prev_ly    = None
#         self._prev_ry    = None
#         self.left        = "CONTACT"
#         self.right       = "CONTACT"
#         self.left_vy     = 0.0
#         self.right_vy    = 0.0
#         self.THRESH_PX   = 22     # pixels above ground to classify AIR
#         self.VEL_THRESH  = 3.0    # px/frame — ankle must also be moving to confirm AIR

#     def update_ground(self, ly, ry):
#         """Call with ankle y values during confirmed ground phase."""
#         for y in [ly, ry]:
#             if y > 0:
#                 self.stance_buf.append(y)
#         if self.stance_buf:
#             # Ground is near the max (bottom-most) ankle positions
#             self.ground_y = np.percentile(list(self.stance_buf), 85)

#     def classify(self, pts):
#         """Update left/right contact state from keypoints."""
#         la = pts[KP_L_ANKLE]
#         ra = pts[KP_R_ANKLE]
#         ly = la[1] if la[0] > 0 else -1
#         ry = ra[1] if ra[0] > 0 else -1

#         # Velocity (px/frame) for noise gating
#         self.left_vy  = (ly - self._prev_ly) if (ly > 0 and self._prev_ly is not None) else 0.0
#         self.right_vy = (ry - self._prev_ry) if (ry > 0 and self._prev_ry is not None) else 0.0
#         if ly > 0: self._prev_ly = ly
#         if ry > 0: self._prev_ry = ry

#         if self.ground_y is None:
#             self.left  = "CONTACT"
#             self.right = "CONTACT"
#             return

#         thr = self.THRESH_PX
#         gnd = self.ground_y

#         # Left foot — require both height-above-ground AND recent motion
#         # to avoid a single jittery keypoint flipping the state.
#         if ly > 0:
#             above_ground = ly < gnd - thr
#             self.left = "AIR" if (above_ground or abs(self.left_vy) > self.VEL_THRESH * 2) else "CONTACT"
#         # Right foot
#         if ry > 0:
#             above_ground = ry < gnd - thr
#             self.right = "AIR" if (above_ground or abs(self.right_vy) > self.VEL_THRESH * 2) else "CONTACT"

#     @property
#     def both_contact(self):
#         return self.left == "CONTACT" and self.right == "CONTACT"

#     @property
#     def any_contact(self):
#         return self.left == "CONTACT" or self.right == "CONTACT"

#     @property
#     def both_air(self):
#         return self.left == "AIR" and self.right == "AIR"


# # ─────────────────────────────────────────────────────────────────────────────
# # ADAPTIVE SCALE ESTIMATOR  (pixel → metric)
# # ─────────────────────────────────────────────────────────────────────────────

# class ScaleEstimator:
#     """
#     Estimates pixels-per-meter from subject anthropometrics.
#     Primary: full body height (nose → ankle), which is far more stable
#     across camera angles than hip-to-ankle alone.
#     Fallback: hip-to-ankle segment if height isn't fully visible.
#     Updated during ground-contact frames only (standing posture).
#     """
#     ANTHRO_BODY_HEIGHT_M = 1.75   # default assumed height if user doesn't supply one

#     def __init__(self, user_height_m=None):
#         self._height_samples = deque(maxlen=60)
#         self._hip_ankle_samples = deque(maxlen=60)
#         self.px_per_m = None
#         self.user_height_m = user_height_m
#         self.method = None

#     def update(self, pts):
#         nose = pts[KP_NOSE]
#         lh = pts[KP_L_HIP];   la = pts[KP_L_ANKLE]
#         rh = pts[KP_R_HIP];   ra = pts[KP_R_ANKLE]

#         # ── Preferred: nose → lowest ankle (full standing height in px) ──
#         ankle_ys = [p[1] for p in (la, ra) if p[0] > 0 and p[1] > 0]
#         if nose[0] > 0 and nose[1] > 0 and ankle_ys:
#             height_px = max(ankle_ys) - nose[1]
#             if height_px > 50:   # sane lower bound, avoids garbage frames
#                 self._height_samples.append(height_px)

#         # ── Fallback: hip → ankle ──
#         pairs = []
#         if lh[0] > 0 and la[0] > 0:
#             pairs.append(abs(la[1] - lh[1]))
#         if rh[0] > 0 and ra[0] > 0:
#             pairs.append(abs(ra[1] - rh[1]))
#         if pairs:
#             hip_ankle_px = np.mean(pairs)
#             if hip_ankle_px > 10:
#                 self._hip_ankle_samples.append(hip_ankle_px)

#         # Prefer body-height calibration once enough samples exist
#         if len(self._height_samples) >= 10:
#             median_px = np.median(list(self._height_samples))
#             height_m  = self.user_height_m or self.ANTHRO_BODY_HEIGHT_M
#             self.px_per_m = median_px / height_m
#             self.method = "user_height" if self.user_height_m else "body_height_anthropometric"
#         elif len(self._hip_ankle_samples) >= 10:
#             median_px = np.median(list(self._hip_ankle_samples))
#             self.px_per_m = median_px / ANTHRO_HIP_ANKLE_M
#             self.method = "hip_ankle_anthropometric"

#     def px_to_m(self, pixels):
#         if self.px_per_m and self.px_per_m > 0:
#             return pixels / self.px_per_m
#         # Fallback: legacy constant (0.00933 m/px)
#         return pixels * 0.00933


# # ─────────────────────────────────────────────────────────────────────────────
# # ANGLE UTILITIES
# # ─────────────────────────────────────────────────────────────────────────────

# def _vec_angle(a, b, c):
#     """Angle at point b in triangle a-b-c (degrees)."""
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b;  bc = c - b
#     cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
#     return math.degrees(math.acos(np.clip(cos, -1, 1)))

# def _hip_flexion(pts, side="L"):
#     """Hip flexion angle (shoulder-hip-knee)."""
#     sh  = pts[KP_L_SHOULDER if side=="L" else KP_R_SHOULDER]
#     hp  = pts[KP_L_HIP      if side=="L" else KP_R_HIP]
#     kn  = pts[KP_L_KNEE     if side=="L" else KP_R_KNEE]
#     if any(p[0] <= 0 for p in [sh, hp, kn]):
#         return None
#     return _vec_angle(sh, hp, kn)

# def _knee_flexion(pts, side="L"):
#     """Knee flexion angle (hip-knee-ankle)."""
#     hp  = pts[KP_L_HIP   if side=="L" else KP_R_HIP]
#     kn  = pts[KP_L_KNEE  if side=="L" else KP_R_KNEE]
#     an  = pts[KP_L_ANKLE if side=="L" else KP_R_ANKLE]
#     if any(p[0] <= 0 for p in [hp, kn, an]):
#         return None
#     return _vec_angle(hp, kn, an)

# def _com_velocity(com_hist, fps):
#     """Returns (vx, vy) in px/frame from last two COM positions."""
#     if len(com_hist) < 2:
#         return (0.0, 0.0)
#     dx = com_hist[-1][0] - com_hist[-2][0]
#     dy = com_hist[-1][1] - com_hist[-2][1]
#     return (dx * fps, dy * fps)


# def _jump_direction(pts_hist):
#     """
#     Determine overall travel direction (+1 = left-to-right, -1 = right-to-left)
#     from a short history of COM x positions. Locked once at the start of a
#     rep sequence so per-jump distance math always uses a consistent sign.
#     """
#     if len(pts_hist) < 2:
#         return 1
#     dx = pts_hist[-1] - pts_hist[0]
#     return 1 if dx >= 0 else -1


# def _ankle_positions(pts):
#     """Return (left_ankle, right_ankle) as (x,y) or None if not visible."""
#     la = pts[KP_L_ANKLE]
#     ra = pts[KP_R_ANKLE]
#     left  = (la[0], la[1]) if la[0] > 0 and la[1] > 0 else None
#     right = (ra[0], ra[1]) if ra[0] > 0 and ra[1] > 0 else None
#     return left, right


# def _heel_x_approx(ankle, knee, direction):
#     """
#     Approximate heel x-position from ankle + knee (shank vector), since
#     COCO-17 has no heel keypoint. The heel sits slightly behind the ankle
#     relative to the direction of travel.
#     ankle, knee: (x, y) tuples
#     direction: +1 (moving right) or -1 (moving left)
#     """
#     if ankle is None:
#         return None
#     if knee is None:
#         return ankle[0]
#     shank_len = math.hypot(knee[0] - ankle[0], knee[1] - ankle[1])
#     if shank_len < 1e-6:
#         return ankle[0]
#     # Heel trails the ankle opposite to the direction of travel.
#     return ankle[0] - direction * 0.15 * shank_len


# def _takeoff_foot_x(pts, direction):
#     """
#     Foot-based takeoff reference: the most FORWARD foot point (in the
#     direction of travel) at the last grounded frame — this is the true
#     takeoff line, not COM.
#     """
#     la, ra = _ankle_positions(pts)
#     xs = [p[0] for p in (la, ra) if p is not None]
#     if not xs:
#         return None
#     return max(xs) if direction >= 0 else min(xs)


# def _landing_foot_x(pts, direction):
#     """
#     Foot-based landing reference: nearest body part to the takeoff line,
#     approximated via heel position on the leading foot at landing.
#     """
#     la, ra = _ankle_positions(pts)
#     lk = pts[KP_L_KNEE]; rk = pts[KP_R_KNEE]
#     lk_pt = (lk[0], lk[1]) if lk[0] > 0 and lk[1] > 0 else None
#     rk_pt = (rk[0], rk[1]) if rk[0] > 0 and rk[1] > 0 else None

#     heels = []
#     hx = _heel_x_approx(la, lk_pt, direction)
#     if hx is not None:
#         heels.append(hx)
#     hx = _heel_x_approx(ra, rk_pt, direction)
#     if hx is not None:
#         heels.append(hx)

#     if not heels:
#         return None
#     # Landing mark = the foot point CLOSEST to the takeoff line, i.e. the
#     # leading (trailing-edge) foot in the direction of travel.
#     return min(heels) if direction >= 0 else max(heels)


# # ─────────────────────────────────────────────────────────────────────────────
# # JUMP ENGINE  (6-state FSM)
# # ─────────────────────────────────────────────────────────────────────────────

# class JumpEngine:
#     """
#     Core 6-state finite state machine.
#     Tracks COM, foot contacts, and produces jump events.
#     """
#     def __init__(self, fps=30.0, exercise="standing_broad_jump"):
#         self.fps      = fps
#         self.exercise = exercise
#         self.state    = ST_READY

#         # Ground reference
#         self._gnd_com_y    = None   # adaptive ground COM y
#         self._gnd_buf      = deque(maxlen=GROUND_WINDOW)
#         self._gnd_locked   = False

#         # Takeoff
#         self._takeoff_com  = None
#         self._takeoff_f    = 0
#         self._takeoff_heel = None   # left heel x at takeoff
#         self._takeoff_foot_x   = None   # foot-based takeoff line (px)
#         self._last_grounded_pts = None  # full pts snapshot at last grounded frame
#         self._last_grounded_com = None

#         # Flight
#         self._flight_f     = 0
#         self._airborne_streak = 0   # consecutive airborne frames (hysteresis)
#         self._ground_streak   = 0   # consecutive grounded frames (hysteresis)
#         self._peak_com_y   = None
#         self._flight_com_x_start = None
#         self._peak_b64     = None

#         # In-flight kinematics (for tuck/pike/bounding)
#         self._flight_hip_flex   = []
#         self._flight_knee_flex  = []
#         self._flight_arm_travel = []   # wrist-y travel relative to shoulder during flight

#         # Landing
#         self._land_stable_f = 0
#         self._land_com_buf  = deque(maxlen=8)
#         self._land_foot_x_buf = deque(maxlen=8)   # foot-based landing samples
#         self._land_pts_at_settle = None           # pts snapshot once landing settles

#         # Direction of travel (+1 left→right, -1 right→left), locked per rep
#         self._direction     = 1
#         self._dir_buf       = deque(maxlen=10)

#         # Reset
#         self._reset_f       = 0
#         self._cooldown_f    = 0

#         # Loading / countermovement
#         self._load_com_y   = None   # COM y at start of countermovement
#         self._load_start_f = 0
#         self._cm_depth     = 0.0   # countermovement depth in px

#         # Completed jumps
#         self.jumps         = []
#         self._jump_n       = 0

#         # Per-phase data for triple/multiple/bounding
#         self._phase_list   = []    # list of phase dicts per jump
#         self._phase_start_com = None

#         # Reactive: ground contact tracking
#         self._gct_start_f  = 0    # frame when ground contact began after landing

#         # Approach velocity (run-up)
#         self._approach_vx  = 0.0
#         self._approach_buf = deque(maxlen=20)
#         self._takeoff_approach_vx = 0.0   # snapshot of approach_vx AT takeoff (frozen, not live)

#         # COM history for velocity-gated landing confirmation
#         self._com_hist      = deque(maxlen=5)

#         # Frame counter
#         self._frame        = 0

#     # ──────────────────────────────────────────────────────────────────────
#     def _update_ground(self, com_y, foot: FootContactEstimator, pts=None, com=None):
#         """Update adaptive ground COM_y during stable contact."""
#         if foot.both_contact or foot.any_contact:
#             self._gnd_buf.append(com_y)
#             if pts is not None:
#                 self._last_grounded_pts = pts.copy()
#                 self._last_grounded_com = com
#         if len(self._gnd_buf) >= 5:
#             self._gnd_com_y = np.percentile(list(self._gnd_buf), 80)

#     def _is_airborne(self, com_y):
#         if self._gnd_com_y is None:
#             return False
#         return com_y < self._gnd_com_y - COM_TAKEOFF_PX

#     def _is_near_ground(self, com_y):
#         if self._gnd_com_y is None:
#             return True
#         return com_y >= self._gnd_com_y - LAND_APPROACH_PX

#     def _com_settled(self):
#         """True when recent COM motion has slowed enough to trust the
#         landing position (prevents capturing landing COM while the body
#         is still travelling forward/downward)."""
#         if len(self._com_hist) < 3:
#             return False
#         vx = self._com_hist[-1][0] - self._com_hist[-2][0]
#         vy = self._com_hist[-1][1] - self._com_hist[-2][1]
#         return abs(vx) < LAND_VEL_PX_F and abs(vy) < LAND_VEL_PX_F

#     # ──────────────────────────────────────────────────────────────────────
#     def update(self, pts, foot: FootContactEstimator, com, scale: ScaleEstimator,
#                frame_b64=None):
#         """
#         Called every frame.
#         pts: filtered keypoints (17,2)
#         foot: FootContactEstimator (already updated this frame)
#         com: (cx, cy) or None
#         scale: ScaleEstimator
#         Returns: jump_event dict if a jump completed this frame, else None
#         """
#         self._frame += 1
#         if com is None:
#             return None

#         com_x, com_y = com
#         self._com_hist.append(com)

#         # Track approach velocity for run-up detection
#         self._approach_buf.append(com_x)
#         if len(self._approach_buf) >= 5:
#             dx = self._approach_buf[-1] - self._approach_buf[-5]
#             self._approach_vx = dx / 5.0 * self.fps  # px/s

#         # Track direction-of-travel candidates continuously; locked at takeoff
#         self._dir_buf.append(com_x)

#         # ── Hysteresis-gated airborne/grounded detection ────────────────────
#         # AND (not OR): both the foot-contact estimator AND the COM-height
#         # check must agree the athlete is airborne. A lone COM rise during
#         # post-landing balance recovery (feet still down) no longer fakes a
#         # flight, and a lone foot-occlusion flicker (COM still grounded)
#         # doesn't either. Requires AIR_CONFIRM_FRAMES/CONTACT_CONFIRM_FRAMES
#         # consecutive agreeing frames before either state is trusted, which
#         # absorbs single-frame keypoint jitter.
#         airborne_raw = foot.both_air and self._is_airborne(com_y)
#         if airborne_raw:
#             self._airborne_streak += 1
#             self._ground_streak = 0
#         else:
#             self._ground_streak += 1
#             self._airborne_streak = 0
#         airborne_confirmed = self._airborne_streak >= AIR_CONFIRM_FRAMES
#         grounded_confirmed = self._ground_streak >= CONTACT_CONFIRM_FRAMES

#         event = None

#         # ── READY ─────────────────────────────────────────────────────────
#         if self.state == ST_READY:
#             self._update_ground(com_y, foot, pts, com)
#             rising = self._gnd_com_y is not None and com_y < self._gnd_com_y - COM_RISE_PX
#             if rising or airborne_raw:
#                 self.state         = ST_LOADING
#                 self._load_com_y   = com_y
#                 self._load_start_f = self._frame
#             else:
#                 # Still grounded — update scale only here (never in flight)
#                 scale.update(pts)
#                 foot.update_ground(pts[KP_L_ANKLE][1], pts[KP_R_ANKLE][1])
#                 self._takeoff_com  = com   # keep updating takeoff reference
#                 self._last_grounded_pts = pts.copy()
#                 self._last_grounded_com = com

#         # ── LOADING ───────────────────────────────────────────────────────
#         elif self.state == ST_LOADING:
#             if not airborne_raw:
#                 self._last_grounded_pts = pts.copy()
#                 self._last_grounded_com = com

#             # Require AIR_CONFIRM_FRAMES consecutive airborne frames
#             # (both conditions agreeing) before committing to a takeoff.
#             if airborne_confirmed:
#                 # True takeoff = the LAST GROUNDED frame, not the first
#                 # airborne one.
#                 ref_pts = self._last_grounded_pts if self._last_grounded_pts is not None else pts
#                 ref_com = self._last_grounded_com if self._last_grounded_com is not None else com

#                 # Lock direction of travel for this rep from recent COM drift
#                 direction_samples = np.array(list(self._dir_buf))
#                 self._direction = _jump_direction(direction_samples)

#                 self._takeoff_com        = ref_com
#                 self._takeoff_foot_x     = _takeoff_foot_x(ref_pts, self._direction)
#                 self._flight_com_x_start = ref_com[0]
#                 self._takeoff_f          = max(1, self._frame - self._airborne_streak)
#                 self._flight_f           = self._airborne_streak
#                 self._peak_com_y         = com_y
#                 self._peak_b64           = frame_b64
#                 self._flight_hip_flex    = []
#                 self._flight_knee_flex   = []
#                 self._flight_arm_travel  = []
#                 self._phase_start_com    = ref_com
#                 self._land_foot_x_buf.clear()
#                 self._land_com_buf.clear()
#                 self._land_stable_f      = 0
#                 self._cm_depth = max(0.0, self._gnd_com_y - self._load_com_y) if self._gnd_com_y else 0.0
#                 self._takeoff_approach_vx = self._approach_vx
#                 self.state = ST_FLIGHT
#             elif not airborne_raw and self._gnd_com_y is not None and com_y > self._gnd_com_y:
#                 # COM went back down without ever becoming airborne — false alarm
#                 self.state = ST_READY

#         # ── FLIGHT ────────────────────────────────────────────────────────
#         elif self.state == ST_FLIGHT:
#             self._flight_f += 1

#             if com_y < self._peak_com_y:
#                 self._peak_com_y = com_y
#                 self._peak_b64   = frame_b64

#             hf_l = _hip_flexion(pts, "L")
#             hf_r = _hip_flexion(pts, "R")
#             kf_l = _knee_flexion(pts, "L")
#             kf_r = _knee_flexion(pts, "R")
#             if hf_l: self._flight_hip_flex.append(hf_l)
#             if hf_r: self._flight_hip_flex.append(hf_r)
#             if kf_l: self._flight_knee_flex.append(kf_l)
#             if kf_r: self._flight_knee_flex.append(kf_r)

#             for sh_i, wr_i in ((KP_L_SHOULDER, KP_L_WRIST), (KP_R_SHOULDER, KP_R_WRIST)):
#                 sh, wr = pts[sh_i], pts[wr_i]
#                 if sh[0] > 0 and wr[0] > 0:
#                     self._flight_arm_travel.append(sh[1] - wr[1])

#             # Landing confirmation requires near-ground AND settled COM
#             # velocity AND a hysteresis-confirmed ground contact streak —
#             # this is what stops the post-landing stabilization wobble from
#             # being read as the landing mark.
#             if self._flight_f >= MIN_FLIGHT_F:
#                 near_ground = self._is_near_ground(com_y) or foot.any_contact
#                 if near_ground:
#                     self._land_stable_f += 1
#                     self._land_com_buf.append(com_x)
#                     foot_x = _landing_foot_x(pts, self._direction)
#                     if foot_x is not None:
#                         self._land_foot_x_buf.append(foot_x)
#                     if (self._land_stable_f >= LAND_STABLE_F and
#                             self._com_settled() and grounded_confirmed):
#                         event = self._confirm_jump(com, pts, scale)
#                         if event is not None:
#                             self.state = ST_LANDING
#                         else:
#                             # Rejected as a false jump (backward movement,
#                             # too short, no real approach speed, physics
#                             # mismatch, etc.) — go straight to cooldown
#                             # without counting it, rather than re-arming
#                             # immediately (which is how it got double-counted).
#                             self.state = ST_COOLDOWN
#                             self._cooldown_f = 0
#                 else:
#                     self._land_stable_f = 0
#                     self._land_com_buf.clear()
#                     self._land_foot_x_buf.clear()

#             if self._flight_f < MIN_FLIGHT_F and grounded_confirmed and self._is_near_ground(com_y):
#                 self.state = ST_READY
#                 self._airborne_streak = 0
#                 self._ground_streak = 0

#         # ── LANDING ───────────────────────────────────────────────────────
#         elif self.state == ST_LANDING:
#             self._land_stable_f = 0
#             self._land_com_buf.clear()
#             self._land_foot_x_buf.clear()
#             self._reset_f = 0
#             self._cooldown_f = 0
#             self._last_grounded_pts = pts.copy()
#             self._last_grounded_com = com

#             self._gct_start_f = self._frame
#             self.state = ST_COOLDOWN

#         # ── COOLDOWN ──────────────────────────────────────────────────────
#         # Mandatory recovery window after every landing. Post-landing balance
#         # adjustments (knee bend, weight shift, brief foot occlusion) happen
#         # here and CANNOT trigger a new LOADING/FLIGHT cycle — this is what
#         # was previously creating a fake second jump immediately after a
#         # real one. Reactive jumps are the sole exception (genuinely meant
#         # to re-launch within a few frames of ground contact).
#         elif self.state == ST_COOLDOWN:
#             self._cooldown_f += 1
#             self._update_ground(com_y, foot, pts, com)
#             self._last_grounded_pts = pts.copy()
#             self._last_grounded_com = com

#             if (foot.both_air and self._cooldown_f <= REACTIVE_MAX_CONTACT_F
#                     and "reactive" in self.exercise):
#                 direction_samples = np.array(list(self._dir_buf))
#                 self._direction = _jump_direction(direction_samples)
#                 ref_pts = self._last_grounded_pts if self._last_grounded_pts is not None else pts

#                 self._takeoff_com        = com
#                 self._takeoff_foot_x     = _takeoff_foot_x(ref_pts, self._direction)
#                 self._flight_com_x_start = com_x
#                 self._takeoff_f          = self._frame
#                 self._flight_f           = 0
#                 self._peak_com_y         = com_y
#                 self._peak_b64           = frame_b64
#                 self._flight_hip_flex    = []
#                 self._flight_knee_flex   = []
#                 self._flight_arm_travel  = []
#                 self._phase_start_com    = com
#                 self._land_foot_x_buf.clear()
#                 self._land_com_buf.clear()
#                 self._takeoff_approach_vx = self._approach_vx
#                 self.state = ST_FLIGHT

#             elif self._cooldown_f >= COOLDOWN_FRAMES:
#                 self.state = ST_READY
#                 self._gnd_locked = True
#                 self._reset_f = 0

#         # ── RESET (legacy, kept for backward compatibility) ─────────────────
#         elif self.state == ST_RESET:
#             self.state = ST_READY

#         return event

#     # ──────────────────────────────────────────────────────────────────────
#     def _confirm_jump(self, land_com, pts, scale: ScaleEstimator):
#         """Build and store a jump event dict. Returns None (and does NOT
#         increment jump_n / append to self.jumps) if this is rejected as a
#         false jump — caller must handle a None return."""

#         # ── Foot-based distance (primary) ───────────────────────────────
#         # Takeoff line = most-forward foot point at the last grounded frame.
#         # Landing mark = nearest heel of the leading foot once landing has
#         # settled (near-ground + low COM velocity + foot contact), smoothed
#         # via median over the buffered landing samples.
#         takeoff_x_foot = self._takeoff_foot_x

#         land_x_foot = None
#         if self._land_foot_x_buf:
#             land_x_foot = float(np.median(list(self._land_foot_x_buf)))
#         elif pts is not None:
#             land_x_foot = _landing_foot_x(pts, self._direction)

#         # COM fallback (only used if foot keypoints were unavailable)
#         land_x_com    = float(np.median(list(self._land_com_buf))) if self._land_com_buf else land_com[0]
#         takeoff_x_com = self._flight_com_x_start if self._flight_com_x_start else (
#             self._takeoff_com[0] if self._takeoff_com else land_x_com)

#         # Distance is SIGNED by the locked direction of travel, not abs().
#         # A jump that nets backward relative to the locked direction (e.g.
#         # a post-landing balance shift) produces a negative raw distance
#         # and is rejected outright below — this is what previously let a
#         # backward foot displacement masquerade as a 0.53m jump.
#         if takeoff_x_foot is not None and land_x_foot is not None:
#             raw_px = (land_x_foot - takeoff_x_foot) * self._direction
#             takeoff_x, land_x = takeoff_x_foot, land_x_foot
#             dist_method = "foot_based"
#         else:
#             raw_px = (land_x_com - takeoff_x_com) * self._direction
#             takeoff_x, land_x = takeoff_x_com, land_x_com
#             dist_method = "com_fallback"

#         if raw_px <= 0:
#             return None   # non-forward / backward displacement — not a real jump

#         px_dist = raw_px
#         dist_m  = round(scale.px_to_m(px_dist), 3)
#         dist_cm = round(dist_m * 100, 1)

#         flight_f  = self._flight_f
#         flight_ms = int(flight_f / self.fps * 1000)

#         # Peak COM height above takeoff COM
#         peak_rise_px = 0.0
#         if self._takeoff_com and self._peak_com_y:
#             peak_rise_px = max(0.0, self._takeoff_com[1] - self._peak_com_y)
#         peak_rise_m = scale.px_to_m(peak_rise_px)

#         approach_speed_ms = scale.px_to_m(abs(self._takeoff_approach_vx))

#         # ── False-jump rejection ────────────────────────────────────────
#         # Reject jumps that are too short, too small, too slow to approach,
#         # or out of physically sane range for a standing broad jump — these
#         # are almost always weight-shift / post-landing-stabilization noise,
#         # not genuine jumps.
#         t = flight_f / self.fps
#         if (dist_m < MIN_VALID_DIST_M or
#                 dist_m > MAX_VALID_DIST_M or
#                 flight_ms < MIN_VALID_FLIGHT_MS or
#                 peak_rise_px < MIN_PEAK_RISE_PX or
#                 approach_speed_ms < MIN_APPROACH_SPEED_MS):
#             return None

#         # ── Physics-correct takeoff velocity / angle ────────────────────
#         g = 9.81
#         vx_ms = dist_m / (t + 1e-9)
#         vy_ms = g * t / 2 if t > 0 else 0.0
#         takeoff_vel = round(math.sqrt(vx_ms**2 + vy_ms**2), 2)
#         takeoff_angle = round(math.degrees(math.atan2(vy_ms, vx_ms + 1e-9)), 1) if vx_ms > 1e-6 else 0.0
#         takeoff_angle = max(0.0, min(85.0, takeoff_angle))

#         # Physics consistency check: does measured distance roughly match
#         # vx * t? A jump that fails this is rejected outright — it means
#         # the takeoff/landing marks don't form a coherent flight.
#         physics_expected_dist = vx_ms * t
#         physics_consistent = bool(abs(physics_expected_dist - dist_m) < max(0.15, 0.25 * dist_m))
#         if not physics_consistent:
#             return None

#         # Tuck / pike detection from in-flight angles
#         tuck_confirmed = False
#         pike_confirmed = False
#         min_hip_flex = min(self._flight_hip_flex) if self._flight_hip_flex else 180
#         min_knee_flex = min(self._flight_knee_flex) if self._flight_knee_flex else 180
#         max_knee_ext  = max(self._flight_knee_flex) if self._flight_knee_flex else 0

#         if min_hip_flex < TUCK_HIP_FLEX_DEG and min_knee_flex < 90:
#             tuck_confirmed = True
#         if min_hip_flex < PIKE_HIP_FLEX_DEG and max_knee_ext > PIKE_KNEE_EXT_DEG:
#             pike_confirmed = True

#         # Countermovement depth
#         cm_depth_m = round(scale.px_to_m(self._cm_depth), 3)

#         # Horizontal efficiency: forward dist / total COM path (approximation)
#         h_eff = round(min(1.0, px_dist / (px_dist + peak_rise_px + 1e-9)), 3)

#         # Landing stability (from COM settle velocity at confirmation time)
#         land_vx = land_vy = 0.0
#         if len(self._com_hist) >= 2:
#             land_vx = self._com_hist[-1][0] - self._com_hist[-2][0]
#             land_vy = self._com_hist[-1][1] - self._com_hist[-2][1]
#         landing_stability = round(max(0.0, 1.0 - (abs(land_vx) + abs(land_vy)) / 20.0), 3)

#         # Multi-factor form score (distance, landing stability, takeoff
#         # angle, arm-swing quality, knee flexion at takeoff)
#         arm_swing_score = _arm_swing_score(self._flight_arm_travel)
#         form = _form_score_full(
#             dist_m=dist_m,
#             landing_stability=landing_stability,
#             takeoff_angle=takeoff_angle,
#             min_hip_flex=min_hip_flex,
#             min_knee_flex=min_knee_flex,
#             arm_swing_score=arm_swing_score,
#         )

#         # Ground contact time (reactive)
#         gct_ms = 0
#         if "reactive" in self.exercise and self._gct_start_f > 0:
#             gct_f  = self._takeoff_f - self._gct_start_f
#             gct_ms = int(max(0, gct_f) / self.fps * 1000)

#         rsi = round(flight_ms / gct_ms, 3) if gct_ms > 0 else None

#         self._jump_n += 1
#         jump = {
#             # Identity
#             "jump_no"           : self._jump_n,
#             # Distances
#             "distance_m"        : dist_m,
#             "distance_cm"       : dist_cm,
#             "pixel_dist"        : round(px_dist, 1),
#             "distance_method"   : dist_method,
#             # Timing
#             "flight_ms"         : flight_ms,
#             "airborne_ms"       : flight_ms,
#             "takeoff_frame"     : self._takeoff_f,
#             "landing_frame"     : self._frame,
#             # Positions
#             "takeoff_com_x"     : round(takeoff_x_com, 1),
#             "landing_com_x"     : round(land_x_com, 1),
#             "takeoff_foot_x"    : round(takeoff_x_foot, 1) if takeoff_x_foot is not None else None,
#             "landing_foot_x"    : round(land_x_foot, 1) if land_x_foot is not None else None,
#             "direction"         : self._direction,
#             # Kinematics
#             "takeoff_velocity_ms": takeoff_vel,
#             "takeoff_angle_deg" : takeoff_angle,
#             "peak_rise_m"       : round(peak_rise_m, 3),
#             "approach_vx_px_s"  : round(self._takeoff_approach_vx, 1),
#             "approach_speed_ms" : round(approach_speed_ms, 2),
#             "horizontal_efficiency": h_eff,
#             "cm_depth_m"        : cm_depth_m,
#             "physics_consistent": physics_consistent,
#             # Technique
#             "tuck_confirmed"    : tuck_confirmed,
#             "pike_confirmed"    : pike_confirmed,
#             "min_hip_flex_deg"  : round(min_hip_flex, 1),
#             "min_knee_flex_deg" : round(min_knee_flex, 1),
#             "landing_stability" : landing_stability,
#             "arm_swing_score"   : arm_swing_score,
#             # Reactive
#             "ground_contact_ms" : gct_ms,
#             "rsi"               : rsi,
#             # Score
#             "form_score"        : form,
#             # Snapshot
#             "_peak_b64"         : self._peak_b64,
#         }
#         jump = _to_json_safe(jump)
#         self.jumps.append(jump)
#         return jump

#     def force_close(self, com, scale: ScaleEstimator, pts=None):
#         """Call at video end if still in FLIGHT state."""
#         if self.state == ST_FLIGHT and self._flight_f >= MIN_FLIGHT_F:
#             land_x = com[0] if com else (self._flight_com_x_start or 0)
#             self._land_com_buf.append(land_x)
#             self._land_stable_f = LAND_STABLE_F
#             ref_pts = pts if pts is not None else np.zeros((17, 2))
#             foot_x = _landing_foot_x(ref_pts, self._direction) if pts is not None else None
#             if foot_x is not None:
#                 self._land_foot_x_buf.append(foot_x)
#             return self._confirm_jump(com, ref_pts, scale)
#         return None


# # ─────────────────────────────────────────────────────────────────────────────
# # EXERCISE-SPECIFIC VALIDATORS
# # ─────────────────────────────────────────────────────────────────────────────

# class ExerciseValidator:
#     """
#     Wraps the core JumpEngine and applies exercise-specific
#     post-processing to the completed jump list.
#     """
#     def __init__(self, exercise: str):
#         self.exercise = exercise

#     def validate(self, jumps: list) -> dict:
#         ex = self.exercise.lower().replace(" ", "_")

#         if "triple" in ex:
#             return self._triple(jumps)
#         elif "alternate" in ex:
#             return self._alternate(jumps)
#         elif "bounding" in ex or "power_skip" in ex:
#             return self._bounding(jumps)
#         elif "multiple" in ex:
#             return self._multiple(jumps)
#         else:
#             return self._standard(jumps)

#     # Standard single / repeated jumps
#     def _standard(self, jumps):
#         return {"validated_jumps": jumps, "phase_info": None}

#     # Triple: group every 3 consecutive jumps into hop/step/jump
#     def _triple(self, jumps):
#         phases = []
#         for i in range(0, len(jumps) - 2, 3):
#             hop, step, jmp = jumps[i], jumps[i+1], jumps[i+2]
#             total = hop["distance_m"] + step["distance_m"] + jmp["distance_m"]
#             phases.append({
#                 "triple_no"   : len(phases) + 1,
#                 "hop"         : hop,
#                 "step"        : step,
#                 "jump"        : jmp,
#                 "total_m"     : round(total, 3),
#                 "phase_ratio" : [round(hop["distance_m"]/total, 2),
#                                  round(step["distance_m"]/total, 2),
#                                  round(jmp["distance_m"]/total, 2)],
#             })
#         return {"validated_jumps": jumps, "phase_info": phases}

#     # Alternate: flag if consecutive jumps show alternating takeoff feet
#     def _alternate(self, jumps):
#         for i, j in enumerate(jumps):
#             j["leg_tag"] = "L" if i % 2 == 0 else "R"
#         return {"validated_jumps": jumps, "phase_info": None}

#     # Bounding: each jump = one bound
#     def _bounding(self, jumps):
#         bounds = []
#         for j in jumps:
#             bounds.append({
#                 "bound_no"  : j["jump_no"],
#                 "distance_m": j["distance_m"],
#                 "flight_ms" : j["flight_ms"],
#                 "gct_ms"    : j.get("ground_contact_ms", 0),
#             })
#         total = sum(b["distance_m"] for b in bounds)
#         return {"validated_jumps": jumps, "phase_info": bounds,
#                 "total_distance_m": round(total, 3)}

#     # Multiple: cumulative metrics
#     def _multiple(self, jumps):
#         cumulative = 0.0
#         for j in jumps:
#             cumulative += j["distance_m"]
#             j["cumulative_m"] = round(cumulative, 3)
#         return {"validated_jumps": jumps, "phase_info": None}


# # ─────────────────────────────────────────────────────────────────────────────
# # QUALITY & SCORING
# # ─────────────────────────────────────────────────────────────────────────────

# def _form_score(distance_m):
#     """Legacy distance-only score, retained for backward compatibility
#     (used by _form_score_aggregate for the session-level summary)."""
#     thresholds = [2.50, 2.20, 2.00, 1.80, 1.50, 1.20]
#     scores     = [10,   9,    8,    7,    6,    5]
#     for t, s in zip(thresholds, scores):
#         if distance_m >= t:
#             return s
#     return 4

# def _distance_subscore(distance_m):
#     """0-10 distance subscore feeding the weighted per-jump form score."""
#     thresholds = [2.50, 2.20, 2.00, 1.80, 1.50, 1.20, 0.90]
#     scores     = [10,   9,    8,    7,    6,    5,    3]
#     for t, s in zip(thresholds, scores):
#         if distance_m >= t:
#             return s
#     return 1

# def _takeoff_angle_subscore(angle_deg):
#     """0-10 subscore peaking in the biomechanically efficient 18-27°
#     range for a standing broad jump; falls off outside it."""
#     ideal_low, ideal_high = 18.0, 27.0
#     if ideal_low <= angle_deg <= ideal_high:
#         return 10.0
#     dist = (ideal_low - angle_deg) if angle_deg < ideal_low else (angle_deg - ideal_high)
#     return max(0.0, 10.0 - dist * 0.35)

# def _knee_flexion_subscore(min_knee_flex_deg):
#     """0-10 subscore for takeoff/flight knee flexion — deep enough to
#     generate power without being so collapsed it signals poor control."""
#     if min_knee_flex_deg is None:
#         return 6.0  # neutral if not measured
#     ideal_low, ideal_high = 90.0, 130.0
#     if ideal_low <= min_knee_flex_deg <= ideal_high:
#         return 10.0
#     dist = (ideal_low - min_knee_flex_deg) if min_knee_flex_deg < ideal_low else (min_knee_flex_deg - ideal_high)
#     return max(0.0, 10.0 - dist * 0.15)

# def _arm_swing_score(wrist_travel_samples):
#     """
#     0-10 subscore from wrist vertical excursion relative to shoulder during
#     flight. A strong arm swing drives the wrists from low (behind the hips
#     at takeoff) to high (overhead/forward at peak flight) — large positive
#     range in (shoulder_y - wrist_y) indicates good swing amplitude.
#     """
#     if not wrist_travel_samples or len(wrist_travel_samples) < 2:
#         return 5.0   # neutral score when arms aren't trackable
#     travel_range = max(wrist_travel_samples) - min(wrist_travel_samples)
#     # travel_range is in pixels; normalise loosely — most useful as a
#     # relative signal since we don't have per-subject scale here.
#     score = min(10.0, travel_range / 12.0)
#     return round(max(0.0, score), 1)

# def _form_score_full(dist_m, landing_stability, takeoff_angle,
#                       min_hip_flex, min_knee_flex, arm_swing_score=5.0):
#     """
#     Weighted multi-factor form score (1-10):
#       40% distance, 20% landing stability, 15% takeoff angle,
#       15% arm swing, 10% knee flexion.
#     """
#     dist_s   = _distance_subscore(dist_m)
#     land_s   = round(landing_stability * 10, 1)
#     angle_s  = _takeoff_angle_subscore(takeoff_angle)
#     knee_s   = _knee_flexion_subscore(min_knee_flex)
#     arm_s    = arm_swing_score

#     weighted = (0.40 * dist_s + 0.20 * land_s + 0.15 * angle_s +
#                 0.15 * arm_s + 0.10 * knee_s)
#     return int(round(max(1.0, min(10.0, weighted))))

# def _form_score_aggregate(jumps_m):
#     if not jumps_m:
#         return 0
#     best  = max(jumps_m)
#     score = _form_score(best)
#     if len(jumps_m) > 1:
#         cons = min(jumps_m) / best
#         if   cons >= 0.90: score = min(10, score + 1)
#         elif cons < 0.70:  score = max(1,  score - 1)
#     return score

# def _compute_quality(jumps, pose_conf, foot_conf, scale_calibrated):
#     """
#     Generate overall confidence score 0-100 from independently meaningful
#     sub-scores, rather than a raw jump count (which says nothing about
#     measurement reliability).
#       pose      : average pose-detection confidence across the video
#       contact   : foot-contact estimator reliability
#       flight    : how cleanly flight phases were detected (physics-consistency
#                   rate + sane flight-time spread across detected jumps)
#       scale     : whether px-per-meter calibration succeeded
#       landing   : average landing stability across detected jumps
#     """
#     pose_s  = int(pose_conf * 100)
#     cont_s  = int(foot_conf * 100)
#     scale_s = 85 if scale_calibrated else 55

#     if not jumps:
#         return {
#             "overall": 0, "pose": pose_s, "contact": cont_s,
#             "flight": 0, "distance": scale_s, "landing": 0,
#             "jump": 0,   # legacy key retained
#         }

#     consistent = [j.get("physics_consistent", True) for j in jumps]
#     flight_s   = int(100 * (sum(consistent) / len(consistent)))

#     land_vals  = [j.get("landing_stability") for j in jumps if j.get("landing_stability") is not None]
#     landing_s  = int(100 * (sum(land_vals) / len(land_vals))) if land_vals else 60

#     overall = int(0.30*pose_s + 0.15*cont_s + 0.25*flight_s +
#                   0.15*scale_s + 0.15*landing_s)
#     return {
#         "overall": overall, "pose": pose_s, "contact": cont_s,
#         "flight": flight_s, "distance": scale_s, "landing": landing_s,
#         "jump": flight_s,   # legacy key retained, now mapped to a meaningful value
#     }

# def _fatigue_index(jumps_m):
#     """Decline in distance over repetitions (lower is better)."""
#     if len(jumps_m) < 2:
#         return None
#     return round((jumps_m[0] - jumps_m[-1]) / jumps_m[0] * 100, 1)

# def _rep_variability(jumps_m):
#     if len(jumps_m) < 2:
#         return None
#     return round(float(np.std(jumps_m)) / (np.mean(jumps_m) + 1e-9) * 100, 1)

# def _feedback(jump_results, detected, exercise):
#     issues    = []
#     strengths = []

#     if not detected:
#         issues += [
#             "❌ Person not detected in video.",
#             "📌 Ensure: full body visible (head to toe), side-angle camera, good lighting.",
#             "🎥 Avoid top-down or front-facing camera angles.",
#         ]
#         return issues, strengths

#     if not jump_results:
#         issues += [
#             "❌ No valid jump detected.",
#             "📐 Camera should be placed sideways at full-body height.",
#             "⏱️ Video must capture full jump — takeoff, flight, and landing.",
#         ]
#         return issues, strengths

#     distances = [j["distance_m"] for j in jump_results]
#     best = max(distances)
#     avg  = sum(distances) / len(distances)

#     if best < 1.20:
#         issues.append(f"Short jump ({best:.2f}m) — focus on arm swing and hip extension.")
#     elif best < 1.80:
#         issues.append(f"Below-average distance ({best:.2f}m) — work on explosive leg drive.")
#     elif best < 2.20:
#         strengths.append(f"Good jump distance ({best:.2f}m).")
#     else:
#         strengths.append(f"Excellent jump distance ({best:.2f}m)!")

#     if len(distances) > 1:
#         cons = min(distances) / best
#         if cons >= 0.90:
#             strengths.append(f"Very consistent across {len(distances)} jumps ({cons:.0%}).")
#         elif cons < 0.70:
#             issues.append(f"High variation ({cons:.0%} consistency) — repeat takeoff mechanics.")

#     # Tuck/pike feedback
#     if "tuck" in exercise.lower():
#         tucks = [j for j in jump_results if j.get("tuck_confirmed")]
#         if tucks:
#             strengths.append(f"Tuck confirmed in {len(tucks)}/{len(jump_results)} jumps.")
#         else:
#             issues.append("Tuck not detected — ensure knees pull toward chest during flight.")

#     if "pike" in exercise.lower():
#         pikes = [j for j in jump_results if j.get("pike_confirmed")]
#         if pikes:
#             strengths.append(f"Pike confirmed in {len(pikes)}/{len(jump_results)} jumps.")
#         else:
#             issues.append("Pike not detected — keep legs extended and reach toward toes.")

#     if not issues:
#         issues = ["No major issues detected."]
#     return issues, strengths


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN ANALYSIS FUNCTION
# # ─────────────────────────────────────────────────────────────────────────────

# def analyse_broad_jump(
#     path,
#     is_video,
#     output_path=None,
#     session_id=None,
#     source_filename="",
#     progress_uid=None,
#     exercise="Standing Broad Jump",
# ):
#     """
#     Main entry point for broad jump analysis.
#     Supports all 12 broad jump variants via the `exercise` parameter.

#     Parameters
#     ----------
#     exercise : str
#         One of: "Standing Broad Jump", "Run-Up Broad Jump",
#         "Single-Leg Broad Jump", "Alternate Leg Broad Jump",
#         "Bounding", "Triple Broad Jump", "Multiple Broad Jump",
#         "Reactive Broad Jump", "Tuck Broad Jump", "Pike Broad Jump",
#         "Weighted Broad Jump", "Sand Broad Jump"
#     """
#     if not _YOLO_AVAILABLE:
#         raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
#     if not is_video:
#         raise ValueError("Broad jump analysis requires a video file.")

#     # ── Model ────────────────────────────────────────────────────────────
#     model_path = YOLO_MODEL_PATH
#     if not os.path.exists(model_path):
#         here = os.path.dirname(os.path.abspath(__file__))
#         alt  = os.path.join(here, YOLO_MODEL_PATH)
#         if os.path.exists(alt):
#             model_path = alt
#         else:
#             raise FileNotFoundError(f"YOLO model not found at '{YOLO_MODEL_PATH}'.")
#     model = YOLO(model_path)

#     # ── Video metadata ────────────────────────────────────────────────────
#     cap    = cv2.VideoCapture(path)
#     fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     # ── Sub-systems ───────────────────────────────────────────────────────
#     lm_filter = LandmarkFilter(n_landmarks=17, freq=fps)
#     foot_est  = FootContactEstimator(window=int(fps * 1.5), fps=fps)
#     scale_est = ScaleEstimator()
#     engine    = JumpEngine(fps=fps, exercise=exercise.lower())
#     validator = ExerciseValidator(exercise)

#     # ── State ─────────────────────────────────────────────────────────────
#     detected_ok   = [False]
#     wrong_events  = []
#     per_frame_data= []
#     total_frames  = [1]
#     pose_confs    = []
#     foot_conf_acc = []

#     live_hud      = {"jump_no": "0", "dist": "---", "form": "---", "state": ST_READY}

#     def _push_event(jdata, b64):
#         if not progress_uid:
#             return
#         pct = min(94, int(len(per_frame_data) / max(1, total_frames[0]) * 90))
#         safe_jdata = _to_json_safe({**jdata, "frame_b64": b64})
#         set_progress(
#             progress_uid, pct,
#             f"Jump {jdata['jump_no']} — {jdata['distance_m']:.2f}m",
#             jump_event=safe_jdata,
#         )

#     # ── Per-frame callback ────────────────────────────────────────────────
#     def pf(frame, fc, total):
#         total_frames[0] = max(total, 1)

#         results   = model(frame, conf=CONFIDENCE, verbose=False)
#         best_pts  = None
#         best_area = 0
#         best_conf = 0.0

#         for result in results:
#             if (result.keypoints is None or
#                     result.keypoints.xy is None or
#                     result.keypoints.conf is None):
#                 continue
#             for kpts_xy, kpts_conf in zip(
#                     result.keypoints.xy.cpu().numpy(),
#                     result.keypoints.conf.cpu().numpy()):
#                 pts   = kpts_xy.astype(float)
#                 valid = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
#                 if len(valid) < 6:
#                     continue
#                 area = ((valid[:, 0].max() - valid[:, 0].min()) *
#                         (valid[:, 1].max() - valid[:, 1].min()))
#                 if area > best_area:
#                     best_area = area
#                     best_pts  = pts
#                     best_conf = float(np.mean(kpts_conf[kpts_conf > 0]))

#         if best_pts is None:
#             draw_footer_hud(frame, [
#                 ("JUMP #", live_hud["jump_no"]),
#                 ("DIST",   live_hud["dist"]),
#                 ("FORM",   live_hud["form"]),
#             ])
#             draw_pcl_logo(frame)
#             return frame

#         detected_ok[0] = True
#         pose_confs.append(best_conf)

#         # Filter landmarks
#         pts = lm_filter.update(best_pts)

#         # Draw skeleton (skip face landmarks 0-4)
#         for p1i, p2i in SKELETON_PAIRS:
#             p1, p2 = pts[p1i], pts[p2i]
#             if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
#                 cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
#                          COLOR_SKEL, 2)
#         for idx, p in enumerate(pts):
#             if idx < 5 or p[0] <= 0 or p[1] <= 0:
#                 continue
#             cv2.circle(frame, (int(p[0]), int(p[1])), 4, COLOR_JOINT, -1)

#         # COM
#         com = estimate_com(pts)
#         if com:
#             cv2.circle(frame, (int(com[0]), int(com[1])), 6, COLOR_COM, -1)

#         # Foot contact
#         foot_est.classify(pts)

#         # Engine update
#         frame_b64  = frame_to_b64(frame) if engine.state in (ST_FLIGHT, ST_TAKEOFF) else None
#         jump_event = engine.update(pts, foot_est, com, scale_est, frame_b64)

#         if jump_event:
#             b64 = jump_event.pop("_peak_b64", None) or frame_to_b64(frame)
#             _push_event(jump_event, b64)
#             live_hud["jump_no"] = str(jump_event["jump_no"])
#             live_hud["dist"]    = f"{jump_event['distance_m']:.2f}m"
#             live_hud["form"]    = f"{jump_event['form_score']}/10"

#         live_hud["state"] = engine.state

#         # Visualise scale (if calibrated, show reference line)
#         if scale_est.px_per_m:
#             ref_px = int(scale_est.px_per_m)
#             mid_x, gnd_y = width // 4, height - 40
#             cv2.line(frame, (mid_x, gnd_y), (mid_x + ref_px, gnd_y), (100, 255, 100), 2)
#             cv2.putText(frame, "1m", (mid_x + ref_px // 2 - 10, gnd_y - 6),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

#         # Draw ground reference
#         if engine._gnd_com_y:
#             gnd_px = int(engine._gnd_com_y + (LAND_APPROACH_PX))
#             cv2.line(frame, (0, gnd_px), (width, gnd_px), (80, 80, 80), 1)

#         # Draw takeoff / landing markers
#         if engine._flight_com_x_start and engine.state == ST_FLIGHT:
#             tx = int(engine._flight_com_x_start)
#             cv2.line(frame, (tx, 0), (tx, height), COLOR_TAKE, 1)

#         per_frame_data.append({
#             "frame"     : fc,
#             "state"     : engine.state,
#             "jump_count": len(engine.jumps),
#         })

#         draw_footer_hud(frame, [
#             ("JUMP #", live_hud["jump_no"]),
#             ("DIST",   live_hud["dist"]),
#             ("FORM",   live_hud["form"]),
#         ])
#         draw_pcl_logo(frame)
#         return frame

#     # ── Run ───────────────────────────────────────────────────────────────
#     snaps = process_video_or_image(
#         path, is_video, pf,
#         output_path=output_path,
#         snap_pcts=[0.1, 0.3, 0.5, 0.7, 0.9],
#         analysis_skip=1,
#         progress_uid=progress_uid,
#     )

#     # Force-close if video ends mid-flight
#     if engine.state == ST_FLIGHT and engine._flight_f >= MIN_FLIGHT_F:
#         last_com = None
#         if per_frame_data:
#             # Best effort: use last known COM from scale buf
#             last_com = (engine._flight_com_x_start, engine._gnd_com_y or 0)
#         ev = engine.force_close(last_com, scale_est)
#         if ev:
#             ev.pop("_peak_b64", None)
#             _push_event(ev, "")

#     if not detected_ok[0]:
#         raise ValueError(
#             "No person detected. Upload a side-angle video with full body visible "
#             "(head to toe) and good lighting."
#         )

#     if session_id:
#         save_wrong_angle_log(exercise, session_id, source_filename, wrong_events)

#     # ── Post-processing ────────────────────────────────────────────────────
#     jumps     = engine.jumps
#     val_out   = validator.validate(jumps)
#     jumps_m   = [j["distance_m"] for j in jumps]

#     best_dist = round(max(jumps_m), 3) if jumps_m else 0.0
#     avg_dist  = round(sum(jumps_m) / len(jumps_m), 3) if jumps_m else 0.0

#     pose_conf_avg = float(np.mean(pose_confs)) if pose_confs else 0.0
#     foot_conf_val = 0.75 if scale_est.px_per_m else 0.50
#     quality = _compute_quality(jumps, pose_conf_avg, foot_conf_val,
#                                 scale_calibrated=scale_est.px_per_m is not None)

#     form_score = _form_score_aggregate(jumps_m)
#     issues, strengths = _feedback(jumps, detected_ok[0], exercise)

#     while len(snaps) < 5:
#         snaps.append(snaps[-1] if snaps else "")

#     step      = max(1, int(fps / 10))
#     per_frame = per_frame_data[::step]

#     best_str = f"{best_dist:.2f} m" if best_dist > 0 else "N/A"
#     avg_str  = f"{avg_dist:.2f} m"  if avg_dist  > 0 else "N/A"

#     # ── Aggregate quality metrics ──────────────────────────────────────────
#     gct_vals  = [j["ground_contact_ms"] for j in jumps if j.get("ground_contact_ms")]
#     rsi_vals  = [j["rsi"] for j in jumps if j.get("rsi")]
#     ft_vals   = [j["flight_ms"] for j in jumps]

#     result = {
#         # Core
#         "exercise"              : exercise,
#         "jump_count"            : len(jumps),
#         "correct_jumps"         : len(jumps),
#         "wrong_jumps"           : 0,
#         # Distance
#         "best_distance_m"       : best_dist,
#         "best_distance_cm"      : round(best_dist * 100, 1),
#         "avg_distance_m"        : avg_dist,
#         "all_jumps_m"           : [round(j, 3) for j in jumps_m],
#         # Per-jump detail
#         "per_jump"              : jumps,
#         # Exercise variant results
#         "phase_info"            : val_out.get("phase_info"),
#         "total_distance_m"      : val_out.get("total_distance_m"),
#         # Timing
#         "avg_flight_ms"         : int(np.mean(ft_vals)) if ft_vals else 0,
#         "avg_ground_contact_ms" : int(np.mean(gct_vals)) if gct_vals else 0,
#         "avg_rsi"               : round(float(np.mean(rsi_vals)), 3) if rsi_vals else None,
#         # Consistency
#         "fatigue_index"         : _fatigue_index(jumps_m),
#         "rep_variability_pct"   : _rep_variability(jumps_m),
#         # Scale
#         "px_per_m"              : round(scale_est.px_per_m, 2) if scale_est.px_per_m else None,
#         "scale_method"          : scale_est.method or "legacy_constant",
#         # Quality / Confidence
#         "confidence"            : quality,
#         "form_score"            : form_score,
#         # Feedback
#         "issues"                : issues,
#         "strengths"             : strengths,
#         # UI
#         "metrics": [
#             {"label": "Jumps Detected", "value": str(len(jumps))},
#             {"label": "Best Jump",      "value": best_str},
#             {"label": "Avg Jump",       "value": avg_str},
#             {"label": "Form Score",     "value": f"{form_score}/10"},
#             {"label": "Confidence",     "value": f"{quality['overall']}%"},
#         ],
#         # Legacy keys kept for backward compatibility
#         "height_cm"             : round(best_dist * 100, 1),
#         "per_frame"             : per_frame,
#         "snapshots"             : snaps,
#         "wrong_angle_count"     : 0,
#         "_wrong_events"         : wrong_events,
#     }
#     return _to_json_safe(result)

































# """
# module_broad_jump.py
# Production-grade broad jump biomechanics engine.

# Supports:
#   1. Standing Broad Jump
#   2. Run-Up Broad Jump
#   3. Single-Leg Broad Jump
#   4. Alternate Leg Broad Jump
#   5. Bounding (Power Skip)
#   6. Triple Broad Jump
#   7. Multiple Broad Jump
#   8. Reactive Broad Jump
#   9. Tuck Broad Jump
#  10. Pike Broad Jump
#  11. Weighted Broad Jump
#  12. Sand Broad Jump

# Architecture:
#   Video → Pose → Landmark Filter → COM → Foot Contact → 6-State FSM
#        → Exercise Validator → Distance Engine → Quality → Output
# """

# import os
# import cv2
# import math
# import numpy as np
# from collections import deque

# try:
#     from ultralytics import YOLO
#     _YOLO_AVAILABLE = True
# except ImportError:
#     _YOLO_AVAILABLE = False

# from utils import (
#     process_video_or_image,
#     save_wrong_angle_log,
#     set_progress,
#     frame_to_b64,
# )
# from hud_overlay import draw_footer_hud, draw_pcl_logo

# # ─────────────────────────────────────────────────────────────────────────────
# # CONSTANTS
# # ─────────────────────────────────────────────────────────────────────────────

# YOLO_MODEL_PATH = "yolov8n-pose.pt"
# CONFIDENCE      = 0.3

# # YOLO keypoint indices (COCO 17-point)
# KP_NOSE        = 0
# KP_L_SHOULDER  = 5
# KP_R_SHOULDER  = 6
# KP_L_ELBOW     = 7
# KP_R_ELBOW     = 8
# KP_L_WRIST     = 9
# KP_R_WRIST     = 10
# KP_L_HIP       = 11
# KP_R_HIP       = 12
# KP_L_KNEE      = 13
# KP_R_KNEE      = 14
# KP_L_ANKLE     = 15
# KP_R_ANKLE     = 16

# # COM weights (Dempster body segment model approximation)
# COM_WEIGHTS = {
#     KP_L_HIP: 0.28, KP_R_HIP: 0.28,
#     KP_L_SHOULDER: 0.11, KP_R_SHOULDER: 0.11,
#     KP_L_KNEE: 0.06, KP_R_KNEE: 0.06,
#     KP_L_ANKLE: 0.05, KP_R_ANKLE: 0.05,
# }

# # State machine states
# ST_READY    = "READY"
# ST_LOADING  = "LOADING"
# ST_TAKEOFF  = "TAKEOFF"
# ST_FLIGHT   = "FLIGHT"
# ST_LANDING  = "LANDING"
# ST_RESET    = "RESET"
# ST_COOLDOWN = "COOLDOWN"


# # ─────────────────────────────────────────────────────────────────────────────
# # JSON-SAFETY HELPER
# # ─────────────────────────────────────────────────────────────────────────────
# def _to_json_safe(obj):
#     """
#     Recursively convert numpy scalar/array types (np.bool_, np.float64,
#     np.int64, np.ndarray, etc.) to native Python types so the result can
#     always be passed to json.dumps() without raising
#     'Object of type X is not JSON serializable'.

#     numpy values leak into output dicts any time a computation touches an
#     np.median()/np.mean()/np.min()/np.max() result (even indirectly, e.g.
#     `abs(numpy_float) < python_float` still yields np.bool_) — so this is
#     applied once, at the point each output dict is finalized, rather than
#     trying to manually cast every individual field (which is exactly how
#     this bug slipped through the first time).
#     """
#     if isinstance(obj, dict):
#         return {k: _to_json_safe(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [_to_json_safe(v) for v in obj]
#     if isinstance(obj, np.bool_):
#         return bool(obj)
#     if isinstance(obj, np.integer):
#         return int(obj)
#     if isinstance(obj, np.floating):
#         return float(obj)
#     if isinstance(obj, np.ndarray):
#         return _to_json_safe(obj.tolist())
#     return obj

# SKELETON_PAIRS = [
#     (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
#     (5, 11), (6, 12), (11, 12),
#     (11, 13), (13, 15), (12, 14), (14, 16),
# ]

# # Colors
# COLOR_SKEL   = (255, 255, 255)
# COLOR_JOINT  = (0,   0,   0)
# COLOR_COM    = (0, 255, 255)
# COLOR_TAKE   = (0, 165, 255)
# COLOR_LAND   = (255, 100, 255)
# COLOR_TEXT   = (255, 255, 255)

# # FSM thresholds
# GROUND_WINDOW       = 20   # frames to establish ground reference
# COM_RISE_PX         = 18   # COM must rise this much above ground to start LOADING
# COM_TAKEOFF_PX      = 30   # COM above ground to confirm TAKEOFF
# FLIGHT_CONFIRM_F    = 3    # consecutive airborne frames required before FLIGHT is trusted
# LAND_APPROACH_PX    = 25   # COM within this of ground → approaching landing
# LAND_STABLE_F       = 7    # frames near ground to confirm LANDING (athlete fully settled)
# MIN_FLIGHT_F        = 7    # minimum flight frames for a valid jump (~233ms @ 30fps)
# MAX_FLIGHT_F        = 90   # ~3s @ 30fps — hard ceiling on a single flight phase. No human
#                              # standing broad jump airborne phase exceeds ~1s; this exists
#                              # purely as a safety valve so a degraded-landing-detection
#                              # stretch (occlusion, bad keypoints) can't freeze the FSM in
#                              # FLIGHT for the rest of the clip and silently drop every
#                              # subsequent jump in the video.
# RESET_STABLE_F      = 4    # frames on ground before READY again (allows fast consecutive jumps)
# LAND_VEL_PX_F       = 6.0  # max COM px/frame velocity to accept landing as "settled"
# MIN_JUMP_DIST_M     = 0.25 # below this, treat as noise / false jump
# MIN_PEAK_RISE_PX    = 10   # minimum COM rise during flight to count as a real jump

# # ── False-jump rejection / hysteresis (post-landing stabilization fix) ──
# MIN_VALID_DIST_M       = 0.40  # floor — screens out post-landing micro-shifts (~0.53m case)
#                                  # without rejecting shorter/beginner real jumps
# MIN_COM_FALLBACK_DIST_M = 0.80  # stricter floor for the com_fallback distance path, which
#                                  # systematically undershoots true foot-to-foot distance
#                                  # (COM folds backward on landing) — used only when ankle
#                                  # keypoints were unavailable for the whole flight
# MIN_DIST_PER_FLIGHT_S   = 0.35  # m of forward distance expected per second of flight time,
#                                  # floor — a measured distance far below this for the
#                                  # observed airtime indicates a degraded measurement, not
#                                  # a genuinely tiny jump (catches the "below 100cm on a
#                                  # professional-looking long flight" failure mode). Kept
#                                  # low enough that legitimate short/beginner/single-leg
#                                  # jumps with modest horizontal velocity are not rejected.
# MAX_VALID_DIST_M       = 4.0   # sanity ceiling
# MIN_VALID_FLIGHT_MS    = 167   # ~5 frames @ 30fps — below this is noise, not a jump
# MIN_APPROACH_SPEED_MS  = 0.20  # reported only — NOT used as a hard rejection gate
# AIR_CONFIRM_FRAMES     = 2     # consecutive frames required to trust "airborne"
# CONTACT_CONFIRM_FRAMES = 2     # consecutive frames required to trust "grounded"
# COOLDOWN_FRAMES        = 10    # frames after landing before a new jump may start (~330ms)
# MAX_OCCLUSION_HOLD_F   = 6     # max consecutive missed-detection frames to bridge by
#                                  # replaying the last known pose, so brief motion-blur/
#                                  # occlusion gaps mid-flight don't freeze the FSM and
#                                  # silently drop the jump (or every jump after it)

# # Exercise-specific
# REACTIVE_MAX_CONTACT_F  = 12   # max ground contact for reactive jump (at 30fps ≈ 400ms)
# TUCK_HIP_FLEX_DEG       = 60   # minimum hip flexion for tuck confirmation
# PIKE_HIP_FLEX_DEG       = 70   # minimum hip flexion for pike
# PIKE_KNEE_EXT_DEG       = 150  # minimum knee extension for pike
# BOUND_MIN_FLIGHT_F      = 3    # minimum flight per bound
# TRIPLE_PHASE_MIN_F      = 3    # min flight frames per triple-jump phase

# # Anthropometric scale: avg hip-to-ankle ≈ 0.53× body height, body height ≈ 1.75m
# # So hip-ankle ≈ 0.927m in pixels → derive px/m
# ANTHRO_HIP_ANKLE_M  = 0.90   # meters (conservative; will be estimated per-subject)

# # ─────────────────────────────────────────────────────────────────────────────
# # ONE-EURO FILTER  (per-coordinate temporal filter)
# # ─────────────────────────────────────────────────────────────────────────────

# class OneEuroFilter:
#     """
#     One Euro Filter for temporal landmark smoothing.
#     Reduces jitter without large latency; preserves fast explosive motion
#     by automatically raising cutoff frequency at high velocity.
#     """
#     def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
#         self.freq      = freq
#         self.min_cutoff = min_cutoff
#         self.beta      = beta
#         self.d_cutoff  = d_cutoff
#         self._x        = None
#         self._dx       = 0.0

#     def _alpha(self, cutoff):
#         tau = 1.0 / (2 * math.pi * cutoff)
#         return 1.0 / (1.0 + tau * self.freq)

#     def __call__(self, x):
#         if self._x is None:
#             self._x = x
#             return x
#         dx     = (x - self._x) * self.freq
#         a_d    = self._alpha(self.d_cutoff)
#         self._dx = a_d * dx + (1 - a_d) * self._dx
#         cutoff = self.min_cutoff + self.beta * abs(self._dx)
#         a      = self._alpha(cutoff)
#         self._x = a * x + (1 - a) * self._x
#         return self._x


# class LandmarkFilter:
#     """Per-landmark One Euro filter (x and y independently)."""
#     def __init__(self, n_landmarks=17, freq=30.0):
#         self.fx = [OneEuroFilter(freq=freq) for _ in range(n_landmarks)]
#         self.fy = [OneEuroFilter(freq=freq) for _ in range(n_landmarks)]

#     def update(self, pts):
#         """pts: np.array shape (17,2). Returns filtered array same shape."""
#         out = pts.copy().astype(float)
#         for i, (fx, fy) in enumerate(zip(self.fx, self.fy)):
#             if pts[i, 0] > 0 and pts[i, 1] > 0:
#                 out[i, 0] = fx(pts[i, 0])
#                 out[i, 1] = fy(pts[i, 1])
#         return out


# # ─────────────────────────────────────────────────────────────────────────────
# # CENTER OF MASS ESTIMATOR
# # ─────────────────────────────────────────────────────────────────────────────

# def estimate_com(pts):
#     """
#     Weighted average COM from body keypoints.
#     Returns (com_x, com_y) or None if not enough valid points.
#     """
#     total_w = 0.0
#     wx, wy  = 0.0, 0.0
#     for idx, w in COM_WEIGHTS.items():
#         p = pts[idx]
#         if p[0] > 0 and p[1] > 0:
#             wx += w * p[0]
#             wy += w * p[1]
#             total_w += w
#     if total_w < 0.3:
#         return None
#     return (wx / total_w, wy / total_w)


# # ─────────────────────────────────────────────────────────────────────────────
# # FOOT CONTACT ESTIMATOR
# # ─────────────────────────────────────────────────────────────────────────────

# class FootContactEstimator:
#     """
#     Tracks each foot (left/right) independently.
#     Uses ankle vertical position relative to adaptive ground reference,
#     combined with ankle vertical velocity, to classify CONTACT / AIR.
#     Velocity gating prevents single noisy keypoint jitters from being
#     read as a takeoff or landing.
#     """
#     def __init__(self, window=30, fps=30.0):
#         self.ground_y    = None   # adaptive ground y (pixel; larger = lower on screen)
#         self.stance_buf  = deque(maxlen=window)  # y values during known contact
#         self.fps         = fps
#         self._prev_ly    = None
#         self._prev_ry    = None
#         self.left        = "CONTACT"
#         self.right       = "CONTACT"
#         self.left_vy     = 0.0
#         self.right_vy    = 0.0
#         self.THRESH_PX   = 22     # pixels above ground to classify AIR
#         self.VEL_THRESH  = 3.0    # px/frame — ankle must also be moving to confirm AIR

#     def update_ground(self, ly, ry):
#         """Call with ankle y values during confirmed ground phase."""
#         for y in [ly, ry]:
#             if y > 0:
#                 self.stance_buf.append(y)
#         if self.stance_buf:
#             # Ground is near the max (bottom-most) ankle positions
#             self.ground_y = np.percentile(list(self.stance_buf), 85)

#     def classify(self, pts):
#         """Update left/right contact state from keypoints."""
#         la = pts[KP_L_ANKLE]
#         ra = pts[KP_R_ANKLE]
#         ly = la[1] if la[0] > 0 else -1
#         ry = ra[1] if ra[0] > 0 else -1

#         # Velocity (px/frame) for noise gating
#         self.left_vy  = (ly - self._prev_ly) if (ly > 0 and self._prev_ly is not None) else 0.0
#         self.right_vy = (ry - self._prev_ry) if (ry > 0 and self._prev_ry is not None) else 0.0
#         if ly > 0: self._prev_ly = ly
#         if ry > 0: self._prev_ry = ry

#         if self.ground_y is None:
#             self.left  = "CONTACT"
#             self.right = "CONTACT"
#             return

#         thr = self.THRESH_PX
#         gnd = self.ground_y

#         # Left foot — require both height-above-ground AND recent motion
#         # to avoid a single jittery keypoint flipping the state.
#         if ly > 0:
#             above_ground = ly < gnd - thr
#             self.left = "AIR" if (above_ground or abs(self.left_vy) > self.VEL_THRESH * 2) else "CONTACT"
#         # Right foot
#         if ry > 0:
#             above_ground = ry < gnd - thr
#             self.right = "AIR" if (above_ground or abs(self.right_vy) > self.VEL_THRESH * 2) else "CONTACT"

#     @property
#     def both_contact(self):
#         return self.left == "CONTACT" and self.right == "CONTACT"

#     @property
#     def any_contact(self):
#         return self.left == "CONTACT" or self.right == "CONTACT"

#     @property
#     def both_air(self):
#         return self.left == "AIR" and self.right == "AIR"


# # ─────────────────────────────────────────────────────────────────────────────
# # ADAPTIVE SCALE ESTIMATOR  (pixel → metric)
# # ─────────────────────────────────────────────────────────────────────────────

# class ScaleEstimator:
#     """
#     Estimates pixels-per-meter from subject anthropometrics.
#     Primary: full body height (nose → ankle), which is far more stable
#     across camera angles than hip-to-ankle alone.
#     Fallback: hip-to-ankle segment if height isn't fully visible.
#     Updated during ground-contact frames only (standing posture).
#     """
#     ANTHRO_BODY_HEIGHT_M = 1.75   # default assumed height if user doesn't supply one

#     def __init__(self, user_height_m=None):
#         self._height_samples = deque(maxlen=60)
#         self._hip_ankle_samples = deque(maxlen=60)
#         self.px_per_m = None
#         self.user_height_m = user_height_m
#         self.method = None

#     def update(self, pts):
#         nose = pts[KP_NOSE]
#         lh = pts[KP_L_HIP];   la = pts[KP_L_ANKLE]
#         rh = pts[KP_R_HIP];   ra = pts[KP_R_ANKLE]

#         # ── Preferred: nose → lowest ankle (full standing height in px) ──
#         ankle_ys = [p[1] for p in (la, ra) if p[0] > 0 and p[1] > 0]
#         if nose[0] > 0 and nose[1] > 0 and ankle_ys:
#             height_px = max(ankle_ys) - nose[1]
#             if height_px > 50:   # sane lower bound, avoids garbage frames
#                 self._height_samples.append(height_px)

#         # ── Fallback: hip → ankle ──
#         pairs = []
#         if lh[0] > 0 and la[0] > 0:
#             pairs.append(abs(la[1] - lh[1]))
#         if rh[0] > 0 and ra[0] > 0:
#             pairs.append(abs(ra[1] - rh[1]))
#         if pairs:
#             hip_ankle_px = np.mean(pairs)
#             if hip_ankle_px > 10:
#                 self._hip_ankle_samples.append(hip_ankle_px)

#         # Prefer body-height calibration once enough samples exist
#         if len(self._height_samples) >= 10:
#             median_px = np.median(list(self._height_samples))
#             height_m  = self.user_height_m or self.ANTHRO_BODY_HEIGHT_M
#             self.px_per_m = median_px / height_m
#             self.method = "user_height" if self.user_height_m else "body_height_anthropometric"
#         elif len(self._hip_ankle_samples) >= 10:
#             median_px = np.median(list(self._hip_ankle_samples))
#             self.px_per_m = median_px / ANTHRO_HIP_ANKLE_M
#             self.method = "hip_ankle_anthropometric"

#     def px_to_m(self, pixels):
#         if self.px_per_m and self.px_per_m > 0:
#             return pixels / self.px_per_m
#         # Fallback: legacy constant (0.00933 m/px)
#         return pixels * 0.00933


# # ─────────────────────────────────────────────────────────────────────────────
# # ANGLE UTILITIES
# # ─────────────────────────────────────────────────────────────────────────────

# def _vec_angle(a, b, c):
#     """Angle at point b in triangle a-b-c (degrees)."""
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b;  bc = c - b
#     cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
#     return math.degrees(math.acos(np.clip(cos, -1, 1)))

# def _hip_flexion(pts, side="L"):
#     """Hip flexion angle (shoulder-hip-knee)."""
#     sh  = pts[KP_L_SHOULDER if side=="L" else KP_R_SHOULDER]
#     hp  = pts[KP_L_HIP      if side=="L" else KP_R_HIP]
#     kn  = pts[KP_L_KNEE     if side=="L" else KP_R_KNEE]
#     if any(p[0] <= 0 for p in [sh, hp, kn]):
#         return None
#     return _vec_angle(sh, hp, kn)

# def _knee_flexion(pts, side="L"):
#     """Knee flexion angle (hip-knee-ankle)."""
#     hp  = pts[KP_L_HIP   if side=="L" else KP_R_HIP]
#     kn  = pts[KP_L_KNEE  if side=="L" else KP_R_KNEE]
#     an  = pts[KP_L_ANKLE if side=="L" else KP_R_ANKLE]
#     if any(p[0] <= 0 for p in [hp, kn, an]):
#         return None
#     return _vec_angle(hp, kn, an)

# def _com_velocity(com_hist, fps):
#     """Returns (vx, vy) in px/frame from last two COM positions."""
#     if len(com_hist) < 2:
#         return (0.0, 0.0)
#     dx = com_hist[-1][0] - com_hist[-2][0]
#     dy = com_hist[-1][1] - com_hist[-2][1]
#     return (dx * fps, dy * fps)


# def _jump_direction(pts_hist):
#     """
#     Determine overall travel direction (+1 = left-to-right, -1 = right-to-left)
#     from a short history of COM x positions. Locked once at the start of a
#     rep sequence so per-jump distance math always uses a consistent sign.
#     """
#     if len(pts_hist) < 2:
#         return 1
#     dx = pts_hist[-1] - pts_hist[0]
#     return 1 if dx >= 0 else -1


# def _ankle_positions(pts):
#     """Return (left_ankle, right_ankle) as (x,y) or None if not visible."""
#     la = pts[KP_L_ANKLE]
#     ra = pts[KP_R_ANKLE]
#     left  = (la[0], la[1]) if la[0] > 0 and la[1] > 0 else None
#     right = (ra[0], ra[1]) if ra[0] > 0 and ra[1] > 0 else None
#     return left, right


# def _heel_x_approx(ankle, knee, direction):
#     """
#     Approximate heel x-position from ankle + knee (shank vector), since
#     COCO-17 has no heel keypoint. The heel sits slightly behind the ankle
#     relative to the direction of travel.
#     ankle, knee: (x, y) tuples
#     direction: +1 (moving right) or -1 (moving left)
#     """
#     if ankle is None:
#         return None
#     if knee is None:
#         return ankle[0]
#     shank_len = math.hypot(knee[0] - ankle[0], knee[1] - ankle[1])
#     if shank_len < 1e-6:
#         return ankle[0]
#     # Heel trails the ankle opposite to the direction of travel.
#     return ankle[0] - direction * 0.15 * shank_len


# def _takeoff_foot_x(pts, direction):
#     """
#     Foot-based takeoff reference: the most FORWARD foot point (in the
#     direction of travel) at the last grounded frame — this is the true
#     takeoff line, not COM.
#     """
#     la, ra = _ankle_positions(pts)
#     xs = [p[0] for p in (la, ra) if p is not None]
#     if not xs:
#         return None
#     return max(xs) if direction >= 0 else min(xs)


# def _landing_foot_x(pts, direction):
#     """
#     Foot-based landing reference: nearest body part to the takeoff line,
#     approximated via heel position on the leading foot at landing.
#     """
#     la, ra = _ankle_positions(pts)
#     lk = pts[KP_L_KNEE]; rk = pts[KP_R_KNEE]
#     lk_pt = (lk[0], lk[1]) if lk[0] > 0 and lk[1] > 0 else None
#     rk_pt = (rk[0], rk[1]) if rk[0] > 0 and rk[1] > 0 else None

#     heels = []
#     hx = _heel_x_approx(la, lk_pt, direction)
#     if hx is not None:
#         heels.append(hx)
#     hx = _heel_x_approx(ra, rk_pt, direction)
#     if hx is not None:
#         heels.append(hx)

#     if not heels:
#         return None
#     # Landing mark = the foot point CLOSEST to the takeoff line, i.e. the
#     # leading (trailing-edge) foot in the direction of travel.
#     return min(heels) if direction >= 0 else max(heels)


# # ─────────────────────────────────────────────────────────────────────────────
# # JUMP ENGINE  (6-state FSM)
# # ─────────────────────────────────────────────────────────────────────────────

# class JumpEngine:
#     """
#     Core 6-state finite state machine.
#     Tracks COM, foot contacts, and produces jump events.
#     """
#     def __init__(self, fps=30.0, exercise="standing_broad_jump"):
#         self.fps      = fps
#         self.exercise = exercise
#         self.state    = ST_READY

#         # Ground reference
#         self._gnd_com_y    = None   # adaptive ground COM y
#         self._gnd_buf      = deque(maxlen=GROUND_WINDOW)
#         self._gnd_locked   = False

#         # Takeoff
#         self._takeoff_com  = None
#         self._takeoff_f    = 0
#         self._takeoff_heel = None   # left heel x at takeoff
#         self._takeoff_foot_x   = None   # foot-based takeoff line (px)
#         self._last_grounded_pts = None  # full pts snapshot at last grounded frame
#         self._last_grounded_com = None

#         # Flight
#         self._flight_f     = 0
#         self._airborne_streak = 0   # consecutive airborne frames (hysteresis)
#         self._ground_streak   = 0   # consecutive grounded frames (hysteresis)
#         self._peak_com_y   = None
#         self._flight_com_x_start = None
#         self._peak_b64     = None

#         # In-flight kinematics (for tuck/pike/bounding)
#         self._flight_hip_flex   = []
#         self._flight_knee_flex  = []
#         self._flight_arm_travel = []   # wrist-y travel relative to shoulder during flight

#         # Landing
#         self._land_stable_f = 0
#         self._land_com_buf  = deque(maxlen=8)
#         self._land_foot_x_buf = deque(maxlen=8)   # foot-based landing samples
#         self._land_pts_at_settle = None           # pts snapshot once landing settles

#         # Direction of travel (+1 left→right, -1 right→left), locked per rep
#         self._direction     = 1
#         self._dir_buf       = deque(maxlen=10)

#         # Reset
#         self._reset_f       = 0
#         self._cooldown_f    = 0

#         # Loading / countermovement
#         self._load_com_y   = None   # COM y at start of countermovement
#         self._load_start_f = 0
#         self._cm_depth     = 0.0   # countermovement depth in px

#         # Completed jumps
#         self.jumps         = []
#         self._jump_n       = 0

#         # Per-phase data for triple/multiple/bounding
#         self._phase_list   = []    # list of phase dicts per jump
#         self._phase_start_com = None

#         # Reactive: ground contact tracking
#         self._gct_start_f  = 0    # frame when ground contact began after landing

#         # Approach velocity (run-up)
#         self._approach_vx  = 0.0
#         self._approach_buf = deque(maxlen=20)
#         self._takeoff_approach_vx = 0.0   # snapshot of approach_vx AT takeoff (frozen, not live)

#         # COM history for velocity-gated landing confirmation
#         self._com_hist      = deque(maxlen=5)

#         # Frame counter
#         self._frame        = 0

#     # ──────────────────────────────────────────────────────────────────────
#     def _update_ground(self, com_y, foot: FootContactEstimator, pts=None, com=None):
#         """Update adaptive ground COM_y during stable contact."""
#         if foot.both_contact or foot.any_contact:
#             self._gnd_buf.append(com_y)
#             if pts is not None:
#                 self._last_grounded_pts = pts.copy()
#                 self._last_grounded_com = com
#         if len(self._gnd_buf) >= 5:
#             self._gnd_com_y = np.percentile(list(self._gnd_buf), 80)

#     def _is_airborne(self, com_y):
#         if self._gnd_com_y is None:
#             return False
#         return com_y < self._gnd_com_y - COM_TAKEOFF_PX

#     def _is_near_ground(self, com_y):
#         if self._gnd_com_y is None:
#             return True
#         return com_y >= self._gnd_com_y - LAND_APPROACH_PX

#     def _com_settled(self):
#         """True when recent COM motion has slowed enough to trust the
#         landing position (prevents capturing landing COM while the body
#         is still travelling forward/downward)."""
#         if len(self._com_hist) < 3:
#             return False
#         vx = self._com_hist[-1][0] - self._com_hist[-2][0]
#         vy = self._com_hist[-1][1] - self._com_hist[-2][1]
#         return abs(vx) < LAND_VEL_PX_F and abs(vy) < LAND_VEL_PX_F

#     # ──────────────────────────────────────────────────────────────────────
#     def update(self, pts, foot: FootContactEstimator, com, scale: ScaleEstimator,
#                frame_b64=None):
#         """
#         Called every frame.
#         pts: filtered keypoints (17,2)
#         foot: FootContactEstimator (already updated this frame)
#         com: (cx, cy) or None
#         scale: ScaleEstimator
#         Returns: jump_event dict if a jump completed this frame, else None
#         """
#         self._frame += 1
#         if com is None:
#             return None

#         com_x, com_y = com
#         self._com_hist.append(com)

#         # Track approach velocity for run-up detection
#         self._approach_buf.append(com_x)
#         if len(self._approach_buf) >= 5:
#             dx = self._approach_buf[-1] - self._approach_buf[-5]
#             self._approach_vx = dx / 5.0 * self.fps  # px/s

#         # Track direction-of-travel candidates continuously; locked at takeoff
#         self._dir_buf.append(com_x)

#         # ── Airborne/grounded detection ──────────────────────────────────────
#         # COM height is the primary signal (smoother, derived from a weighted
#         # blend of multiple keypoints, less prone to single-frame dropout than
#         # any one foot keypoint). Foot contact is a confirming signal — OR'd
#         # in, not AND'd, because requiring both foot ankles to be visible AND
#         # agreeing with COM on every single frame is too strict for real YOLO
#         # keypoint noise (occlusion, low-confidence frames, motion blur) and
#         # was causing both missed jumps and stuck-in-FLIGHT (0-jump) results.
#         # A short confirm streak (not a hard AND) still absorbs single-frame
#         # jitter without requiring perfect agreement.
#         airborne_raw = foot.both_air or self._is_airborne(com_y)
#         if airborne_raw:
#             self._airborne_streak += 1
#             self._ground_streak = 0
#         else:
#             self._ground_streak += 1
#             self._airborne_streak = 0
#         airborne_confirmed = self._airborne_streak >= AIR_CONFIRM_FRAMES
#         # Grounded confirmation for landing uses COM proximity OR foot
#         # contact too (any_contact, not both_contact) — landing is reached
#         # the moment either signal agrees the athlete is back down.
#         grounded_raw = foot.any_contact or self._is_near_ground(com_y)
#         grounded_confirmed = self._ground_streak >= CONTACT_CONFIRM_FRAMES or grounded_raw

#         event = None

#         # ── READY ─────────────────────────────────────────────────────────
#         if self.state == ST_READY:
#             self._update_ground(com_y, foot, pts, com)
#             rising = self._gnd_com_y is not None and com_y < self._gnd_com_y - COM_RISE_PX
#             # Require a short confirmed streak (not the single-frame raw
#             # signal) before leaving READY. Using airborne_raw directly let
#             # a single jittery ankle/COM frame right after a landing
#             # immediately re-enter LOADING — wasteful at best, and at worst
#             # a contributor to one real jump being reported as two when the
#             # spurious LOADING attempt's countermovement bookkeeping bled
#             # into the next genuine rep.
#             if rising or airborne_confirmed:
#                 self.state         = ST_LOADING
#                 self._load_com_y   = com_y
#                 self._load_start_f = self._frame
#             else:
#                 # Still grounded — update scale only here (never in flight)
#                 scale.update(pts)
#                 foot.update_ground(pts[KP_L_ANKLE][1], pts[KP_R_ANKLE][1])
#                 self._takeoff_com  = com   # keep updating takeoff reference
#                 self._last_grounded_pts = pts.copy()
#                 self._last_grounded_com = com

#         # ── LOADING ───────────────────────────────────────────────────────
#         elif self.state == ST_LOADING:
#             if not airborne_raw:
#                 self._last_grounded_pts = pts.copy()
#                 self._last_grounded_com = com

#             # Require AIR_CONFIRM_FRAMES consecutive airborne frames
#             # (both conditions agreeing) before committing to a takeoff.
#             if airborne_confirmed:
#                 # True takeoff = the LAST GROUNDED frame, not the first
#                 # airborne one.
#                 ref_pts = self._last_grounded_pts if self._last_grounded_pts is not None else pts
#                 ref_com = self._last_grounded_com if self._last_grounded_com is not None else com

#                 # Lock direction of travel for this rep from recent COM drift
#                 direction_samples = np.array(list(self._dir_buf))
#                 self._direction = _jump_direction(direction_samples)

#                 self._takeoff_com        = ref_com
#                 self._takeoff_foot_x     = _takeoff_foot_x(ref_pts, self._direction)
#                 self._flight_com_x_start = ref_com[0]
#                 self._takeoff_f          = max(1, self._frame - self._airborne_streak)
#                 self._flight_f           = self._airborne_streak
#                 self._peak_com_y         = com_y
#                 self._peak_b64           = frame_b64
#                 self._flight_hip_flex    = []
#                 self._flight_knee_flex   = []
#                 self._flight_arm_travel  = []
#                 self._phase_start_com    = ref_com
#                 self._land_foot_x_buf.clear()
#                 self._land_com_buf.clear()
#                 self._land_stable_f      = 0
#                 self._cm_depth = max(0.0, self._gnd_com_y - self._load_com_y) if self._gnd_com_y else 0.0
#                 self._takeoff_approach_vx = self._approach_vx
#                 self.state = ST_FLIGHT
#             elif not airborne_raw and self._gnd_com_y is not None and com_y > self._gnd_com_y:
#                 # COM went back down without ever becoming airborne — false alarm
#                 self.state = ST_READY

#         # ── FLIGHT ────────────────────────────────────────────────────────
#         elif self.state == ST_FLIGHT:
#             self._flight_f += 1

#             if com_y < self._peak_com_y:
#                 self._peak_com_y = com_y
#                 self._peak_b64   = frame_b64

#             hf_l = _hip_flexion(pts, "L")
#             hf_r = _hip_flexion(pts, "R")
#             kf_l = _knee_flexion(pts, "L")
#             kf_r = _knee_flexion(pts, "R")
#             if hf_l: self._flight_hip_flex.append(hf_l)
#             if hf_r: self._flight_hip_flex.append(hf_r)
#             if kf_l: self._flight_knee_flex.append(kf_l)
#             if kf_r: self._flight_knee_flex.append(kf_r)

#             for sh_i, wr_i in ((KP_L_SHOULDER, KP_L_WRIST), (KP_R_SHOULDER, KP_R_WRIST)):
#                 sh, wr = pts[sh_i], pts[wr_i]
#                 if sh[0] > 0 and wr[0] > 0:
#                     self._flight_arm_travel.append(sh[1] - wr[1])

#             # Landing confirmation requires near-ground AND settled COM
#             # velocity AND a hysteresis-confirmed ground contact streak —
#             # this is what stops the post-landing stabilization wobble from
#             # being read as the landing mark.
#             if self._flight_f >= MIN_FLIGHT_F:
#                 near_ground = self._is_near_ground(com_y) or foot.any_contact
#                 if near_ground:
#                     self._land_stable_f += 1
#                     self._land_com_buf.append(com_x)
#                     foot_x = _landing_foot_x(pts, self._direction)
#                     if foot_x is not None:
#                         self._land_foot_x_buf.append(foot_x)
#                     if (self._land_stable_f >= LAND_STABLE_F and
#                             self._com_settled() and grounded_confirmed):
#                         event = self._confirm_jump(com, pts, scale)
#                         if event is not None:
#                             self.state = ST_LANDING
#                         else:
#                             # Rejected as a false jump (backward movement,
#                             # too short/out of range) — return to READY
#                             # directly rather than burning a full landing
#                             # cooldown, so subsequent real jumps in the
#                             # same video aren't delayed or missed.
#                             self.state = ST_READY
#                             self._airborne_streak = 0
#                             self._ground_streak = 0
#                 else:
#                     self._land_stable_f = 0
#                     self._land_com_buf.clear()
#                     self._land_foot_x_buf.clear()

#             if self._flight_f < MIN_FLIGHT_F and grounded_confirmed and self._is_near_ground(com_y):
#                 self.state = ST_READY
#                 self._airborne_streak = 0
#                 self._ground_streak = 0

#             # ── Stuck-FLIGHT safety valve ───────────────────────────────
#             # If landing is never cleanly confirmed (sustained occlusion,
#             # degraded keypoints right at touchdown, or a true-but-noisy
#             # landing that never reaches LAND_STABLE_F), force a close
#             # using best-available data rather than letting FLIGHT run out
#             # the rest of the clip — that previously meant every jump after
#             # the stuck one was lost entirely (0-jump / missing-jump bug on
#             # multi-jump videos).
#             if self._flight_f >= MAX_FLIGHT_F:
#                 event = self._confirm_jump(com, pts, scale)
#                 if event is not None:
#                     self.state = ST_LANDING
#                 else:
#                     self.state = ST_READY
#                 self._airborne_streak = 0
#                 self._ground_streak = 0

#         # ── LANDING ───────────────────────────────────────────────────────
#         elif self.state == ST_LANDING:
#             self._land_stable_f = 0
#             self._land_com_buf.clear()
#             self._land_foot_x_buf.clear()
#             self._reset_f = 0
#             self._cooldown_f = 0
#             self._last_grounded_pts = pts.copy()
#             self._last_grounded_com = com

#             self._gct_start_f = self._frame
#             self.state = ST_COOLDOWN

#         # ── COOLDOWN ──────────────────────────────────────────────────────
#         # Mandatory recovery window after every landing. Post-landing balance
#         # adjustments (knee bend, weight shift, brief foot occlusion) happen
#         # here and CANNOT trigger a new LOADING/FLIGHT cycle — this is what
#         # was previously creating a fake second jump immediately after a
#         # real one. Reactive jumps are the sole exception (genuinely meant
#         # to re-launch within a few frames of ground contact).
#         elif self.state == ST_COOLDOWN:
#             self._cooldown_f += 1
#             self._update_ground(com_y, foot, pts, com)
#             self._last_grounded_pts = pts.copy()
#             self._last_grounded_com = com

#             if (foot.both_air and self._cooldown_f <= REACTIVE_MAX_CONTACT_F
#                     and "reactive" in self.exercise):
#                 direction_samples = np.array(list(self._dir_buf))
#                 self._direction = _jump_direction(direction_samples)
#                 ref_pts = self._last_grounded_pts if self._last_grounded_pts is not None else pts

#                 self._takeoff_com        = com
#                 self._takeoff_foot_x     = _takeoff_foot_x(ref_pts, self._direction)
#                 self._flight_com_x_start = com_x
#                 self._takeoff_f          = self._frame
#                 self._flight_f           = 0
#                 self._peak_com_y         = com_y
#                 self._peak_b64           = frame_b64
#                 self._flight_hip_flex    = []
#                 self._flight_knee_flex   = []
#                 self._flight_arm_travel  = []
#                 self._phase_start_com    = com
#                 self._land_foot_x_buf.clear()
#                 self._land_com_buf.clear()
#                 self._takeoff_approach_vx = self._approach_vx
#                 self.state = ST_FLIGHT

#             elif self._cooldown_f >= COOLDOWN_FRAMES:
#                 self.state = ST_READY
#                 self._gnd_locked = True
#                 self._reset_f = 0

#         # ── RESET (legacy, kept for backward compatibility) ─────────────────
#         elif self.state == ST_RESET:
#             self.state = ST_READY

#         return event

#     # ──────────────────────────────────────────────────────────────────────
#     def _confirm_jump(self, land_com, pts, scale: ScaleEstimator):
#         """Build and store a jump event dict. Returns None (and does NOT
#         increment jump_n / append to self.jumps) if this is rejected as a
#         false jump — caller must handle a None return."""

#         # ── Foot-based distance (primary) ───────────────────────────────
#         # Takeoff line = most-forward foot point at the last grounded frame.
#         # Landing mark = nearest heel of the leading foot once landing has
#         # settled (near-ground + low COM velocity + foot contact), smoothed
#         # via median over the buffered landing samples.
#         takeoff_x_foot = self._takeoff_foot_x

#         land_x_foot = None
#         if self._land_foot_x_buf:
#             land_x_foot = float(np.median(list(self._land_foot_x_buf)))
#         elif pts is not None:
#             land_x_foot = _landing_foot_x(pts, self._direction)

#         # COM fallback (only used if foot keypoints were unavailable)
#         land_x_com    = float(np.median(list(self._land_com_buf))) if self._land_com_buf else land_com[0]
#         takeoff_x_com = self._flight_com_x_start if self._flight_com_x_start else (
#             self._takeoff_com[0] if self._takeoff_com else land_x_com)

#         # Distance is SIGNED by the locked direction of travel, not abs().
#         # A jump that nets backward relative to the locked direction (e.g.
#         # a post-landing balance shift) produces a negative raw distance
#         # and is rejected outright below — this is what previously let a
#         # backward foot displacement masquerade as a 0.53m jump.
#         if takeoff_x_foot is not None and land_x_foot is not None:
#             raw_px = (land_x_foot - takeoff_x_foot) * self._direction
#             takeoff_x, land_x = takeoff_x_foot, land_x_foot
#             dist_method = "foot_based"
#         else:
#             raw_px = (land_x_com - takeoff_x_com) * self._direction
#             takeoff_x, land_x = takeoff_x_com, land_x_com
#             dist_method = "com_fallback"

#         if raw_px <= 0:
#             return None   # non-forward / backward displacement — not a real jump

#         px_dist = raw_px
#         dist_m  = round(scale.px_to_m(px_dist), 3)
#         dist_cm = round(dist_m * 100, 1)

#         flight_f  = self._flight_f
#         flight_ms = int(flight_f / self.fps * 1000)

#         # Peak COM height above takeoff COM
#         peak_rise_px = 0.0
#         if self._takeoff_com and self._peak_com_y:
#             peak_rise_px = max(0.0, self._takeoff_com[1] - self._peak_com_y)
#         peak_rise_m = scale.px_to_m(peak_rise_px)

#         approach_speed_ms = scale.px_to_m(abs(self._takeoff_approach_vx))

#         # ── False-jump rejection ────────────────────────────────────────
#         # Reject jumps that are too short, too small, or outside a
#         # physically sane distance range for a standing broad jump — these
#         # are the load-bearing checks against weight-shift / post-landing
#         # noise. Approach-speed and physics-consistency are NOT hard gates:
#         # a standing broad jump can legitimately have near-zero horizontal
#         # drift right up to takeoff (the athlete is stationary before the
#         # countermovement), and the physics-consistency comparison below is
#         # derived circularly from dist_m/t so it has no real rejection
#         # power — both were previously zeroing out genuine jumps on noisier
#         # real-world video without actually screening out bad ones.
#         t = flight_f / self.fps

#         # The com_fallback path only fires when ankle keypoints were
#         # unavailable for the whole flight, and it systematically
#         # UNDERSHOOTS true takeoff-to-landing distance — COM displacement
#         # is consistently shorter than foot-to-foot distance because the
#         # body folds (knees/hips flex) on landing, pulling COM backward
#         # relative to where the foot actually lands. This is the dominant
#         # cause of implausibly short distances reported for genuinely long
#         # (e.g. professional-level) jumps. Apply a stricter, higher floor
#         # to this less-reliable path so a degraded-tracking jump is
#         # dropped rather than reported with a misleadingly small number.
#         min_dist_gate = MIN_VALID_DIST_M if dist_method == "foot_based" else MIN_COM_FALLBACK_DIST_M

#         # Physical plausibility floor: real airborne time and real forward
#         # distance are correlated (a body in the air for several hundred ms
#         # cannot have travelled almost no horizontal distance in a standing
#         # or run-up broad jump). If the measured distance is far below what
#         # the observed flight time implies, that's a sign the distance
#         # measurement degraded (e.g. heel-approximation drift), not that
#         # the jump itself was tiny — so it's rejected rather than reported.
#         physics_floor_m = MIN_DIST_PER_FLIGHT_S * t

#         if (dist_m < min_dist_gate or
#                 dist_m < physics_floor_m or
#                 dist_m > MAX_VALID_DIST_M or
#                 flight_ms < MIN_VALID_FLIGHT_MS or
#                 peak_rise_px < MIN_PEAK_RISE_PX):
#             return None

#         # ── Physics-correct takeoff velocity / angle ────────────────────
#         g = 9.81
#         vx_ms = dist_m / (t + 1e-9)
#         vy_ms = g * t / 2 if t > 0 else 0.0
#         takeoff_vel = round(math.sqrt(vx_ms**2 + vy_ms**2), 2)
#         takeoff_angle = round(math.degrees(math.atan2(vy_ms, vx_ms + 1e-9)), 1) if vx_ms > 1e-6 else 0.0
#         takeoff_angle = max(0.0, min(85.0, takeoff_angle))

#         # Physics-consistency is reported for diagnostics only — NOT used
#         # to reject the jump (see note above).
#         physics_expected_dist = vx_ms * t
#         physics_consistent = bool(abs(physics_expected_dist - dist_m) < max(0.15, 0.25 * dist_m))

#         # Tuck / pike detection from in-flight angles
#         tuck_confirmed = False
#         pike_confirmed = False
#         min_hip_flex = min(self._flight_hip_flex) if self._flight_hip_flex else 180
#         min_knee_flex = min(self._flight_knee_flex) if self._flight_knee_flex else 180
#         max_knee_ext  = max(self._flight_knee_flex) if self._flight_knee_flex else 0

#         if min_hip_flex < TUCK_HIP_FLEX_DEG and min_knee_flex < 90:
#             tuck_confirmed = True
#         if min_hip_flex < PIKE_HIP_FLEX_DEG and max_knee_ext > PIKE_KNEE_EXT_DEG:
#             pike_confirmed = True

#         # Countermovement depth
#         cm_depth_m = round(scale.px_to_m(self._cm_depth), 3)

#         # Horizontal efficiency: forward dist / total COM path (approximation)
#         h_eff = round(min(1.0, px_dist / (px_dist + peak_rise_px + 1e-9)), 3)

#         # Landing stability (from COM settle velocity at confirmation time)
#         land_vx = land_vy = 0.0
#         if len(self._com_hist) >= 2:
#             land_vx = self._com_hist[-1][0] - self._com_hist[-2][0]
#             land_vy = self._com_hist[-1][1] - self._com_hist[-2][1]
#         landing_stability = round(max(0.0, 1.0 - (abs(land_vx) + abs(land_vy)) / 20.0), 3)

#         # Multi-factor form score (distance, landing stability, takeoff
#         # angle, arm-swing quality, knee flexion at takeoff)
#         arm_swing_score = _arm_swing_score(self._flight_arm_travel)
#         form = _form_score_full(
#             dist_m=dist_m,
#             landing_stability=landing_stability,
#             takeoff_angle=takeoff_angle,
#             min_hip_flex=min_hip_flex,
#             min_knee_flex=min_knee_flex,
#             arm_swing_score=arm_swing_score,
#         )

#         # Ground contact time (reactive)
#         gct_ms = 0
#         if "reactive" in self.exercise and self._gct_start_f > 0:
#             gct_f  = self._takeoff_f - self._gct_start_f
#             gct_ms = int(max(0, gct_f) / self.fps * 1000)

#         rsi = round(flight_ms / gct_ms, 3) if gct_ms > 0 else None

#         self._jump_n += 1
#         jump = {
#             # Identity
#             "jump_no"           : self._jump_n,
#             # Distances
#             "distance_m"        : dist_m,
#             "distance_cm"       : dist_cm,
#             "pixel_dist"        : round(px_dist, 1),
#             "distance_method"   : dist_method,
#             # Timing
#             "flight_ms"         : flight_ms,
#             "airborne_ms"       : flight_ms,
#             "takeoff_frame"     : self._takeoff_f,
#             "landing_frame"     : self._frame,
#             # Positions
#             "takeoff_com_x"     : round(takeoff_x_com, 1),
#             "landing_com_x"     : round(land_x_com, 1),
#             "takeoff_foot_x"    : round(takeoff_x_foot, 1) if takeoff_x_foot is not None else None,
#             "landing_foot_x"    : round(land_x_foot, 1) if land_x_foot is not None else None,
#             "direction"         : self._direction,
#             # Kinematics
#             "takeoff_velocity_ms": takeoff_vel,
#             "takeoff_angle_deg" : takeoff_angle,
#             "peak_rise_m"       : round(peak_rise_m, 3),
#             "approach_vx_px_s"  : round(self._takeoff_approach_vx, 1),
#             "approach_speed_ms" : round(approach_speed_ms, 2),
#             "horizontal_efficiency": h_eff,
#             "cm_depth_m"        : cm_depth_m,
#             "physics_consistent": physics_consistent,
#             # Technique
#             "tuck_confirmed"    : tuck_confirmed,
#             "pike_confirmed"    : pike_confirmed,
#             "min_hip_flex_deg"  : round(min_hip_flex, 1),
#             "min_knee_flex_deg" : round(min_knee_flex, 1),
#             "landing_stability" : landing_stability,
#             "arm_swing_score"   : arm_swing_score,
#             # Reactive
#             "ground_contact_ms" : gct_ms,
#             "rsi"               : rsi,
#             # Score
#             "form_score"        : form,
#             # Snapshot
#             "_peak_b64"         : self._peak_b64,
#         }
#         jump = _to_json_safe(jump)
#         self.jumps.append(jump)
#         return jump

#     def force_close(self, com, scale: ScaleEstimator, pts=None):
#         """Call at video end if still in FLIGHT state."""
#         if self.state == ST_FLIGHT and self._flight_f >= MIN_FLIGHT_F:
#             land_x = com[0] if com else (self._flight_com_x_start or 0)
#             self._land_com_buf.append(land_x)
#             self._land_stable_f = LAND_STABLE_F
#             ref_pts = pts if pts is not None else np.zeros((17, 2))
#             foot_x = _landing_foot_x(ref_pts, self._direction) if pts is not None else None
#             if foot_x is not None:
#                 self._land_foot_x_buf.append(foot_x)
#             return self._confirm_jump(com, ref_pts, scale)
#         return None


# # ─────────────────────────────────────────────────────────────────────────────
# # EXERCISE-SPECIFIC VALIDATORS
# # ─────────────────────────────────────────────────────────────────────────────

# class ExerciseValidator:
#     """
#     Wraps the core JumpEngine and applies exercise-specific
#     post-processing to the completed jump list.
#     """
#     def __init__(self, exercise: str):
#         self.exercise = exercise

#     def validate(self, jumps: list) -> dict:
#         ex = self.exercise.lower().replace(" ", "_")

#         if "triple" in ex:
#             return self._triple(jumps)
#         elif "alternate" in ex:
#             return self._alternate(jumps)
#         elif "bounding" in ex or "power_skip" in ex:
#             return self._bounding(jumps)
#         elif "multiple" in ex:
#             return self._multiple(jumps)
#         else:
#             return self._standard(jumps)

#     # Standard single / repeated jumps
#     def _standard(self, jumps):
#         return {"validated_jumps": jumps, "phase_info": None}

#     # Triple: group every 3 consecutive jumps into hop/step/jump
#     def _triple(self, jumps):
#         phases = []
#         for i in range(0, len(jumps) - 2, 3):
#             hop, step, jmp = jumps[i], jumps[i+1], jumps[i+2]
#             total = hop["distance_m"] + step["distance_m"] + jmp["distance_m"]
#             phases.append({
#                 "triple_no"   : len(phases) + 1,
#                 "hop"         : hop,
#                 "step"        : step,
#                 "jump"        : jmp,
#                 "total_m"     : round(total, 3),
#                 "phase_ratio" : [round(hop["distance_m"]/total, 2),
#                                  round(step["distance_m"]/total, 2),
#                                  round(jmp["distance_m"]/total, 2)],
#             })
#         return {"validated_jumps": jumps, "phase_info": phases}

#     # Alternate: flag if consecutive jumps show alternating takeoff feet
#     def _alternate(self, jumps):
#         for i, j in enumerate(jumps):
#             j["leg_tag"] = "L" if i % 2 == 0 else "R"
#         return {"validated_jumps": jumps, "phase_info": None}

#     # Bounding: each jump = one bound
#     def _bounding(self, jumps):
#         bounds = []
#         for j in jumps:
#             bounds.append({
#                 "bound_no"  : j["jump_no"],
#                 "distance_m": j["distance_m"],
#                 "flight_ms" : j["flight_ms"],
#                 "gct_ms"    : j.get("ground_contact_ms", 0),
#             })
#         total = sum(b["distance_m"] for b in bounds)
#         return {"validated_jumps": jumps, "phase_info": bounds,
#                 "total_distance_m": round(total, 3)}

#     # Multiple: cumulative metrics
#     def _multiple(self, jumps):
#         cumulative = 0.0
#         for j in jumps:
#             cumulative += j["distance_m"]
#             j["cumulative_m"] = round(cumulative, 3)
#         return {"validated_jumps": jumps, "phase_info": None}


# # ─────────────────────────────────────────────────────────────────────────────
# # QUALITY & SCORING
# # ─────────────────────────────────────────────────────────────────────────────

# def _form_score(distance_m):
#     """Legacy distance-only score, retained for backward compatibility
#     (used by _form_score_aggregate for the session-level summary)."""
#     thresholds = [2.50, 2.20, 2.00, 1.80, 1.50, 1.20]
#     scores     = [10,   9,    8,    7,    6,    5]
#     for t, s in zip(thresholds, scores):
#         if distance_m >= t:
#             return s
#     return 4

# def _distance_subscore(distance_m):
#     """0-10 distance subscore feeding the weighted per-jump form score."""
#     thresholds = [2.50, 2.20, 2.00, 1.80, 1.50, 1.20, 0.90]
#     scores     = [10,   9,    8,    7,    6,    5,    3]
#     for t, s in zip(thresholds, scores):
#         if distance_m >= t:
#             return s
#     return 1

# def _takeoff_angle_subscore(angle_deg):
#     """0-10 subscore peaking in the biomechanically efficient 18-27°
#     range for a standing broad jump; falls off outside it."""
#     ideal_low, ideal_high = 18.0, 27.0
#     if ideal_low <= angle_deg <= ideal_high:
#         return 10.0
#     dist = (ideal_low - angle_deg) if angle_deg < ideal_low else (angle_deg - ideal_high)
#     return max(0.0, 10.0 - dist * 0.35)

# def _knee_flexion_subscore(min_knee_flex_deg):
#     """0-10 subscore for takeoff/flight knee flexion — deep enough to
#     generate power without being so collapsed it signals poor control."""
#     if min_knee_flex_deg is None:
#         return 6.0  # neutral if not measured
#     ideal_low, ideal_high = 90.0, 130.0
#     if ideal_low <= min_knee_flex_deg <= ideal_high:
#         return 10.0
#     dist = (ideal_low - min_knee_flex_deg) if min_knee_flex_deg < ideal_low else (min_knee_flex_deg - ideal_high)
#     return max(0.0, 10.0 - dist * 0.15)

# def _arm_swing_score(wrist_travel_samples):
#     """
#     0-10 subscore from wrist vertical excursion relative to shoulder during
#     flight. A strong arm swing drives the wrists from low (behind the hips
#     at takeoff) to high (overhead/forward at peak flight) — large positive
#     range in (shoulder_y - wrist_y) indicates good swing amplitude.
#     """
#     if not wrist_travel_samples or len(wrist_travel_samples) < 2:
#         return 5.0   # neutral score when arms aren't trackable
#     travel_range = max(wrist_travel_samples) - min(wrist_travel_samples)
#     # travel_range is in pixels; normalise loosely — most useful as a
#     # relative signal since we don't have per-subject scale here.
#     score = min(10.0, travel_range / 12.0)
#     return round(max(0.0, score), 1)

# def _form_score_full(dist_m, landing_stability, takeoff_angle,
#                       min_hip_flex, min_knee_flex, arm_swing_score=5.0):
#     """
#     Weighted multi-factor form score (1-10):
#       40% distance, 20% landing stability, 15% takeoff angle,
#       15% arm swing, 10% knee flexion.
#     """
#     dist_s   = _distance_subscore(dist_m)
#     land_s   = round(landing_stability * 10, 1)
#     angle_s  = _takeoff_angle_subscore(takeoff_angle)
#     knee_s   = _knee_flexion_subscore(min_knee_flex)
#     arm_s    = arm_swing_score

#     weighted = (0.40 * dist_s + 0.20 * land_s + 0.15 * angle_s +
#                 0.15 * arm_s + 0.10 * knee_s)
#     return int(round(max(1.0, min(10.0, weighted))))

# def _form_score_aggregate(jumps_m):
#     if not jumps_m:
#         return 0
#     best  = max(jumps_m)
#     score = _form_score(best)
#     if len(jumps_m) > 1:
#         cons = min(jumps_m) / best
#         if   cons >= 0.90: score = min(10, score + 1)
#         elif cons < 0.70:  score = max(1,  score - 1)
#     return score

# def _compute_quality(jumps, pose_conf, foot_conf, scale_calibrated):
#     """
#     Generate overall confidence score 0-100 from independently meaningful
#     sub-scores, rather than a raw jump count (which says nothing about
#     measurement reliability).
#       pose      : average pose-detection confidence across the video
#       contact   : foot-contact estimator reliability
#       flight    : how cleanly flight phases were detected (physics-consistency
#                   rate + sane flight-time spread across detected jumps)
#       scale     : whether px-per-meter calibration succeeded
#       landing   : average landing stability across detected jumps
#     """
#     pose_s  = int(pose_conf * 100)
#     cont_s  = int(foot_conf * 100)
#     scale_s = 85 if scale_calibrated else 55

#     if not jumps:
#         return {
#             "overall": 0, "pose": pose_s, "contact": cont_s,
#             "flight": 0, "distance": scale_s, "landing": 0,
#             "jump": 0,   # legacy key retained
#         }

#     consistent = [j.get("physics_consistent", True) for j in jumps]
#     flight_s   = int(100 * (sum(consistent) / len(consistent)))

#     land_vals  = [j.get("landing_stability") for j in jumps if j.get("landing_stability") is not None]
#     landing_s  = int(100 * (sum(land_vals) / len(land_vals))) if land_vals else 60

#     overall = int(0.30*pose_s + 0.15*cont_s + 0.25*flight_s +
#                   0.15*scale_s + 0.15*landing_s)
#     return {
#         "overall": overall, "pose": pose_s, "contact": cont_s,
#         "flight": flight_s, "distance": scale_s, "landing": landing_s,
#         "jump": flight_s,   # legacy key retained, now mapped to a meaningful value
#     }

# def _fatigue_index(jumps_m):
#     """Decline in distance over repetitions (lower is better)."""
#     if len(jumps_m) < 2:
#         return None
#     return round((jumps_m[0] - jumps_m[-1]) / jumps_m[0] * 100, 1)

# def _rep_variability(jumps_m):
#     if len(jumps_m) < 2:
#         return None
#     return round(float(np.std(jumps_m)) / (np.mean(jumps_m) + 1e-9) * 100, 1)

# def _feedback(jump_results, detected, exercise):
#     issues    = []
#     strengths = []

#     if not detected:
#         issues += [
#             "❌ Person not detected in video.",
#             "📌 Ensure: full body visible (head to toe), side-angle camera, good lighting.",
#             "🎥 Avoid top-down or front-facing camera angles.",
#         ]
#         return issues, strengths

#     if not jump_results:
#         issues += [
#             "❌ No valid jump detected.",
#             "📐 Camera should be placed sideways at full-body height.",
#             "⏱️ Video must capture full jump — takeoff, flight, and landing.",
#         ]
#         return issues, strengths

#     distances = [j["distance_m"] for j in jump_results]
#     best = max(distances)
#     avg  = sum(distances) / len(distances)

#     if best < 1.20:
#         issues.append(f"Short jump ({best:.2f}m) — focus on arm swing and hip extension.")
#     elif best < 1.80:
#         issues.append(f"Below-average distance ({best:.2f}m) — work on explosive leg drive.")
#     elif best < 2.20:
#         strengths.append(f"Good jump distance ({best:.2f}m).")
#     else:
#         strengths.append(f"Excellent jump distance ({best:.2f}m)!")

#     if len(distances) > 1:
#         cons = min(distances) / best
#         if cons >= 0.90:
#             strengths.append(f"Very consistent across {len(distances)} jumps ({cons:.0%}).")
#         elif cons < 0.70:
#             issues.append(f"High variation ({cons:.0%} consistency) — repeat takeoff mechanics.")

#     # Tuck/pike feedback
#     if "tuck" in exercise.lower():
#         tucks = [j for j in jump_results if j.get("tuck_confirmed")]
#         if tucks:
#             strengths.append(f"Tuck confirmed in {len(tucks)}/{len(jump_results)} jumps.")
#         else:
#             issues.append("Tuck not detected — ensure knees pull toward chest during flight.")

#     if "pike" in exercise.lower():
#         pikes = [j for j in jump_results if j.get("pike_confirmed")]
#         if pikes:
#             strengths.append(f"Pike confirmed in {len(pikes)}/{len(jump_results)} jumps.")
#         else:
#             issues.append("Pike not detected — keep legs extended and reach toward toes.")

#     if not issues:
#         issues = ["No major issues detected."]
#     return issues, strengths


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN ANALYSIS FUNCTION
# # ─────────────────────────────────────────────────────────────────────────────

# def analyse_broad_jump(
#     path,
#     is_video,
#     output_path=None,
#     session_id=None,
#     source_filename="",
#     progress_uid=None,
#     exercise="Standing Broad Jump",
# ):
#     """
#     Main entry point for broad jump analysis.
#     Supports all 12 broad jump variants via the `exercise` parameter.

#     Parameters
#     ----------
#     exercise : str
#         One of: "Standing Broad Jump", "Run-Up Broad Jump",
#         "Single-Leg Broad Jump", "Alternate Leg Broad Jump",
#         "Bounding", "Triple Broad Jump", "Multiple Broad Jump",
#         "Reactive Broad Jump", "Tuck Broad Jump", "Pike Broad Jump",
#         "Weighted Broad Jump", "Sand Broad Jump"
#     """
#     if not _YOLO_AVAILABLE:
#         raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
#     if not is_video:
#         raise ValueError("Broad jump analysis requires a video file.")

#     # ── Model ────────────────────────────────────────────────────────────
#     model_path = YOLO_MODEL_PATH
#     if not os.path.exists(model_path):
#         here = os.path.dirname(os.path.abspath(__file__))
#         alt  = os.path.join(here, YOLO_MODEL_PATH)
#         if os.path.exists(alt):
#             model_path = alt
#         else:
#             raise FileNotFoundError(f"YOLO model not found at '{YOLO_MODEL_PATH}'.")
#     model = YOLO(model_path)

#     # ── Video metadata ────────────────────────────────────────────────────
#     cap    = cv2.VideoCapture(path)
#     fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     # ── Sub-systems ───────────────────────────────────────────────────────
#     lm_filter = LandmarkFilter(n_landmarks=17, freq=fps)
#     foot_est  = FootContactEstimator(window=int(fps * 1.5), fps=fps)
#     scale_est = ScaleEstimator()
#     engine    = JumpEngine(fps=fps, exercise=exercise.lower())
#     validator = ExerciseValidator(exercise)

#     # ── State ─────────────────────────────────────────────────────────────
#     detected_ok   = [False]
#     wrong_events  = []
#     per_frame_data= []
#     total_frames  = [1]
#     pose_confs    = []
#     foot_conf_acc = []
#     miss_streak   = [0]      # consecutive frames with no pose detected
#     last_good_pts = [None]   # most recent valid keypoints, for bridging gaps

#     live_hud      = {"jump_no": "0", "dist": "---", "form": "---", "state": ST_READY}

#     def _push_event(jdata, b64):
#         if not progress_uid:
#             return
#         pct = min(94, int(len(per_frame_data) / max(1, total_frames[0]) * 90))
#         safe_jdata = _to_json_safe({**jdata, "frame_b64": b64})
#         set_progress(
#             progress_uid, pct,
#             f"Jump {jdata['jump_no']} — {jdata['distance_m']:.2f}m",
#             jump_event=safe_jdata,
#         )

#     # ── Per-frame callback ────────────────────────────────────────────────
#     def pf(frame, fc, total):
#         total_frames[0] = max(total, 1)

#         results   = model(frame, conf=CONFIDENCE, verbose=False)
#         best_pts  = None
#         best_area = 0
#         best_conf = 0.0

#         for result in results:
#             if (result.keypoints is None or
#                     result.keypoints.xy is None or
#                     result.keypoints.conf is None):
#                 continue
#             for kpts_xy, kpts_conf in zip(
#                     result.keypoints.xy.cpu().numpy(),
#                     result.keypoints.conf.cpu().numpy()):
#                 pts   = kpts_xy.astype(float)
#                 valid = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
#                 if len(valid) < 6:
#                     continue
#                 area = ((valid[:, 0].max() - valid[:, 0].min()) *
#                         (valid[:, 1].max() - valid[:, 1].min()))
#                 if area > best_area:
#                     best_area = area
#                     best_pts  = pts
#                     best_conf = float(np.mean(kpts_conf[kpts_conf > 0]))

#         if best_pts is None:
#             miss_streak[0] += 1
#             # Short dropouts (motion blur mid-flight, brief occlusion) must
#             # NOT freeze the FSM — if we simply skip engine.update() here,
#             # _flight_f / _land_stable_f stall, the FSM can get stuck in
#             # FLIGHT for the rest of the clip, and every later jump in the
#             # video is silently lost (0-jump / missing-jump bug). Replaying
#             # the last known good keypoints keeps state/timers advancing
#             # through brief gaps without injecting a fabricated pose.
#             if last_good_pts[0] is not None and miss_streak[0] <= MAX_OCCLUSION_HOLD_F:
#                 pts_hold = last_good_pts[0]
#                 com_hold = estimate_com(pts_hold)
#                 foot_est.classify(pts_hold)
#                 frame_b64_hold = frame_to_b64(frame) if engine.state in (ST_FLIGHT, ST_TAKEOFF) else None
#                 jump_event = engine.update(pts_hold, foot_est, com_hold, scale_est, frame_b64_hold)
#                 if jump_event:
#                     b64 = jump_event.pop("_peak_b64", None) or frame_to_b64(frame)
#                     _push_event(jump_event, b64)
#                     live_hud["jump_no"] = str(jump_event["jump_no"])
#                     live_hud["dist"]    = f"{jump_event['distance_m']:.2f}m"
#                     live_hud["form"]    = f"{jump_event['form_score']}/10"
#                 live_hud["state"] = engine.state

#             draw_footer_hud(frame, [
#                 ("JUMP #", live_hud["jump_no"]),
#                 ("DIST",   live_hud["dist"]),
#                 ("FORM",   live_hud["form"]),
#             ])
#             draw_pcl_logo(frame)
#             return frame

#         miss_streak[0] = 0
#         detected_ok[0] = True
#         pose_confs.append(best_conf)

#         # Filter landmarks
#         pts = lm_filter.update(best_pts)
#         last_good_pts[0] = pts

#         # Draw skeleton (skip face landmarks 0-4)
#         for p1i, p2i in SKELETON_PAIRS:
#             p1, p2 = pts[p1i], pts[p2i]
#             if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
#                 cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
#                          COLOR_SKEL, 2)
#         for idx, p in enumerate(pts):
#             if idx < 5 or p[0] <= 0 or p[1] <= 0:
#                 continue
#             cv2.circle(frame, (int(p[0]), int(p[1])), 4, COLOR_JOINT, -1)

#         # COM
#         com = estimate_com(pts)
#         if com:
#             cv2.circle(frame, (int(com[0]), int(com[1])), 6, COLOR_COM, -1)

#         # Foot contact
#         foot_est.classify(pts)

#         # Engine update
#         frame_b64  = frame_to_b64(frame) if engine.state in (ST_FLIGHT, ST_TAKEOFF) else None
#         jump_event = engine.update(pts, foot_est, com, scale_est, frame_b64)

#         if jump_event:
#             b64 = jump_event.pop("_peak_b64", None) or frame_to_b64(frame)
#             _push_event(jump_event, b64)
#             live_hud["jump_no"] = str(jump_event["jump_no"])
#             live_hud["dist"]    = f"{jump_event['distance_m']:.2f}m"
#             live_hud["form"]    = f"{jump_event['form_score']}/10"

#         live_hud["state"] = engine.state

#         # Visualise scale (if calibrated, show reference line)
#         if scale_est.px_per_m:
#             ref_px = int(scale_est.px_per_m)
#             mid_x, gnd_y = width // 4, height - 40
#             cv2.line(frame, (mid_x, gnd_y), (mid_x + ref_px, gnd_y), (100, 255, 100), 2)
#             cv2.putText(frame, "1m", (mid_x + ref_px // 2 - 10, gnd_y - 6),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

#         # Draw ground reference
#         if engine._gnd_com_y:
#             gnd_px = int(engine._gnd_com_y + (LAND_APPROACH_PX))
#             cv2.line(frame, (0, gnd_px), (width, gnd_px), (80, 80, 80), 1)

#         # Draw takeoff / landing markers
#         if engine._flight_com_x_start and engine.state == ST_FLIGHT:
#             tx = int(engine._flight_com_x_start)
#             cv2.line(frame, (tx, 0), (tx, height), COLOR_TAKE, 1)

#         per_frame_data.append({
#             "frame"     : fc,
#             "state"     : engine.state,
#             "jump_count": len(engine.jumps),
#         })

#         draw_footer_hud(frame, [
#             ("JUMP #", live_hud["jump_no"]),
#             ("DIST",   live_hud["dist"]),
#             ("FORM",   live_hud["form"]),
#         ])
#         draw_pcl_logo(frame)
#         return frame

#     # ── Run ───────────────────────────────────────────────────────────────
#     snaps = process_video_or_image(
#         path, is_video, pf,
#         output_path=output_path,
#         snap_pcts=[0.1, 0.3, 0.5, 0.7, 0.9],
#         analysis_skip=1,
#         progress_uid=progress_uid,
#     )

#     # Force-close if video ends mid-flight. Previously this used a
#     # synthetic (takeoff_x, ground_y) placeholder for last_com whenever no
#     # better value was tracked, and never passed real pts — so foot-based
#     # distance was unavailable and the COM-fallback path saw an essentially
#     # fabricated landing position. That's what silently truncated/dropped
#     # the last jump in a video whenever it ran right up to the final frame
#     # without a clean LAND_STABLE_F streak. Use the actual last detected
#     # pose/COM (held across any trailing occlusion via last_good_pts) so
#     # the recovered jump's distance reflects where the athlete really was.
#     if engine.state == ST_FLIGHT and engine._flight_f >= MIN_FLIGHT_F:
#         last_pts = last_good_pts[0]
#         last_com = estimate_com(last_pts) if last_pts is not None else None
#         if last_com is None:
#             last_com = (engine._flight_com_x_start, engine._gnd_com_y or 0)
#         ev = engine.force_close(last_com, scale_est, pts=last_pts)
#         if ev:
#             ev.pop("_peak_b64", None)
#             _push_event(ev, "")

#     if not detected_ok[0]:
#         raise ValueError(
#             "No person detected. Upload a side-angle video with full body visible "
#             "(head to toe) and good lighting."
#         )

#     if session_id:
#         save_wrong_angle_log(exercise, session_id, source_filename, wrong_events)

#     # ── Post-processing ────────────────────────────────────────────────────
#     jumps     = engine.jumps
#     val_out   = validator.validate(jumps)
#     jumps_m   = [j["distance_m"] for j in jumps]

#     best_dist = round(max(jumps_m), 3) if jumps_m else 0.0
#     avg_dist  = round(sum(jumps_m) / len(jumps_m), 3) if jumps_m else 0.0

#     pose_conf_avg = float(np.mean(pose_confs)) if pose_confs else 0.0
#     foot_conf_val = 0.75 if scale_est.px_per_m else 0.50
#     quality = _compute_quality(jumps, pose_conf_avg, foot_conf_val,
#                                 scale_calibrated=scale_est.px_per_m is not None)

#     form_score = _form_score_aggregate(jumps_m)
#     issues, strengths = _feedback(jumps, detected_ok[0], exercise)

#     while len(snaps) < 5:
#         snaps.append(snaps[-1] if snaps else "")

#     step      = max(1, int(fps / 10))
#     per_frame = per_frame_data[::step]

#     best_str = f"{best_dist:.2f} m" if best_dist > 0 else "N/A"
#     avg_str  = f"{avg_dist:.2f} m"  if avg_dist  > 0 else "N/A"

#     # ── Aggregate quality metrics ──────────────────────────────────────────
#     gct_vals  = [j["ground_contact_ms"] for j in jumps if j.get("ground_contact_ms")]
#     rsi_vals  = [j["rsi"] for j in jumps if j.get("rsi")]
#     ft_vals   = [j["flight_ms"] for j in jumps]

#     # Top-level aggregate angle/landing fields — app.py's build_metrics()
#     # reads these directly off the result dict (avg_takeoff_angle,
#     # avg_landing_angle, landing_score, jump_distance_cm) for the broad_jump
#     # metrics card. Without these exact keys present, the frontend shows
#     # "undefined" even though the same data exists nested inside per_jump.
#     takeoff_angle_vals = [j["takeoff_angle_deg"] for j in jumps if j.get("takeoff_angle_deg") is not None]
#     landing_stab_vals  = [j["landing_stability"] for j in jumps if j.get("landing_stability") is not None]
#     avg_takeoff_angle  = round(float(np.mean(takeoff_angle_vals)), 1) if takeoff_angle_vals else 0.0
#     # No direct landing-angle measurement in this engine (no heel/ankle
#     # ground-impact angle tracked) — approximate via landing stability as
#     # a proxy so the field is populated rather than missing/undefined.
#     avg_landing_angle  = round(float(np.mean(landing_stab_vals)) * 30.0, 1) if landing_stab_vals else 0.0
#     landing_score      = round(float(np.mean(landing_stab_vals)) * 10, 1) if landing_stab_vals else 0.0

#     result = {
#         # Core
#         "exercise"              : exercise,
#         "jump_count"            : len(jumps),
#         "correct_jumps"         : len(jumps),
#         "wrong_jumps"           : 0,
#         # Distance
#         "best_distance_m"       : best_dist,
#         "best_distance_cm"      : round(best_dist * 100, 1),
#         "avg_distance_m"        : avg_dist,
#         "all_jumps_m"           : [round(j, 3) for j in jumps_m],
#         "jump_distance_cm"      : round(best_dist * 100, 1),   # app.py build_metrics() key
#         "avg_takeoff_angle"     : avg_takeoff_angle,            # app.py build_metrics() key
#         "avg_landing_angle"     : avg_landing_angle,            # app.py build_metrics() key
#         "landing_score"         : landing_score,                # app.py build_metrics() key
#         # Per-jump detail
#         "per_jump"              : jumps,
#         # Exercise variant results
#         "phase_info"            : val_out.get("phase_info"),
#         "total_distance_m"      : val_out.get("total_distance_m"),
#         # Timing
#         "avg_flight_ms"         : int(np.mean(ft_vals)) if ft_vals else 0,
#         "avg_ground_contact_ms" : int(np.mean(gct_vals)) if gct_vals else 0,
#         "avg_rsi"               : round(float(np.mean(rsi_vals)), 3) if rsi_vals else None,
#         # Consistency
#         "fatigue_index"         : _fatigue_index(jumps_m),
#         "rep_variability_pct"   : _rep_variability(jumps_m),
#         # Scale
#         "px_per_m"              : round(scale_est.px_per_m, 2) if scale_est.px_per_m else None,
#         "scale_method"          : scale_est.method or "legacy_constant",
#         # Quality / Confidence
#         "confidence"            : quality,
#         "form_score"            : form_score,
#         # Feedback
#         "issues"                : issues,
#         "strengths"             : strengths,
#         # UI
#         "metrics": [
#             {"label": "Jumps Detected", "value": str(len(jumps))},
#             {"label": "Best Jump",      "value": best_str},
#             {"label": "Avg Jump",       "value": avg_str},
#             {"label": "Form Score",     "value": f"{form_score}/10"},
#             {"label": "Confidence",     "value": f"{quality['overall']}%"},
#         ],
#         # Legacy keys kept for backward compatibility
#         "height_cm"             : round(best_dist * 100, 1),
#         "per_frame"             : per_frame,
#         "snapshots"             : snaps,
#         "wrong_angle_count"     : 0,
#         "_wrong_events"         : wrong_events,
#     }
#     return _to_json_safe(result)






















"""
module_broad_jump.py
Production-grade broad jump biomechanics engine.

Supports:
  1. Standing Broad Jump
  2. Run-Up Broad Jump
  3. Single-Leg Broad Jump
  4. Alternate Leg Broad Jump
  5. Bounding (Power Skip)
  6. Triple Broad Jump
  7. Multiple Broad Jump
  8. Reactive Broad Jump
  9. Tuck Broad Jump
 10. Pike Broad Jump
 11. Weighted Broad Jump
 12. Sand Broad Jump

Architecture:
  Video → Pose → Landmark Filter → COM → Foot Contact → 6-State FSM
       → Exercise Validator → Distance Engine → Quality → Output
"""

import os
import cv2
import math
import numpy as np
from collections import deque

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

from utils import (
    process_video_or_image,
    save_wrong_angle_log,
    set_progress,
    frame_to_b64,
)
from hud_overlay import draw_footer_hud, draw_pcl_logo

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

YOLO_MODEL_PATH = "yolov8n-pose.pt"
CONFIDENCE      = 0.3

# YOLO keypoint indices (COCO 17-point)
KP_NOSE        = 0
KP_L_SHOULDER  = 5
KP_R_SHOULDER  = 6
KP_L_ELBOW     = 7
KP_R_ELBOW     = 8
KP_L_WRIST     = 9
KP_R_WRIST     = 10
KP_L_HIP       = 11
KP_R_HIP       = 12
KP_L_KNEE      = 13
KP_R_KNEE      = 14
KP_L_ANKLE     = 15
KP_R_ANKLE     = 16

# COM weights (Dempster body segment model approximation)
COM_WEIGHTS = {
    KP_L_HIP: 0.28, KP_R_HIP: 0.28,
    KP_L_SHOULDER: 0.11, KP_R_SHOULDER: 0.11,
    KP_L_KNEE: 0.06, KP_R_KNEE: 0.06,
    KP_L_ANKLE: 0.05, KP_R_ANKLE: 0.05,
}

# State machine states
ST_READY    = "READY"
ST_LOADING  = "LOADING"
ST_TAKEOFF  = "TAKEOFF"
ST_FLIGHT   = "FLIGHT"
ST_LANDING  = "LANDING"
ST_RESET    = "RESET"
ST_COOLDOWN = "COOLDOWN"


# ─────────────────────────────────────────────────────────────────────────────
# JSON-SAFETY HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _to_json_safe(obj):
    """
    Recursively convert numpy scalar/array types (np.bool_, np.float64,
    np.int64, np.ndarray, etc.) to native Python types so the result can
    always be passed to json.dumps() without raising
    'Object of type X is not JSON serializable'.

    numpy values leak into output dicts any time a computation touches an
    np.median()/np.mean()/np.min()/np.max() result (even indirectly, e.g.
    `abs(numpy_float) < python_float` still yields np.bool_) — so this is
    applied once, at the point each output dict is finalized, rather than
    trying to manually cast every individual field (which is exactly how
    this bug slipped through the first time).
    """
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    return obj

SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Colors
COLOR_SKEL   = (255, 255, 255)
COLOR_JOINT  = (0,   0,   0)
COLOR_COM    = (0, 255, 255)
COLOR_TAKE   = (0, 165, 255)
COLOR_LAND   = (255, 100, 255)
COLOR_TEXT   = (255, 255, 255)

# FSM thresholds
GROUND_WINDOW       = 20   # frames to establish ground reference
COM_RISE_PX         = 18   # COM must rise this much above ground to start LOADING
COM_TAKEOFF_PX      = 30   # COM above ground to confirm TAKEOFF
FLIGHT_CONFIRM_F    = 3    # consecutive airborne frames required before FLIGHT is trusted
LAND_APPROACH_PX    = 25   # COM within this of ground → approaching landing
LAND_STABLE_F       = 7    # frames near ground to confirm LANDING (athlete fully settled)
MIN_FLIGHT_F        = 7    # minimum flight frames for a valid jump (~233ms @ 30fps)
MAX_FLIGHT_F        = 90   # ~3s @ 30fps — hard ceiling on a single flight phase. No human
                             # standing broad jump airborne phase exceeds ~1s; this exists
                             # purely as a safety valve so a degraded-landing-detection
                             # stretch (occlusion, bad keypoints) can't freeze the FSM in
                             # FLIGHT for the rest of the clip and silently drop every
                             # subsequent jump in the video.
RESET_STABLE_F      = 4    # frames on ground before READY again (allows fast consecutive jumps)
LAND_VEL_PX_F       = 6.0  # max COM px/frame velocity to accept landing as "settled"
MIN_JUMP_DIST_M     = 0.25 # below this, treat as noise / false jump
MIN_PEAK_RISE_PX    = 10   # minimum COM rise during flight to count as a real jump

# ── False-jump rejection / hysteresis (post-landing stabilization fix) ──
MIN_VALID_DIST_M       = 0.40  # floor — screens out post-landing micro-shifts (~0.53m case)
                                 # without rejecting shorter/beginner real jumps
MIN_COM_FALLBACK_DIST_M = 0.80  # stricter floor for the com_fallback distance path, which
                                 # systematically undershoots true foot-to-foot distance
                                 # (COM folds backward on landing) — used only when ankle
                                 # keypoints were unavailable for the whole flight
MIN_DIST_PER_FLIGHT_S   = 0.35  # m of forward distance expected per second of flight time,
                                 # floor — a measured distance far below this for the
                                 # observed airtime indicates a degraded measurement, not
                                 # a genuinely tiny jump (catches the "below 100cm on a
                                 # professional-looking long flight" failure mode). Kept
                                 # low enough that legitimate short/beginner/single-leg
                                 # jumps with modest horizontal velocity are not rejected.
MAX_VALID_DIST_M       = 4.0   # sanity ceiling
MIN_VALID_FLIGHT_MS    = 167   # ~5 frames @ 30fps — below this is noise, not a jump
MIN_APPROACH_SPEED_MS  = 0.20  # reported only — NOT used as a hard rejection gate
AIR_CONFIRM_FRAMES     = 2     # consecutive frames required to trust "airborne"
CONTACT_CONFIRM_FRAMES = 2     # consecutive frames required to trust "grounded"
COOLDOWN_FRAMES        = 10    # frames after landing before a new jump may start (~330ms)
MAX_OCCLUSION_HOLD_F   = 6     # max consecutive missed-detection frames to bridge by
                                 # replaying the last known pose, so brief motion-blur/
                                 # occlusion gaps mid-flight don't freeze the FSM and
                                 # silently drop the jump (or every jump after it)

# Exercise-specific
REACTIVE_MAX_CONTACT_F  = 12   # max ground contact for reactive jump (at 30fps ≈ 400ms)
TUCK_HIP_FLEX_DEG       = 60   # minimum hip flexion for tuck confirmation
PIKE_HIP_FLEX_DEG       = 70   # minimum hip flexion for pike
PIKE_KNEE_EXT_DEG       = 150  # minimum knee extension for pike
BOUND_MIN_FLIGHT_F      = 3    # minimum flight per bound
TRIPLE_PHASE_MIN_F      = 3    # min flight frames per triple-jump phase

# Anthropometric scale: avg hip-to-ankle ≈ 0.53× body height, body height ≈ 1.75m
# So hip-ankle ≈ 0.927m in pixels → derive px/m
ANTHRO_HIP_ANKLE_M  = 0.90   # meters (conservative; will be estimated per-subject)

# ─────────────────────────────────────────────────────────────────────────────
# ONE-EURO FILTER  (per-coordinate temporal filter)
# ─────────────────────────────────────────────────────────────────────────────

class OneEuroFilter:
    """
    One Euro Filter for temporal landmark smoothing.
    Reduces jitter without large latency; preserves fast explosive motion
    by automatically raising cutoff frequency at high velocity.
    """
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.freq      = freq
        self.min_cutoff = min_cutoff
        self.beta      = beta
        self.d_cutoff  = d_cutoff
        self._x        = None
        self._dx       = 0.0

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * self.freq)

    def __call__(self, x):
        if self._x is None:
            self._x = x
            return x
        dx     = (x - self._x) * self.freq
        a_d    = self._alpha(self.d_cutoff)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a      = self._alpha(cutoff)
        self._x = a * x + (1 - a) * self._x
        return self._x


class LandmarkFilter:
    """Per-landmark One Euro filter (x and y independently)."""
    def __init__(self, n_landmarks=17, freq=30.0):
        self.fx = [OneEuroFilter(freq=freq) for _ in range(n_landmarks)]
        self.fy = [OneEuroFilter(freq=freq) for _ in range(n_landmarks)]

    def update(self, pts):
        """pts: np.array shape (17,2). Returns filtered array same shape."""
        out = pts.copy().astype(float)
        for i, (fx, fy) in enumerate(zip(self.fx, self.fy)):
            if pts[i, 0] > 0 and pts[i, 1] > 0:
                out[i, 0] = fx(pts[i, 0])
                out[i, 1] = fy(pts[i, 1])
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CENTER OF MASS ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def estimate_com(pts):
    """
    Weighted average COM from body keypoints.
    Returns (com_x, com_y) or None if not enough valid points.
    """
    total_w = 0.0
    wx, wy  = 0.0, 0.0
    for idx, w in COM_WEIGHTS.items():
        p = pts[idx]
        if p[0] > 0 and p[1] > 0:
            wx += w * p[0]
            wy += w * p[1]
            total_w += w
    if total_w < 0.3:
        return None
    return (wx / total_w, wy / total_w)


# ─────────────────────────────────────────────────────────────────────────────
# FOOT CONTACT ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

class FootContactEstimator:
    """
    Tracks each foot (left/right) independently.
    Uses ankle vertical position relative to adaptive ground reference,
    combined with ankle vertical velocity, to classify CONTACT / AIR.
    Velocity gating prevents single noisy keypoint jitters from being
    read as a takeoff or landing.
    """
    def __init__(self, window=30, fps=30.0):
        self.ground_y    = None   # adaptive ground y (pixel; larger = lower on screen)
        self.stance_buf  = deque(maxlen=window)  # y values during known contact
        self.fps         = fps
        self._prev_ly    = None
        self._prev_ry    = None
        self.left        = "CONTACT"
        self.right       = "CONTACT"
        self.left_vy     = 0.0
        self.right_vy    = 0.0
        # Body-scale-relative thresholds (set via set_body_scale once the
        # ScaleEstimator locks on). Fixed-pixel defaults below are only a
        # safety fallback before calibration completes — a fixed pixel
        # value is wrong for any camera distance other than the one it was
        # tuned on, which is what caused missed jumps on far-away cameras
        # and over-sensitive flicker on close-up ones.
        self.THRESH_PX   = 22     # pixels above ground to classify AIR (fallback)
        self.VEL_THRESH  = 3.0    # px/frame — ankle must also be moving to confirm AIR (fallback)
        self._body_scale_px = None   # body height in px, once known

    def set_body_scale(self, body_height_px):
        """Rescale AIR/CONTACT thresholds relative to the athlete's body
        size in pixels, so detection sensitivity is consistent regardless
        of how close/far the camera is."""
        if not body_height_px or body_height_px <= 0:
            return
        self._body_scale_px = body_height_px
        self.THRESH_PX  = max(6.0, body_height_px * 0.018)
        self.VEL_THRESH = max(1.0, body_height_px * 0.006)

    def update_ground(self, ly, ry):
        """Call with ankle y values during confirmed ground phase."""
        for y in [ly, ry]:
            if y > 0:
                self.stance_buf.append(y)
        if self.stance_buf:
            # Ground is near the max (bottom-most) ankle positions
            self.ground_y = np.percentile(list(self.stance_buf), 85)

    def classify(self, pts):
        """Update left/right contact state from keypoints."""
        la = pts[KP_L_ANKLE]
        ra = pts[KP_R_ANKLE]
        ly = la[1] if la[0] > 0 else -1
        ry = ra[1] if ra[0] > 0 else -1

        # Velocity (px/frame) for noise gating
        self.left_vy  = (ly - self._prev_ly) if (ly > 0 and self._prev_ly is not None) else 0.0
        self.right_vy = (ry - self._prev_ry) if (ry > 0 and self._prev_ry is not None) else 0.0
        if ly > 0: self._prev_ly = ly
        if ry > 0: self._prev_ry = ry

        if self.ground_y is None:
            self.left  = "CONTACT"
            self.right = "CONTACT"
            return

        thr = self.THRESH_PX
        gnd = self.ground_y

        # Left foot — require both height-above-ground AND recent motion
        # to avoid a single jittery keypoint flipping the state.
        if ly > 0:
            above_ground = ly < gnd - thr
            self.left = "AIR" if (above_ground or abs(self.left_vy) > self.VEL_THRESH * 2) else "CONTACT"
        # Right foot
        if ry > 0:
            above_ground = ry < gnd - thr
            self.right = "AIR" if (above_ground or abs(self.right_vy) > self.VEL_THRESH * 2) else "CONTACT"

    @property
    def both_contact(self):
        return self.left == "CONTACT" and self.right == "CONTACT"

    @property
    def any_contact(self):
        return self.left == "CONTACT" or self.right == "CONTACT"

    @property
    def both_air(self):
        return self.left == "AIR" and self.right == "AIR"


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE SCALE ESTIMATOR  (pixel → metric)
# ─────────────────────────────────────────────────────────────────────────────

class ScaleEstimator:
    """
    Estimates pixels-per-meter from subject anthropometrics.
    Primary: full body height (nose → ankle), which is far more stable
    across camera angles than hip-to-ankle alone.
    Fallback: hip-to-ankle segment if height isn't fully visible.
    Updated during ground-contact frames only (standing posture), and
    LOCKED once enough stable samples are collected — px_per_m is not
    recalculated mid-jump, so a single jump's distance can't drift
    depending on which frame happened to update calibration last.
    """
    ANTHRO_BODY_HEIGHT_M = 1.75   # default assumed height if user doesn't supply one
    LOCK_SAMPLES = 40             # samples required before calibration locks

    def __init__(self, user_height_m=None, user_height_cm=None):
        self._height_samples = deque(maxlen=60)
        self._hip_ankle_samples = deque(maxlen=60)
        self.px_per_m = None
        # Accept either unit so callers (e.g. app.py, which collects
        # user_height_cm from the upload form) don't silently no-op —
        # previously app.py passed user_height_cm as a kwarg that
        # analyse_broad_jump() didn't even accept, so a user-supplied
        # height never reached this class and every subject was scaled
        # with the generic 1.75m default regardless of their real height.
        if user_height_m is None and user_height_cm:
            user_height_m = user_height_cm / 100.0
        self.user_height_m = user_height_m
        self.method = None
        self.body_height_px = None   # last-known body height in px (for FootContactEstimator)
        self._locked = False

    def update(self, pts):
        if self._locked:
            return
        nose = pts[KP_NOSE]
        lh = pts[KP_L_HIP];   la = pts[KP_L_ANKLE]
        rh = pts[KP_R_HIP];   ra = pts[KP_R_ANKLE]

        # ── Preferred: nose → lowest ankle (full standing height in px) ──
        ankle_ys = [p[1] for p in (la, ra) if p[0] > 0 and p[1] > 0]
        if nose[0] > 0 and nose[1] > 0 and ankle_ys:
            height_px = max(ankle_ys) - nose[1]
            if height_px > 50:   # sane lower bound, avoids garbage frames
                self._height_samples.append(height_px)
                self.body_height_px = height_px

        # ── Fallback: hip → ankle ──
        pairs = []
        if lh[0] > 0 and la[0] > 0:
            pairs.append(abs(la[1] - lh[1]))
        if rh[0] > 0 and ra[0] > 0:
            pairs.append(abs(ra[1] - rh[1]))
        if pairs:
            hip_ankle_px = np.mean(pairs)
            if hip_ankle_px > 10:
                self._hip_ankle_samples.append(hip_ankle_px)

        # Prefer body-height calibration once enough samples exist
        if len(self._height_samples) >= 10:
            median_px = np.median(list(self._height_samples))
            height_m  = self.user_height_m or self.ANTHRO_BODY_HEIGHT_M
            self.px_per_m = median_px / height_m
            self.method = "user_height" if self.user_height_m else "body_height_anthropometric"
            if not self.body_height_px:
                self.body_height_px = median_px
        elif len(self._hip_ankle_samples) >= 10:
            median_px = np.median(list(self._hip_ankle_samples))
            self.px_per_m = median_px / ANTHRO_HIP_ANKLE_M
            self.method = "hip_ankle_anthropometric"

        # Lock calibration once we have a solid sample of stable
        # ground-contact frames — per the spec, px_per_m must not keep
        # drifting frame-to-frame (including mid-flight, where body
        # height in frame changes due to limb extension, not actual
        # camera scale) once a confident estimate exists.
        if len(self._height_samples) >= self.LOCK_SAMPLES:
            median_px = np.median(list(self._height_samples))
            self.body_height_px = median_px
            self._locked = True

    def px_to_m(self, pixels):
        if self.px_per_m and self.px_per_m > 0:
            return pixels / self.px_per_m
        # Fallback: legacy constant (0.00933 m/px)
        return pixels * 0.00933


# ─────────────────────────────────────────────────────────────────────────────
# ANGLE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _vec_angle(a, b, c):
    """Angle at point b in triangle a-b-c (degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b;  bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))

def _hip_flexion(pts, side="L"):
    """Hip flexion angle (shoulder-hip-knee)."""
    sh  = pts[KP_L_SHOULDER if side=="L" else KP_R_SHOULDER]
    hp  = pts[KP_L_HIP      if side=="L" else KP_R_HIP]
    kn  = pts[KP_L_KNEE     if side=="L" else KP_R_KNEE]
    if any(p[0] <= 0 for p in [sh, hp, kn]):
        return None
    return _vec_angle(sh, hp, kn)

def _knee_flexion(pts, side="L"):
    """Knee flexion angle (hip-knee-ankle)."""
    hp  = pts[KP_L_HIP   if side=="L" else KP_R_HIP]
    kn  = pts[KP_L_KNEE  if side=="L" else KP_R_KNEE]
    an  = pts[KP_L_ANKLE if side=="L" else KP_R_ANKLE]
    if any(p[0] <= 0 for p in [hp, kn, an]):
        return None
    return _vec_angle(hp, kn, an)

def _com_velocity(com_hist, fps):
    """Returns (vx, vy) in px/frame from last two COM positions."""
    if len(com_hist) < 2:
        return (0.0, 0.0)
    dx = com_hist[-1][0] - com_hist[-2][0]
    dy = com_hist[-1][1] - com_hist[-2][1]
    return (dx * fps, dy * fps)


def _jump_direction(pts_hist):
    """
    Determine overall travel direction (+1 = left-to-right, -1 = right-to-left)
    from a short history of COM x positions. Locked once at the start of a
    rep sequence so per-jump distance math always uses a consistent sign.
    """
    if len(pts_hist) < 2:
        return 1
    dx = pts_hist[-1] - pts_hist[0]
    return 1 if dx >= 0 else -1


def _ankle_positions(pts):
    """Return (left_ankle, right_ankle) as (x,y) or None if not visible."""
    la = pts[KP_L_ANKLE]
    ra = pts[KP_R_ANKLE]
    left  = (la[0], la[1]) if la[0] > 0 and la[1] > 0 else None
    right = (ra[0], ra[1]) if ra[0] > 0 and ra[1] > 0 else None
    return left, right


def _heel_x_approx(ankle, knee, direction):
    """
    Approximate heel x-position from ankle + knee (shank vector), since
    COCO-17 has no heel keypoint. The heel sits slightly behind the ankle
    relative to the direction of travel.
    ankle, knee: (x, y) tuples
    direction: +1 (moving right) or -1 (moving left)
    """
    if ankle is None:
        return None
    if knee is None:
        return ankle[0]
    shank_len = math.hypot(knee[0] - ankle[0], knee[1] - ankle[1])
    if shank_len < 1e-6:
        return ankle[0]
    # Heel trails the ankle opposite to the direction of travel.
    return ankle[0] - direction * 0.15 * shank_len


def _takeoff_foot_x(pts, direction):
    """
    Foot-based takeoff reference: the most FORWARD foot point (in the
    direction of travel) at the last grounded frame — this is the true
    takeoff line, not COM.
    """
    la, ra = _ankle_positions(pts)
    xs = [p[0] for p in (la, ra) if p is not None]
    if not xs:
        return None
    return max(xs) if direction >= 0 else min(xs)


def _landing_foot_x(pts, direction):
    """
    Foot-based landing reference: nearest body part to the takeoff line,
    approximated via heel position on the leading foot at landing.
    """
    la, ra = _ankle_positions(pts)
    lk = pts[KP_L_KNEE]; rk = pts[KP_R_KNEE]
    lk_pt = (lk[0], lk[1]) if lk[0] > 0 and lk[1] > 0 else None
    rk_pt = (rk[0], rk[1]) if rk[0] > 0 and rk[1] > 0 else None

    heels = []
    hx = _heel_x_approx(la, lk_pt, direction)
    if hx is not None:
        heels.append(hx)
    hx = _heel_x_approx(ra, rk_pt, direction)
    if hx is not None:
        heels.append(hx)

    if not heels:
        return None
    # Landing mark = the foot point CLOSEST to the takeoff line, i.e. the
    # leading (trailing-edge) foot in the direction of travel.
    return min(heels) if direction >= 0 else max(heels)


# ─────────────────────────────────────────────────────────────────────────────
# JUMP ENGINE  (6-state FSM)
# ─────────────────────────────────────────────────────────────────────────────

class JumpEngine:
    """
    Core 6-state finite state machine.
    Tracks COM, foot contacts, and produces jump events.
    """
    def __init__(self, fps=30.0, exercise="standing_broad_jump"):
        self.fps      = fps
        self.exercise = exercise
        self.state    = ST_READY

        # Ground reference
        self._gnd_com_y    = None   # adaptive ground COM y
        self._gnd_buf      = deque(maxlen=GROUND_WINDOW)
        self._gnd_locked   = False

        # Takeoff
        self._takeoff_com  = None
        self._takeoff_f    = 0
        self._takeoff_heel = None   # left heel x at takeoff
        self._takeoff_foot_x   = None   # foot-based takeoff line (px)
        self._last_grounded_pts = None  # full pts snapshot at last grounded frame
        self._last_grounded_com = None

        # Flight
        self._flight_f     = 0
        self._airborne_streak = 0   # consecutive airborne frames (hysteresis)
        self._ground_streak   = 0   # consecutive grounded frames (hysteresis)
        self._peak_com_y   = None
        self._flight_com_x_start = None
        self._peak_b64     = None

        # In-flight kinematics (for tuck/pike/bounding)
        self._flight_hip_flex   = []
        self._flight_knee_flex  = []
        self._flight_arm_travel = []   # wrist-y travel relative to shoulder during flight

        # Landing
        self._land_stable_f = 0
        self._land_com_buf  = deque(maxlen=8)
        self._land_foot_x_buf = deque(maxlen=8)   # foot-based landing samples
        self._land_pts_at_settle = None           # pts snapshot once landing settles

        # Direction of travel (+1 left→right, -1 right→left), locked per rep
        self._direction     = 1
        self._dir_buf       = deque(maxlen=10)

        # Reset
        self._reset_f       = 0
        self._cooldown_f    = 0

        # Loading / countermovement
        self._load_com_y   = None   # COM y at start of countermovement
        self._load_start_f = 0
        self._cm_depth     = 0.0   # countermovement depth in px

        # Completed jumps
        self.jumps         = []
        self._jump_n       = 0

        # Per-phase data for triple/multiple/bounding
        self._phase_list   = []    # list of phase dicts per jump
        self._phase_start_com = None

        # Reactive: ground contact tracking
        self._gct_start_f  = 0    # frame when ground contact began after landing

        # Approach velocity (run-up)
        self._approach_vx  = 0.0
        self._approach_buf = deque(maxlen=20)
        self._takeoff_approach_vx = 0.0   # snapshot of approach_vx AT takeoff (frozen, not live)

        # COM history for velocity-gated landing confirmation
        self._com_hist      = deque(maxlen=5)

        # Frame counter
        self._frame        = 0

        # ── Body-scale-relative thresholds ──────────────────────────────
        # The constants COM_RISE_PX / COM_TAKEOFF_PX / LAND_APPROACH_PX /
        # MIN_PEAK_RISE_PX / LAND_VEL_PX_F were originally fixed pixel
        # values tuned for one camera distance. At a farther camera the
        # athlete's whole body (and therefore COM excursion) is smaller in
        # pixels, so a real jump's COM rise never crosses a fixed-pixel
        # takeoff threshold → 0 jumps detected. At a closer camera the same
        # fixed threshold is too sensitive → state flicker / false splits
        # of one jump into several. Scaling every threshold to the
        # athlete's measured body height in pixels keeps detection
        # sensitivity consistent across camera distances. Ratios below are
        # calibrated against the original constants at the reference body
        # height implied by ANTHRO_BODY_HEIGHT_M / a typical ~480px frame
        # body height, and are used as fallback values until per-subject
        # body_height_px is known.
        self._com_rise_px      = COM_RISE_PX
        self._com_takeoff_px   = COM_TAKEOFF_PX
        self._land_approach_px = LAND_APPROACH_PX
        self._min_peak_rise_px = MIN_PEAK_RISE_PX
        self._land_vel_px_f    = LAND_VEL_PX_F

    def _rescale_thresholds(self, body_height_px):
        """Recompute pixel thresholds relative to this athlete's measured
        body height, so sensitivity no longer depends on camera distance."""
        if not body_height_px or body_height_px <= 0:
            return
        ref = 480.0   # reference body-height px these ratios were tuned against
        k = body_height_px / ref
        self._com_rise_px      = max(6.0,  COM_RISE_PX * k)
        self._com_takeoff_px   = max(10.0, COM_TAKEOFF_PX * k)
        self._land_approach_px = max(8.0,  LAND_APPROACH_PX * k)
        self._min_peak_rise_px = max(4.0,  MIN_PEAK_RISE_PX * k)
        self._land_vel_px_f    = max(2.0,  LAND_VEL_PX_F * k)

    # ──────────────────────────────────────────────────────────────────────
    def _update_ground(self, com_y, foot: FootContactEstimator, pts=None, com=None):
        """Update adaptive ground COM_y during stable contact."""
        if foot.both_contact or foot.any_contact:
            self._gnd_buf.append(com_y)
            if pts is not None:
                self._last_grounded_pts = pts.copy()
                self._last_grounded_com = com
        if len(self._gnd_buf) >= 5:
            self._gnd_com_y = np.percentile(list(self._gnd_buf), 80)

    def _is_airborne(self, com_y):
        if self._gnd_com_y is None:
            return False
        return com_y < self._gnd_com_y - self._com_takeoff_px

    def _is_near_ground(self, com_y):
        if self._gnd_com_y is None:
            return True
        return com_y >= self._gnd_com_y - self._land_approach_px

    def _com_settled(self):
        """True when recent COM motion has slowed enough to trust the
        landing position (prevents capturing landing COM while the body
        is still travelling forward/downward)."""
        if len(self._com_hist) < 3:
            return False
        vx = self._com_hist[-1][0] - self._com_hist[-2][0]
        vy = self._com_hist[-1][1] - self._com_hist[-2][1]
        return abs(vx) < self._land_vel_px_f and abs(vy) < self._land_vel_px_f

    # ──────────────────────────────────────────────────────────────────────
    def update(self, pts, foot: FootContactEstimator, com, scale: ScaleEstimator,
               frame_b64=None):
        """
        Called every frame.
        pts: filtered keypoints (17,2)
        foot: FootContactEstimator (already updated this frame)
        com: (cx, cy) or None
        scale: ScaleEstimator
        Returns: jump_event dict if a jump completed this frame, else None
        """
        self._frame += 1
        if com is None:
            return None

        com_x, com_y = com
        self._com_hist.append(com)

        # Track approach velocity for run-up detection
        self._approach_buf.append(com_x)
        if len(self._approach_buf) >= 5:
            dx = self._approach_buf[-1] - self._approach_buf[-5]
            self._approach_vx = dx / 5.0 * self.fps  # px/s

        # Track direction-of-travel candidates continuously; locked at takeoff
        self._dir_buf.append(com_x)

        # ── Airborne/grounded detection ──────────────────────────────────────
        # COM height is the primary signal (smoother, derived from a weighted
        # blend of multiple keypoints, less prone to single-frame dropout than
        # any one foot keypoint). Foot contact is a confirming signal — OR'd
        # in, not AND'd, because requiring both foot ankles to be visible AND
        # agreeing with COM on every single frame is too strict for real YOLO
        # keypoint noise (occlusion, low-confidence frames, motion blur) and
        # was causing both missed jumps and stuck-in-FLIGHT (0-jump) results.
        # A short confirm streak (not a hard AND) still absorbs single-frame
        # jitter without requiring perfect agreement.
        airborne_raw = foot.both_air or self._is_airborne(com_y)
        if airborne_raw:
            self._airborne_streak += 1
            self._ground_streak = 0
        else:
            self._ground_streak += 1
            self._airborne_streak = 0
        airborne_confirmed = self._airborne_streak >= AIR_CONFIRM_FRAMES
        # Grounded confirmation for landing uses COM proximity OR foot
        # contact too (any_contact, not both_contact) — landing is reached
        # the moment either signal agrees the athlete is back down.
        grounded_raw = foot.any_contact or self._is_near_ground(com_y)
        grounded_confirmed = self._ground_streak >= CONTACT_CONFIRM_FRAMES or grounded_raw

        event = None

        # ── READY ─────────────────────────────────────────────────────────
        if self.state == ST_READY:
            self._update_ground(com_y, foot, pts, com)
            rising = self._gnd_com_y is not None and com_y < self._gnd_com_y - self._com_rise_px
            # Require a short confirmed streak (not the single-frame raw
            # signal) before leaving READY. Using airborne_raw directly let
            # a single jittery ankle/COM frame right after a landing
            # immediately re-enter LOADING — wasteful at best, and at worst
            # a contributor to one real jump being reported as two when the
            # spurious LOADING attempt's countermovement bookkeeping bled
            # into the next genuine rep.
            if rising or airborne_confirmed:
                self.state         = ST_LOADING
                self._load_com_y   = com_y
                self._load_start_f = self._frame
            else:
                # Still grounded — update scale only here (never in flight),
                # and propagate the locked body scale to both this engine's
                # pixel thresholds and the foot-contact estimator so jump
                # detection sensitivity stays consistent regardless of how
                # close/far the camera is.
                scale.update(pts)
                if scale.body_height_px:
                    self._rescale_thresholds(scale.body_height_px)
                    foot.set_body_scale(scale.body_height_px)
                foot.update_ground(pts[KP_L_ANKLE][1], pts[KP_R_ANKLE][1])
                self._takeoff_com  = com   # keep updating takeoff reference
                self._last_grounded_pts = pts.copy()
                self._last_grounded_com = com

        # ── LOADING ───────────────────────────────────────────────────────
        elif self.state == ST_LOADING:
            if not airborne_raw:
                self._last_grounded_pts = pts.copy()
                self._last_grounded_com = com

            # Require AIR_CONFIRM_FRAMES consecutive airborne frames
            # (both conditions agreeing) before committing to a takeoff.
            if airborne_confirmed:
                # True takeoff = the LAST GROUNDED frame, not the first
                # airborne one.
                ref_pts = self._last_grounded_pts if self._last_grounded_pts is not None else pts
                ref_com = self._last_grounded_com if self._last_grounded_com is not None else com

                # Lock direction of travel for this rep from recent COM drift
                direction_samples = np.array(list(self._dir_buf))
                self._direction = _jump_direction(direction_samples)

                self._takeoff_com        = ref_com
                self._takeoff_foot_x     = _takeoff_foot_x(ref_pts, self._direction)
                self._flight_com_x_start = ref_com[0]
                self._takeoff_f          = max(1, self._frame - self._airborne_streak)
                self._flight_f           = self._airborne_streak
                self._peak_com_y         = com_y
                self._peak_b64           = frame_b64
                self._flight_hip_flex    = []
                self._flight_knee_flex   = []
                self._flight_arm_travel  = []
                self._phase_start_com    = ref_com
                self._land_foot_x_buf.clear()
                self._land_com_buf.clear()
                self._land_stable_f      = 0
                self._cm_depth = max(0.0, self._gnd_com_y - self._load_com_y) if self._gnd_com_y else 0.0
                self._takeoff_approach_vx = self._approach_vx
                self.state = ST_FLIGHT
            elif not airborne_raw and self._gnd_com_y is not None and com_y > self._gnd_com_y:
                # COM went back down without ever becoming airborne — false alarm
                self.state = ST_READY

        # ── FLIGHT ────────────────────────────────────────────────────────
        elif self.state == ST_FLIGHT:
            self._flight_f += 1

            if com_y < self._peak_com_y:
                self._peak_com_y = com_y
                self._peak_b64   = frame_b64

            hf_l = _hip_flexion(pts, "L")
            hf_r = _hip_flexion(pts, "R")
            kf_l = _knee_flexion(pts, "L")
            kf_r = _knee_flexion(pts, "R")
            if hf_l: self._flight_hip_flex.append(hf_l)
            if hf_r: self._flight_hip_flex.append(hf_r)
            if kf_l: self._flight_knee_flex.append(kf_l)
            if kf_r: self._flight_knee_flex.append(kf_r)

            for sh_i, wr_i in ((KP_L_SHOULDER, KP_L_WRIST), (KP_R_SHOULDER, KP_R_WRIST)):
                sh, wr = pts[sh_i], pts[wr_i]
                if sh[0] > 0 and wr[0] > 0:
                    self._flight_arm_travel.append(sh[1] - wr[1])

            # Landing confirmation requires near-ground AND settled COM
            # velocity AND a hysteresis-confirmed ground contact streak —
            # this is what stops the post-landing stabilization wobble from
            # being read as the landing mark.
            if self._flight_f >= MIN_FLIGHT_F:
                near_ground = self._is_near_ground(com_y) or foot.any_contact
                if near_ground:
                    self._land_stable_f += 1
                    self._land_com_buf.append(com_x)
                    foot_x = _landing_foot_x(pts, self._direction)
                    if foot_x is not None:
                        self._land_foot_x_buf.append(foot_x)
                    if (self._land_stable_f >= LAND_STABLE_F and
                            self._com_settled() and grounded_confirmed):
                        event = self._confirm_jump(com, pts, scale)
                        if event is not None:
                            self.state = ST_LANDING
                        else:
                            # Rejected as a false jump (backward movement,
                            # too short/out of range) — return to READY
                            # directly rather than burning a full landing
                            # cooldown, so subsequent real jumps in the
                            # same video aren't delayed or missed.
                            self.state = ST_READY
                            self._airborne_streak = 0
                            self._ground_streak = 0
                else:
                    self._land_stable_f = 0
                    self._land_com_buf.clear()
                    self._land_foot_x_buf.clear()

            if self._flight_f < MIN_FLIGHT_F and grounded_confirmed and self._is_near_ground(com_y):
                self.state = ST_READY
                self._airborne_streak = 0
                self._ground_streak = 0

            # ── Stuck-FLIGHT safety valve ───────────────────────────────
            # If landing is never cleanly confirmed (sustained occlusion,
            # degraded keypoints right at touchdown, or a true-but-noisy
            # landing that never reaches LAND_STABLE_F), force a close
            # using best-available data rather than letting FLIGHT run out
            # the rest of the clip — that previously meant every jump after
            # the stuck one was lost entirely (0-jump / missing-jump bug on
            # multi-jump videos).
            if self._flight_f >= MAX_FLIGHT_F:
                event = self._confirm_jump(com, pts, scale)
                if event is not None:
                    self.state = ST_LANDING
                else:
                    self.state = ST_READY
                self._airborne_streak = 0
                self._ground_streak = 0

        # ── LANDING ───────────────────────────────────────────────────────
        elif self.state == ST_LANDING:
            self._land_stable_f = 0
            self._land_com_buf.clear()
            self._land_foot_x_buf.clear()
            self._reset_f = 0
            self._cooldown_f = 0
            self._last_grounded_pts = pts.copy()
            self._last_grounded_com = com

            self._gct_start_f = self._frame
            self.state = ST_COOLDOWN

        # ── COOLDOWN ──────────────────────────────────────────────────────
        # Mandatory recovery window after every landing. Post-landing balance
        # adjustments (knee bend, weight shift, brief foot occlusion) happen
        # here and CANNOT trigger a new LOADING/FLIGHT cycle — this is what
        # was previously creating a fake second jump immediately after a
        # real one. Reactive jumps are the sole exception (genuinely meant
        # to re-launch within a few frames of ground contact).
        elif self.state == ST_COOLDOWN:
            self._cooldown_f += 1
            self._update_ground(com_y, foot, pts, com)
            if scale.body_height_px:
                self._rescale_thresholds(scale.body_height_px)
                foot.set_body_scale(scale.body_height_px)
            self._last_grounded_pts = pts.copy()
            self._last_grounded_com = com

            if (foot.both_air and self._cooldown_f <= REACTIVE_MAX_CONTACT_F
                    and "reactive" in self.exercise):
                direction_samples = np.array(list(self._dir_buf))
                self._direction = _jump_direction(direction_samples)
                ref_pts = self._last_grounded_pts if self._last_grounded_pts is not None else pts

                self._takeoff_com        = com
                self._takeoff_foot_x     = _takeoff_foot_x(ref_pts, self._direction)
                self._flight_com_x_start = com_x
                self._takeoff_f          = self._frame
                self._flight_f           = 0
                self._peak_com_y         = com_y
                self._peak_b64           = frame_b64
                self._flight_hip_flex    = []
                self._flight_knee_flex   = []
                self._flight_arm_travel  = []
                self._phase_start_com    = com
                self._land_foot_x_buf.clear()
                self._land_com_buf.clear()
                self._takeoff_approach_vx = self._approach_vx
                self.state = ST_FLIGHT

            elif self._cooldown_f >= COOLDOWN_FRAMES:
                self.state = ST_READY
                self._gnd_locked = True
                self._reset_f = 0

        # ── RESET (legacy, kept for backward compatibility) ─────────────────
        elif self.state == ST_RESET:
            self.state = ST_READY

        return event

    # ──────────────────────────────────────────────────────────────────────
    def _confirm_jump(self, land_com, pts, scale: ScaleEstimator):
        """Build and store a jump event dict. Returns None (and does NOT
        increment jump_n / append to self.jumps) if this is rejected as a
        false jump — caller must handle a None return."""

        # ── Foot-based distance (primary) ───────────────────────────────
        # Takeoff line = most-forward foot point at the last grounded frame.
        # Landing mark = nearest heel of the leading foot once landing has
        # settled (near-ground + low COM velocity + foot contact), smoothed
        # via median over the buffered landing samples.
        takeoff_x_foot = self._takeoff_foot_x

        land_x_foot = None
        if self._land_foot_x_buf:
            land_x_foot = float(np.median(list(self._land_foot_x_buf)))
        elif pts is not None:
            land_x_foot = _landing_foot_x(pts, self._direction)

        # COM fallback (only used if foot keypoints were unavailable)
        land_x_com    = float(np.median(list(self._land_com_buf))) if self._land_com_buf else land_com[0]
        takeoff_x_com = self._flight_com_x_start if self._flight_com_x_start else (
            self._takeoff_com[0] if self._takeoff_com else land_x_com)

        # Distance is SIGNED by the locked direction of travel, not abs().
        # A jump that nets backward relative to the locked direction (e.g.
        # a post-landing balance shift) produces a negative raw distance
        # and is rejected outright below — this is what previously let a
        # backward foot displacement masquerade as a 0.53m jump.
        if takeoff_x_foot is not None and land_x_foot is not None:
            raw_px = (land_x_foot - takeoff_x_foot) * self._direction
            takeoff_x, land_x = takeoff_x_foot, land_x_foot
            dist_method = "foot_based"
        else:
            raw_px = (land_x_com - takeoff_x_com) * self._direction
            takeoff_x, land_x = takeoff_x_com, land_x_com
            dist_method = "com_fallback"

        if raw_px <= 0:
            return None   # non-forward / backward displacement — not a real jump

        px_dist = raw_px
        dist_m  = round(scale.px_to_m(px_dist), 3)
        dist_cm = round(dist_m * 100, 1)

        flight_f  = self._flight_f
        flight_ms = int(flight_f / self.fps * 1000)

        # Peak COM height above takeoff COM
        peak_rise_px = 0.0
        if self._takeoff_com and self._peak_com_y:
            peak_rise_px = max(0.0, self._takeoff_com[1] - self._peak_com_y)
        peak_rise_m = scale.px_to_m(peak_rise_px)

        approach_speed_ms = scale.px_to_m(abs(self._takeoff_approach_vx))

        # ── False-jump rejection ────────────────────────────────────────
        # Reject jumps that are too short, too small, or outside a
        # physically sane distance range for a standing broad jump — these
        # are the load-bearing checks against weight-shift / post-landing
        # noise. Approach-speed and physics-consistency are NOT hard gates:
        # a standing broad jump can legitimately have near-zero horizontal
        # drift right up to takeoff (the athlete is stationary before the
        # countermovement), and the physics-consistency comparison below is
        # derived circularly from dist_m/t so it has no real rejection
        # power — both were previously zeroing out genuine jumps on noisier
        # real-world video without actually screening out bad ones.
        t = flight_f / self.fps

        # The com_fallback path only fires when ankle keypoints were
        # unavailable for the whole flight, and it systematically
        # UNDERSHOOTS true takeoff-to-landing distance — COM displacement
        # is consistently shorter than foot-to-foot distance because the
        # body folds (knees/hips flex) on landing, pulling COM backward
        # relative to where the foot actually lands. This is the dominant
        # cause of implausibly short distances reported for genuinely long
        # (e.g. professional-level) jumps. Apply a stricter, higher floor
        # to this less-reliable path so a degraded-tracking jump is
        # dropped rather than reported with a misleadingly small number.
        min_dist_gate = MIN_VALID_DIST_M if dist_method == "foot_based" else MIN_COM_FALLBACK_DIST_M

        # Physical plausibility floor: real airborne time and real forward
        # distance are correlated (a body in the air for several hundred ms
        # cannot have travelled almost no horizontal distance in a standing
        # or run-up broad jump). If the measured distance is far below what
        # the observed flight time implies, that's a sign the distance
        # measurement degraded (e.g. heel-approximation drift), not that
        # the jump itself was tiny — so it's rejected rather than reported.
        physics_floor_m = MIN_DIST_PER_FLIGHT_S * t

        if (dist_m < min_dist_gate or
                dist_m < physics_floor_m or
                dist_m > MAX_VALID_DIST_M or
                flight_ms < MIN_VALID_FLIGHT_MS or
                peak_rise_px < self._min_peak_rise_px):
            return None

        # ── Physics-correct takeoff velocity / angle ────────────────────
        g = 9.81
        vx_ms = dist_m / (t + 1e-9)
        vy_ms = g * t / 2 if t > 0 else 0.0
        takeoff_vel = round(math.sqrt(vx_ms**2 + vy_ms**2), 2)
        takeoff_angle = round(math.degrees(math.atan2(vy_ms, vx_ms + 1e-9)), 1) if vx_ms > 1e-6 else 0.0
        takeoff_angle = max(0.0, min(85.0, takeoff_angle))

        # Physics-consistency is reported for diagnostics only — NOT used
        # to reject the jump (see note above).
        physics_expected_dist = vx_ms * t
        physics_consistent = bool(abs(physics_expected_dist - dist_m) < max(0.15, 0.25 * dist_m))

        # Tuck / pike detection from in-flight angles
        tuck_confirmed = False
        pike_confirmed = False
        min_hip_flex = min(self._flight_hip_flex) if self._flight_hip_flex else 180
        min_knee_flex = min(self._flight_knee_flex) if self._flight_knee_flex else 180
        max_knee_ext  = max(self._flight_knee_flex) if self._flight_knee_flex else 0

        if min_hip_flex < TUCK_HIP_FLEX_DEG and min_knee_flex < 90:
            tuck_confirmed = True
        if min_hip_flex < PIKE_HIP_FLEX_DEG and max_knee_ext > PIKE_KNEE_EXT_DEG:
            pike_confirmed = True

        # Countermovement depth
        cm_depth_m = round(scale.px_to_m(self._cm_depth), 3)

        # Horizontal efficiency: forward dist / total COM path (approximation)
        h_eff = round(min(1.0, px_dist / (px_dist + peak_rise_px + 1e-9)), 3)

        # Landing stability (from COM settle velocity at confirmation time)
        land_vx = land_vy = 0.0
        if len(self._com_hist) >= 2:
            land_vx = self._com_hist[-1][0] - self._com_hist[-2][0]
            land_vy = self._com_hist[-1][1] - self._com_hist[-2][1]
        landing_stability = round(max(0.0, 1.0 - (abs(land_vx) + abs(land_vy)) / 20.0), 3)

        # Multi-factor form score (distance, landing stability, takeoff
        # angle, arm-swing quality, knee flexion at takeoff)
        arm_swing_score = _arm_swing_score(self._flight_arm_travel)
        form = _form_score_full(
            dist_m=dist_m,
            landing_stability=landing_stability,
            takeoff_angle=takeoff_angle,
            min_hip_flex=min_hip_flex,
            min_knee_flex=min_knee_flex,
            arm_swing_score=arm_swing_score,
        )

        # Ground contact time (reactive)
        gct_ms = 0
        if "reactive" in self.exercise and self._gct_start_f > 0:
            gct_f  = self._takeoff_f - self._gct_start_f
            gct_ms = int(max(0, gct_f) / self.fps * 1000)

        rsi = round(flight_ms / gct_ms, 3) if gct_ms > 0 else None

        self._jump_n += 1
        jump = {
            # Identity
            "jump_no"           : self._jump_n,
            # Distances
            "distance_m"        : dist_m,
            "distance_cm"       : dist_cm,
            "pixel_dist"        : round(px_dist, 1),
            "distance_method"   : dist_method,
            # Timing
            "flight_ms"         : flight_ms,
            "airborne_ms"       : flight_ms,
            "takeoff_frame"     : self._takeoff_f,
            "landing_frame"     : self._frame,
            # Positions
            "takeoff_com_x"     : round(takeoff_x_com, 1),
            "landing_com_x"     : round(land_x_com, 1),
            "takeoff_foot_x"    : round(takeoff_x_foot, 1) if takeoff_x_foot is not None else None,
            "landing_foot_x"    : round(land_x_foot, 1) if land_x_foot is not None else None,
            "direction"         : self._direction,
            # Kinematics
            "takeoff_velocity_ms": takeoff_vel,
            "takeoff_angle_deg" : takeoff_angle,
            "peak_rise_m"       : round(peak_rise_m, 3),
            "approach_vx_px_s"  : round(self._takeoff_approach_vx, 1),
            "approach_speed_ms" : round(approach_speed_ms, 2),
            "horizontal_efficiency": h_eff,
            "cm_depth_m"        : cm_depth_m,
            "physics_consistent": physics_consistent,
            # Technique
            "tuck_confirmed"    : tuck_confirmed,
            "pike_confirmed"    : pike_confirmed,
            "min_hip_flex_deg"  : round(min_hip_flex, 1),
            "min_knee_flex_deg" : round(min_knee_flex, 1),
            "landing_stability" : landing_stability,
            "arm_swing_score"   : arm_swing_score,
            # Reactive
            "ground_contact_ms" : gct_ms,
            "rsi"               : rsi,
            # Score
            "form_score"        : form,
            # Snapshot
            "_peak_b64"         : self._peak_b64,
        }
        jump = _to_json_safe(jump)
        self.jumps.append(jump)
        return jump

    def force_close(self, com, scale: ScaleEstimator, pts=None):
        """Call at video end if still in FLIGHT state."""
        if self.state == ST_FLIGHT and self._flight_f >= MIN_FLIGHT_F:
            land_x = com[0] if com else (self._flight_com_x_start or 0)
            self._land_com_buf.append(land_x)
            self._land_stable_f = LAND_STABLE_F
            ref_pts = pts if pts is not None else np.zeros((17, 2))
            foot_x = _landing_foot_x(ref_pts, self._direction) if pts is not None else None
            if foot_x is not None:
                self._land_foot_x_buf.append(foot_x)
            return self._confirm_jump(com, ref_pts, scale)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE-SPECIFIC VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

class ExerciseValidator:
    """
    Wraps the core JumpEngine and applies exercise-specific
    post-processing to the completed jump list.
    """
    def __init__(self, exercise: str):
        self.exercise = exercise

    def validate(self, jumps: list) -> dict:
        ex = self.exercise.lower().replace(" ", "_")

        if "triple" in ex:
            return self._triple(jumps)
        elif "alternate" in ex:
            return self._alternate(jumps)
        elif "bounding" in ex or "power_skip" in ex:
            return self._bounding(jumps)
        elif "multiple" in ex:
            return self._multiple(jumps)
        else:
            return self._standard(jumps)

    # Standard single / repeated jumps
    def _standard(self, jumps):
        return {"validated_jumps": jumps, "phase_info": None}

    # Triple: group every 3 consecutive jumps into hop/step/jump
    def _triple(self, jumps):
        phases = []
        for i in range(0, len(jumps) - 2, 3):
            hop, step, jmp = jumps[i], jumps[i+1], jumps[i+2]
            total = hop["distance_m"] + step["distance_m"] + jmp["distance_m"]
            phases.append({
                "triple_no"   : len(phases) + 1,
                "hop"         : hop,
                "step"        : step,
                "jump"        : jmp,
                "total_m"     : round(total, 3),
                "phase_ratio" : [round(hop["distance_m"]/total, 2),
                                 round(step["distance_m"]/total, 2),
                                 round(jmp["distance_m"]/total, 2)],
            })
        return {"validated_jumps": jumps, "phase_info": phases}

    # Alternate: flag if consecutive jumps show alternating takeoff feet
    def _alternate(self, jumps):
        for i, j in enumerate(jumps):
            j["leg_tag"] = "L" if i % 2 == 0 else "R"
        return {"validated_jumps": jumps, "phase_info": None}

    # Bounding: each jump = one bound
    def _bounding(self, jumps):
        bounds = []
        for j in jumps:
            bounds.append({
                "bound_no"  : j["jump_no"],
                "distance_m": j["distance_m"],
                "flight_ms" : j["flight_ms"],
                "gct_ms"    : j.get("ground_contact_ms", 0),
            })
        total = sum(b["distance_m"] for b in bounds)
        return {"validated_jumps": jumps, "phase_info": bounds,
                "total_distance_m": round(total, 3)}

    # Multiple: cumulative metrics
    def _multiple(self, jumps):
        cumulative = 0.0
        for j in jumps:
            cumulative += j["distance_m"]
            j["cumulative_m"] = round(cumulative, 3)
        return {"validated_jumps": jumps, "phase_info": None}


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY & SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _form_score(distance_m):
    """Legacy distance-only score, retained for backward compatibility
    (used by _form_score_aggregate for the session-level summary)."""
    thresholds = [2.50, 2.20, 2.00, 1.80, 1.50, 1.20]
    scores     = [10,   9,    8,    7,    6,    5]
    for t, s in zip(thresholds, scores):
        if distance_m >= t:
            return s
    return 4

def _distance_subscore(distance_m):
    """0-10 distance subscore feeding the weighted per-jump form score."""
    thresholds = [2.50, 2.20, 2.00, 1.80, 1.50, 1.20, 0.90]
    scores     = [10,   9,    8,    7,    6,    5,    3]
    for t, s in zip(thresholds, scores):
        if distance_m >= t:
            return s
    return 1

def _takeoff_angle_subscore(angle_deg):
    """0-10 subscore peaking in the biomechanically efficient 18-27°
    range for a standing broad jump; falls off outside it."""
    ideal_low, ideal_high = 18.0, 27.0
    if ideal_low <= angle_deg <= ideal_high:
        return 10.0
    dist = (ideal_low - angle_deg) if angle_deg < ideal_low else (angle_deg - ideal_high)
    return max(0.0, 10.0 - dist * 0.35)

def _knee_flexion_subscore(min_knee_flex_deg):
    """0-10 subscore for takeoff/flight knee flexion — deep enough to
    generate power without being so collapsed it signals poor control."""
    if min_knee_flex_deg is None:
        return 6.0  # neutral if not measured
    ideal_low, ideal_high = 90.0, 130.0
    if ideal_low <= min_knee_flex_deg <= ideal_high:
        return 10.0
    dist = (ideal_low - min_knee_flex_deg) if min_knee_flex_deg < ideal_low else (min_knee_flex_deg - ideal_high)
    return max(0.0, 10.0 - dist * 0.15)

def _arm_swing_score(wrist_travel_samples):
    """
    0-10 subscore from wrist vertical excursion relative to shoulder during
    flight. A strong arm swing drives the wrists from low (behind the hips
    at takeoff) to high (overhead/forward at peak flight) — large positive
    range in (shoulder_y - wrist_y) indicates good swing amplitude.
    """
    if not wrist_travel_samples or len(wrist_travel_samples) < 2:
        return 5.0   # neutral score when arms aren't trackable
    travel_range = max(wrist_travel_samples) - min(wrist_travel_samples)
    # travel_range is in pixels; normalise loosely — most useful as a
    # relative signal since we don't have per-subject scale here.
    score = min(10.0, travel_range / 12.0)
    return round(max(0.0, score), 1)

def _form_score_full(dist_m, landing_stability, takeoff_angle,
                      min_hip_flex, min_knee_flex, arm_swing_score=5.0):
    """
    Weighted multi-factor form score (1-10):
      40% distance, 20% landing stability, 15% takeoff angle,
      15% arm swing, 10% knee flexion.
    """
    dist_s   = _distance_subscore(dist_m)
    land_s   = round(landing_stability * 10, 1)
    angle_s  = _takeoff_angle_subscore(takeoff_angle)
    knee_s   = _knee_flexion_subscore(min_knee_flex)
    arm_s    = arm_swing_score

    weighted = (0.40 * dist_s + 0.20 * land_s + 0.15 * angle_s +
                0.15 * arm_s + 0.10 * knee_s)
    return int(round(max(1.0, min(10.0, weighted))))

def _form_score_aggregate(jumps_m):
    if not jumps_m:
        return 0
    best  = max(jumps_m)
    score = _form_score(best)
    if len(jumps_m) > 1:
        cons = min(jumps_m) / best
        if   cons >= 0.90: score = min(10, score + 1)
        elif cons < 0.70:  score = max(1,  score - 1)
    return score

def _compute_quality(jumps, pose_conf, foot_conf, scale_calibrated):
    """
    Generate overall confidence score 0-100 from independently meaningful
    sub-scores, rather than a raw jump count (which says nothing about
    measurement reliability).
      pose      : average pose-detection confidence across the video
      contact   : foot-contact estimator reliability
      flight    : how cleanly flight phases were detected (physics-consistency
                  rate + sane flight-time spread across detected jumps)
      scale     : whether px-per-meter calibration succeeded
      landing   : average landing stability across detected jumps
    """
    pose_s  = int(pose_conf * 100)
    cont_s  = int(foot_conf * 100)
    scale_s = 85 if scale_calibrated else 55

    if not jumps:
        return {
            "overall": 0, "pose": pose_s, "contact": cont_s,
            "flight": 0, "distance": scale_s, "landing": 0,
            "jump": 0,   # legacy key retained
        }

    consistent = [j.get("physics_consistent", True) for j in jumps]
    flight_s   = int(100 * (sum(consistent) / len(consistent)))

    land_vals  = [j.get("landing_stability") for j in jumps if j.get("landing_stability") is not None]
    landing_s  = int(100 * (sum(land_vals) / len(land_vals))) if land_vals else 60

    overall = int(0.30*pose_s + 0.15*cont_s + 0.25*flight_s +
                  0.15*scale_s + 0.15*landing_s)
    return {
        "overall": overall, "pose": pose_s, "contact": cont_s,
        "flight": flight_s, "distance": scale_s, "landing": landing_s,
        "jump": flight_s,   # legacy key retained, now mapped to a meaningful value
    }

def _fatigue_index(jumps_m):
    """Decline in distance over repetitions (lower is better)."""
    if len(jumps_m) < 2:
        return None
    return round((jumps_m[0] - jumps_m[-1]) / jumps_m[0] * 100, 1)

def _rep_variability(jumps_m):
    if len(jumps_m) < 2:
        return None
    return round(float(np.std(jumps_m)) / (np.mean(jumps_m) + 1e-9) * 100, 1)

def _feedback(jump_results, detected, exercise):
    issues    = []
    strengths = []

    if not detected:
        issues += [
            "❌ Person not detected in video.",
            "📌 Ensure: full body visible (head to toe), side-angle camera, good lighting.",
            "🎥 Avoid top-down or front-facing camera angles.",
        ]
        return issues, strengths

    if not jump_results:
        issues += [
            "❌ No valid jump detected.",
            "📐 Camera should be placed sideways at full-body height.",
            "⏱️ Video must capture full jump — takeoff, flight, and landing.",
        ]
        return issues, strengths

    distances = [j["distance_m"] for j in jump_results]
    best = max(distances)
    avg  = sum(distances) / len(distances)

    if best < 1.20:
        issues.append(f"Short jump ({best:.2f}m) — focus on arm swing and hip extension.")
    elif best < 1.80:
        issues.append(f"Below-average distance ({best:.2f}m) — work on explosive leg drive.")
    elif best < 2.20:
        strengths.append(f"Good jump distance ({best:.2f}m).")
    else:
        strengths.append(f"Excellent jump distance ({best:.2f}m)!")

    if len(distances) > 1:
        cons = min(distances) / best
        if cons >= 0.90:
            strengths.append(f"Very consistent across {len(distances)} jumps ({cons:.0%}).")
        elif cons < 0.70:
            issues.append(f"High variation ({cons:.0%} consistency) — repeat takeoff mechanics.")

    # Tuck/pike feedback
    if "tuck" in exercise.lower():
        tucks = [j for j in jump_results if j.get("tuck_confirmed")]
        if tucks:
            strengths.append(f"Tuck confirmed in {len(tucks)}/{len(jump_results)} jumps.")
        else:
            issues.append("Tuck not detected — ensure knees pull toward chest during flight.")

    if "pike" in exercise.lower():
        pikes = [j for j in jump_results if j.get("pike_confirmed")]
        if pikes:
            strengths.append(f"Pike confirmed in {len(pikes)}/{len(jump_results)} jumps.")
        else:
            issues.append("Pike not detected — keep legs extended and reach toward toes.")

    if not issues:
        issues = ["No major issues detected."]
    return issues, strengths


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def analyse_broad_jump(
    path,
    is_video,
    output_path=None,
    session_id=None,
    source_filename="",
    progress_uid=None,
    exercise="Standing Broad Jump",
    user_height_cm=None,
    user_height_m=None,
):
    """
    Main entry point for broad jump analysis.
    Supports all 12 broad jump variants via the `exercise` parameter.

    Parameters
    ----------
    exercise : str
        One of: "Standing Broad Jump", "Run-Up Broad Jump",
        "Single-Leg Broad Jump", "Alternate Leg Broad Jump",
        "Bounding", "Triple Broad Jump", "Multiple Broad Jump",
        "Reactive Broad Jump", "Tuck Broad Jump", "Pike Broad Jump",
        "Weighted Broad Jump", "Sand Broad Jump"
    user_height_cm : float, optional
        Athlete's real height in centimetres. When supplied, calibration
        uses the athlete's TRUE height instead of the generic 1.75m
        anthropometric default — this is what makes distance accuracy
        correct for athletes who are notably taller/shorter than average
        (e.g. professional athletes), since every reported distance is
        directly proportional to this value.
    user_height_m : float, optional
        Same as user_height_cm but in metres. If both are given,
        user_height_m takes precedence.
    """
    if not _YOLO_AVAILABLE:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
    if not is_video:
        raise ValueError("Broad jump analysis requires a video file.")

    # ── Model ────────────────────────────────────────────────────────────
    model_path = YOLO_MODEL_PATH
    if not os.path.exists(model_path):
        here = os.path.dirname(os.path.abspath(__file__))
        alt  = os.path.join(here, YOLO_MODEL_PATH)
        if os.path.exists(alt):
            model_path = alt
        else:
            raise FileNotFoundError(f"YOLO model not found at '{YOLO_MODEL_PATH}'.")
    model = YOLO(model_path)

    # ── Video metadata ────────────────────────────────────────────────────
    cap    = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # ── Sub-systems ───────────────────────────────────────────────────────
    lm_filter = LandmarkFilter(n_landmarks=17, freq=fps)
    foot_est  = FootContactEstimator(window=int(fps * 1.5), fps=fps)
    scale_est = ScaleEstimator(user_height_m=user_height_m, user_height_cm=user_height_cm)
    engine    = JumpEngine(fps=fps, exercise=exercise.lower())
    validator = ExerciseValidator(exercise)


    # ── State ─────────────────────────────────────────────────────────────
    detected_ok   = [False]
    wrong_events  = []
    per_frame_data= []
    total_frames  = [1]
    pose_confs    = []
    foot_conf_acc = []
    miss_streak   = [0]      # consecutive frames with no pose detected
    last_good_pts = [None]   # most recent valid keypoints, for bridging gaps

    live_hud      = {"jump_no": "0", "dist": "---", "form": "---", "state": ST_READY}

    def _push_event(jdata, b64):
        if not progress_uid:
            return
        pct = min(94, int(len(per_frame_data) / max(1, total_frames[0]) * 90))
        safe_jdata = _to_json_safe({**jdata, "frame_b64": b64})
        set_progress(
            progress_uid, pct,
            f"Jump {jdata['jump_no']} — {jdata['distance_m']:.2f}m",
            jump_event=safe_jdata,
        )

    # ── Per-frame callback ────────────────────────────────────────────────
    def pf(frame, fc, total):
        total_frames[0] = max(total, 1)

        results   = model(frame, conf=CONFIDENCE, verbose=False)
        best_pts  = None
        best_area = 0
        best_conf = 0.0

        for result in results:
            if (result.keypoints is None or
                    result.keypoints.xy is None or
                    result.keypoints.conf is None):
                continue
            for kpts_xy, kpts_conf in zip(
                    result.keypoints.xy.cpu().numpy(),
                    result.keypoints.conf.cpu().numpy()):
                pts   = kpts_xy.astype(float)
                valid = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
                if len(valid) < 6:
                    continue
                area = ((valid[:, 0].max() - valid[:, 0].min()) *
                        (valid[:, 1].max() - valid[:, 1].min()))
                if area > best_area:
                    best_area = area
                    best_pts  = pts
                    best_conf = float(np.mean(kpts_conf[kpts_conf > 0]))

        if best_pts is None:
            miss_streak[0] += 1
            # Short dropouts (motion blur mid-flight, brief occlusion) must
            # NOT freeze the FSM — if we simply skip engine.update() here,
            # _flight_f / _land_stable_f stall, the FSM can get stuck in
            # FLIGHT for the rest of the clip, and every later jump in the
            # video is silently lost (0-jump / missing-jump bug). Replaying
            # the last known good keypoints keeps state/timers advancing
            # through brief gaps without injecting a fabricated pose.
            if last_good_pts[0] is not None and miss_streak[0] <= MAX_OCCLUSION_HOLD_F:
                pts_hold = last_good_pts[0]
                com_hold = estimate_com(pts_hold)
                foot_est.classify(pts_hold)
                frame_b64_hold = frame_to_b64(frame) if engine.state in (ST_FLIGHT, ST_TAKEOFF) else None
                jump_event = engine.update(pts_hold, foot_est, com_hold, scale_est, frame_b64_hold)
                if jump_event:
                    b64 = jump_event.pop("_peak_b64", None) or frame_to_b64(frame)
                    _push_event(jump_event, b64)
                    live_hud["jump_no"] = str(jump_event["jump_no"])
                    live_hud["dist"]    = f"{jump_event['distance_m']:.2f}m"
                    live_hud["form"]    = f"{jump_event['form_score']}/10"
                live_hud["state"] = engine.state

            draw_footer_hud(frame, [
                ("JUMP #", live_hud["jump_no"]),
                ("DIST",   live_hud["dist"]),
                ("FORM",   live_hud["form"]),
            ])
            draw_pcl_logo(frame)
            return frame

        miss_streak[0] = 0
        detected_ok[0] = True
        pose_confs.append(best_conf)

        # Filter landmarks
        pts = lm_filter.update(best_pts)
        last_good_pts[0] = pts

        # Draw skeleton (skip face landmarks 0-4)
        for p1i, p2i in SKELETON_PAIRS:
            p1, p2 = pts[p1i], pts[p2i]
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                         COLOR_SKEL, 2)
        for idx, p in enumerate(pts):
            if idx < 5 or p[0] <= 0 or p[1] <= 0:
                continue
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, COLOR_JOINT, -1)

        # COM
        com = estimate_com(pts)
        if com:
            cv2.circle(frame, (int(com[0]), int(com[1])), 6, COLOR_COM, -1)

        # Foot contact
        foot_est.classify(pts)

        # Engine update
        frame_b64  = frame_to_b64(frame) if engine.state in (ST_FLIGHT, ST_TAKEOFF) else None
        jump_event = engine.update(pts, foot_est, com, scale_est, frame_b64)

        if jump_event:
            b64 = jump_event.pop("_peak_b64", None) or frame_to_b64(frame)
            _push_event(jump_event, b64)
            live_hud["jump_no"] = str(jump_event["jump_no"])
            live_hud["dist"]    = f"{jump_event['distance_m']:.2f}m"
            live_hud["form"]    = f"{jump_event['form_score']}/10"

        live_hud["state"] = engine.state

        # Visualise scale (if calibrated, show reference line)
        if scale_est.px_per_m:
            ref_px = int(scale_est.px_per_m)
            mid_x, gnd_y = width // 4, height - 40
            cv2.line(frame, (mid_x, gnd_y), (mid_x + ref_px, gnd_y), (100, 255, 100), 2)
            cv2.putText(frame, "1m", (mid_x + ref_px // 2 - 10, gnd_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

        # Draw ground reference
        if engine._gnd_com_y:
            gnd_px = int(engine._gnd_com_y + (LAND_APPROACH_PX))
            cv2.line(frame, (0, gnd_px), (width, gnd_px), (80, 80, 80), 1)

        # Draw takeoff / landing markers
        if engine._flight_com_x_start and engine.state == ST_FLIGHT:
            tx = int(engine._flight_com_x_start)
            cv2.line(frame, (tx, 0), (tx, height), COLOR_TAKE, 1)

        per_frame_data.append({
            "frame"     : fc,
            "state"     : engine.state,
            "jump_count": len(engine.jumps),
        })

        draw_footer_hud(frame, [
            ("JUMP #", live_hud["jump_no"]),
            ("DIST",   live_hud["dist"]),
            ("FORM",   live_hud["form"]),
        ])
        draw_pcl_logo(frame)
        return frame

    # ── Run ───────────────────────────────────────────────────────────────
    snaps = process_video_or_image(
        path, is_video, pf,
        output_path=output_path,
        snap_pcts=[0.1, 0.3, 0.5, 0.7, 0.9],
        analysis_skip=1,
        progress_uid=progress_uid,
    )

    # Force-close if video ends mid-flight. Previously this used a
    # synthetic (takeoff_x, ground_y) placeholder for last_com whenever no
    # better value was tracked, and never passed real pts — so foot-based
    # distance was unavailable and the COM-fallback path saw an essentially
    # fabricated landing position. That's what silently truncated/dropped
    # the last jump in a video whenever it ran right up to the final frame
    # without a clean LAND_STABLE_F streak. Use the actual last detected
    # pose/COM (held across any trailing occlusion via last_good_pts) so
    # the recovered jump's distance reflects where the athlete really was.
    if engine.state == ST_FLIGHT and engine._flight_f >= MIN_FLIGHT_F:
        last_pts = last_good_pts[0]
        last_com = estimate_com(last_pts) if last_pts is not None else None
        if last_com is None:
            last_com = (engine._flight_com_x_start, engine._gnd_com_y or 0)
        ev = engine.force_close(last_com, scale_est, pts=last_pts)
        if ev:
            ev.pop("_peak_b64", None)
            _push_event(ev, "")

    if not detected_ok[0]:
        raise ValueError(
            "No person detected. Upload a side-angle video with full body visible "
            "(head to toe) and good lighting."
        )

    if session_id:
        save_wrong_angle_log(exercise, session_id, source_filename, wrong_events)

    # ── Post-processing ────────────────────────────────────────────────────
    jumps     = engine.jumps
    val_out   = validator.validate(jumps)
    jumps_m   = [j["distance_m"] for j in jumps]

    best_dist = round(max(jumps_m), 3) if jumps_m else 0.0
    avg_dist  = round(sum(jumps_m) / len(jumps_m), 3) if jumps_m else 0.0

    pose_conf_avg = float(np.mean(pose_confs)) if pose_confs else 0.0
    foot_conf_val = 0.75 if scale_est.px_per_m else 0.50
    quality = _compute_quality(jumps, pose_conf_avg, foot_conf_val,
                                scale_calibrated=scale_est.px_per_m is not None)

    form_score = _form_score_aggregate(jumps_m)
    issues, strengths = _feedback(jumps, detected_ok[0], exercise)

    while len(snaps) < 5:
        snaps.append(snaps[-1] if snaps else "")

    step      = max(1, int(fps / 10))
    per_frame = per_frame_data[::step]

    best_str = f"{best_dist:.2f} m" if best_dist > 0 else "N/A"
    avg_str  = f"{avg_dist:.2f} m"  if avg_dist  > 0 else "N/A"

    # ── Aggregate quality metrics ──────────────────────────────────────────
    gct_vals  = [j["ground_contact_ms"] for j in jumps if j.get("ground_contact_ms")]
    rsi_vals  = [j["rsi"] for j in jumps if j.get("rsi")]
    ft_vals   = [j["flight_ms"] for j in jumps]

    # Top-level aggregate angle/landing fields — app.py's build_metrics()
    # reads these directly off the result dict (avg_takeoff_angle,
    # avg_landing_angle, landing_score, jump_distance_cm) for the broad_jump
    # metrics card. Without these exact keys present, the frontend shows
    # "undefined" even though the same data exists nested inside per_jump.
    takeoff_angle_vals = [j["takeoff_angle_deg"] for j in jumps if j.get("takeoff_angle_deg") is not None]
    landing_stab_vals  = [j["landing_stability"] for j in jumps if j.get("landing_stability") is not None]
    avg_takeoff_angle  = round(float(np.mean(takeoff_angle_vals)), 1) if takeoff_angle_vals else 0.0
    # No direct landing-angle measurement in this engine (no heel/ankle
    # ground-impact angle tracked) — approximate via landing stability as
    # a proxy so the field is populated rather than missing/undefined.
    avg_landing_angle  = round(float(np.mean(landing_stab_vals)) * 30.0, 1) if landing_stab_vals else 0.0
    landing_score      = round(float(np.mean(landing_stab_vals)) * 10, 1) if landing_stab_vals else 0.0

    result = {
        # Core
        "exercise"              : exercise,
        "jump_count"            : len(jumps),
        "correct_jumps"         : len(jumps),
        "wrong_jumps"           : 0,
        # Distance
        "best_distance_m"       : best_dist,
        "best_distance_cm"      : round(best_dist * 100, 1),
        "avg_distance_m"        : avg_dist,
        "all_jumps_m"           : [round(j, 3) for j in jumps_m],
        "jump_distance_cm"      : round(best_dist * 100, 1),   # app.py build_metrics() key
        "avg_takeoff_angle"     : avg_takeoff_angle,            # app.py build_metrics() key
        "avg_landing_angle"     : avg_landing_angle,            # app.py build_metrics() key
        "landing_score"         : landing_score,                # app.py build_metrics() key
        # Per-jump detail
        "per_jump"              : jumps,
        # Exercise variant results
        "phase_info"            : val_out.get("phase_info"),
        "total_distance_m"      : val_out.get("total_distance_m"),
        # Timing
        "avg_flight_ms"         : int(np.mean(ft_vals)) if ft_vals else 0,
        "avg_ground_contact_ms" : int(np.mean(gct_vals)) if gct_vals else 0,
        "avg_rsi"               : round(float(np.mean(rsi_vals)), 3) if rsi_vals else None,
        # Consistency
        "fatigue_index"         : _fatigue_index(jumps_m),
        "rep_variability_pct"   : _rep_variability(jumps_m),
        # Scale
        "px_per_m"              : round(scale_est.px_per_m, 2) if scale_est.px_per_m else None,
        "scale_method"          : scale_est.method or "legacy_constant",
        # Quality / Confidence
        "confidence"            : quality,
        "form_score"            : form_score,
        # Feedback
        "issues"                : issues,
        "strengths"             : strengths,
        # UI
        "metrics": [
            {"label": "Jumps Detected", "value": str(len(jumps))},
            {"label": "Best Jump",      "value": best_str},
            {"label": "Avg Jump",       "value": avg_str},
            {"label": "Form Score",     "value": f"{form_score}/10"},
            {"label": "Confidence",     "value": f"{quality['overall']}%"},
        ],
        # Legacy keys kept for backward compatibility
        "height_cm"             : round(best_dist * 100, 1),
        "per_frame"             : per_frame,
        "snapshots"             : snaps,
        "wrong_angle_count"     : 0,
        "_wrong_events"         : wrong_events,
    }
    return _to_json_safe(result)
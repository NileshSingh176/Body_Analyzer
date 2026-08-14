import cv2
import numpy as np
import json
from utils import (
    mp_pose, get_landmark, calculate_angle, draw_angle_arc,
    process_video_or_image, save_wrong_angle_log, draw_pose_skyblue,
    set_progress, frame_to_b64, LandmarkSmoother,
)
from hud_overlay import draw_footer_hud, draw_pcl_logo, expand_canvas_for_lhs, draw_lhs_panel

# ════════════════════════════════════════════════════════════════
# module_vertical_jump.py — CMJ v2
#
# Countermovement Vertical Jump (CMJ) with Arm Swing
# Full 5-phase biomechanical analysis:
#   1. STANDING    — initial posture & alignment
#   2. CROUCH      — countermovement depth & arm loading
#   3. TAKEOFF     — triple extension & arm swing sync
#   4. FLIGHT      — airborne height & body position
#   5. LANDING     — shock absorption, knee valgus, symmetry
#
# Per-jump classification:  GOOD / AVERAGE / POOR
# Injury risk detection:    ACL / Patellar / Ankle / Hip
# Scores:  Form /10 · Landing /10 · Explosiveness /10 · InjuryRisk
#
# v2 fixes:
#   • Adaptive AIR_THRESH scaled to athlete's normalised body span
#     (fixes underestimation when athlete is far from camera)
#   • Still-only baseline: never updates during crouch or motion
#     (fixes baseline drift that shrinks apparent elevation)
#   • Velocity gate at takeoff: requires upward hip velocity to
#     confirm a real jump vs. body-sway or deep crouch
#   • Debounced landing counter: 3 consecutive frames below
#     threshold before ending flight (prevents early cutoff)
#   • FPS-scaled frame-clipping correction instead of fixed +2
#   • Peak-elevation displacement blend as secondary cross-check
# ════════════════════════════════════════════════════════════════

# ── Biomechanical thresholds ──────────────────────────────────
CROUCH_IDEAL_MIN   = 85
CROUCH_IDEAL_MAX   = 120
CROUCH_TOO_SHALLOW = 140
CROUCH_TOO_DEEP    = 70

TRIPLE_EXT_MIN     = 160
TRIPLE_EXT_GOOD    = 170

LAND_IDEAL_MIN     = 110
LAND_IDEAL_MAX     = 140
LAND_STIFF         = 150
LAND_COLLAPSE      = 80

VALGUS_THRESHOLD   = 10
TRUNK_LEAN_MAX     = 30
HIP_DROP_MAX       = 8

HEIGHT_POOR        = 25
HEIGHT_AVERAGE     = 40
HEIGHT_GOOD        = 55

PHASE_STANDING  = "STANDING"
PHASE_CROUCH    = "CROUCH"
PHASE_TAKEOFF   = "TAKEOFF"
PHASE_FLIGHT    = "FLIGHT"
PHASE_LANDING   = "LANDING"

# ── Airborne detection constants ──────────────────────────────
AIR_THRESH_NORM    = 0.030   # normalised hip units (fallback) — relaxed from 0.045
VEL_TAKEOFF_MIN    = 0.0015  # upward hip velocity to confirm takeoff — relaxed from 0.004
MIN_AIR_FRAMES     = 3       # must be airborne >= 3 frames (was 4)
LANDING_HOLD       = 3       # consecutive frames below threshold to confirm landing
STILL_VEL_THRESH   = 0.004   # max hip velocity for "standing still" — relaxed from 0.002


# ════════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════════

def _safe_angle(a, b, c):
    if a is None or b is None or c is None:
        return None
    return calculate_angle(a, b, c)


def _lm_xy(lm, idx, min_vis=0.35):
    p = lm[idx]
    if p.visibility < min_vis:
        return None
    return [p.x, p.y]


def _bilateral_knee(lm):
    lh = _lm_xy(lm, 23); lk = _lm_xy(lm, 25); la = _lm_xy(lm, 27)
    rh = _lm_xy(lm, 24); rk = _lm_xy(lm, 26); ra = _lm_xy(lm, 28)
    l_k = _safe_angle(lh, lk, la)
    r_k = _safe_angle(rh, rk, ra)
    if l_k is not None and r_k is not None:
        return l_k, r_k, (l_k + r_k) / 2
    elif l_k is not None:
        return l_k, l_k, l_k
    elif r_k is not None:
        return r_k, r_k, r_k
    return None, None, None


def _bilateral_hip(lm):
    ls = _lm_xy(lm, 11); lh = _lm_xy(lm, 23); lk = _lm_xy(lm, 25)
    rs = _lm_xy(lm, 12); rh = _lm_xy(lm, 24); rk = _lm_xy(lm, 26)
    l_h = _safe_angle(ls, lh, lk)
    r_h = _safe_angle(rs, rh, rk)
    if l_h and r_h:
        return (l_h + r_h) / 2
    return l_h or r_h or 170.0


def _ankle_angle(lm, side="left"):
    if side == "left":
        k = _lm_xy(lm, 25); a = _lm_xy(lm, 27); f = _lm_xy(lm, 31)
    else:
        k = _lm_xy(lm, 26); a = _lm_xy(lm, 28); f = _lm_xy(lm, 32)
    return _safe_angle(k, a, f)


def _elbow_angle(lm, side="left"):
    if side == "left":
        s = _lm_xy(lm, 11); e = _lm_xy(lm, 13); w = _lm_xy(lm, 15)
    else:
        s = _lm_xy(lm, 12); e = _lm_xy(lm, 14); w = _lm_xy(lm, 16)
    return _safe_angle(s, e, w)


def _trunk_lean(lm):
    ls = _lm_xy(lm, 11); rs = _lm_xy(lm, 12)
    lh = _lm_xy(lm, 23); rh = _lm_xy(lm, 24)
    if not (ls and rs and lh and rh):
        return 0.0
    sx = (ls[0] + rs[0]) / 2; sy = (ls[1] + rs[1]) / 2
    hx = (lh[0] + rh[0]) / 2; hy = (lh[1] + rh[1]) / 2
    dx = sx - hx; dy = hy - sy
    if dy == 0:
        return 90.0
    return round(abs(np.degrees(np.arctan2(abs(dx), abs(dy)))), 1)


def _pelvic_drop(lm):
    lh = lm[23]; rh = lm[24]
    if lh.visibility < 0.3 or rh.visibility < 0.3:
        return 0.0
    hip_width = abs(lh.x - rh.x)
    if hip_width < 0.001:
        return 0.0
    return round(abs(lh.y - rh.y) / hip_width * 100, 1)


def _knee_valgus(lm, side="left"):
    if side == "left":
        h = _lm_xy(lm, 23); k = _lm_xy(lm, 25); a = _lm_xy(lm, 27)
    else:
        h = _lm_xy(lm, 24); k = _lm_xy(lm, 26); a = _lm_xy(lm, 28)
    if not (h and k and a):
        return 0.0
    ha = [a[0]-h[0], a[1]-h[1]]
    ha_len = np.hypot(*ha)
    if ha_len < 1e-6:
        return 0.0
    hk = [k[0]-h[0], k[1]-h[1]]
    cross = ha[0]*hk[1] - ha[1]*hk[0]
    dist  = abs(cross) / ha_len
    sign  = 1 if (side == "left" and cross > 0) or (side == "right" and cross < 0) else -1
    return round(float(np.degrees(np.arctan2(dist, ha_len)) * sign), 1)


def _arm_swing_range(wrist_y_history, shoulder_y_ref):
    if not wrist_y_history or shoulder_y_ref is None:
        return 0.0
    lo = max(wrist_y_history)
    hi = min(wrist_y_history)
    return round(abs(lo - hi), 3)


def _estimate_height_cm(lm, samples):
    MIN_VIS = 0.4
    def y(i): return lm[i].y
    def v(i): return lm[i].visibility

    hip_y = sh_y = ank_y = None
    if v(23) > MIN_VIS and v(24) > MIN_VIS: hip_y = (y(23)+y(24))/2
    elif v(23) > MIN_VIS: hip_y = y(23)
    elif v(24) > MIN_VIS: hip_y = y(24)
    if v(27) > MIN_VIS and v(28) > MIN_VIS: ank_y = (y(27)+y(28))/2
    elif v(27) > MIN_VIS: ank_y = y(27)
    elif v(28) > MIN_VIS: ank_y = y(28)
    if v(11) > MIN_VIS and v(12) > MIN_VIS: sh_y = (y(11)+y(12))/2
    elif v(11) > MIN_VIS: sh_y = y(11)
    elif v(12) > MIN_VIS: sh_y = y(12)

    est = None
    if hip_y is not None and ank_y is not None:
        span = abs(ank_y - hip_y)
        if span > 0.04: est = (span / 0.53) * 170.0
    if est is None and sh_y is not None and ank_y is not None:
        span = abs(ank_y - sh_y)
        if span > 0.07: est = (span / 0.85) * 170.0
    if est is None and sh_y is not None and hip_y is not None:
        span = abs(hip_y - sh_y)
        if span > 0.02: est = (span / 0.32) * 170.0
    if est is None and v(0) > MIN_VIS and ank_y is not None:
        span = abs(ank_y - y(0))
        if span > 0.08: est = (span / 0.92) * 170.0

    if est is not None:
        est = max(100.0, min(215.0, round(est, 0)))
        samples.append(est)
    if len(samples) >= 3:
        recent = samples[-30:]
        return float(sorted(recent)[len(recent)//2])
    return 0.0


# ════════════════════════════════════════════════════════════════
# CMJ Classifier
# ════════════════════════════════════════════════════════════════

def classify_cmj(jump_data):
    score     = 100
    issues    = []
    strengths = []
    injuries  = []
    remedies  = {}

    height_cm    = jump_data.get("height_cm", 0)
    crouch_angle = jump_data.get("crouch_min_knee", 130)
    takeoff_ext  = jump_data.get("takeoff_max_ext", 150)
    land_knee    = jump_data.get("landing_avg_knee", 120)
    valgus_l     = jump_data.get("valgus_left",  0)
    valgus_r     = jump_data.get("valgus_right", 0)
    max_valgus   = max(abs(valgus_l), abs(valgus_r))
    trunk_lean   = jump_data.get("crouch_trunk_lean", 0)
    arm_swing    = jump_data.get("arm_swing_range", 0)
    asym_knee    = jump_data.get("asymmetry_knee", 0)

    # 1. Jump Height (25 pts)
    if height_cm >= HEIGHT_GOOD:
        strengths.append(f"Excellent jump height ({height_cm:.1f} cm) — elite explosive power")
    elif height_cm >= HEIGHT_AVERAGE:
        strengths.append(f"Good jump height ({height_cm:.1f} cm)")
        score -= 5
    elif height_cm >= HEIGHT_POOR:
        issues.append(f"Average jump height ({height_cm:.1f} cm) — below athletic standard")
        score -= 15
        remedies["low_height"] = {
            "cause": "Insufficient elastic energy storage or weak triple extension",
            "exercises": ["Jump squats 3×10", "Box jumps", "Power cleans",
                          "Plyometric step-ups", "Band-resisted squats for hip drive"]
        }
    else:
        issues.append(f"Low jump height ({height_cm:.1f} cm) — significant power deficit")
        score -= 25
        remedies["low_height"] = {
            "cause": "Weak glutes/quads, poor stretch-shortening cycle, insufficient arm swing",
            "exercises": ["Goblet squats 4×8", "Jump squats 3×12", "Broad jumps",
                          "Depth drops", "Ankle mobility drills"]
        }

    # 2. Crouch Depth (15 pts)
    if crouch_angle is not None:
        if CROUCH_IDEAL_MIN <= crouch_angle <= CROUCH_IDEAL_MAX:
            strengths.append(f"Ideal crouch depth ({crouch_angle:.0f}°) — optimal elastic energy storage")
        elif crouch_angle > CROUCH_TOO_SHALLOW:
            issues.append(f"Insufficient countermovement ({crouch_angle:.0f}°) — barely bending knees. "
                          f"Aim for 90–120° to maximise stretch-shortening cycle.")
            score -= 15
            remedies["shallow_crouch"] = {
                "cause": "Stiff ankles or fear of depth reduce elastic energy loaded in tendons",
                "exercises": ["Ankle dorsiflexion stretches", "Goblet squat to depth",
                              "Pause squats (2s hold at bottom)", "Wall ankle stretches"]
            }
        elif CROUCH_IDEAL_MAX < crouch_angle <= CROUCH_TOO_SHALLOW:
            issues.append(f"Shallow countermovement ({crouch_angle:.0f}°) — try to reach 90–120°")
            score -= 8
            remedies["shallow_crouch"] = {
                "cause": "Partial elastic loading limits power output",
                "exercises": ["Pause squats", "Depth jump practice", "Goblet squats"]
            }
        elif crouch_angle < CROUCH_TOO_DEEP:
            issues.append(f"Excessive crouch depth ({crouch_angle:.0f}°) — too deep loses elastic energy")
            score -= 8
            remedies["deep_crouch"] = {
                "cause": "Over-crouching dissipates stored elastic energy before toe-off",
                "exercises": ["Box squat to controlled depth",
                              "Countermovement drills with cue 'quick dip'",
                              "Reactive jump drills"]
            }

    # 3. Triple Extension (20 pts)
    if takeoff_ext is not None:
        if takeoff_ext >= TRIPLE_EXT_GOOD:
            strengths.append(f"Full triple extension ({takeoff_ext:.0f}°) — maximum propulsive force")
        elif takeoff_ext >= TRIPLE_EXT_MIN:
            strengths.append(f"Good hip-knee-ankle extension ({takeoff_ext:.0f}°)")
            score -= 5
        elif takeoff_ext >= 140:
            issues.append(f"Incomplete triple extension ({takeoff_ext:.0f}°) — ankle plantarflexion weak.")
            score -= 12
            remedies["weak_extension"] = {
                "cause": "Weak ankle plantarflexors or poor neural drive for full extension",
                "exercises": ["Calf raises 4×15", "Single-leg calf jumps", "Ankle jump drills",
                              "Banded ankle plantarflexion", "Seated calf raises"]
            }
        else:
            issues.append(f"Weak takeoff extension ({takeoff_ext:.0f}°) — significant power lost.")
            score -= 20
            remedies["weak_extension"] = {
                "cause": "Weak hip extensors or inefficient neuromuscular activation sequence",
                "exercises": ["Hip thrusts 4×10", "Broad jumps", "Olympic lift derivatives",
                              "Explosive step-ups", "Banded hip extension"]
            }

    # 4. Landing Mechanics (20 pts)
    if land_knee is not None:
        if LAND_IDEAL_MIN <= land_knee <= LAND_IDEAL_MAX:
            strengths.append(f"Controlled soft landing ({land_knee:.0f}°) — good shock absorption")
        elif land_knee > LAND_STIFF:
            issues.append(f"Stiff/hard landing ({land_knee:.0f}°) — knees not bending enough on contact.")
            score -= 18
            injuries.append("Patellar Tendinopathy risk — high impact forces at landing")
            injuries.append("Tibial stress fracture risk — inadequate shock absorption")
            remedies["stiff_landing"] = {
                "cause": "Insufficient eccentric quadriceps strength or poor landing cue awareness",
                "exercises": ["Drop landings (focus on soft knees)", "Eccentric squats 3×8",
                              "Box landing drills", "Landing cue: 'Land quietly like a cat'",
                              "Single-leg landing balance drills"]
            }
        elif land_knee < LAND_COLLAPSE:
            issues.append(f"Knee collapse on landing ({land_knee:.0f}°) — excessive knee bend.")
            score -= 15
            injuries.append("Patellofemoral pain syndrome risk")
            remedies["landing_collapse"] = {
                "cause": "Weak quadriceps unable to control eccentric load at impact",
                "exercises": ["Eccentric single-leg squats", "Bulgarian split squats",
                              "Step-down exercises", "Wall sits progressive"]
            }
        elif land_knee < LAND_IDEAL_MIN:
            issues.append(f"Moderately hard landing ({land_knee:.0f}°) — increase knee flexion on contact")
            score -= 8

    # 5. Knee Valgus / ACL Risk (10 pts)
    if max_valgus > VALGUS_THRESHOLD:
        issues.append(f"Knee valgus detected (up to {max_valgus:.1f}°) — HIGH ACL injury risk.")
        score -= 10
        injuries.append("ACL tear risk — dynamic knee valgus under landing load")
        injuries.append("Medial meniscus stress")
        remedies["knee_valgus"] = {
            "cause": "Weak hip abductors / glute medius allowing femoral internal rotation",
            "exercises": ["Glute bridges 3×15", "Clamshells with band 3×20",
                          "Resistance band lateral walks",
                          "Single-leg squat with knee-over-toe cue",
                          "Bulgarian split squats focusing on knee tracking"]
        }
    elif max_valgus > VALGUS_THRESHOLD * 0.6:
        issues.append(f"Mild knee valgus tendency ({max_valgus:.1f}°) — monitor and strengthen glutes")
        score -= 4

    # 6. Trunk Lean
    if trunk_lean > TRUNK_LEAN_MAX:
        issues.append(f"Excessive forward trunk lean during crouch ({trunk_lean:.0f}°).")
        score -= 8
        remedies["trunk_lean"] = {
            "cause": "Tight hip flexors, weak core, or poor ankle mobility",
            "exercises": ["Hip flexor stretches", "Front squats", "Goblet squats",
                          "Thoracic mobility work", "Plank progressions"]
        }

    # 7. Bilateral Symmetry
    if asym_knee > 15:
        issues.append(f"Large left-right asymmetry at crouch ({asym_knee:.0f}°).")
        score -= 6
        remedies["asymmetry"] = {
            "cause": "Strength or mobility imbalance between legs",
            "exercises": ["Single-leg squats each side", "Bulgarian split squats",
                          "Single-leg RDL", "Unilateral calf raises"]
        }
    elif asym_knee > 8:
        issues.append(f"Moderate asymmetry ({asym_knee:.0f}°) — consider unilateral strength work")
        score -= 2
        if "asymmetry" not in remedies:
            remedies["asymmetry"] = {
                "cause": "Minor bilateral imbalance",
                "exercises": ["Single-leg squats", "Unilateral RDL", "Step-ups alternating"]
            }

    # 8. Arm Swing
    if arm_swing > 0.35:
        strengths.append("Good arm swing — contributes effectively to jump height")
    elif arm_swing > 0.15:
        issues.append("Moderate arm swing — extend arm drive fully for +5–10cm height")
        score -= 5
        remedies["arm_swing"] = {
            "cause": "Incomplete arm loading or poor coordination of arm-leg timing",
            "exercises": ["Arm swing drill (seated): practise full range lat-to-overhead",
                          "Standing arm swing jumps (no leg bend)", "Mirror drill for timing"]
        }
    elif arm_swing > 0:
        issues.append("Minimal arm swing detected — arms add up to 10cm of jump height when used fully")
        score -= 10
        remedies["arm_swing"] = {
            "cause": "Arms not engaged in stretch-shortening cycle",
            "exercises": ["Arm swing drills", "Counter-arm jump practise",
                          "Box jumps with focus on arm drive", "Reactive arm-swing jumps"]
        }

    score = max(0, min(100, score))
    if score >= 75:
        category = "GOOD"
    elif score >= 50:
        category = "AVERAGE"
    else:
        category = "POOR"

    return {
        "category": category,
        "score":    score,
        "issues":   issues,
        "strengths":strengths,
        "injuries": injuries,
        "remedies": remedies,
    }


def _injury_risk_level(injuries, valgus, land_knee, land_pelvic):
    n   = len(injuries)
    acl = any("ACL" in i for i in injuries)
    if acl or n >= 3:
        return "HIGH", injuries
    elif n >= 1 or valgus > VALGUS_THRESHOLD * 0.6 or (land_knee and land_knee > LAND_STIFF):
        return "MEDIUM", injuries
    return "LOW", injuries


def _score_landing(avg_la):
    if   avg_la >= 150: return 3
    elif avg_la >= 135: return 6
    elif avg_la >= 115: return 9
    elif avg_la >= 95:  return 7
    elif avg_la >= 80:  return 5
    else:               return 3


def _score_explosiveness(height_cm, flight_ms):
    h_score = min(10, max(0, (height_cm - HEIGHT_POOR) / (HEIGHT_GOOD - HEIGHT_POOR) * 10))
    return round(h_score, 1)


def _score_form(cls_score):
    return round(max(0, min(10, cls_score / 10)), 1)


# ════════════════════════════════════════════════════════════════
# Hip velocity helper
# ════════════════════════════════════════════════════════════════

def _hip_velocity(buf):
    """
    Estimate upward hip velocity from a rolling buffer of hip_y values.
    Positive = hip rising (jumping direction).
    hip_y is in normalised image coords where y increases downward,
    so rising = decreasing y, meaning earlier value minus later value > 0.
    """
    if len(buf) < 3:
        return 0.0
    deltas = [buf[i-1] - buf[i] for i in range(1, len(buf))]
    return float(np.mean(deltas[-2:]))


# ════════════════════════════════════════════════════════════════
# Main analysis function
# ════════════════════════════════════════════════════════════════

def analyse_vertical_jump(path, is_video, output_path=None, session_id=None,
                           source_filename="", progress_uid=None):

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    # ── Per-frame collected data ──────────────────────────────
    knee_angles_all  = []
    hip_angles_all   = []
    all_hip_y        = []
    all_knee_fc      = []
    wrong_events     = []
    lm_smoother      = LandmarkSmoother(alpha=0.40)
    fps_capture      = [30.0]
    height_samples   = []
    person_height_cm = [0.0]

    left_wrist_y_all  = []
    right_wrist_y_all = []
    shoulder_y_ref    = [None]

    valgus_l_all  = []
    valgus_r_all  = []
    trunk_lean_all = []
    pelvic_all    = []
    ankle_l_all   = []
    ankle_r_all   = []

    # ── Baseline / airborne detection state ───────────────────
    _baseline_y          = [None]
    _grounded_still_buf  = []      # hip_y only while standing still
    _hip_y_buf           = []      # rolling window for velocity calc
    _air_start_fc        = [None]
    _in_air              = [False]
    _peak_frame          = [None]
    _peak_elev           = [0.0]
    _landing_ctr         = [0]     # debounce counter for landing

    # Hip-ankle span buffer (for adaptive threshold)
    _span_buf            = []

    jump_results  = []
    jump_counter  = [0]
    total_frames  = [1]

    cmj_hip_traj = []
    phase_log    = []

    live_hud = {"jmp_ht": "---", "flt_time": "---", "lnd_scr": "---",
                "form": "---", "jump_no": "0", "category": "---"}

    def _push_jump_event(jump_data, frame_b64):
        if not progress_uid:
            return
        pct   = min(94, int(len(all_hip_y) / max(1, total_frames[0]) * 90))
        label = (f"Jump {jump_data['jump_no']} — "
                 f"{jump_data['height_cm']}cm — {jump_data.get('category','?')}")
        set_progress(progress_uid, pct, label,
                     jump_event={**jump_data, "frame_b64": frame_b64})

    # ──────────────────────────────────────────────────────────
    def pf(frame, fc, total):
        nonlocal live_hud
        total_frames[0] = max(total, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            canvas = expand_canvas_for_lhs(frame)
            draw_lhs_panel(canvas, _build_hud(live_hud))
            draw_pcl_logo(canvas)
            return canvas

        lm = res.pose_landmarks.landmark

        # ── Joint angles ──────────────────────────────────────
        l_k, r_k, avg_k = _bilateral_knee(lm)
        hip_a  = _bilateral_hip(lm)
        t_lean = _trunk_lean(lm)
        pelv   = _pelvic_drop(lm)
        vl     = _knee_valgus(lm, "left")
        vr     = _knee_valgus(lm, "right")
        ank_l  = _ankle_angle(lm, "left")
        ank_r  = _ankle_angle(lm, "right")
        el_l   = _elbow_angle(lm, "left")

        h_est = _estimate_height_cm(lm, height_samples)
        if h_est > 0:
            person_height_cm[0] = h_est

        lw = lm[15]; rw = lm[16]
        if lw.visibility > 0.3:
            left_wrist_y_all.append(lw.y)
        if rw.visibility > 0.3:
            right_wrist_y_all.append(rw.y)
        ls = lm[11]; rs = lm[12]
        if ls.visibility > 0.3 and rs.visibility > 0.3:
            shoulder_y_ref[0] = (ls.y + rs.y) / 2

        if avg_k is not None:
            knee_angles_all.append(avg_k)
            all_knee_fc.append(fc)
        if hip_a is not None:
            hip_angles_all.append(hip_a)
        valgus_l_all.append(vl)
        valgus_r_all.append(vr)
        trunk_lean_all.append(t_lean)
        pelvic_all.append(pelv)
        if ank_l: ankle_l_all.append(ank_l)
        if ank_r: ankle_r_all.append(ank_r)

        cur_hip_y = (lm[23].y + lm[24].y) / 2
        all_hip_y.append((fc, cur_hip_y))
        cmj_hip_traj.append((fc, cur_hip_y))

        # ── Rolling hip velocity ───────────────────────────────
        _hip_y_buf.append(cur_hip_y)
        if len(_hip_y_buf) > 10:
            _hip_y_buf.pop(0)
        hip_vel = _hip_velocity(_hip_y_buf)

        # ── Adaptive hip-ankle span ───────────────────────────
        # When athlete is far from camera their normalised span is smaller.
        # Scale AIR_THRESH to always represent the same fraction of their
        # visible lower-body length, regardless of camera distance.
        hip_ank_span = None
        if lm[23].visibility > 0.3 and lm[27].visibility > 0.3:
            hip_ank_span = abs(lm[23].y - lm[27].y)
        elif lm[24].visibility > 0.3 and lm[28].visibility > 0.3:
            hip_ank_span = abs(lm[24].y - lm[28].y)

        if hip_ank_span and hip_ank_span > 0.05:
            _span_buf.append(hip_ank_span)
            if len(_span_buf) > 60:
                _span_buf.pop(0)

        # Use median span of recent grounded frames for stability
        stable_span = float(np.median(_span_buf)) if _span_buf else None
        if stable_span and stable_span > 0.05:
            AIR_THRESH = max(0.018, stable_span * 0.04)   # was 0.025 / 0.06 — more sensitive
        else:
            AIR_THRESH = AIR_THRESH_NORM

        # ── Still-only baseline update ─────────────────────────
        # Only sample baseline when grounded AND hip is nearly stationary.
        # This prevents crouch frames from dragging the baseline down,
        # which would shrink the apparent elevation at true takeoff.
        if not _in_air[0] and abs(hip_vel) < STILL_VEL_THRESH:
            _grounded_still_buf.append(cur_hip_y)
            if len(_grounded_still_buf) > 120:
                _grounded_still_buf.pop(0)
            if len(_grounded_still_buf) >= 3:
                # 5th percentile = tallest standing posture seen so far
                _baseline_y[0] = float(np.percentile(_grounded_still_buf, 5))

        baseline = _baseline_y[0]
        fps_live  = fps_capture[0] if fps_capture[0] > 0 else 30.0

        # ── Draw skeleton & angle arcs ─────────────────────────
        pts = lm_smoother.smooth(lm, frame.shape[1], frame.shape[0])
        draw_pose_skyblue(frame, res.pose_landmarks)
        if avg_k is not None:
            draw_angle_arc(frame, lm[25], avg_k, bad=(avg_k > 150 or avg_k < 80))
        if hip_a is not None:
            draw_angle_arc(frame, lm[23], hip_a, bad=(hip_a < 130), color=(0, 200, 255))
        if abs(vl) > VALGUS_THRESHOLD:
            draw_angle_arc(frame, lm[25], avg_k if avg_k else 90,
                           bad=True, color=(0, 0, 255))

        # ── Airborne detection ────────────────────────────────
        if baseline is not None:
            elevation = baseline - cur_hip_y   # positive = hip above standing

            if not _in_air[0]:
                # Require BOTH elevation threshold AND upward velocity.
                # Velocity gate prevents a deep crouch (which also raises
                # hip_y transiently) from being mistaken for takeoff.
                if elevation > AIR_THRESH and hip_vel > VEL_TAKEOFF_MIN:
                    _in_air[0]       = True
                    _air_start_fc[0] = fc
                    _peak_frame[0]   = None
                    _peak_elev[0]    = elevation
                    _landing_ctr[0]  = 0

            else:
                # Track peak elevation frame for snapshot
                if elevation > _peak_elev[0]:
                    _peak_elev[0]  = elevation
                    _peak_frame[0] = frame_to_b64(frame)

                # Debounced landing: require LANDING_HOLD consecutive frames
                # below threshold before closing the flight window.
                # This prevents a single jittery frame from cutting flight short.
                LAND_THRESH = AIR_THRESH * 0.4
                if elevation < LAND_THRESH:
                    _landing_ctr[0] += 1
                else:
                    _landing_ctr[0] = 0

                if _landing_ctr[0] >= LANDING_HOLD:
                    air_start = _air_start_fc[0]
                    # True landing frame = current frame minus debounce hold
                    air_end   = fc - LANDING_HOLD + 1

                    # ── FPS-scaled frame-clipping correction ───────────
                    # The threshold is crossed mid-frame on both sides.
                    # Add proportional frames to recover lost flight time:
                    #   • 1 clip frame at 30fps  (~33ms each side)
                    #   • 2 clip frames at 60fps (~16ms each side)
                    clip_frames = max(1, round(fps_live / 30.0))
                    air_start   = max(0, air_start - clip_frames)
                    air_end     = air_end + clip_frames

                    air_frames  = air_end - air_start

                    if air_frames >= MIN_AIR_FRAMES:
                        flight_s  = air_frames / fps_live
                        flight_ms = round(flight_s * 1000)
                        flight_ms = max(0, min(1100, flight_ms))
                        t_half    = flight_s / 2.0
                        height_cm = round(0.5 * 9.81 * t_half**2 * 100, 1)
                        height_cm = max(0.0, min(120.0, height_cm))

                        # ── Peak-elevation cross-check ─────────────────
                        # If the normalised peak elevation implies a
                        # meaningfully higher jump than flight-time gives,
                        # blend 30% of the displacement estimate in.
                        # Uses athlete's own hip-ankle span as a ruler so
                        # it works at any camera distance.
                        if stable_span and stable_span > 0.05 and person_height_cm[0] > 0:
                            px_per_cm = stable_span / (0.53 * person_height_cm[0])
                            if px_per_cm > 0:
                                elev_cm = _peak_elev[0] / px_per_cm
                                if elev_cm > height_cm * 1.2:
                                    height_cm = round(0.3 * elev_cm + 0.7 * height_cm, 1)
                                    height_cm = min(120.0, height_cm)

                        # ── Phase windows ──────────────────────────────
                        pre_s    = int(fps_live * 0.8)
                        to_win   = int(fps_live * 0.25)
                        land_win = int(fps_live * 0.6)

                        idx_stand_start = max(0, air_start - pre_s)
                        idx_crouch_end  = max(0, air_start - to_win)
                        idx_to_start    = idx_crouch_end
                        idx_land_end    = min(len(all_knee_fc)-1, air_end + land_win)

                        def fc_to_pos(tfc):
                            if not all_knee_fc:
                                return 0
                            return min(range(len(all_knee_fc)),
                                       key=lambda i: abs(all_knee_fc[i]-tfc),
                                       default=0)

                        p_stand  = fc_to_pos(idx_stand_start)
                        p_crouch = fc_to_pos(idx_crouch_end)
                        p_to     = fc_to_pos(idx_to_start)
                        p_air    = fc_to_pos(air_start)
                        p_land   = fc_to_pos(air_end)
                        p_land_e = fc_to_pos(idx_land_end)

                        stand_k  = knee_angles_all[p_stand:p_crouch]
                        crouch_k = knee_angles_all[p_crouch:p_air]
                        to_k     = knee_angles_all[p_to:p_air]
                        land_k   = knee_angles_all[p_land:p_land_e]

                        crouch_min = float(np.min(crouch_k)) if crouch_k else None
                        to_max     = float(np.max(to_k))     if to_k     else None
                        land_avg   = float(np.mean(land_k))  if land_k   else None

                        asym_knee = 0.0
                        if l_k is not None and r_k is not None:
                            asym_knee = abs(l_k - r_k)

                        def _slice_fc(arr, start_fc, end_fc):
                            n_total = len(all_hip_y)
                            si = max(0, int(start_fc / max(1, total_frames[0]) * len(arr)))
                            ei = min(len(arr), int(end_fc / max(1, total_frames[0]) * len(arr)))
                            return arr[si:ei]

                        land_vl = _slice_fc(valgus_l_all, air_end, idx_land_end)
                        land_vr = _slice_fc(valgus_r_all, air_end, idx_land_end)
                        max_vl  = float(max(abs(v) for v in land_vl)) if land_vl else 0.0
                        max_vr  = float(max(abs(v) for v in land_vr)) if land_vr else 0.0

                        crouch_trunk = _slice_fc(trunk_lean_all, idx_stand_start, air_start)
                        avg_trunk    = float(np.mean(crouch_trunk)) if crouch_trunk else 0.0

                        land_pelv = _slice_fc(pelvic_all, air_end, idx_land_end)
                        avg_pelv  = float(np.mean(land_pelv)) if land_pelv else 0.0

                        wrist_in_jump = left_wrist_y_all[
                            max(0, len(left_wrist_y_all)-int(fps_live*2)):
                        ]
                        arm_swing = _arm_swing_range(wrist_in_jump, shoulder_y_ref[0])

                        jd_input = {
                            "height_cm":           height_cm,
                            "crouch_min_knee":     crouch_min,
                            "takeoff_max_ext":     to_max,
                            "landing_avg_knee":    land_avg,
                            "valgus_left":         max_vl,
                            "valgus_right":        max_vr,
                            "crouch_trunk_lean":   avg_trunk,
                            "landing_pelvic_drop": avg_pelv,
                            "arm_swing_range":     arm_swing,
                            "asymmetry_knee":      asym_knee,
                        }
                        cls = classify_cmj(jd_input)

                        risk_lvl, risk_list = _injury_risk_level(
                            cls["injuries"], max(max_vl, max_vr), land_avg, avg_pelv)

                        exp_score  = _score_explosiveness(height_cm, flight_ms)
                        form_score = _score_form(cls["score"])
                        lnd_score  = _score_landing(land_avg) if land_avg else 5

                        jump_counter[0] += 1
                        jno = jump_counter[0]

                        jump_data = {
                            "jump_no":           jno,
                            "height_cm":         height_cm,
                            "flight_ms":         flight_ms,
                            "category":          cls["category"],
                            "cls_score":         cls["score"],
                            "form_score":        form_score,
                            "landing_score":     lnd_score,
                            "explosiveness":     exp_score,
                            "injury_risk":       risk_lvl,
                            "injuries":          risk_list,
                            "crouch_min_knee":   round(crouch_min, 1) if crouch_min else None,
                            "takeoff_ext":       round(to_max, 1)     if to_max    else None,
                            "landing_avg_knee":  round(land_avg, 1)   if land_avg  else None,
                            "valgus_left":       round(max_vl, 1),
                            "valgus_right":      round(max_vr, 1),
                            "trunk_lean":        round(avg_trunk, 1),
                            "arm_swing":         round(arm_swing, 3),
                            "issues":            cls["issues"],
                            "strengths":         cls["strengths"],
                            "remedies":          cls["remedies"],
                            "phases": {
                                "standing": {
                                    "description": "Initial upright posture before countermovement",
                                    "avg_knee": round(float(np.mean(stand_k)), 1) if stand_k else None,
                                },
                                "crouch": {
                                    "description": "Rapid countermovement — loads elastic energy",
                                    "min_knee":    round(crouch_min, 1) if crouch_min else None,
                                    "trunk_lean":  round(avg_trunk, 1),
                                    "ideal_range": "90–120°",
                                },
                                "takeoff": {
                                    "description": "Triple extension — peak propulsive force",
                                    "max_ext": round(to_max, 1) if to_max else None,
                                    "ideal":   "≥160°",
                                },
                                "flight": {
                                    "description": "Airborne — vertical displacement determines height",
                                    "flight_ms": flight_ms,
                                    "height_cm": height_cm,
                                    "formula":   "h = g × (t/2)² / 2",
                                },
                                "landing": {
                                    "description": "Shock absorption — eccentric muscle control critical",
                                    "avg_knee":    round(land_avg, 1) if land_avg else None,
                                    "pelvic_drop": round(avg_pelv, 1),
                                    "valgus_l":    round(max_vl, 1),
                                    "valgus_r":    round(max_vr, 1),
                                    "ideal_range": "110–140°",
                                },
                            },
                        }
                        jump_results.append(jump_data)

                        peak_b64 = _peak_frame[0] or frame_to_b64(frame)
                        _push_jump_event(jump_data, peak_b64)

                        live_hud = {
                            "jump_no":  str(jno),
                            "jmp_ht":   f"{height_cm}cm",
                            "flt_time": f"{round(flight_ms/1000.0, 2)}s",
                            "lnd_scr":  f"{lnd_score}/10",
                            "form":     f"{form_score}/10",
                            "category": cls["category"],
                        }

                    # Reset airborne state
                    _in_air[0]       = False
                    _air_start_fc[0] = None
                    _peak_frame[0]   = None
                    _peak_elev[0]    = 0.0
                    _landing_ctr[0]  = 0

        # ── HUD ───────────────────────────────────────────────
        canvas = expand_canvas_for_lhs(frame)
        draw_lhs_panel(canvas, [
            ("JUMP #",  live_hud.get("jump_no",  "0")),
            ("HT",      live_hud.get("jmp_ht",   "---")),
            ("FORM",    live_hud.get("form",      "---")),
            ("QUALITY", live_hud.get("category", "---")),
        ])
        draw_pcl_logo(canvas)
        return canvas

    def _build_hud(h):
        return [
            ("JUMP #",  h.get("jump_no",  "0")),
            ("HT",      h.get("jmp_ht",   "---")),
            ("FORM",    h.get("form",      "---")),
            ("QUALITY", h.get("category", "---")),
        ]

    # ── Capture FPS ───────────────────────────────────────────
    if is_video:
        _cap = cv2.VideoCapture(path)
        if _cap.isOpened():
            fps_capture[0] = _cap.get(cv2.CAP_PROP_FPS) or 30.0
        _cap.release()

    snaps = process_video_or_image(
        path, is_video, pf,
        output_path=output_path,
        snap_pcts=[0.1, 0.3, 0.5, 0.7, 0.9],
        analysis_skip=1,
        progress_uid=progress_uid,
    )
    pose.close()

    if session_id:
        save_wrong_angle_log("vertical_jump", session_id, source_filename, wrong_events)
    if is_video and len(all_hip_y) < 10:
        raise ValueError("No reliable vertical jump motion detected.")

    fps = fps_capture[0]

    if height_samples:
        all_s = sorted(height_samples)
        person_height_cm[0] = float(all_s[len(all_s)//2])

    # ── Fallback batch detection if live detector missed all jumps ─
    if len(jump_results) == 0 and len(all_hip_y) >= 10:
        hip_y_vals = [y for _, y in all_hip_y]
        n          = len(hip_y_vals)

        # 5th percentile = tallest standing posture (same logic as live baseline)
        standing_y = float(np.percentile(hip_y_vals, 5))
        hip_range  = max(hip_y_vals) - min(hip_y_vals)
        air_thresh = max(0.008, hip_range * 0.20)
        min_air_f  = max(2, int(0.06 * fps))

        flags  = [(standing_y - y) > air_thresh for _, y in all_hip_y]
        blocks = []
        in_b = False; bs = 0
        for i, flag in enumerate(flags):
            if flag and not in_b:  in_b = True;  bs = i
            elif not flag and in_b: in_b = False; blocks.append((bs, i-1))
        if in_b:
            blocks.append((bs, len(flags)-1))
        blocks = [(s, e) for s, e in blocks if (e-s+1) >= min_air_f]

        if blocks:
            best         = max(blocks, key=lambda b: b[1]-b[0])
            air_s, air_e = best

            # Apply same FPS-scaled clip correction as live detector
            clip_frames = max(1, round(fps / 30.0))
            air_s = max(0, air_s - clip_frames)
            air_e = min(len(all_hip_y) - 1, air_e + clip_frames)

            ft_s      = (all_hip_y[air_e][0] - all_hip_y[air_s][0]) / fps
            flight_ms = max(0, min(900, round(ft_s * 1000)))
            t_half    = ft_s / 2.0
            height_cm = max(0.0, min(80.0, round(0.5 * 9.81 * t_half**2 * 100, 1)))

            mid         = n // 2
            avg_takeoff = round(float(np.min(knee_angles_all[:mid//2])), 1) if knee_angles_all else 0.0
            avg_landing = round(float(np.mean(knee_angles_all[mid:])),   1) if knee_angles_all else 0.0
            avg_vl      = float(np.mean([abs(v) for v in valgus_l_all[mid:]])) if valgus_l_all else 0.0
            avg_vr      = float(np.mean([abs(v) for v in valgus_r_all[mid:]])) if valgus_r_all else 0.0
            avg_trunk   = float(np.mean(trunk_lean_all[:mid])) if trunk_lean_all else 0.0
            avg_pelv    = float(np.mean(pelvic_all[mid:]))     if pelvic_all     else 0.0
            wrist_hist  = left_wrist_y_all[max(0, len(left_wrist_y_all)-int(fps*2)):]
            arm_sw      = _arm_swing_range(wrist_hist, shoulder_y_ref[0])

            jd_input = {
                "height_cm":           height_cm,
                "crouch_min_knee":     avg_takeoff,
                "takeoff_max_ext":     float(np.max(knee_angles_all[:mid])) if knee_angles_all else None,
                "landing_avg_knee":    avg_landing,
                "valgus_left":         avg_vl,
                "valgus_right":        avg_vr,
                "crouch_trunk_lean":   avg_trunk,
                "landing_pelvic_drop": avg_pelv,
                "arm_swing_range":     arm_sw,
                "asymmetry_knee":      0.0,
            }
            cls      = classify_cmj(jd_input)
            risk_lvl, risk_list = _injury_risk_level(
                cls["injuries"], max(avg_vl, avg_vr), avg_landing, avg_pelv)
            exp_score  = _score_explosiveness(height_cm, flight_ms)
            form_score = _score_form(cls["score"])
            lnd_score  = _score_landing(avg_landing)

            jump_results.append({
                "jump_no":          1,
                "height_cm":        height_cm,
                "flight_ms":        flight_ms,
                "category":         cls["category"],
                "cls_score":        cls["score"],
                "form_score":       form_score,
                "landing_score":    lnd_score,
                "explosiveness":    exp_score,
                "injury_risk":      risk_lvl,
                "injuries":         risk_list,
                "crouch_min_knee":  avg_takeoff,
                "takeoff_ext":      round(float(np.max(knee_angles_all[:mid])), 1) if knee_angles_all else None,
                "landing_avg_knee": avg_landing,
                "valgus_left":      round(avg_vl, 1),
                "valgus_right":     round(avg_vr, 1),
                "trunk_lean":       round(avg_trunk, 1),
                "arm_swing":        round(arm_sw, 3),
                "issues":           cls["issues"],
                "strengths":        cls["strengths"],
                "remedies":         cls["remedies"],
                "phases":           {},
            })

    # ── Aggregate across all jumps ────────────────────────────
    jump_count  = len(jump_results)
    best_height = round(max(j["height_cm"]     for j in jump_results), 1) if jump_results else 0.0
    avg_height  = round(np.mean([j["height_cm"]     for j in jump_results]), 1) if jump_results else 0.0
    avg_flight  = round(np.mean([j["flight_ms"]     for j in jump_results]))    if jump_results else 0
    avg_lnd_scr = round(np.mean([j["landing_score"] for j in jump_results]))    if jump_results else 0
    avg_frm_scr = round(np.mean([j["form_score"]    for j in jump_results]), 1) if jump_results else 4.0
    avg_exp     = round(np.mean([j["explosiveness"] for j in jump_results]), 1)  if jump_results else 0.0
    best_cls    = max(jump_results, key=lambda j: j["cls_score"]) if jump_results else {}
    best_cat    = best_cls.get("category", "N/A")

    risk_order   = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    overall_risk = max(
        (j["injury_risk"] for j in jump_results),
        key=lambda r: risk_order.get(r, 0),
        default="LOW"
    )

    all_issues    = []
    all_strengths = []
    all_remedies  = {}
    seen_i = set(); seen_s = set()
    for j in jump_results:
        for s in j.get("strengths", []):
            if s not in seen_s: seen_s.add(s); all_strengths.append(s)
        for i in j.get("issues", []):
            if i not in seen_i: seen_i.add(i); all_issues.append(i)
        all_remedies.update(j.get("remedies", {}))

    if not all_issues:
        all_issues = ["No major CMJ form issues detected — excellent movement quality!"]

    all_injuries = list({inj for j in jump_results for inj in j.get("injuries", [])})

    h_str  = f"{int(person_height_cm[0])}cm" if person_height_cm[0] > 0 else "N/A"
    ht_str = f"{best_height}cm"  if best_height > 0 else "N/A"
    avg_flight_s = round(avg_flight / 1000.0, 2) if avg_flight > 0 else 0.0
    ft_str = f"{avg_flight_s}s"  if avg_flight  > 0 else "N/A"

    return {
        "exercise":            "Countermovement Vertical Jump (CMJ)",
        "jump_count":          jump_count,
        "correct_jumps":       sum(1 for j in jump_results if j["category"] == "GOOD"),
        "wrong_jumps":         sum(1 for j in jump_results if j["category"] == "POOR"),
        "jump_height_cm":      best_height,
        "avg_height_cm":       avg_height,
        "flight_time_ms":      avg_flight,
        "person_height_cm":    person_height_cm[0],
        "form_score":          avg_frm_scr,
        "landing_score":       avg_lnd_scr,
        "explosiveness_score": avg_exp,
        "injury_risk":         overall_risk,
        "overall_category":    best_cat,
        "avg_takeoff_angle":   round(float(np.mean([j["crouch_min_knee"] for j in jump_results
                                                     if j.get("crouch_min_knee") is not None
                                                     and j["crouch_min_knee"] > 0])), 1)
                               if any(j.get("crouch_min_knee") for j in jump_results) else 0.0,
        "avg_landing_angle":   round(float(np.mean([j["landing_avg_knee"] for j in jump_results
                                                     if j.get("landing_avg_knee") is not None
                                                     and j["landing_avg_knee"] > 0])), 1)
                               if any(j.get("landing_avg_knee") for j in jump_results) else 0.0,
        "issues":              all_issues,
        "strengths":           all_strengths,
        "remedies":            all_remedies,
        "injury_list":         all_injuries,
        "per_jump":            jump_results,
        "snapshots":           snaps,
        "wrong_angle_count":   len(wrong_events),
        "_wrong_events":       wrong_events,
        "metrics": [
            {"label": "Jumps Detected",  "value": str(jump_count)},
            {"label": "Best Height",     "value": ht_str},
            {"label": "Avg Height",      "value": f"{avg_height}cm"},
            {"label": "Flight Time",     "value": ft_str},
            {"label": "Overall Quality", "value": best_cat},
            {"label": "Form Score",      "value": f"{avg_frm_scr}/10"},
            {"label": "Landing Score",   "value": f"{avg_lnd_scr}/10"},
            {"label": "Explosiveness",   "value": f"{avg_exp}/10"},
            {"label": "Injury Risk",     "value": overall_risk},
            {"label": "Person Height",   "value": h_str},
        ],
    }
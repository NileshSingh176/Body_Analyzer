import cv2
import numpy as np
import os
from utils import (
    mp_pose,
    get_landmark, calculate_angle, draw_angle_arc,
    RollingMean, LandmarkSmoother, process_video_or_image,
    save_wrong_angle_log, draw_pose_skyblue,
)
from hud_overlay import draw_footer_hud, draw_pcl_logo

REP_RISE_DEG = 28

# ── Color constants (BGR) ─────────────────────────────────────
_C_LEFT_LEG  = (255, 255,   0)   # cyan
_C_RIGHT_LEG = (  0, 140, 255)   # orange
_C_LEFT_ARM  = (100, 220,   0)   # green
_C_RIGHT_ARM = (220, 220,   0)   # yellow
_C_TORSO     = (220, 220, 220)   # white

_LM_LEFT_LEG  = [23, 25, 27, 29, 31]
_LM_RIGHT_LEG = [24, 26, 28, 30, 32]
_LM_LEFT_ARM  = [11, 13, 15, 17, 19, 21]
_LM_RIGHT_ARM = [12, 14, 16, 18, 20, 22]
_LM_TORSO     = [11, 12, 23, 24]

_CONN_LEFT_LEG  = [(23,25),(25,27),(27,29),(27,31),(29,31)]
_CONN_RIGHT_LEG = [(24,26),(26,28),(28,30),(28,32),(30,32)]
_CONN_LEFT_ARM  = [(11,13),(13,15),(15,17),(15,19),(15,21),(17,19)]
_CONN_RIGHT_ARM = [(12,14),(14,16),(16,18),(16,20),(16,22),(18,20)]
_CONN_TORSO     = [(11,12),(11,23),(12,24),(23,24)]

# Locked landmark indices — angle calc only uses same-side points
_SIDE_LANDMARKS = {
    "left":  {"hip": 23, "knee": 25, "ankle": 27, "shoulder": 11, "knee_lm": 25},
    "right": {"hip": 24, "knee": 26, "ankle": 28, "shoulder": 12, "knee_lm": 26},
}


def _px(lm_obj, w, h):
    return (int(lm_obj.x * w), int(lm_obj.y * h))


def draw_pose_bicolor(frame, pose_landmarks, active_leg="left", bad=False, pts=None):
    if pose_landmarks is None:
        return
    lm = pose_landmarks.landmark
    h, w = frame.shape[:2]

    # Use pre-smoothed pixel coords if provided, else compute raw
    def _spx(idx):
        if pts is not None:
            return (int(pts[idx][0]), int(pts[idx][1]))
        return _px(lm[idx], w, h)

    if active_leg == "left":
        c_active      = _C_LEFT_LEG if not bad else (0, 0, 230)
        c_inactive    = tuple(int(v * 0.45) for v in _C_RIGHT_LEG)
        conn_active   = _CONN_LEFT_LEG
        conn_inactive = _CONN_RIGHT_LEG
        lm_active     = _LM_LEFT_LEG
        lm_inactive   = _LM_RIGHT_LEG
    else:
        c_active      = _C_RIGHT_LEG if not bad else (0, 0, 230)
        c_inactive    = tuple(int(v * 0.45) for v in _C_LEFT_LEG)
        conn_active   = _CONN_RIGHT_LEG
        conn_inactive = _CONN_LEFT_LEG
        lm_active     = _LM_RIGHT_LEG
        lm_inactive   = _LM_LEFT_LEG

    def draw_connections(connections, color, thickness):
        # Shadow pass
        shadow = tuple(max(0, int(c * 0.35)) for c in color)
        overlay = frame.copy()
        for a, b in connections:
            if lm[a].visibility > 0.15 and lm[b].visibility > 0.15:
                cv2.line(overlay, _spx(a), _spx(b), shadow, thickness + 4, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
        # Bright pass
        for a, b in connections:
            if lm[a].visibility > 0.15 and lm[b].visibility > 0.15:
                cv2.line(frame, _spx(a), _spx(b), color, thickness, cv2.LINE_AA)

    def draw_keypoints(indices, color, radius):
        for i in indices:
            if lm[i].visibility > 0.15:
                cx, cy = _spx(i)
                cv2.circle(frame, (cx, cy), radius + 2, (30, 30, 30), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), radius,     (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), max(1, radius - 2), color, -1, cv2.LINE_AA)

    draw_connections(_CONN_TORSO, _C_TORSO, 2)
    draw_keypoints(_LM_TORSO, _C_TORSO, 4)
    draw_connections(_CONN_LEFT_ARM,  _C_LEFT_ARM,  2)
    draw_connections(_CONN_RIGHT_ARM, _C_RIGHT_ARM, 2)
    draw_keypoints(_LM_LEFT_ARM,  _C_LEFT_ARM,  4)
    draw_keypoints(_LM_RIGHT_ARM, _C_RIGHT_ARM, 4)
    draw_connections(conn_inactive, c_inactive, 2)
    draw_keypoints(lm_inactive, c_inactive, 4)
    draw_connections(conn_active, c_active, 3)
    draw_keypoints(lm_active, c_active, 6)

    items = [
        (_C_LEFT_LEG  if not bad or active_leg != "left"  else (0, 0, 230),
         f"LEFT LEG {'(ACTIVE)' if active_leg == 'left' else ''}"),
        (_C_RIGHT_LEG if not bad or active_leg != "right" else (0, 0, 230),
         f"RIGHT LEG {'(ACTIVE)' if active_leg == 'right' else ''}"),
        (_C_LEFT_ARM,  "LEFT ARM"),
        (_C_RIGHT_ARM, "RIGHT ARM"),
        (_C_TORSO,     "TORSO"),
    ]
    x0, y0 = 10, 10
    for i, (color, label) in enumerate(items):
        y = y0 + i * 18
        cv2.rectangle(frame, (x0, y), (x0 + 12, y + 12), color, -1)
        cv2.rectangle(frame, (x0, y), (x0 + 12, y + 12), (30, 30, 30), 1)
        cv2.putText(frame, label, (x0 + 16, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)


def draw_angle_arc_colored(frame, kne_lm, angle, active_leg, bad=False):
    draw_angle_arc(frame, kne_lm, angle, bad=bad)


def get_locked_angle(lm, side, min_vis=0.20):
    idx      = _SIDE_LANDMARKS[side]
    hip_lm   = lm[idx["hip"]]
    knee_lm  = lm[idx["knee"]]
    ankle_lm = lm[idx["ankle"]]
    if (hip_lm.visibility   < min_vis or
            knee_lm.visibility  < min_vis or
            ankle_lm.visibility < min_vis):
        return None
    return calculate_angle(
        [hip_lm.x,   hip_lm.y],
        [knee_lm.x,  knee_lm.y],
        [ankle_lm.x, ankle_lm.y],
    )


def get_locked_hip_angle(lm, side, min_vis=0.20):
    idx      = _SIDE_LANDMARKS[side]
    shldr_lm = lm[idx["shoulder"]]
    hip_lm   = lm[idx["hip"]]
    knee_lm  = lm[idx["knee"]]
    if (shldr_lm.visibility < min_vis or
            hip_lm.visibility   < min_vis or
            knee_lm.visibility  < min_vis):
        return None
    return calculate_angle(
        [shldr_lm.x, shldr_lm.y],
        [hip_lm.x,   hip_lm.y],
        [knee_lm.x,  knee_lm.y],
    )


# ── Injury risk database ──────────────────────────────────────
_INJURY_DB = {
    "ACL": {
        "name": "ACL Tear Risk",
        "cause": (
            "Dynamic knee valgus (inward knee collapse) increases anterior tibial "
            "shear and ACL tensile load. Hip abductor weakness allows femoral "
            "adduction and internal rotation, collapsing the knee medially."
        ),
        "trigger": "knee_valgus > 10° OR pelvic_drop > 10",
        "remedies": [
            "Glute bridge 3×15 — activates glute max and medius",
            "Clamshells with resistance band 3×20 — isolates glute medius",
            "Lateral band walks 3×15 steps — hip abductor strengthening",
            "Single-leg RDL 3×10 — hip stability under load",
            "Knee-over-toe box step-ups — trains proper knee tracking",
            "Mirror single-leg squat drill — visual feedback on knee position",
        ],
    },
    "PFPS": {
        "name": "Patellofemoral Pain Syndrome (Runner's Knee)",
        "cause": (
            "Excessive hip adduction and internal rotation shift the patella "
            "laterally, increasing patellofemoral contact pressure."
        ),
        "trigger": "trunk_lean > 9° AND pelvic_drop > 10 (Powers 2010)",
        "remedies": [
            "Vastus medialis activation drills (terminal knee extension with band)",
            "Hip external rotation strengthening — side-lying leg raises",
            "Step-down eccentric exercise 3×10 — patellar tendon loading",
            "Foam rolling iliotibial band and quadriceps",
            "Straight-leg raises 3×15 — VMO isolation",
            "Short-foot arch activation exercises",
        ],
    },
    "ITB": {
        "name": "IT Band Syndrome",
        "cause": (
            "Excessive hip adduction and contralateral pelvic drop increase "
            "iliotibial band tension and lateral knee compression."
        ),
        "trigger": "hip_drop > 10 AND trunk_lean > 15°",
        "remedies": [
            "IT band foam rolling daily",
            "Hip abductor strengthening — clamshells, lateral walks",
            "TFL stretching — pigeon pose, figure-4 stretch",
            "Glute medius strengthening — standing hip abduction",
            "Reduce training volume temporarily",
        ],
    },
    "PATELLAR_TENDON": {
        "name": "Patellar Tendinopathy",
        "cause": (
            "Insufficient knee flexion depth combined with rapid loading "
            "concentrates stress at the patellar tendon insertion."
        ),
        "trigger": "knee_depth > 120° (insufficient flexion) AND movement = jerky",
        "remedies": [
            "Eccentric single-leg decline squat 3×15 — gold-standard tendon loading",
            "Isometric quad holds (70° knee flexion) 4×45s — pain relief",
            "Gradual depth progression with slow tempo (3-1-3 cadence)",
            "Calf raises to improve ankle mobility and reduce tendon compensation",
        ],
    },
    "HIP_IMPINGEMENT": {
        "name": "Hip Impingement (FAI Risk)",
        "cause": (
            "Excessive forward trunk lean increases hip flexion beyond 90°, "
            "compressing the anterior hip capsule and labrum."
        ),
        "trigger": "trunk_lean > 30° (severe forward lean)",
        "remedies": [
            "Hip flexor stretching — couch stretch, kneeling lunge stretch",
            "Thoracic spine mobility drills — cat-cow, thoracic rotations",
            "Goblet squat with upright torso cue 3×10",
            "Ankle dorsiflexion mobility — wall ankle stretches",
            "Core anti-flexion strengthening — Pallof press",
        ],
    },
    "ANKLE_INSTABILITY": {
        "name": "Ankle Instability / Chronic Lateral Ankle Sprain",
        "cause": (
            "Heel lift and excessive foot pronation indicate reduced ankle "
            "dorsiflexion and poor subtalar control."
        ),
        "trigger": "heel_lift > 3 OR sway_SD > 7",
        "remedies": [
            "Ankle dorsiflexion mobilisation — banded ankle distraction",
            "Single-leg balance on unstable surface (BOSU, foam pad) 3×30s",
            "Calf raises (eccentric) 3×15 — Achilles-soleus loading",
            "Tibialis anterior strengthening — towel scrunches, toe raises",
            "Proprioception drills — eyes-closed single-leg stand",
        ],
    },
    "LOWER_BACK": {
        "name": "Lower Back Pain / Lumbar Overload",
        "cause": (
            "Severe forward trunk lean transfers load from the hips and quads "
            "to the lumbar erectors, increasing disc compressive forces."
        ),
        "trigger": "trunk_lean > 35°",
        "remedies": [
            "Hip hinge pattern retraining — Romanian deadlift with dowel cue",
            "Glute strengthening to reduce lumbar compensation",
            "Core stability — dead bug, bird-dog progressions",
            "Ankle mobility work to allow upright squat posture",
            "Foam rolling thoracic spine",
        ],
    },
    "MENISCUS": {
        "name": "Medial Meniscus Stress",
        "cause": (
            "Combined knee valgus and deep knee flexion (>90°) increases medial "
            "compartment compressive load on the meniscus."
        ),
        "trigger": "knee_valgus > 10° AND knee_depth < 80°",
        "remedies": [
            "Limit squat depth to pain-free range during rehabilitation",
            "Knee valgus correction via glute medius strengthening",
            "Quadriceps strengthening in limited range",
            "Physiotherapy assessment recommended",
        ],
    },
}


def _detect_injuries(knee_valgus, torso_lean, hip_drop,
                     heel_lift, balance_sway, knee_depth):
    # FIXED: All thresholds raised to match the corrected scoring thresholds.
    # Valgus is excluded from ACL trigger (proxy value, not real geometry).
    detected = []
    if hip_drop > 20:                           # was > 10 — noise was triggering this
        lvl = "HIGH" if hip_drop > 28 else "MEDIUM"
        detected.append({**_INJURY_DB["ACL"], "risk_level": lvl})
    elif hip_drop > 14:                         # was > 7
        detected.append({**_INJURY_DB["ACL"], "risk_level": "LOW"})
    if torso_lean > 20 and hip_drop > 20:       # was > 9 and > 10
        detected.append({**_INJURY_DB["PFPS"], "risk_level": "HIGH"})
    elif torso_lean > 20 or hip_drop > 14:      # was > 9 or > 7
        detected.append({**_INJURY_DB["PFPS"], "risk_level": "MEDIUM"})
    if hip_drop > 20 and torso_lean > 30:       # was > 10 and > 15
        detected.append({**_INJURY_DB["ITB"], "risk_level": "MEDIUM"})
    if knee_depth > 130 and balance_sway > 10:  # was > 120 and > 5
        detected.append({**_INJURY_DB["PATELLAR_TENDON"], "risk_level": "MEDIUM"})
    elif knee_depth > 145:                      # was > 130
        detected.append({**_INJURY_DB["PATELLAR_TENDON"], "risk_level": "LOW"})
    if torso_lean > 45:                         # was > 30
        detected.append({**_INJURY_DB["HIP_IMPINGEMENT"], "risk_level": "MEDIUM"})
    if heel_lift > 3 or balance_sway > 14:      # was > 7
        detected.append({**_INJURY_DB["ANKLE_INSTABILITY"], "risk_level": "MEDIUM"})
    elif heel_lift > 1 or balance_sway > 8:     # was > 4
        detected.append({**_INJURY_DB["ANKLE_INSTABILITY"], "risk_level": "LOW"})
    if torso_lean > 50:                         # was > 35
        detected.append({**_INJURY_DB["LOWER_BACK"], "risk_level": "MEDIUM"})
    # Meniscus: removed valgus proxy — not reliable from side view
    seen, unique = set(), []
    for d in detected:
        k = d["name"]
        if k not in seen:
            seen.add(k)
            unique.append(d)
    return unique


def _overall_injury_risk(injuries):
    order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    if not injuries:
        return "LOW"
    return max((i["risk_level"] for i in injuries), key=lambda r: order.get(r, 0))


def classify_single_leg_squat(
    knee_angle, knee_valgus, torso_lean, hip_drop,
    heel_lift, balance_sway, foot_pronation, movement_smoothness
):
    score = 100
    suggestions = []

    # FIXED: Crossley thresholds raised to match clinical standards.
    # Require >= 2 criteria to be positive — professional athletes will
    # naturally trigger one criterion, which is normal biomechanical variance.
    crossley = {
        "trunk_lean":      torso_lean  > 15,   # was 8  — too strict for any lean
        "pelvic_drop":     hip_drop    > 12,   # was 5  — MediaPipe noise alone hits 5
        "knee_valgus":     knee_valgus > 15,   # was 10 — proxy from pelvic drop, not real
        "hip_adduction":   hip_drop    > 14,   # was 7
        "loss_of_balance": balance_sway > 8,   # was 4
    }
    crossley_count    = sum(crossley.values())
    crossley_positive = crossley_count >= 2    # was >= 1 — triggered on almost every rep

    if knee_angle < 60:
        pass
    elif knee_angle <= 90:
        pass
    elif knee_angle <= 120:
        score -= 10
        suggestions.append(
            "Increase squat depth — aim for 60–90° knee flexion. "
            "Ankle mobility and hip flexibility work will help reach optimal depth."
        )
    else:
        score -= 20
        suggestions.append(
            "Very shallow squat depth detected (knee barely bending). "
            "Focus on quad strengthening and ankle dorsiflexion mobility."
        )

    # FIXED: Valgus scoring disabled. The valgus value passed in is estimated
    # as `pelvic_drop * 1.4` — not real knee geometry. This means pelvic drop
    # was being penalised TWICE (once directly, once via fake valgus), which
    # caused most professional reps to be marked wrong. Real knee valgus
    # measurement requires frontal-view video; side-view cannot determine it.
    # Crossley knee_valgus criterion is also raised above to reflect this.
    if knee_valgus < 5:
        pass   # all cases pass — valgus scoring inactive until real geometry available

    # FIXED: Trunk lean thresholds raised. Forward lean is biomechanically
    # normal in single-leg squats, especially pistol variants. Penalties
    # are halved — trunk lean should not dominate the score.
    if torso_lean < 15:       # was < 8 — any slight lean triggered penalty
        pass
    elif torso_lean <= 30:    # was <= 20
        score -= 4            # was -7 — halved
        suggestions.append(
            f"Moderate trunk lean ({torso_lean:.0f}°) — engage core and keep "
            "chest tall. Ankle dorsiflexion and hip mobility may be limiting upright posture."
        )
    elif torso_lean <= 45:    # was <= 35
        score -= 7            # was -12 — halved
        suggestions.append(
            f"Excessive forward lean ({torso_lean:.0f}°) — compensating for "
            "limited ankle mobility. Practice wall ankle stretches and goblet squats."
        )
    else:
        score -= 10           # was -15 — reduced
        suggestions.append(
            f"Severe forward lean ({torso_lean:.0f}°) — overloading lumbar spine. "
            "Improve thoracic mobility, core anti-flexion strength, and ankle dorsiflexion."
        )

    # FIXED: Pelvic drop thresholds raised. MediaPipe side-view noise alone
    # can generate 6–10 index units without any real pelvic drop occurring.
    if hip_drop < 12:         # was < 5 — caught sensor noise as a fault
        pass
    elif hip_drop <= 20:      # was <= 10
        score -= 8            # was -10
        suggestions.append(
            f"Pelvic drop detected (index {hip_drop:.1f}) — weak glute medius. "
            "Add clamshells, lateral band walks, and side-lying hip abduction."
        )
    else:                     # was > 10
        score -= 16           # was -20
        suggestions.append(
            f"Significant pelvic drop (index {hip_drop:.1f}) — major glute medius "
            "weakness. High risk for PFPS and IT band syndrome. "
            "Prioritise hip abductor strengthening before progressing load."
        )

    if heel_lift < 1:
        pass
    elif heel_lift <= 3:
        score -= 5
        suggestions.append(
            "Heel lifting from ground — limited ankle dorsiflexion. "
            "Keep heel firmly planted. Do daily ankle mobility drills."
        )
    else:
        score -= 10
        suggestions.append(
            "Significant heel lift detected — poor ankle dorsiflexion severely "
            "limits squat mechanics. Do banded ankle stretches and calf stretches daily."
        )

    # FIXED: Sway thresholds raised. std-dev of smoothed knee angles naturally
    # runs 3–6 even in professional athletes due to controlled movement.
    if balance_sway < 8:      # was < 3
        pass
    elif balance_sway <= 14:  # was <= 6
        score -= 4            # was -5
        suggestions.append(
            "Moderate balance instability — practice single-leg balance drills "
            "(eyes open then closed) and proprioceptive training on unstable surfaces."
        )
    else:
        score -= 8            # was -10
        suggestions.append(
            "High instability detected — build single-leg stance endurance (30s holds) "
            "and proprioception before adding squat depth."
        )

    if foot_pronation < 5:
        pass
    elif foot_pronation <= 10:
        score -= 3
        suggestions.append(
            "Mild foot pronation — arch collapsing inward. "
            "Strengthen foot intrinsics with towel scrunches and short-foot drills."
        )
    else:
        score -= 7
        suggestions.append(
            "Foot arch collapse — contributing to knee valgus chain. "
            "Consider arch support, foot intrinsic strengthening, and footwear review."
        )

    if movement_smoothness == "smooth":
        pass
    elif movement_smoothness == "moderate":
        score -= 5
        suggestions.append(
            "Inconsistent movement tempo — use a 3-second descent cue for "
            "better eccentric muscle activation and joint control."
        )
    else:
        score -= 10
        suggestions.append(
            "Jerky/uncontrolled movement — slow down significantly. "
            "Eccentric control is critical for injury prevention and strength gains."
        )

    score    = max(0, min(100, score))
    # FIXED: Category thresholds lowered. With the original penalties, a
    # textbook rep could still score 73 (under the old 80 threshold) just
    # from mild lean + mild sway. Now 65+ = GOOD, 40+ = BAD, else POOR.
    category = "GOOD" if score >= 65 else ("BAD" if score >= 40 else "POOR")
    injuries     = _detect_injuries(
        knee_valgus=knee_valgus, torso_lean=torso_lean,
        hip_drop=hip_drop, heel_lift=heel_lift,
        balance_sway=balance_sway, knee_depth=knee_angle,
    )
    overall_risk = _overall_injury_risk(injuries)

    phase_notes = {
        "idle": (
            "Good starting position — feet hip-width apart, arms extended for balance."
            if torso_lean < 8 else
            "Starting position: keep spine neutral and shoulders back."
        ),
        "descent": (
            "Controlled eccentric loading observed."
            if movement_smoothness == "smooth" else
            "Descent phase: slow to 3 seconds for optimal muscle activation."
        ),
        "bottom": (
            f"Bottom position: knee at {knee_angle:.0f}°. "
            + ("Ideal depth achieved." if knee_angle <= 90 else
               "Increase flexion depth toward 90° over time.")
        ),
        "ascent":         "Ascent phase: drive through heel and extend hip fully.",
        "stabilisation": (
            "Good return to neutral."
            if balance_sway < 3 else
            "Work on stabilisation: pause 1s at top for proprioceptive training."
        ),
    }

    return {
        "category":          category,
        "score":             score,
        "suggestions":       suggestions,
        "crossley":          crossley,
        "crossley_positive": crossley_positive,
        "crossley_count":    crossley_count,
        "injuries":          injuries,
        "overall_risk":      overall_risk,
        "phase_notes":       phase_notes,
    }


def analyse_single_leg_squat(path, is_video, output_path=None,
                              session_id=None, source_filename="", progress_uid=None):

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    rep_count    = 0
    correct_reps = 0
    wrong_reps   = 0
    stage        = "up"

    knee_angles       = []
    hip_angles        = []
    pelvic_drops      = []
    frame_data        = []
    frame_data_detail = []
    smoother_L        = RollingMean(5)
    smoother_R        = RollingMean(5)
    lm_smoother       = LandmarkSmoother(alpha=0.40)
    wrong_events      = []
    classify_results  = []

    rep_min_knee    = 180
    rep_max_knee    = 0
    rep_worst_pelv  = 0.0
    rep_frame_knees = []
    rep_hip_angles  = []

    seen_standing        = False
    standing_frame_count = 0

    frames_in_down  = 0
    stuck_threshold = 120
    cooldown_frames = 0
    COOLDOWN        = 6
    MIN_DOWN_FRAMES = 6

    total_frames = [0]
    video_fps    = [30.0]

    WARMUP_FRAMES  = 0   # unused — kept for FPS probe compatibility
    warmup_done    = False
    thresholds_set = False
    down_thresh    = 115.0
    up_thresh      = 140.0
    baseline_knee  = 160.0

    POST_WARMUP_CONFIRM_FRAMES     = 3
    post_warmup_standing_confirmed = False
    post_warmup_standing_frames    = 0

    # Leg locked during a rep — prevents mid-rep leg-switch resets
    leg_locked_during_rep = False

    active_leg        = "left"
    LEG_SWITCH_WINDOW = 20
    leg_vote_buffer   = []
    MIN_LEG_VIS       = 0.20
    leg_rep_counts    = {"left": 0, "right": 0}

    # Spike rejection: track previous knee value to detect occlusion jumps
    prev_knee_k  = None
    SPIKE_THRESH = 35.0   # degrees per frame — larger = occlusion spike

    log_lines = []
    os.makedirs("logs", exist_ok=True)
    log_path  = os.path.join("logs", "single_leg_squat_debug.txt")

    def log(msg):
        log_lines.append(msg)

    def detect_active_leg(lm):
        l_ok = (lm[27].visibility >= MIN_LEG_VIS and lm[25].visibility >= MIN_LEG_VIS)
        r_ok = (lm[28].visibility >= MIN_LEG_VIS and lm[26].visibility >= MIN_LEG_VIS)
        if not l_ok and not r_ok:
            return active_leg
        if l_ok and not r_ok:
            return "left"
        if r_ok and not l_ok:
            return "right"
        delta_y = lm[27].y - lm[28].y
        if abs(delta_y) >= 0.04:
            return "left" if delta_y > 0 else "right"
        try:
            l_ang = get_locked_angle(lm, "left",  min_vis=0.10)
            r_ang = get_locked_angle(lm, "right", min_vis=0.10)
            if l_ang is None and r_ang is None:
                return active_leg
            if l_ang is None:
                return "right"
            if r_ang is None:
                return "left"
            return "left" if l_ang <= r_ang else "right"
        except Exception:
            return active_leg

    def get_side_landmarks(lm, side):
        idx = _SIDE_LANDMARKS[side]
        return (
            [lm[idx["hip"]].x,      lm[idx["hip"]].y],
            [lm[idx["knee"]].x,     lm[idx["knee"]].y],
            [lm[idx["ankle"]].x,    lm[idx["ankle"]].y],
            [lm[idx["shoulder"]].x, lm[idx["shoulder"]].y],
            lm[idx["knee_lm"]],
        )

    def estimate_smoothness(angle_list):
        if len(angle_list) < 4:
            return "smooth"
        diffs = [abs(angle_list[i] - angle_list[i-1]) for i in range(1, len(angle_list))]
        jerk  = np.mean(diffs) + np.std(diffs)
        return "smooth" if jerk < 2.5 else ("moderate" if jerk < 5.0 else "jerky")

    def finalise_rep(fc, forced=False):
        nonlocal rep_count, correct_reps, wrong_reps, stage
        nonlocal rep_min_knee, rep_max_knee, rep_worst_pelv
        nonlocal frames_in_down, cooldown_frames, leg_locked_during_rep
        nonlocal rep_frame_knees, rep_hip_angles

        rep_count      += 1
        leg_rep_counts[active_leg] += 1
        stage           = "up"
        frames_in_down  = 0
        cooldown_frames = COOLDOWN
        leg_locked_during_rep = False

        depth_ok    = rep_min_knee < (down_thresh + 5)
        lockout_ok  = (rep_max_knee > rep_min_knee + REP_RISE_DEG * 0.8) and not forced
        # FIXED: Pelvic drop threshold raised from 10 to 20 to match new scoring.
        pelv_ok     = rep_worst_pelv < 20          # was < 10 — noise alone could fail this
        # FIXED: Valgus proxy multiplier reduced (was 1.5, causing double-counting
        # with pelvic drop). Capped at 15 — above that we still don't show a warning
        # since it's not real geometry from side view.
        knee_valgus_cls  = min(14.0, rep_worst_pelv * 0.8)   # was min(25, pelv*1.5)

        # FIXED: rep_correct uses score-based logic (>= 60) instead of hard AND.
        # A single tiny pelv/lockout issue was failing the whole rep — too binary.
        avg_hip          = float(np.mean(rep_hip_angles)) if rep_hip_angles else 90.0
        torso_lean_cls   = max(0.0, 180.0 - avg_hip)
        balance_sway_cls = float(np.std(rep_frame_knees)) if len(rep_frame_knees) > 1 else 0.0
        smoothness_cls   = estimate_smoothness(rep_frame_knees)

        cls = classify_single_leg_squat(
            knee_angle=rep_min_knee, knee_valgus=knee_valgus_cls,
            torso_lean=torso_lean_cls, hip_drop=rep_worst_pelv,
            heel_lift=0.0, balance_sway=balance_sway_cls,
            foot_pronation=0.0, movement_smoothness=smoothness_cls,
        )
        classify_results.append(cls)

        # Score-based correct/wrong: >= 60 is correct. This avoids one small
        # boolean flag failing an otherwise excellent rep.
        rep_correct = cls["score"] >= 60

        log(f"  REP {rep_count} [{active_leg}]: min={rep_min_knee:.1f} "
            f"max={rep_max_knee:.1f} rise={rep_max_knee-rep_min_knee:.1f} "
            f"pelv={rep_worst_pelv:.1f} score={cls['score']} correct={rep_correct} "
            f"forced={forced} cat={cls['category']} risk={cls['overall_risk']}")

        if rep_correct:
            correct_reps += 1
        else:
            wrong_reps += 1
            # Log specific faults for wrong_events even though score drives correct/wrong
            if rep_min_knee >= (down_thresh + 5):
                wrong_events.append({
                    "frame": fc, "joint": "knee_depth",
                    "angle_deg": round(rep_min_knee, 1),
                    "note": f"Insufficient depth — {rep_min_knee:.0f}° (need < {down_thresh+5:.0f}°)"
                })
            if forced and (rep_max_knee - rep_min_knee) < REP_RISE_DEG * 0.8:
                wrong_events.append({
                    "frame": fc, "joint": "knee_lockout",
                    "angle_deg": round(rep_max_knee, 1),
                    "note": (f"Did not return to top — only rose {rep_max_knee-rep_min_knee:.0f}° "
                             f"(need {REP_RISE_DEG*0.8:.0f}°+) [incomplete]")
                })
            if rep_worst_pelv >= 20:
                wrong_events.append({
                    "frame": fc, "joint": "pelvis",
                    "angle_deg": round(rep_worst_pelv, 1),
                    "note": f"Pelvic drop {rep_worst_pelv:.1f} (limit 20) — strengthen hip abductors"
                })

        inj_short = ", ".join(
            f"{inj['name'].split('(')[0].split('/')[0].strip()} ({inj['risk_level'][:3]})"
            for inj in cls["injuries"]
        ) if cls["injuries"] else "None"

        top_sugg = cls["suggestions"][:2]
        sugg_str = (" · ".join(s[:60] + ("…" if len(s) > 60 else "") for s in top_sugg)
                    if top_sugg else "Good form ✓")

        crossley_flags = ", ".join(
            k.replace("_", " ").title()
            for k, v in cls["crossley"].items() if v
        ) or "None"

        frame_data.append({
            "rep":          rep_count,
            "leg":          active_leg[0].upper(),
            "knee_depth":   f"{rep_min_knee:.0f}°",
            "knee_valgus":  f"{knee_valgus_cls:.0f}°",
            "trunk_lean":   f"{torso_lean_cls:.0f}°",
            "pelvic_drop":  f"{rep_worst_pelv:.1f}",
            "balance_sway": f"{balance_sway_cls:.1f}",
            "quality":      cls["category"],
            "score":        f"{cls['score']}/100",
            "correct":      "✓" if rep_correct else "✗",
            "crossley":     crossley_flags,
            "injury_risk":  cls["overall_risk"],
            "injuries":     inj_short,
            "corrections":  sugg_str,
        })

        frame_data_detail.append({
            "rep":              rep_count,
            "leg":              active_leg,
            "min_knee_raw":     round(rep_min_knee, 1),
            "max_knee_raw":     round(rep_max_knee, 1),
            "rise":             round(rep_max_knee - rep_min_knee, 1),
            "depth_ok":         rep_min_knee < (down_thresh + 5),
            "lockout_ok":       (rep_max_knee - rep_min_knee) >= REP_RISE_DEG * 0.8,
            "pelv_ok":          rep_worst_pelv < 20,
            "forced":           forced,
            "crossley_detail":  cls["crossley"],
            "injuries_detail":  cls["injuries"],
            "suggestions_full": cls["suggestions"],
            "phase_notes":      cls["phase_notes"],
            "cls_score_raw":    cls["score"],
        })

        rep_min_knee    = 180
        rep_max_knee    = 0
        rep_worst_pelv  = 0.0
        rep_frame_knees = []
        rep_hip_angles  = []

    def reset_for_leg_switch(fc, new_leg):
        nonlocal active_leg, stage, frames_in_down, cooldown_frames
        nonlocal rep_min_knee, rep_max_knee, rep_worst_pelv
        nonlocal rep_frame_knees, rep_hip_angles
        nonlocal seen_standing, standing_frame_count
        nonlocal post_warmup_standing_confirmed, post_warmup_standing_frames
        nonlocal leg_locked_during_rep
        log(f"  → LEG SWITCH fc={fc}: {active_leg} → {new_leg}")
        active_leg                     = new_leg
        stage                          = "up"
        frames_in_down                 = 0
        cooldown_frames                = COOLDOWN
        rep_min_knee                   = 180
        rep_max_knee                   = 0
        rep_worst_pelv                 = 0.0
        rep_frame_knees                = []
        rep_hip_angles                 = []
        seen_standing                  = False
        standing_frame_count           = 0
        post_warmup_standing_confirmed = False
        post_warmup_standing_frames    = 0
        leg_locked_during_rep          = False

    def pf(frame, fc, total):
        nonlocal stage, frames_in_down, seen_standing
        nonlocal standing_frame_count, cooldown_frames
        nonlocal rep_min_knee, rep_max_knee, rep_worst_pelv
        nonlocal thresholds_set, down_thresh, up_thresh, baseline_knee
        nonlocal rep_frame_knees, rep_hip_angles, warmup_done
        nonlocal active_leg, leg_vote_buffer, WARMUP_FRAMES
        nonlocal post_warmup_standing_confirmed, post_warmup_standing_frames
        nonlocal leg_locked_during_rep, prev_knee_k
        total_frames[0] = fc

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            return frame

        lm = res.pose_landmarks.landmark

        # ── Active leg detection via majority vote ────────────────
        # Leg is LOCKED during a rep — no resets mid-movement.
        # This fixes the side-profile occlusion over-count bug (Issue B).
        guess = detect_active_leg(lm)
        leg_vote_buffer.append(guess)
        if len(leg_vote_buffer) > LEG_SWITCH_WINDOW:
            leg_vote_buffer.pop(0)

        left_v   = leg_vote_buffer.count("left")
        right_v  = leg_vote_buffer.count("right")
        majority = "left" if left_v >= right_v else "right"

        majority_vis_ok = (
            (majority == "left"  and lm[25].visibility >= MIN_LEG_VIS
                                 and lm[27].visibility >= MIN_LEG_VIS) or
            (majority == "right" and lm[26].visibility >= MIN_LEG_VIS
                                 and lm[28].visibility >= MIN_LEG_VIS)
        )
        if (majority != active_leg
                and stage == "up"
                and not leg_locked_during_rep
                and len(leg_vote_buffer) == LEG_SWITCH_WINDOW
                and majority_vis_ok):
            reset_for_leg_switch(fc, majority)

        # ── Locked angle calculation ──────────────────────────────
        _, _, _, _, kne_lm = get_side_landmarks(lm, active_leg)
        smoother = smoother_L if active_leg == "left" else smoother_R

        raw_k = get_locked_angle(lm, active_leg, min_vis=MIN_LEG_VIS)

        # Spike rejection: if landmark glitches (arm over knee, brace),
        # the angle can jump >SPIKE_THRESH in one frame. Discard the spike
        # and hold the previous smoothed value. Fixes Issue C (phantom reps
        # from arm-over-knee occlusion during toe-hold pistol squats).
        if raw_k is not None and prev_knee_k is not None:
            if abs(raw_k - prev_knee_k) > SPIKE_THRESH:
                log(f"  SPIKE fc={fc} raw={raw_k:.1f} prev={prev_knee_k:.1f} — rejected")
                raw_k = None

        if raw_k is not None:
            knee_k = smoother.update(raw_k)
        else:
            knee_k = smoother.value if hasattr(smoother, "value") else 160.0
        prev_knee_k = knee_k
        knee_angles.append(knee_k)

        raw_hip = get_locked_hip_angle(lm, active_leg, min_vis=MIN_LEG_VIS)
        hip = raw_hip if raw_hip is not None else (hip_angles[-1] if hip_angles else 90.0)
        hip_angles.append(hip)

        pelvic = round(abs(lm[23].y - lm[24].y) * 100, 2)
        pelvic_drops.append(pelvic)

        # ── Instant threshold setup (no warmup delay) ────────────
        # Thresholds are set on the very first frame using safe defaults.
        # The first frame's knee angle is sampled to pick between
        # "standing" defaults (160° baseline) and "already squatting"
        # defaults (person starts mid-rep).
        if not thresholds_set:
            first_angle = get_locked_angle(lm, active_leg, min_vis=0.10)
            if first_angle is not None and first_angle < 130:
                # Person already mid-squat at frame 0 — use wider defaults
                baseline_knee = 160.0
                down_thresh   = 115.0
                up_thresh     = 140.0
                log(f"INSTANT thresholds (mid-squat start) down={down_thresh} up={up_thresh}")
            else:
                # Person standing — use adaptive defaults based on first angle
                baseline_knee = float(first_angle) if first_angle else 160.0
                baseline_knee = max(150.0, min(175.0, baseline_knee))
                down_thresh   = max(115, baseline_knee - 25)
                up_thresh     = max(down_thresh + 20, baseline_knee - 15)
                log(f"INSTANT thresholds base={baseline_knee:.1f} "
                    f"down={down_thresh:.1f} up={up_thresh:.1f}")
            thresholds_set = True
            warmup_done    = True
            # Pre-seed standing confirmation if person is clearly upright
            if first_angle is not None and first_angle > (up_thresh - 10):
                seen_standing = True
                post_warmup_standing_confirmed = True
                log("  seen_standing + confirmed pre-seeded from frame 0")

        # ── Per-rep trackers ──────────────────────────────────────
        if stage == "down":
            rep_min_knee   = min(rep_min_knee, knee_k)
            rep_worst_pelv = max(rep_worst_pelv, pelvic)
            rep_frame_knees.append(knee_k)
            rep_hip_angles.append(hip)

        if fc % 10 == 0:
            log(f"  fc={fc:04d} leg={active_leg} knee={knee_k:.1f} stage={stage} "
                f"seen={seen_standing} fd={frames_in_down} cool={cooldown_frames} "
                f"votes=L{left_v}:R{right_v}")

        bad_k    = (stage == "down" and frames_in_down > MIN_DOWN_FRAMES
                    and knee_k > down_thresh + 10)
        bad_pelv = pelvic > 8

        if knee_k > (up_thresh - 10):
            seen_standing = True

        if warmup_done and not post_warmup_standing_confirmed:
            if knee_k > up_thresh:
                post_warmup_standing_frames += 1
                if post_warmup_standing_frames >= POST_WARMUP_CONFIRM_FRAMES:
                    post_warmup_standing_confirmed = True
                    log(f"  post_warmup_standing_confirmed fc={fc}")
            else:
                post_warmup_standing_frames = 0

        if cooldown_frames > 0:
            cooldown_frames -= 1

        # ── State machine ─────────────────────────────────────────
        if stage == "up":
            if (knee_k < down_thresh and seen_standing
                    and post_warmup_standing_confirmed
                    and cooldown_frames == 0):
                stage                 = "down"
                frames_in_down        = 0
                leg_locked_during_rep = True   # lock leg for duration of rep
                rep_min_knee          = 180
                rep_max_knee          = knee_k
                rep_worst_pelv        = 0.0
                rep_frame_knees       = []
                rep_hip_angles        = []
                log(f"  → DOWN fc={fc} leg={active_leg} knee={knee_k:.1f} "
                    f"thresh={down_thresh:.1f}")

        elif stage == "down":
            frames_in_down += 1
            rep_max_knee    = max(rep_max_knee, knee_k)

            rose_enough  = (frames_in_down > MIN_DOWN_FRAMES
                            and knee_k > rep_min_knee + REP_RISE_DEG)
            above_thresh = (frames_in_down > MIN_DOWN_FRAMES
                            and knee_k > up_thresh)
            rep_complete = rose_enough or above_thresh

            if rep_complete:
                log(f"  → LOCKOUT fc={fc} knee={knee_k:.1f} "
                    f"rise={knee_k-rep_min_knee:.1f} fd={frames_in_down}")
                finalise_rep(fc, forced=False)
            elif frames_in_down >= stuck_threshold:
                log(f"  → STUCK FORCED fc={fc} fd={frames_in_down}")
                finalise_rep(fc, forced=True)

        # ── Drawing ───────────────────────────────────────────────
        smooth_pts = lm_smoother.smooth(lm, frame.shape[1], frame.shape[0])
        draw_pose_bicolor(frame, res.pose_landmarks, active_leg=active_leg,
                          bad=(bad_k or bad_pelv), pts=smooth_pts)
        draw_angle_arc_colored(frame, kne_lm, knee_k, active_leg=active_leg,
                               bad=(bad_k or bad_pelv))
        draw_footer_hud(frame, [
            ("REPS",    str(rep_count)),
            ("CORRECT", str(correct_reps)),
            ("WRONG",   str(wrong_reps)),
            ("LEG",     active_leg[0].upper()),
        ])
        draw_pcl_logo(frame)
        return frame

    # ── Run ───────────────────────────────────────────────────────
    log(f"=== single_leg_squat v16 | session={session_id} | file={source_filename} ===")

    try:
        cap_probe = cv2.VideoCapture(path)
        fps_probe  = cap_probe.get(cv2.CAP_PROP_FPS)
        cap_probe.release()
        if fps_probe and fps_probe > 0:
            WARMUP_FRAMES = max(20, int(fps_probe * WARMUP_SECONDS))
            video_fps[0]  = fps_probe
            log(f"FPS={fps_probe:.0f} → WARMUP_FRAMES={WARMUP_FRAMES}")
    except Exception:
        WARMUP_FRAMES = 45

    snaps = process_video_or_image(path, is_video, pf, output_path=output_path,
                                   analysis_skip=1, progress_uid=progress_uid)
    pose.close()

    if stage == "down" and rep_min_knee < down_thresh + 10 and frames_in_down >= MIN_DOWN_FRAMES:
        log(f"  → END-OF-VIDEO FORCE fc={total_frames[0]}")
        finalise_rep(fc=total_frames[0], forced=True)

    l_reps = leg_rep_counts["left"]
    r_reps = leg_rep_counts["right"]
    log(f"=== FINAL: reps={rep_count} correct={correct_reps} wrong={wrong_reps} "
        f"L={l_reps} R={r_reps} ===")

    try:
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))
    except Exception:
        pass

    if session_id:
        save_wrong_angle_log("single_leg_squat", session_id, source_filename, wrong_events)

    if is_video and len(knee_angles) < 10:
        raise ValueError("No reliable single-leg squat motion detected.")

    avg_k = round(np.mean(knee_angles),  1) if knee_angles  else 0
    min_k = round(np.min(knee_angles),   1) if knee_angles  else 0
    avg_h = round(np.mean(hip_angles),   1) if hip_angles   else 0
    avg_p = round(np.mean(pelvic_drops), 2) if pelvic_drops else 0

    if classify_results:
        overall_cls_score    = round(np.mean([r["score"] for r in classify_results]))
        # FIXED: Match the per-rep category thresholds (65/40 instead of 80/50)
        overall_cls_category = ("GOOD" if overall_cls_score >= 65 else
                                "BAD" if overall_cls_score >= 40 else "POOR")
        seen_sugg, all_sugg = set(), []
        for r in classify_results:
            for s in r["suggestions"]:
                if s not in seen_sugg:
                    seen_sugg.add(s)
                    all_sugg.append(s)
        crossley_totals = {k: sum(1 for r in classify_results if r["crossley"].get(k))
                           for k in ["trunk_lean", "pelvic_drop", "knee_valgus",
                                     "hip_adduction", "loss_of_balance"]}
        n_reps_cls       = max(1, len(classify_results))
        crossley_summary = {k: v >= (n_reps_cls * 0.5) for k, v in crossley_totals.items()}
        crossley_positive_overall = any(crossley_summary.values())

        risk_order  = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        injury_map  = {}
        for r in classify_results:
            for inj in r.get("injuries", []):
                k = inj["name"]
                if k not in injury_map or (risk_order.get(inj["risk_level"], 0) >
                                            risk_order.get(injury_map[k]["risk_level"], 0)):
                    injury_map[k] = inj
        all_injuries        = list(injury_map.values())
        overall_injury_risk = _overall_injury_risk(all_injuries)

        all_remedies = {}
        for inj in all_injuries:
            all_remedies[inj["name"]] = {
                "risk_level": inj["risk_level"],
                "cause":      inj["cause"],
                "remedies":   inj["remedies"],
            }
    else:
        overall_cls_score         = 70
        overall_cls_category      = "AVERAGE"
        all_sugg                  = []
        crossley_summary          = {}
        crossley_positive_overall = False
        all_injuries              = []
        overall_injury_risk       = "LOW"
        all_remedies              = {}

    issues, strengths = [], []

    depth_target = down_thresh + 5
    if min_k > depth_target + 20:
        issues.append(f"Very shallow squat ({min_k}° min knee) — aim for ≈90° flexion")
    elif min_k > depth_target:
        issues.append(f"Insufficient squat depth ({min_k}°) — aim for 60°–90° knee flexion")
    elif min_k < 65:
        strengths.append(f"Excellent squat depth — pistol range ({min_k}°)")
    else:
        strengths.append(f"Good single-leg squat depth ({min_k}°)")

    # FIXED: Only flag pelvic drop above BAD threshold (20). Values 5-12 are
    # MediaPipe noise in side-view; flagging them creates false issues.
    if avg_p > 20:            # was > 10
        issues.append(f"Pelvic drop detected ({avg_p:.1f}) — strengthen glute medius")
    elif avg_p < 8:           # was < 4
        strengths.append(f"Excellent pelvic stability ({avg_p:.1f})")
    else:
        strengths.append(f"Good pelvic control ({avg_p:.1f})")

    torso_lean_est = max(0.0, 180.0 - avg_h)
    # FIXED: Only raise issues at the raised thresholds. Mild lean is not an issue.
    if torso_lean_est > 45:   # was > 35
        issues.append(f"Severe forward lean ({torso_lean_est:.0f}°) — improve core and ankle dorsiflexion")
    elif torso_lean_est > 30: # was > 20
        issues.append(f"Excessive forward torso lean ({torso_lean_est:.0f}°) — keep chest tall")
    else:
        strengths.append(f"Good torso position maintained ({torso_lean_est:.0f}° lean)")

    if correct_reps == rep_count and rep_count > 0:
        strengths.append("All reps correct!")
    elif correct_reps > 0:
        strengths.append(f"{correct_reps}/{rep_count} reps with good form")

    if l_reps > 0 and r_reps > 0:
        if abs(l_reps - r_reps) <= 1:
            strengths.append(f"Balanced reps: Left {l_reps} / Right {r_reps}")
        else:
            issues.append(f"Rep imbalance — Left: {l_reps}, Right: {r_reps}. Aim for equal reps.")

    if crossley_positive_overall:
        triggered = [k.replace("_", " ").title()
                     for k, v in crossley_summary.items() if v]
        issues.append(f"Positive SLS test — Crossley criteria: {', '.join(triggered)}")
    else:
        strengths.append("Negative SLS test — no Crossley pathologic criteria in majority of reps")

    if overall_injury_risk == "HIGH":
        issues.append("HIGH injury risk detected — see remedies below")
    elif overall_injury_risk == "MEDIUM":
        issues.append("MEDIUM injury risk — corrective exercises recommended")

    if overall_cls_category == "GOOD":
        strengths.append(f"Overall movement quality: GOOD ({overall_cls_score}/100)")
    elif overall_cls_category == "BAD":
        issues.append(f"Overall movement quality: BAD ({overall_cls_score}/100) — moderate compensations")
    else:
        issues.append(f"Overall movement quality: POOR ({overall_cls_score}/100) — focus on mobility and strength")

    for s in all_sugg:
        if s not in issues:
            issues.append(s)

    if not issues:
        issues = ["No major form issues detected — excellent form!"]

    rep_score    = (max(4, min(10, round(10 - (len(wrong_events) / rep_count) * 1.5)))
                    if rep_count > 0 else 4)
    cls_score_10 = round(overall_cls_score / 10)
    form_score   = max(4, min(10, round((rep_score + cls_score_10) / 2)))

    avg_balance_sway = float(np.mean(
        [float(r["balance_sway"]) for r in frame_data if r.get("balance_sway", "0") != "0"]
    )) if frame_data else 0.0
    control_score = max(1, min(10, round(10 - avg_balance_sway * 0.8)))

    return {
        "exercise":             "Single Leg Squat",
        "rep_count":            rep_count,
        "correct_reps":         correct_reps,
        "wrong_reps":           wrong_reps,
        "left_reps":            l_reps,
        "right_reps":           r_reps,
        "avg_knee_angle":       avg_k,
        "min_knee_angle":       min_k,
        "avg_hip_angle":        avg_h,
        "avg_pelvic_drop":      avg_p,
        "form_score":           form_score,
        "control_score":        control_score,
        "overall_category":     overall_cls_category,
        "overall_cls_score":    overall_cls_score,
        "crossley_summary":     crossley_summary,
        "crossley_positive":    crossley_positive_overall,
        "injury_risk":          overall_injury_risk,
        "injuries":             [f"{i['name']} ({i['risk_level']})" for i in all_injuries],
        "_injuries_detail":     all_injuries,
        "remedies":             all_remedies,
        "issues":               issues,
        "strengths":            strengths,
        "per_rep":              frame_data,
        "per_rep_detail":       frame_data_detail,
        "snapshots":            snaps,
        "wrong_angle_count":    len(wrong_events),
        "_wrong_events":        wrong_events,
        "_classify_results":    classify_results,
        "metrics": [
            {"label": "Total Reps",          "value": str(rep_count)},
            {"label": "Correct Reps",        "value": str(correct_reps)},
            {"label": "Wrong Reps",          "value": str(wrong_reps)},
            {"label": "Avg Knee Angle",      "value": f"{avg_k}°"},
            {"label": "Min Knee (Depth)",    "value": f"{min_k}°"},
            {"label": "Hip Angle",           "value": f"{avg_h}°"},
            {"label": "Pelvic Drop Index",   "value": f"{avg_p}"},
            {"label": "Form Score",          "value": f"{form_score}/10"},
            {"label": "Control & Stillness", "value": f"{control_score}/10"},
            {"label": "Movement Quality",    "value": overall_cls_category},
            {"label": "Injury Risk",         "value": overall_injury_risk},
            {"label": "SLS Test",            "value": "POSITIVE" if crossley_positive_overall else "NEGATIVE"},
        ],
    }
 

import cv2
import numpy as np
import os
import math
from collections import deque

from utils import (
    mp_pose,
    calculate_angle, draw_angle_arc,
    RollingMean, LandmarkSmoother,
    process_video_or_image, save_wrong_angle_log,
)
from hud_overlay import draw_pcl_logo

# ─────────────────────────────────────────────────────────────────
# Constants  (tuned for robustness — not strictness)
# ─────────────────────────────────────────────────────────────────
SPIKE_THRESH   = 35.0     # per-frame spike rejection (degrees)
MIN_LEG_VIS    = 0.18     # relaxed — more frames survive
LEG_VOTE_WIN   = 8        # faster leg-switch detection

# Multi-signal thresholds (normalised units unless noted)
KNEE_DROP_THRESH      = 18.0   # degrees below baseline to start descent
HIP_DROP_THRESH       = 0.06   # fraction of body-height
NOSE_DROP_THRESH      = 0.04   # fraction of body-height

# FSM
BOTTOM_VELOCITY_WIN   = 4      # frames to check velocity sign change
ASCENT_KNEE_RISE      = 12.0   # degrees rise from min to call it ascending
ASCENT_HIP_RISE       = 0.03   # hip rising (normalised)
STANDING_KNEE_DROP    = 10.0   # max knee drop to re-enter STANDING

# Rep validation (rep_score >= 1 counts)
MIN_REP_KNEE_DROP     = 15.0   # degrees below baseline to score knee point
MIN_REP_HIP_DROP      = 0.04   # normalised hip drop
MIN_REP_NOSE_DROP     = 0.03   # normalised nose drop

# BGR colours
_C_ACTIVE_GOOD = (0,   220, 120)
_C_ACTIVE_BAD  = (0,   60,  230)
_C_INACTIVE    = (55,  55,  55)
_C_TORSO       = (200, 200, 200)
_C_ARM         = (100, 200, 80)
_NAVY          = (0x5b, 0x1f, 0x00)
_WHITE         = (255, 255, 255)
_GREY          = (170, 170, 170)
_DIVIDER       = (80,  50,  15)

# MediaPipe landmark index sets
_LM_LEFT_LEG   = [23, 25, 27, 29, 31]
_LM_RIGHT_LEG  = [24, 26, 28, 30, 32]
_LM_LEFT_ARM   = [11, 13, 15]
_LM_RIGHT_ARM  = [12, 14, 16]
_LM_TORSO      = [11, 12, 23, 24]

_CONN_LEFT_LEG  = [(23,25),(25,27),(27,29),(27,31),(29,31)]
_CONN_RIGHT_LEG = [(24,26),(26,28),(28,30),(28,32),(30,32)]
_CONN_LEFT_ARM  = [(11,13),(13,15)]
_CONN_RIGHT_ARM = [(12,14),(14,16)]
_CONN_TORSO     = [(11,12),(11,23),(12,24),(23,24)]

_SIDE = {
    "left":  {"hip": 23, "knee": 25, "ankle": 27, "heel": 29,
              "shoulder": 11, "knee_lm": 25,
              "opp_hip": 24, "opp_knee": 26, "opp_ankle": 28},
    "right": {"hip": 24, "knee": 26, "ankle": 28, "heel": 30,
              "shoulder": 12, "knee_lm": 26,
              "opp_hip": 23, "opp_knee": 25, "opp_ankle": 27},
}

# ─────────────────────────────────────────────────────────────────
# Quality thresholds  (relaxed — practical coaching)
# ─────────────────────────────────────────────────────────────────
CHAIR_DEPTH_GOOD  = 120
CHAIR_DEPTH_BAD   = 135
PISTOL_DEPTH_GOOD = 90
PISTOL_DEPTH_BAD  = 110

PELV_GOOD = 8
PELV_BAD  = 14

TRUNK_LEAN_GOOD_CHAIR  = 25
TRUNK_LEAN_BAD_CHAIR   = 40
TRUNK_LEAN_GOOD_PISTOL = 35
TRUNK_LEAN_BAD_PISTOL  = 50

SWAY_GOOD = 6
SWAY_BAD  = 12

VALGUS_GOOD = 8
VALGUS_BAD  = 16


# ─────────────────────────────────────────────────────────────────
# Injury DB
# ─────────────────────────────────────────────────────────────────
_INJURY_DB = {
    "ACL": {
        "name": "ACL Tear Risk",
        "cause": (
            "Dynamic knee valgus (inward knee collapse) increases anterior "
            "tibial shear and ACL tensile load."
        ),
        "remedies": [
            "Glute bridge 3×15 — activates glute max and medius",
            "Clamshells with resistance band 3×20 — isolates glute medius",
            "Lateral band walks 3×15 steps — hip abductor strengthening",
            "Mirror single-leg squat drill — visual feedback on knee position",
        ],
    },
    "PFPS": {
        "name": "Patellofemoral Pain Syndrome",
        "cause": "Excessive hip adduction/internal rotation shifts patella laterally.",
        "remedies": [
            "Vastus medialis activation drills (terminal knee extension with band)",
            "Hip external rotation strengthening — side-lying leg raises",
            "Step-down eccentric exercise 3×10 — patellar tendon loading",
        ],
    },
    "LOWER_BACK": {
        "name": "Lower Back Pain / Lumbar Overload",
        "cause": "Severe forward trunk lean overloads lumbar erectors.",
        "remedies": [
            "Hip hinge pattern retraining — Romanian deadlift with dowel cue",
            "Core stability — dead bug, bird-dog progressions",
            "Ankle mobility work to allow upright squat posture",
        ],
    },
    "ANKLE_INSTABILITY": {
        "name": "Ankle Instability",
        "cause": "Heel lift and balance sway indicate poor ankle control.",
        "remedies": [
            "Ankle dorsiflexion mobilisation — banded ankle distraction",
            "Single-leg balance 3×30s — eyes open then closed",
            "Calf raises (eccentric) 3×15",
        ],
    },
}


def _detect_injuries(valgus, trunk_lean, hip_drop, sway, ex_type):
    detected = []
    if valgus > VALGUS_BAD or hip_drop > PELV_BAD:
        lvl = "HIGH" if valgus > 22 or hip_drop > 18 else "MEDIUM"
        detected.append({**_INJURY_DB["ACL"], "risk_level": lvl})
    elif valgus > VALGUS_GOOD or hip_drop > PELV_GOOD:
        detected.append({**_INJURY_DB["ACL"], "risk_level": "LOW"})

    lean_bad = TRUNK_LEAN_BAD_PISTOL if ex_type == "pistol" else TRUNK_LEAN_BAD_CHAIR
    if trunk_lean > lean_bad:
        detected.append({**_INJURY_DB["LOWER_BACK"], "risk_level": "MEDIUM"})

    if sway > SWAY_BAD:
        detected.append({**_INJURY_DB["ANKLE_INSTABILITY"], "risk_level": "MEDIUM"})
    elif sway > SWAY_GOOD:
        detected.append({**_INJURY_DB["ANKLE_INSTABILITY"], "risk_level": "LOW"})

    seen, unique = set(), []
    for d in detected:
        k = d["name"]
        if k not in seen:
            seen.add(k)
            unique.append(d)
    return unique


def _overall_injury_risk(injuries):
    order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    if not injuries:
        return "LOW"
    return max((i["risk_level"] for i in injuries), key=lambda r: order.get(r, 0))


# ─────────────────────────────────────────────────────────────────
# Exercise type detection (chair vs pistol)
# ─────────────────────────────────────────────────────────────────
def _detect_exercise_type(lm, active_leg, min_vis=0.18):
    opp       = _SIDE[active_leg]
    opp_hip   = lm[opp["opp_hip"]]
    opp_knee  = lm[opp["opp_knee"]]
    opp_ankle = lm[opp["opp_ankle"]]

    if (opp_hip.visibility < min_vis or opp_knee.visibility < min_vis
            or opp_ankle.visibility < min_vis):
        return "chair"

    opp_knee_angle = calculate_angle(
        [opp_hip.x,   opp_hip.y],
        [opp_knee.x,  opp_knee.y],
        [opp_ankle.x, opp_ankle.y],
    )
    foot_raised = (opp_ankle.y < opp_hip.y - 0.05)
    if opp_knee_angle > 140 and foot_raised:
        return "pistol"
    return "chair"


# ─────────────────────────────────────────────────────────────────
# Rep quality classification
# ─────────────────────────────────────────────────────────────────
def _classify_rep(min_knee, hip_drop, sway, rep_hip_angles, ex_type):
    score       = 100
    suggestions = []

    depth_good = PISTOL_DEPTH_GOOD if ex_type == "pistol" else CHAIR_DEPTH_GOOD
    depth_bad  = PISTOL_DEPTH_BAD  if ex_type == "pistol" else CHAIR_DEPTH_BAD

    trunk_lean_good = TRUNK_LEAN_GOOD_PISTOL if ex_type == "pistol" else TRUNK_LEAN_GOOD_CHAIR
    trunk_lean_bad  = TRUNK_LEAN_BAD_PISTOL  if ex_type == "pistol" else TRUNK_LEAN_BAD_CHAIR

    avg_hip    = float(np.mean(rep_hip_angles)) if rep_hip_angles else 155.0
    trunk_lean = max(0.0, 180.0 - avg_hip)
    valgus     = min(30.0, hip_drop * 1.4)

    if min_knee <= depth_good:
        pass
    elif min_knee <= depth_bad:
        score -= 15
        target = "90°" if ex_type == "chair" else "deep below 90°"
        suggestions.append(
            f"Increase squat depth — aim for knee ≈{target}. "
            "Ankle mobility and hip flexibility drills will help."
        )
    else:
        score -= 30
        suggestions.append(
            "Very shallow squat — knee is barely bending. "
            "Focus on quad strengthening and ankle dorsiflexion."
        )

    if trunk_lean <= trunk_lean_good:
        pass
    elif trunk_lean <= trunk_lean_bad:
        score -= 10
        suggestions.append(
            f"Forward lean ({trunk_lean:.0f}°) — engage core, keep chest tall."
        )
    else:
        score -= 20
        suggestions.append(
            f"Excessive forward lean ({trunk_lean:.0f}°) — overloading lower back. "
            "Improve thoracic mobility and ankle dorsiflexion."
        )

    if hip_drop <= PELV_GOOD:
        pass
    elif hip_drop <= PELV_BAD:
        score -= 12
        suggestions.append(
            "Pelvic drop detected — hip tilting to one side. "
            "Strengthen glute medius: clamshells and lateral band walks."
        )
    else:
        score -= 22
        suggestions.append(
            "Significant pelvic drop — major glute medius weakness. "
            "Prioritise hip abductor work before adding depth."
        )

    if valgus > VALGUS_BAD:
        score -= 20
        suggestions.append(
            "Knee collapsing inward — push knee out over 2nd toe. Strengthen glutes urgently."
        )
    elif valgus > VALGUS_GOOD:
        score -= 10
        suggestions.append("Mild knee drift inward — keep knee tracking over 2nd–3rd toe.")

    if sway > SWAY_BAD:
        score -= 10
        suggestions.append(
            "High instability — build single-leg stance (30-second holds) before adding depth."
        )
    elif sway > SWAY_GOOD:
        score -= 5
        suggestions.append("Some balance wobble — slow descent and focus on a fixed point.")

    score    = max(0, min(100, score))
    category = "GOOD" if score >= 75 else ("BAD" if score >= 45 else "POOR")

    crossley = {
        "trunk_lean":      trunk_lean > 9,
        "pelvic_drop":     hip_drop   > 5,
        "knee_valgus":     valgus     > 10,
        "hip_adduction":   hip_drop   > 7,
        "loss_of_balance": sway       > 5,
    }

    injuries     = _detect_injuries(valgus, trunk_lean, hip_drop, sway, ex_type)
    overall_risk = _overall_injury_risk(injuries)

    phase_notes = {
        "descent": ("Controlled descent." if score > 75 else
                    "Slow your descent — aim for 2–3 seconds down."),
        "bottom":  (f"Knee at {min_knee:.0f}°. Good depth." if min_knee <= depth_good else
                    f"Knee at {min_knee:.0f}°. Aim deeper next rep."),
        "ascent":  "Drive through your heel — extend hip fully on the way up.",
    }

    return {
        "category":     category,
        "score":        score,
        "suggestions":  suggestions,
        "crossley":     crossley,
        "injuries":     injuries,
        "overall_risk": overall_risk,
        "phase_notes":  phase_notes,
        "trunk_lean":   trunk_lean,
        "valgus":       valgus,
    }


# ─────────────────────────────────────────────────────────────────
# Skeleton drawing helpers
# ─────────────────────────────────────────────────────────────────
def _draw_connections(frame, lm, connections, color, thickness, pts=None):
    h, w = frame.shape[:2]
    for a, b in connections:
        if lm[a].visibility > 0.18 and lm[b].visibility > 0.18:
            pa = (int(pts[a][0]), int(pts[a][1])) if pts is not None else (int(lm[a].x * w), int(lm[a].y * h))
            pb = (int(pts[b][0]), int(pts[b][1])) if pts is not None else (int(lm[b].x * w), int(lm[b].y * h))
            cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)


def _draw_keypoints(frame, lm, indices, color, radius, pts=None):
    h, w = frame.shape[:2]
    for i in indices:
        if lm[i].visibility > 0.18:
            cx, cy = (int(pts[i][0]), int(pts[i][1])) if pts is not None else (int(lm[i].x * w), int(lm[i].y * h))
            cv2.circle(frame, (cx, cy), radius + 2, (20, 20, 20),  -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius,     (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), max(1, radius-2), color,   -1, cv2.LINE_AA)


def draw_skeleton(frame, pose_landmarks, active_leg, bad=False, pts=None):
    if pose_landmarks is None:
        return
    lm = pose_landmarks.landmark
    c_active = _C_ACTIVE_BAD if bad else _C_ACTIVE_GOOD
    c_inact  = _C_INACTIVE

    if active_leg == "left":
        conn_act, lm_act  = _CONN_LEFT_LEG,  _LM_LEFT_LEG
        conn_in,  lm_in   = _CONN_RIGHT_LEG, _LM_RIGHT_LEG
    else:
        conn_act, lm_act  = _CONN_RIGHT_LEG, _LM_RIGHT_LEG
        conn_in,  lm_in   = _CONN_LEFT_LEG,  _LM_LEFT_LEG

    _draw_connections(frame, lm, _CONN_TORSO,    _C_TORSO, 2, pts)
    _draw_keypoints  (frame, lm, _LM_TORSO,      _C_TORSO, 4, pts)
    _draw_connections(frame, lm, _CONN_LEFT_ARM,  _C_ARM,   2, pts)
    _draw_connections(frame, lm, _CONN_RIGHT_ARM, _C_ARM,   2, pts)
    _draw_keypoints  (frame, lm, _LM_LEFT_ARM,    _C_ARM,   4, pts)
    _draw_keypoints  (frame, lm, _LM_RIGHT_ARM,   _C_ARM,   4, pts)
    _draw_connections(frame, lm, conn_in,  c_inact,  1, pts)
    _draw_keypoints  (frame, lm, lm_in,   c_inact,  3, pts)
    _draw_connections(frame, lm, conn_act, c_active, 3, pts)
    _draw_keypoints  (frame, lm, lm_act,  c_active, 6, pts)

    for i, (col, lbl) in enumerate([
        (c_active, f"ACTIVE ({active_leg.upper()})"),
        (c_inact,  "INACTIVE"),
    ]):
        y = 12 + i * 18
        cv2.rectangle(frame, (6, y), (18, y+12), col, -1)
        cv2.putText(frame, lbl, (22, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220,220,220), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
# Coaching overlay
# ─────────────────────────────────────────────────────────────────
_COACHING_DECAY = 90

def _draw_coaching(frame, text_lines, alpha=0.55):
    if not text_lines:
        return
    h, w  = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.32, min(0.44, h / 1080 * 0.44))
    thick = 1
    pad   = 6
    lh    = int(18 * (h / 720))
    box_h = pad*2 + lh * len(text_lines)
    max_w = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in text_lines)
    box_w = max_w + pad*2
    x0    = w - box_w - 8
    y0    = 8
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0+box_w, y0+box_h), (10,10,10), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for i, line in enumerate(text_lines):
        color = (80, 220, 80) if i == 0 else (200, 200, 200)
        cv2.putText(frame, line,
                    (x0+pad, y0+pad+lh*(i+1)-4),
                    font, scale, color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
# LHS panel (in-place)
# ─────────────────────────────────────────────────────────────────
def _draw_lhs_panel_inplace(frame, items, logo_h=60):
    h, w    = frame.shape[:2]
    panel_w = max(70, int(w * 0.13))
    frame[:, :panel_w] = _NAVY
    cv2.line(frame, (panel_w-1, 0), (panel_w-1, h), _DIVIDER, 2)

    n = len(items)
    if n == 0:
        return panel_w

    reserved_top = logo_h
    avail_h      = h - reserved_top
    cell_h       = max(40, avail_h // n)
    cx = panel_w // 2

    for idx, (lbl, val) in enumerate(items):
        y_top = reserved_top + idx * cell_h
        y_bot = y_top + cell_h
        if idx > 0:
            cv2.line(frame, (6, y_top), (panel_w-6, y_top), _DIVIDER, 1)
        cell_mid = (y_top + y_bot) // 2

        val_s  = str(val)
        vscale = max(0.30, min(0.65, panel_w / 80 * 0.60))
        vthick = 2
        (vw, vh), _ = cv2.getTextSize(val_s, cv2.FONT_HERSHEY_SIMPLEX, vscale, vthick)
        while vw > panel_w - 8 and vscale > 0.28:
            vscale -= 0.04
            (vw, vh), _ = cv2.getTextSize(val_s, cv2.FONT_HERSHEY_SIMPLEX, vscale, vthick)

        val_y = cell_mid - 4
        cv2.putText(frame, val_s,
                    (cx - vw//2, val_y),
                    cv2.FONT_HERSHEY_SIMPLEX, vscale, _WHITE, vthick, cv2.LINE_AA)

        lbl_s  = str(lbl).upper()
        lscale = max(0.24, min(0.34, panel_w / 80 * 0.32))
        (lw2, _), _ = cv2.getTextSize(lbl_s, cv2.FONT_HERSHEY_SIMPLEX, lscale, 1)
        lbl_y = val_y + vh + 10
        if lbl_y < y_bot - 2:
            cv2.putText(frame, lbl_s,
                        (cx - lw2//2, lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, lscale, _GREY, 1, cv2.LINE_AA)

    return panel_w


# ─────────────────────────────────────────────────────────────────
# Signal helpers
# ─────────────────────────────────────────────────────────────────
def _calc_3d_angle(wlm, hip_i, knee_i, ankle_i):
    """3D angle using world landmarks (metric space, rotation-robust)."""
    try:
        h = np.array([wlm[hip_i].x,   wlm[hip_i].y,   wlm[hip_i].z])
        k = np.array([wlm[knee_i].x,  wlm[knee_i].y,  wlm[knee_i].z])
        a = np.array([wlm[ankle_i].x, wlm[ankle_i].y, wlm[ankle_i].z])
        ba, bc = h - k, a - k
        cos_a  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    except Exception:
        return None


def _get_knee_angle_2d(lm, side, min_vis=MIN_LEG_VIS):
    s = _SIDE[side]
    h_, k, a = lm[s["hip"]], lm[s["knee"]], lm[s["ankle"]]
    if h_.visibility < min_vis or k.visibility < min_vis or a.visibility < min_vis:
        return None
    return calculate_angle([h_.x, h_.y], [k.x, k.y], [a.x, a.y])


def _get_knee_angle_best(lm, wlm, side, min_vis=MIN_LEG_VIS):
    """Use 3D world angle if world landmarks available, else 2D."""
    s = _SIDE[side]
    if wlm is not None:
        ang3d = _calc_3d_angle(wlm, s["hip"], s["knee"], s["ankle"])
        if ang3d is not None:
            return ang3d
    return _get_knee_angle_2d(lm, side, min_vis)


def _get_hip_angle(lm, side, min_vis=MIN_LEG_VIS):
    s = _SIDE[side]
    sh, h_, k = lm[s["shoulder"]], lm[s["hip"]], lm[s["knee"]]
    if sh.visibility < min_vis or h_.visibility < min_vis or k.visibility < min_vis:
        return None
    return calculate_angle([sh.x, sh.y], [h_.x, h_.y], [k.x, k.y])


def _pelvic_drop(lm):
    return round(abs(lm[23].y - lm[24].y) * 100, 2)


def _body_height(lm):
    """Normalised body height: mid-shoulder y to mid-ankle y (in normalised coords)."""
    try:
        sh_y = (lm[11].y + lm[12].y) / 2
        an_y = (lm[27].y + lm[28].y) / 2
        bh = abs(an_y - sh_y)
        return max(bh, 0.10)   # guard against bad pose
    except Exception:
        return 0.5


def _landmark_confidence(lm, indices):
    """Average visibility for a list of landmark indices."""
    vis = [lm[i].visibility for i in indices if i < len(lm)]
    return float(np.mean(vis)) if vis else 0.0


# ─────────────────────────────────────────────────────────────────
# Active leg detection  (excursion-based)
# ─────────────────────────────────────────────────────────────────
def _detect_active_leg_excursion(lm, wlm, baseline_L, baseline_R, current_leg, min_vis=MIN_LEG_VIS):
    """
    Whichever knee deviates more from its standing baseline is active.
    Falls back to visibility and ankle-height heuristics.
    """
    l_vis = lm[27].visibility >= min_vis and lm[25].visibility >= min_vis
    r_vis = lm[28].visibility >= min_vis and lm[26].visibility >= min_vis

    if l_vis and not r_vis:
        return "left"
    if r_vis and not l_vis:
        return "right"
    if not l_vis and not r_vis:
        return current_leg

    l_ang = _get_knee_angle_best(lm, wlm, "left",  min_vis=0.12)
    r_ang = _get_knee_angle_best(lm, wlm, "right", min_vis=0.12)

    if l_ang is not None and r_ang is not None and baseline_L > 0 and baseline_R > 0:
        l_drop = baseline_L - l_ang
        r_drop = baseline_R - r_ang
        if abs(l_drop - r_drop) > 5:
            return "left" if l_drop > r_drop else "right"

    # Fallback: ankle height (lower ankle = stance leg)
    delta_y = lm[27].y - lm[28].y
    if abs(delta_y) >= 0.04:
        return "left" if delta_y > 0 else "right"

    return current_leg


# ─────────────────────────────────────────────────────────────────
# FSM state label for HUD
# ─────────────────────────────────────────────────────────────────
_FSM_LABEL = {
    "STANDING":   "STAND",
    "DESCENDING": "DOWN",
    "BOTTOM":     "BOTM",
    "ASCENDING":  "UP",
}


# ─────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────
def analyse_single_leg_squat(
    path, is_video, output_path=None,
    session_id=None, source_filename="", progress_uid=None
):
    """
    Analyse Single-Leg Squat (Chair or Pistol variant) — v19 multi-signal FSM engine.
    """
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.45,
    )

    # ── Counters ──────────────────────────────────────────────────
    rep_count    = 0
    correct_reps = 0
    wrong_reps   = 0
    bad_reps     = 0
    poor_reps    = 0

    # 4-state FSM
    stage = "STANDING"   # STANDING | DESCENDING | BOTTOM | ASCENDING

    knee_angles     = []
    hip_angles_all  = []
    pelvic_drops    = []
    frame_data      = []
    frame_data_detail = []
    classify_results  = []
    wrong_events      = []

    # Smoothers
    smoother_L  = RollingMean(6)
    smoother_R  = RollingMean(6)
    lm_smoother = LandmarkSmoother(alpha=0.38)

    # ── Adaptive baseline buffers ─────────────────────────────────
    # Keep the last N standing-phase knee angles; baseline = 90th pct
    _STANDING_BUF_SIZE = 45
    standing_knee_buf_L = deque(maxlen=_STANDING_BUF_SIZE)
    standing_knee_buf_R = deque(maxlen=_STANDING_BUF_SIZE)
    baseline_knee_L = 160.0
    baseline_knee_R = 160.0

    # Hip and nose standing references (for multi-signal)
    standing_hip_buf  = deque(maxlen=_STANDING_BUF_SIZE)
    standing_nose_buf = deque(maxlen=_STANDING_BUF_SIZE)
    baseline_hip_y  = None
    baseline_nose_y = None

    # ── Per-rep accumulators ──────────────────────────────────────
    rep_min_knee    = 180.0
    rep_max_knee    = 0.0
    rep_worst_pelv  = 0.0
    rep_frame_knees = []
    rep_hip_angles  = []
    rep_hip_min_y   = None    # lowest hip y seen during rep (highest physically)
    rep_nose_min_y  = None
    rep_knee_drop   = 0.0     # max knee drop seen this rep
    rep_hip_drop    = 0.0     # max normalised hip drop seen
    rep_nose_drop   = 0.0     # max normalised nose drop

    # ── FSM velocity tracking (for BOTTOM detection) ──────────────
    hip_y_buf   = deque(maxlen=BOTTOM_VELOCITY_WIN + 2)
    knee_buf_fsm = deque(maxlen=BOTTOM_VELOCITY_WIN + 2)

    # ── Timing ────────────────────────────────────────────────────
    frames_in_stage  = 0
    cooldown_frames  = 0
    stuck_threshold  = 150

    # FPS-aware timing (computed after probing video)
    video_fps        = 30.0
    COOLDOWN         = 8     # will be updated after FPS probe
    MIN_DOWN_FRAMES  = 6     # frames needed in DESCENDING before BOTTOM

    POST_CONFIRM_FRAMES            = 3
    post_warmup_standing_confirmed = False
    post_warmup_standing_frames    = 0
    seen_standing                  = False

    # ── Leg detection ─────────────────────────────────────────────
    active_leg            = "left"
    leg_vote_buffer       = deque(maxlen=LEG_VOTE_WIN)
    leg_locked_during_rep = False
    leg_rep_counts        = {"left": 0, "right": 0}

    # ── Spike rejection ───────────────────────────────────────────
    prev_knee_k = None

    # ── Coaching overlay ──────────────────────────────────────────
    last_coaching   = []
    coaching_frames = 0

    # ── Exercise type ─────────────────────────────────────────────
    exercise_type_votes = []
    ex_type_current     = "chair"

    total_frames = [0]
    _current_lm  = [None]
    _current_wlm = [None]

    log_lines = []
    os.makedirs("logs", exist_ok=True)
    log_path  = os.path.join("logs", "single_leg_squat_debug.txt")

    def log(msg):
        log_lines.append(msg)

    # ── Rep finalisation ──────────────────────────────────────────
    def finalise_rep(fc, forced=False):
        nonlocal rep_count, correct_reps, wrong_reps, bad_reps, poor_reps
        nonlocal stage, frames_in_stage, cooldown_frames, leg_locked_during_rep
        nonlocal rep_min_knee, rep_max_knee, rep_worst_pelv
        nonlocal rep_frame_knees, rep_hip_angles
        nonlocal rep_hip_min_y, rep_nose_min_y
        nonlocal rep_knee_drop, rep_hip_drop, rep_nose_drop
        nonlocal last_coaching, coaching_frames

        rep_count += 1
        leg_rep_counts[active_leg] += 1
        stage                 = "STANDING"
        frames_in_stage       = 0
        cooldown_frames       = COOLDOWN
        leg_locked_during_rep = False

        rep_ex_type = _detect_exercise_type(_current_lm[0], active_leg) if _current_lm[0] else "chair"
        exercise_type_votes.append(rep_ex_type)

        sway = float(np.std(rep_frame_knees)) if len(rep_frame_knees) > 1 else 0.0

        cls = _classify_rep(
            min_knee       = rep_min_knee,
            hip_drop       = rep_worst_pelv,
            sway           = sway,
            rep_hip_angles = rep_hip_angles,
            ex_type        = rep_ex_type,
        )
        classify_results.append(cls)

        category = cls["category"]
        if category == "GOOD":
            correct_reps += 1
        elif category == "BAD":
            wrong_reps += 1; bad_reps += 1
        else:
            wrong_reps += 1; poor_reps += 1

        depth_target = PISTOL_DEPTH_BAD if rep_ex_type == "pistol" else CHAIR_DEPTH_BAD
        if rep_min_knee > depth_target:
            wrong_events.append({
                "frame": fc, "joint": "knee_depth",
                "angle_deg": round(rep_min_knee, 1),
                "note": f"Shallow depth {rep_min_knee:.0f}° ({rep_ex_type})",
            })
        if rep_worst_pelv > PELV_BAD:
            wrong_events.append({
                "frame": fc, "joint": "pelvis",
                "angle_deg": round(rep_worst_pelv, 1),
                "note": f"Pelvic drop {rep_worst_pelv:.1f}",
            })

        top_sugg = cls["suggestions"][:2]
        if category == "GOOD":
            header = f"Rep {rep_count}: GOOD \u2713"
        elif category == "BAD":
            header = f"Rep {rep_count}: BAD \u2014 improve form"
        else:
            header = f"Rep {rep_count}: POOR \u2014 major fault"

        last_coaching   = [header] + ([top_sugg[0][:55]] if top_sugg else [])
        coaching_frames = _COACHING_DECAY

        log(
            f"  REP {rep_count} [{active_leg}|{rep_ex_type}] "
            f"min={rep_min_knee:.1f} max={rep_max_knee:.1f} "
            f"knee_drop={rep_knee_drop:.1f} hip_drop={rep_hip_drop:.3f} "
            f"pelv={rep_worst_pelv:.1f} sway={sway:.1f} "
            f"cat={category} score={cls['score']} forced={forced}"
        )

        inj_short = ", ".join(
            f"{inj['name'].split('/')[0].strip()} ({inj['risk_level'][:3]})"
            for inj in cls["injuries"]
        ) or "None"

        top2_str = " · ".join(
            s[:55] + ("\u2026" if len(s) > 55 else "") for s in top_sugg
        ) or "Good form \u2713"

        crossley_flags = ", ".join(
            k.replace("_", " ").title() for k, v in cls["crossley"].items() if v
        ) or "None"

        frame_data.append({
            "rep":         rep_count,
            "leg":         active_leg[0].upper(),
            "type":        rep_ex_type,
            "knee_depth":  f"{rep_min_knee:.0f}\u00b0",
            "trunk_lean":  f"{cls['trunk_lean']:.0f}\u00b0",
            "pelvic_drop": f"{rep_worst_pelv:.1f}",
            "knee_valgus": f"{cls['valgus']:.0f}\u00b0",
            "sway":        f"{sway:.1f}",
            "quality":     category,
            "score":       f"{cls['score']}/100",
            "correct":     "\u2713" if category == "GOOD" else "\u2717",
            "crossley":    crossley_flags,
            "injury_risk": cls["overall_risk"],
            "injuries":    inj_short,
            "corrections": top2_str,
        })

        frame_data_detail.append({
            "rep":              rep_count,
            "leg":              active_leg,
            "exercise_type":    rep_ex_type,
            "min_knee":         round(rep_min_knee, 1),
            "max_knee":         round(rep_max_knee, 1),
            "rise":             round(rep_max_knee - rep_min_knee, 1),
            "pelvic_drop":      round(rep_worst_pelv, 1),
            "sway":             round(sway, 1),
            "trunk_lean":       round(cls["trunk_lean"], 1),
            "valgus_proxy":     round(cls["valgus"], 1),
            "forced":           forced,
            "category":         category,
            "score":            cls["score"],
            "crossley_detail":  cls["crossley"],
            "injuries_detail":  cls["injuries"],
            "suggestions_full": cls["suggestions"],
            "phase_notes":      cls["phase_notes"],
        })

        # Reset per-rep state
        rep_min_knee    = 180.0
        rep_max_knee    = 0.0
        rep_worst_pelv  = 0.0
        rep_frame_knees = []
        rep_hip_angles  = []
        rep_hip_min_y   = None
        rep_nose_min_y  = None
        rep_knee_drop   = 0.0
        rep_hip_drop    = 0.0
        rep_nose_drop   = 0.0

    def reset_for_leg_switch(fc, new_leg):
        nonlocal active_leg, stage, frames_in_stage, cooldown_frames
        nonlocal rep_min_knee, rep_max_knee, rep_worst_pelv
        nonlocal rep_frame_knees, rep_hip_angles
        nonlocal rep_hip_min_y, rep_nose_min_y
        nonlocal rep_knee_drop, rep_hip_drop, rep_nose_drop
        nonlocal seen_standing, post_warmup_standing_confirmed
        nonlocal post_warmup_standing_frames, leg_locked_during_rep
        log(f"  \u2192 LEG SWITCH fc={fc}: {active_leg} \u2192 {new_leg}")
        active_leg                   = new_leg
        stage                        = "STANDING"
        frames_in_stage              = 0
        cooldown_frames              = COOLDOWN
        rep_min_knee                 = 180.0
        rep_max_knee                 = 0.0
        rep_worst_pelv               = 0.0
        rep_frame_knees              = []
        rep_hip_angles               = []
        rep_hip_min_y                = None
        rep_nose_min_y               = None
        rep_knee_drop                = 0.0
        rep_hip_drop                 = 0.0
        rep_nose_drop                = 0.0
        seen_standing                = False
        post_warmup_standing_confirmed = False
        post_warmup_standing_frames  = 0
        leg_locked_during_rep        = False

    # ── Per-frame callback ────────────────────────────────────────
    def pf(frame, fc, total):
        nonlocal stage, frames_in_stage, seen_standing, cooldown_frames
        nonlocal rep_min_knee, rep_max_knee, rep_worst_pelv
        nonlocal rep_frame_knees, rep_hip_angles
        nonlocal rep_hip_min_y, rep_nose_min_y
        nonlocal rep_knee_drop, rep_hip_drop, rep_nose_drop
        nonlocal baseline_knee_L, baseline_knee_R
        nonlocal baseline_hip_y, baseline_nose_y
        nonlocal active_leg
        nonlocal post_warmup_standing_confirmed, post_warmup_standing_frames
        nonlocal leg_locked_during_rep, prev_knee_k
        nonlocal last_coaching, coaching_frames, ex_type_current
        total_frames[0] = fc

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            if coaching_frames > 0:
                coaching_frames -= 1
            return frame

        lm  = res.pose_landmarks.landmark
        wlm = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None
        _current_lm[0]  = lm
        _current_wlm[0] = wlm

        smooth_pts = lm_smoother.smooth(lm, frame.shape[1], frame.shape[0])

        # ── Confidence gate: freeze state on low-visibility frames ─
        conf = _landmark_confidence(lm, [
            _SIDE[active_leg]["hip"],
            _SIDE[active_leg]["knee"],
            _SIDE[active_leg]["ankle"],
        ])
        if conf < 0.25:
            # Freeze — draw but don't update FSM
            draw_skeleton(frame, res.pose_landmarks, active_leg=active_leg, pts=smooth_pts)
            _draw_lhs_panel_inplace(frame, [
                ("REPS",  str(rep_count)),
                ("RIGHT", str(correct_reps)),
                ("WRONG", str(wrong_reps)),
                ("LEG",   active_leg[0].upper()),
                ("POSE",  _FSM_LABEL.get(stage, stage[:4])),
            ])
            draw_pcl_logo(frame)
            return frame

        # ── Active leg detection via excursion-based majority vote ─
        guess = _detect_active_leg_excursion(
            lm, wlm, baseline_knee_L, baseline_knee_R, active_leg)
        leg_vote_buffer.append(guess)

        left_v  = leg_vote_buffer.count("left")
        right_v = leg_vote_buffer.count("right")
        majority = "left" if left_v >= right_v else "right"

        maj_vis_ok = (
            (majority == "left"  and lm[25].visibility >= MIN_LEG_VIS and lm[27].visibility >= MIN_LEG_VIS) or
            (majority == "right" and lm[26].visibility >= MIN_LEG_VIS and lm[28].visibility >= MIN_LEG_VIS)
        )
        if (majority != active_leg
                and stage == "STANDING"
                and not leg_locked_during_rep
                and len(leg_vote_buffer) >= LEG_VOTE_WIN
                and maj_vis_ok):
            reset_for_leg_switch(fc, majority)

        # ── Knee angle (3D world if possible, else 2D) ────────────
        smoother = smoother_L if active_leg == "left" else smoother_R
        raw_k    = _get_knee_angle_best(lm, wlm, active_leg, min_vis=MIN_LEG_VIS)

        if raw_k is not None and prev_knee_k is not None:
            if abs(raw_k - prev_knee_k) > SPIKE_THRESH:
                log(f"  SPIKE fc={fc} raw={raw_k:.1f} prev={prev_knee_k:.1f}")
                raw_k = None

        if raw_k is not None:
            knee_k = smoother.update(raw_k)
        else:
            knee_k = smoother.update(prev_knee_k) if prev_knee_k is not None else 160.0
        prev_knee_k = knee_k
        knee_angles.append(knee_k)

        raw_hip = _get_hip_angle(lm, active_leg, min_vis=MIN_LEG_VIS)
        hip_k   = raw_hip if raw_hip is not None else (hip_angles_all[-1] if hip_angles_all else 155.0)
        hip_angles_all.append(hip_k)

        pelvic = _pelvic_drop(lm)
        pelvic_drops.append(pelvic)

        # ── Body height normalisation ─────────────────────────────
        bh = _body_height(lm)

        # ── Current signal values ─────────────────────────────────
        s     = _SIDE[active_leg]
        hip_y_now  = lm[s["hip"]].y
        nose_y_now = lm[0].y if lm[0].visibility > 0.15 else None

        # ── Adaptive baseline updates (when STANDING) ─────────────
        if stage == "STANDING":
            if active_leg == "left":
                standing_knee_buf_L.append(knee_k)
            else:
                standing_knee_buf_R.append(knee_k)
            standing_hip_buf.append(hip_y_now)
            if nose_y_now is not None:
                standing_nose_buf.append(nose_y_now)

            # Update baselines from buffers
            if len(standing_knee_buf_L) >= 5:
                baseline_knee_L = float(np.percentile(standing_knee_buf_L, 90))
            if len(standing_knee_buf_R) >= 5:
                baseline_knee_R = float(np.percentile(standing_knee_buf_R, 90))
            if len(standing_hip_buf) >= 5:
                baseline_hip_y = float(np.median(standing_hip_buf))
            if len(standing_nose_buf) >= 5:
                baseline_nose_y = float(np.median(standing_nose_buf))

        baseline_k = baseline_knee_L if active_leg == "left" else baseline_knee_R
        knee_drop  = max(0.0, baseline_k - knee_k)

        hip_drop_norm = 0.0
        if baseline_hip_y is not None:
            hip_drop_norm = max(0.0, (hip_y_now - baseline_hip_y) / bh)

        nose_drop_norm = 0.0
        if baseline_nose_y is not None and nose_y_now is not None:
            nose_drop_norm = max(0.0, (nose_y_now - baseline_nose_y) / bh)

        # ── Multi-signal movement score ───────────────────────────
        movement_score = 0
        if knee_drop  > KNEE_DROP_THRESH:  movement_score += 1
        if hip_drop_norm  > HIP_DROP_THRESH:  movement_score += 1
        if nose_drop_norm > NOSE_DROP_THRESH: movement_score += 1

        # ── Standing detection ────────────────────────────────────
        if knee_drop < STANDING_KNEE_DROP and hip_drop_norm < 0.04:
            seen_standing = True
        if not post_warmup_standing_confirmed and seen_standing:
            if knee_drop < STANDING_KNEE_DROP:
                post_warmup_standing_frames += 1
                if post_warmup_standing_frames >= POST_CONFIRM_FRAMES:
                    post_warmup_standing_confirmed = True
            else:
                post_warmup_standing_frames = 0

        if cooldown_frames > 0:
            cooldown_frames -= 1

        # ── Per-rep accumulation (during active movement phases) ──
        if stage in ("DESCENDING", "BOTTOM", "ASCENDING"):
            rep_min_knee   = min(rep_min_knee, knee_k)
            rep_max_knee   = max(rep_max_knee, knee_k)
            rep_worst_pelv = max(rep_worst_pelv, pelvic)
            rep_frame_knees.append(knee_k)
            rep_hip_angles.append(hip_k)
            rep_knee_drop  = max(rep_knee_drop, knee_drop)
            rep_hip_drop   = max(rep_hip_drop, hip_drop_norm)
            rep_nose_drop  = max(rep_nose_drop, nose_drop_norm)

        # ── FSM transitions ───────────────────────────────────────
        hip_y_buf.append(hip_y_now)
        knee_buf_fsm.append(knee_k)

        if stage == "STANDING":
            if (movement_score >= 2
                    and seen_standing
                    and post_warmup_standing_confirmed
                    and cooldown_frames == 0):
                stage           = "DESCENDING"
                frames_in_stage = 0
                leg_locked_during_rep = True
                rep_min_knee    = knee_k
                rep_max_knee    = knee_k
                rep_worst_pelv  = 0.0
                rep_frame_knees = [knee_k]
                rep_hip_angles  = [hip_k]
                rep_hip_min_y   = hip_y_now
                rep_nose_min_y  = nose_y_now
                rep_knee_drop   = knee_drop
                rep_hip_drop    = hip_drop_norm
                rep_nose_drop   = nose_drop_norm
                ex_type_current = _detect_exercise_type(lm, active_leg)
                log(f"  → DESCENDING fc={fc} leg={active_leg} "
                    f"knee_drop={knee_drop:.1f} hip_drop={hip_drop_norm:.3f} "
                    f"nose_drop={nose_drop_norm:.3f} score={movement_score}")

        elif stage == "DESCENDING":
            frames_in_stage += 1

            # Detect velocity sign change → entered BOTTOM
            if len(hip_y_buf) >= BOTTOM_VELOCITY_WIN:
                recent = list(hip_y_buf)
                mid    = len(recent) // 2
                v_early = recent[mid] - recent[0]     # positive = descending (y increases downward)
                v_late  = recent[-1] - recent[mid]
                # Velocity near zero or reversal after MIN_DOWN_FRAMES
                if (frames_in_stage >= MIN_DOWN_FRAMES
                        and (abs(v_late) < 0.008 or v_late < 0)):
                    stage           = "BOTTOM"
                    frames_in_stage = 0
                    log(f"  → BOTTOM fc={fc} knee={knee_k:.1f} "
                        f"v_early={v_early:.4f} v_late={v_late:.4f}")

            # Stuck protection: if in DESCENDING too long, force to BOTTOM
            if frames_in_stage > stuck_threshold // 2:
                stage           = "BOTTOM"
                frames_in_stage = 0
                log(f"  → BOTTOM (stuck) fc={fc}")

        elif stage == "BOTTOM":
            frames_in_stage += 1

            # Detect ascending: hip rising or knee extending
            hip_rising  = False
            knee_rising = False
            if len(hip_y_buf) >= 3:
                hip_rising  = (hip_y_buf[-1] - hip_y_buf[-3]) < -ASCENT_HIP_RISE * bh
            if len(knee_buf_fsm) >= 3:
                knee_rising = (knee_buf_fsm[-1] - rep_min_knee) > ASCENT_KNEE_RISE

            if hip_rising or knee_rising:
                stage           = "ASCENDING"
                frames_in_stage = 0
                log(f"  → ASCENDING fc={fc} hip_rising={hip_rising} knee_rising={knee_rising}")

            # Timeout in BOTTOM → force ASCENDING (real pause)
            if frames_in_stage > stuck_threshold:
                stage           = "ASCENDING"
                frames_in_stage = 0
                log(f"  → ASCENDING (timeout) fc={fc}")

        elif stage == "ASCENDING":
            frames_in_stage += 1

            # rep_score: count how many signals confirm a real rep happened
            rep_score = 0
            if rep_knee_drop  > MIN_REP_KNEE_DROP:  rep_score += 1
            if rep_hip_drop   > MIN_REP_HIP_DROP:   rep_score += 1
            if rep_nose_drop  > MIN_REP_NOSE_DROP:  rep_score += 1

            # Detect return to standing: movement_score falls back to 0 or 1
            returned_to_top = (movement_score <= 1 and knee_drop < STANDING_KNEE_DROP + 10)

            if returned_to_top:
                if rep_score >= 1:
                    finalise_rep(fc, forced=False)
                    log(f"  → STANDING (rep complete) fc={fc} rep_score={rep_score}")
                else:
                    # Jitter / no real movement — silently reset
                    stage           = "STANDING"
                    frames_in_stage = 0
                    cooldown_frames = COOLDOWN // 2
                    leg_locked_during_rep = False
                    log(f"  → STANDING (jitter, rep_score={rep_score}) fc={fc}")

            # Timeout stuck in ASCENDING — force complete if rep seen
            elif frames_in_stage > stuck_threshold:
                if rep_score >= 1:
                    finalise_rep(fc, forced=True)
                    log(f"  → STANDING (timeout force) fc={fc} rep_score={rep_score}")
                else:
                    stage           = "STANDING"
                    frames_in_stage = 0
                    cooldown_frames = COOLDOWN // 2
                    leg_locked_during_rep = False

        # ── Form flags for drawing ────────────────────────────────
        depth_thresh = PISTOL_DEPTH_BAD if ex_type_current == "pistol" else CHAIR_DEPTH_BAD
        bad_depth    = (stage != "STANDING" and rep_min_knee > depth_thresh and frames_in_stage > MIN_DOWN_FRAMES)
        bad_pelv     = pelvic > PELV_BAD

        # ── Drawing ───────────────────────────────────────────────
        draw_skeleton(frame, res.pose_landmarks,
                      active_leg=active_leg,
                      bad=(bad_depth or bad_pelv),
                      pts=smooth_pts)

        kne_lm = res.pose_landmarks.landmark[_SIDE[active_leg]["knee_lm"]]
        draw_angle_arc(frame, kne_lm, knee_k, bad=(bad_depth or bad_pelv))

        if coaching_frames > 0:
            coaching_frames -= 1

        _draw_lhs_panel_inplace(frame, [
            ("REPS",  str(rep_count)),
            ("RIGHT", str(correct_reps)),
            ("WRONG", str(wrong_reps)),
            ("LEG",   active_leg[0].upper()),
            ("POSE",  _FSM_LABEL.get(stage, stage[:4])),
        ])

        draw_pcl_logo(frame)

        return frame

    # ── Run ───────────────────────────────────────────────────────
    log(f"=== single_leg_squat v19 | session={session_id} | file={source_filename} ===")

    try:
        cap_probe = cv2.VideoCapture(path)
        fps_probe = cap_probe.get(cv2.CAP_PROP_FPS)
        cap_probe.release()
        if fps_probe and fps_probe > 0:
            video_fps       = fps_probe
            COOLDOWN        = max(5, int(fps_probe * 0.25))
            MIN_DOWN_FRAMES = max(4, int(fps_probe * 0.18))
            log(f"FPS={fps_probe:.0f} COOLDOWN={COOLDOWN} MIN_DOWN={MIN_DOWN_FRAMES}")
    except Exception:
        pass

    snaps = process_video_or_image(
        path, is_video, pf,
        output_path=output_path,
        analysis_skip=1,
        progress_uid=progress_uid,
    )
    pose.close()

    # End-of-video: if rep was in progress and real movement seen, count it
    if stage in ("ASCENDING", "BOTTOM"):
        rep_score = 0
        if rep_knee_drop  > MIN_REP_KNEE_DROP:  rep_score += 1
        if rep_hip_drop   > MIN_REP_HIP_DROP:   rep_score += 1
        if rep_nose_drop  > MIN_REP_NOSE_DROP:  rep_score += 1
        if rep_score >= 1:
            log(f"  → END-OF-VIDEO FORCE fc={total_frames[0]} rep_score={rep_score}")
            finalise_rep(fc=total_frames[0], forced=True)

    l_reps = leg_rep_counts["left"]
    r_reps = leg_rep_counts["right"]
    log(f"=== FINAL reps={rep_count} good={correct_reps} bad={bad_reps} "
        f"poor={poor_reps} L={l_reps} R={r_reps} ===")

    try:
        with open(log_path, "w") as fh:
            fh.write("\n".join(log_lines))
    except Exception:
        pass

    if session_id:
        save_wrong_angle_log("single_leg_squat", session_id, source_filename, wrong_events)

    if is_video and len(knee_angles) < 10:
        raise ValueError("No reliable single-leg squat motion detected.")

    # ── Aggregates ────────────────────────────────────────────────
    avg_k = round(float(np.mean(knee_angles)),    1) if knee_angles    else 0.0
    min_k = round(float(np.min(knee_angles)),     1) if knee_angles    else 0.0
    avg_h = round(float(np.mean(hip_angles_all)), 1) if hip_angles_all else 0.0
    avg_p = round(float(np.mean(pelvic_drops)),   2) if pelvic_drops   else 0.0

    # Overall classification
    if classify_results:
        overall_cls_score    = round(float(np.mean([r["score"] for r in classify_results])))
        overall_cls_category = ("GOOD" if overall_cls_score >= 75 else
                                ("BAD"  if overall_cls_score >= 45 else "POOR"))
        seen_sugg, all_sugg = set(), []
        for r in classify_results:
            for s in r["suggestions"]:
                if s not in seen_sugg:
                    seen_sugg.add(s); all_sugg.append(s)

        crossley_totals = {
            k: sum(1 for r in classify_results if r["crossley"].get(k))
            for k in ["trunk_lean", "pelvic_drop", "knee_valgus", "hip_adduction", "loss_of_balance"]
        }
        n_cls = max(1, len(classify_results))
        crossley_summary          = {k: v >= (n_cls * 0.5) for k, v in crossley_totals.items()}
        crossley_positive_overall = any(crossley_summary.values())

        risk_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        injury_map = {}
        for r in classify_results:
            for inj in r.get("injuries", []):
                k = inj["name"]
                if k not in injury_map or (
                    risk_order.get(inj["risk_level"], 0) > risk_order.get(injury_map[k]["risk_level"], 0)
                ):
                    injury_map[k] = inj
        all_injuries        = list(injury_map.values())
        overall_injury_risk = _overall_injury_risk(all_injuries)
        all_remedies        = {
            inj["name"]: {
                "risk_level": inj["risk_level"],
                "cause":      inj["cause"],
                "remedies":   inj["remedies"],
            }
            for inj in all_injuries
        }
    else:
        overall_cls_score         = 70
        overall_cls_category      = "BAD"
        all_sugg                  = []
        crossley_summary          = {}
        crossley_positive_overall = False
        all_injuries              = []
        overall_injury_risk       = "LOW"
        all_remedies              = {}

    # FIX: dominant exercise type from ALL frame votes (more stable)
    if exercise_type_votes:
        pistol_count  = exercise_type_votes.count("pistol")
        dominant_type = "Pistol Squat" if pistol_count > len(exercise_type_votes) * 0.6 else "Chair/Bench Squat"
    else:
        dominant_type = "Chair/Bench Squat"

    # Issues / strengths narrative
    issues, strengths = [], []

    depth_target = CHAIR_DEPTH_GOOD if "Chair" in dominant_type else PISTOL_DEPTH_GOOD
    if min_k > depth_target + 25:
        issues.append(f"Very shallow squat depth ({min_k:.0f}\u00b0) — aim for more knee flexion")
    elif min_k > depth_target:
        issues.append(f"Could go deeper ({min_k:.0f}\u00b0) — work on ankle and hip mobility")
    else:
        strengths.append(f"Good squat depth achieved ({min_k:.0f}\u00b0)")

    if avg_p > PELV_BAD:
        issues.append(f"Persistent pelvic drop ({avg_p:.1f}) — strengthen glute medius")
    elif avg_p > PELV_GOOD:
        issues.append(f"Moderate pelvic drop ({avg_p:.1f}) — add clamshells and lateral walks")
    else:
        strengths.append(f"Good pelvic stability ({avg_p:.1f})")

    torso_lean_est = max(0.0, 180.0 - avg_h)
    lean_limit = TRUNK_LEAN_BAD_PISTOL if "Pistol" in dominant_type else TRUNK_LEAN_BAD_CHAIR
    if torso_lean_est > lean_limit:
        issues.append(f"Excessive forward trunk lean ({torso_lean_est:.0f}\u00b0) — improve ankle & core")
    elif torso_lean_est > lean_limit * 0.7:
        issues.append(f"Moderate trunk lean ({torso_lean_est:.0f}\u00b0) — keep chest tall")
    else:
        strengths.append(f"Good upright posture ({torso_lean_est:.0f}\u00b0 lean)")

    if correct_reps == rep_count and rep_count > 0:
        strengths.append("All reps classified as GOOD!")
    elif correct_reps > 0:
        strengths.append(f"{correct_reps}/{rep_count} reps with good form")

    if l_reps > 0 and r_reps > 0:
        if abs(l_reps - r_reps) <= 1:
            strengths.append(f"Good balance: Left {l_reps} / Right {r_reps} reps")
        else:
            issues.append(f"Side imbalance — Left: {l_reps}, Right: {r_reps}. Train weaker side more.")

    if crossley_positive_overall:
        triggered = [k.replace("_", " ").title() for k, v in crossley_summary.items() if v]
        issues.append(f"SLS Positive — Crossley criteria: {', '.join(triggered)}")
    else:
        strengths.append("SLS Negative — no Crossley criteria in majority of reps")

    if overall_injury_risk == "HIGH":
        issues.append("HIGH injury risk detected — see remedies below")
    elif overall_injury_risk == "MEDIUM":
        issues.append("MEDIUM injury risk — corrective exercises recommended")

    for s in all_sugg:
        if s not in issues:
            issues.append(s)

    if not issues:
        issues = ["No major form issues detected — excellent work!"]

    rep_score    = (max(4, min(10, round(10 - (len(wrong_events) / rep_count) * 1.5)))
                    if rep_count > 0 else 4)
    cls_score_10 = round(overall_cls_score / 10)
    form_score   = max(4, min(10, round((rep_score + cls_score_10) / 2)))

    avg_balance_sway = float(np.mean(
        [float(r["sway"]) for r in frame_data if r.get("sway", "0") != "0"]
    )) if frame_data else 0.0
    control_score = max(1, min(10, round(10 - avg_balance_sway * 0.7)))

    return {
        "exercise":             "Single Leg Squat",
        "exercise_type":        dominant_type,
        "rep_count":            rep_count,
        "correct_reps":         correct_reps,
        "wrong_reps":           wrong_reps,
        "bad_reps":             bad_reps,
        "poor_reps":            poor_reps,
        "left_reps":            l_reps,
        "right_reps":           r_reps,
        "avg_knee_angle":       avg_k,
        "min_knee_angle":       min_k,
        "avg_hip_angle":        avg_h,
        "avg_pelvic_drop":      avg_p,
        "pelvic_drop":          avg_p,
        "form_score":           form_score,
        "control_score":        control_score,
        "overall_category":     overall_cls_category,
        "overall_cls_score":    overall_cls_score,
        "crossley_summary":     crossley_summary,
        "crossley_positive":    crossley_positive_overall,
        "injury_risk":          overall_injury_risk,
        "injuries":             [f"{i['name']} ({i['risk_level']})" for i in all_injuries],
        "_injuries_detail":     all_injuries,
        "remedies":             all_remedies,
        "issues":               issues,
        "strengths":            strengths,
        "per_rep":              frame_data,
        "per_rep_detail":       frame_data_detail,
        "snapshots":            snaps,
        "wrong_angle_count":    len(wrong_events),
        "_wrong_events":        wrong_events,
        "_classify_results":    classify_results,
        "metrics": [
            {"label": "Total Reps",          "value": str(rep_count)},
            {"label": "Correct Reps",        "value": str(correct_reps)},
            {"label": "Wrong Reps",          "value": str(wrong_reps)},
            {"label": "Avg Knee Angle",      "value": f"{avg_k}\u00b0"},
            {"label": "Min Knee (Depth)",    "value": f"{min_k}\u00b0"},
            {"label": "Hip Angle",           "value": f"{avg_h}\u00b0"},
            {"label": "Pelvic Drop Index",   "value": f"{avg_p}"},
            {"label": "Form Score",          "value": f"{form_score}/10"},
            {"label": "Control & Stillness", "value": f"{control_score}/10"},
            {"label": "Movement Quality",    "value": overall_cls_category},
            {"label": "Injury Risk",         "value": overall_injury_risk},
            {"label": "SLS Test",
             "value": "POSITIVE" if crossley_positive_overall else "NEGATIVE"},
        ],
    }
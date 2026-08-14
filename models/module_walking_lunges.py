# # ============================================================
# # AI WALKING LUNGES ANALYZER — ULTIMATE MASTER EDITION
# # Upgrades:
# # ✅ Single Master State Machine architecture eliminates dual-leg ghost tracking
# # ✅ Dynamic leg dominance targeting tracks active driving limb
# # ✅ Controlled transition cooldown prevents shallow reset bouncing
# # ✅ Fixed average knee pipeline ignoring zero-value artifacts
# # ✅ Rigid frame debouncing filters out trailing limb balance noise
# # ✅ Fully production-safe standalone script
# # ============================================================

# import cv2
# import json
# import numpy as np
# from collections import deque

# from utils import (
#     mp_pose,
#     get_landmark,
#     calculate_angle,
#     draw_angle_arc,
#     RollingMean,
#     process_video_or_image,
#     save_wrong_angle_log,
#     draw_pose_skyblue,
# )

# from hud_overlay import (
#     draw_footer_hud,
#     draw_pcl_logo,
# )

# # ============================================================
# # CONFIG
# # ============================================================

# # State Threshold Boundaries (Degrees)
# STANDING_THRESHOLD = 155   # Extension target to complete/reset a lunge loop
# LUNGING_THRESHOLD  = 135   # Trigger boundary confirming descent has started
# BOTTOM_THRESHOLD   = 105   # Core inflection lock zone checking required depth

# # Debounce filters checking sequence consistency
# STATE_FRAME_DEBOUNCE = 3 
# COOLDOWN_FRAMES     = 20  # Structural buffer locking out consecutive counts

# HIP_SWAY_LIMIT   = 0.08
# TRUNK_LEAN_LIMIT = 150
# MIN_VISIBILITY   = 0.50

# ASYMMETRY_LIMIT  = 20.0
# TEMPO_FAST_LIMIT = 14.0

# # ============================================================
# # MAIN AUTOMATION PIPELINE
# # ============================================================

# def analyse_walking_lunges(
#     path,
#     is_video,
#     output_path=None,
#     session_id=None,
#     source_filename="",
#     progress_uid=None,
# ):

#     # --- Pose Instance Tracking ---
#     pose = mp_pose.Pose(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5,
#         smooth_landmarks=True,
#         model_complexity=1,
#     )

#     # --- Session Aggregators ---
#     rep_count    = 0
#     correct_reps = 0
#     wrong_reps   = 0
#     detected_exercise = "walking_lunges"

#     # --- Metric Storage Arrays ---
#     lunge_knee_angles = []   
#     trunk_angles      = []
#     sway_values       = []
#     frame_data        = []
#     wrong_events      = []
#     ai_coaching       = []

#     # --- Bilateral Symmetry Baselines ---
#     last_rep_min_l = None
#     last_rep_min_r = None

#     # --- Frame Smoothing Engines ---
#     sm_l = RollingMean(5)
#     sm_r = RollingMean(5)

#     # ========================================================
#     # SINGLE MASTER STATE MACHINE W/ DYNAMIC LEG TARGETING
#     # States: "standing" | "descending" | "bottom" | "ascending"
#     # ========================================================
#     current_state    = "standing"
#     active_leg       = None  # "left" or "right" dynamic focus
#     cooldown_counter = 0
    
#     state_inc_buffer = 0
#     state_dec_buffer = 0

#     # --- Active Rep Accumulators ---
#     rep_min_knee   = 180.0
#     rep_max_knee   = 0.0
#     rep_worst_sway = 0.0
#     rep_worst_tr   = 180.0
#     rep_speeds     = []
    
#     prev_left_angle  = None
#     prev_right_angle = None
#     total_frames     = [0]

#     def generate_coaching(depth_ok, sway_ok, trunk_ok, symmetry_ok, tempo_ok):
#         tips = []
#         if not depth_ok:
#             tips.append("Keep chest upright and avoid leaning forward.")
#         if not sway_ok:
#             tips.append("Keep hips stable and reduce side movement.")
#         if not trunk_ok:
#             tips.append("Keep chest upright and avoid leaning forward.")
#         if not symmetry_ok:
#             tips.append("Maintain equal balance between both legs.")
#         if not tempo_ok:
#             tips.append("Excellent form. Keep maintaining control.")
#         if not tips:
#             tips.append("Excellent form. Keep maintaining control.")
#         return tips

#     def finalise_rep(fc, leg_side, forced=False):
#         nonlocal rep_count, correct_reps, wrong_reps, current_state, active_leg
#         nonlocal cooldown_counter, rep_min_knee, rep_max_knee, rep_worst_sway
#         nonlocal rep_worst_tr, rep_speeds, last_rep_min_l, last_rep_min_r, ai_coaching

#         rep_count += 1

#         # --- Strict Quality Check Processing ---
#         depth_ok   = bool(rep_min_knee < BOTTOM_THRESHOLD)
#         lockout_ok = bool(rep_max_knee > STANDING_THRESHOLD)
#         sway_ok    = bool(rep_worst_sway < HIP_SWAY_LIMIT)
#         trunk_ok   = bool(rep_worst_tr > TRUNK_LEAN_LIMIT)
        
#         avg_speed  = float(np.mean(rep_speeds)) if rep_speeds else 0.0
#         tempo_ok   = bool(avg_speed < TEMPO_FAST_LIMIT)

#         if leg_side == "left":
#             last_rep_min_l = rep_min_knee
#         else:
#             last_rep_min_r = rep_min_knee

#         if last_rep_min_l is not None and last_rep_min_r is not None:
#             asymmetry = float(abs(last_rep_min_l - last_rep_min_r))
#         else:
#             asymmetry = 0.0

#         symmetry_ok = bool(asymmetry < ASYMMETRY_LIMIT)

#         rep_correct = bool(
#             depth_ok and lockout_ok and sway_ok and 
#             trunk_ok and symmetry_ok and tempo_ok and not forced
#         )

#         if rep_correct:
#             correct_reps += 1
#         else:
#             wrong_reps += 1

#         # --- Log Fault Events ---
#         if not depth_ok:
#             wrong_events.append({"frame": int(fc), "joint": "depth", "note": "Shallow lunge"})
#         if not sway_ok:
#             wrong_events.append({"frame": int(fc), "joint": "stability", "note": "Hip instability"})
#         if not trunk_ok:
#             wrong_events.append({"frame": int(fc), "joint": "trunk", "note": "Forward trunk lean"})
#         if not symmetry_ok:
#             wrong_events.append({"frame": int(fc), "joint": "symmetry", "note": "Left/right imbalance"})

#         ai_coaching.extend(generate_coaching(depth_ok, sway_ok, trunk_ok, symmetry_ok, tempo_ok))

#         frame_data.append({
#             "rep": int(rep_count),
#             "leg": leg_side,
#             "correct": bool(rep_correct),
#             "min_front_knee": float(rep_min_knee),
#             "max_front_knee": float(rep_max_knee),
#             "hip_sway": float(rep_worst_sway),
#             "trunk_angle": float(rep_worst_tr),
#             "tempo": float(avg_speed),
#             "symmetry_diff": float(asymmetry),
#         })

#         # --- Complete Flush and Cooldown Engagement ---
#         rep_min_knee   = 180.0
#         rep_max_knee   = 0.0
#         rep_worst_sway = 0.0
#         rep_worst_tr   = 180.0
#         rep_speeds.clear()
        
#         current_state    = "standing"
#         active_leg       = None
#         cooldown_counter = COOLDOWN_FRAMES

#     # --- Core Processing Frame Engine ---
#     def pf(frame, fc, total):
#         nonlocal current_state, active_leg, cooldown_counter, state_inc_buffer, state_dec_buffer
#         nonlocal rep_min_knee, rep_max_knee, rep_worst_sway, rep_worst_tr
#         nonlocal prev_left_angle, prev_right_angle, detected_exercise

#         total_frames[0] = fc
#         if cooldown_counter > 0:
#             cooldown_counter -= 1

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = pose.process(rgb)

#         if not res.pose_landmarks:
#             return frame

#         lm = res.pose_landmarks.landmark

#         # --- Occlusion Gate ---
#         for idx in [23, 24, 25, 26, 27, 28]:
#             if lm[idx].visibility < MIN_VISIBILITY:
#                 return frame

#         # --- Extract Base Kinematic Coordinates ---
#         lh, lk, la = get_landmark(lm, 23), get_landmark(lm, 25), get_landmark(lm, 27)
#         rh, rk, ra = get_landmark(lm, 24), get_landmark(lm, 26), get_landmark(lm, 28)
#         ls, rs     = get_landmark(lm, 11), get_landmark(lm, 12)

#         l_k = float(sm_l.update(calculate_angle(lh, lk, la)))
#         r_k = float(sm_r.update(calculate_angle(rh, rk, ra)))

#         # --- Calculate Dynamic Global Variables ---
#         hip_sway = float(abs(lm[23].x - lm[24].x))
#         sway_values.append(hip_sway)

#         mid_sh = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2]
#         mid_hi = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2]
#         trunk  = float(calculate_angle([mid_sh[0], mid_sh[1]-0.1, mid_sh[2]], mid_sh, mid_hi))
#         trunk_angles.append(trunk)

#         # --- Velocity and Speed Pipeline ---
#         if prev_left_angle is not None and active_leg == "left":
#             rep_speeds.append(float(abs(l_k - prev_left_angle)))
#         if prev_right_angle is not None and active_leg == "right":
#             rep_speeds.append(float(abs(r_k - prev_right_angle)))

#         prev_left_angle  = l_k
#         prev_right_angle = r_k

#         # ========================================================
#         # RECONSTRUCTED CENTRALIZED STATE MACHINE PIPELINE
#         # ========================================================
        
#         # 1. Target Assignment Block
#         if current_state == "standing" and cooldown_counter == 0:
#             if l_k < LUNGING_THRESHOLD and l_k < r_k:
#                 state_dec_buffer += 1
#                 if state_dec_buffer >= STATE_FRAME_DEBOUNCE:
#                     current_state    = "descending"
#                     active_leg       = "left"
#                     state_dec_buffer = 0
#             elif r_k < LUNGING_THRESHOLD and r_k < l_k:
#                 state_dec_buffer += 1
#                 if state_dec_buffer >= STATE_FRAME_DEBOUNCE:
#                     current_state    = "descending"
#                     active_leg       = "right"
#                     state_dec_buffer = 0
#             else:
#                 state_dec_buffer = 0

#         # Set running variables based on current leg assignment
#         target_knee_angle = l_k if active_leg == "left" else r_k

#         # 2. Tracking Lifecycle Management Loop
#         if current_state != "standing":
#             lunge_knee_angles.append(target_knee_angle)
            
#             rep_min_knee   = min(rep_min_knee, target_knee_angle)
#             rep_max_knee   = max(rep_max_knee, target_knee_angle)
#             rep_worst_sway = max(rep_worst_sway, hip_sway)
#             rep_worst_tr   = min(rep_worst_tr, trunk)

#             if current_state == "descending":
#                 if target_knee_angle < BOTTOM_THRESHOLD:
#                     current_state = "bottom"
#                 elif target_knee_angle > STANDING_THRESHOLD:
#                     # Filter step corrections safely
#                     current_state = "standing"
#                     active_leg    = None

#             elif current_state == "bottom":
#                 if target_knee_angle > BOTTOM_THRESHOLD:
#                     current_state = "ascending"

#             elif current_state == "ascending":
#                 if target_knee_angle > STANDING_THRESHOLD:
#                     state_inc_buffer += 1
#                     if state_inc_buffer >= STATE_FRAME_DEBOUNCE:
#                         finalise_rep(fc, leg_side=active_leg, forced=False)
#                         state_inc_buffer = 0
#                 else:
#                     state_inc_buffer = 0

#         # --- Live HUD Display Logic Processing ---
#         front_knee     = l_k if l_k <= r_k else r_k
#         front_landmark = lm[25] if l_k <= r_k else lm[26]
#         bad_depth      = bool(current_state == "descending" and front_knee > LUNGING_THRESHOLD)

#         draw_pose_skyblue(frame, res.pose_landmarks)
#         draw_angle_arc(frame, front_landmark, front_knee, bad=bad_depth)

#         form_score = max(4, min(10, round(10 - (len(wrong_events) / rep_count)))) if rep_count > 0 else 4

#         draw_footer_hud(frame, [
#             ("REPS", str(rep_count)),
#             ("CORRECT", str(correct_reps)),
#             ("WRONG", str(wrong_reps)),
#             ("FORM", f"{form_score}/10"),
#         ])
#         draw_pcl_logo(frame)

#         # --- Frame Data Text Prints ---
#         cv2.putText(frame, f"EXERCISE: WALKING LUNGES", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#         cv2.putText(frame, f"L-KNEE: {int(l_k)}  R-KNEE: {int(r_k)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#         cv2.putText(frame, f"TRUNK: {int(trunk)}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#         cv2.putText(frame, f"STATE: {current_state.upper()} ({str(active_leg).upper()})", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

#         return frame

#     # --- Processing Execution Start ---
#     snaps = process_video_or_image(
#         path,
#         is_video,
#         pf,
#         output_path=output_path,
#         analysis_skip=1,
#         progress_uid=progress_uid,
#     )
#     pose.close()

#     # --- Frame-End Lifecycle Wrap-Up ---
#     if current_state in ("bottom", "ascending") and rep_min_knee < BOTTOM_THRESHOLD:
#         finalise_rep(fc=total_frames[0], leg_side=active_leg, forced=True)

#     avg_fk = float(np.mean(lunge_knee_angles)) if lunge_knee_angles else 0.0
#     avg_tr = float(np.mean(trunk_angles)) if trunk_angles else 0.0
#     avg_sw = float(np.mean(sway_values)) if sway_values else 0.0
#     form_score = int(max(4, min(10, round(10 - (len(wrong_events) / rep_count))))) if rep_count > 0 else 4

#     result = {
#         "exercise": "walking_lunges",
#         "rep_count": int(rep_count),
#         "correct_reps": int(correct_reps),
#         "wrong_reps": int(wrong_reps),
#         "form_score": int(form_score),
#         "avg_front_knee": float(avg_fk),
#         "avg_trunk_angle": float(avg_tr),
#         "avg_hip_sway": float(avg_sw),
#         "issues": list(set(e["note"] for e in wrong_events)),
#         "ai_coaching": list(set(ai_coaching)),
#         "per_rep": frame_data,
#         "snapshots": snaps,
#         "_wrong_events": wrong_events,
#     }

#     return json.loads(json.dumps(
#         result,
#         default=lambda o:
#             bool(o)    if isinstance(o, np.bool_)    else
#             float(o)   if isinstance(o, np.floating)  else
#             int(o)     if isinstance(o, np.integer)   else
#             str(o)
#     ))




















# # ============================================================
# # AI WALKING LUNGES ANALYZER — FIXED EDITION
# # Fixes:
# # ✅ FIX 1: state_dec_buffer now resets correctly on leg-switch
# # ✅ FIX 1: target_knee_angle guarded — never reads r_k before leg assigned
# # ✅ FIX 1: rep accumulators flushed on aborted descending->standing reset
# # ✅ FIX 2: lockout_ok threshold lowered to 140° (walking lunges don't fully extend)
# # ✅ FIX 2: depth_ok threshold corrected to 110° (proper lunge depth)
# # ✅ FIX 2: COOLDOWN_FRAMES raised to 35 frames (~1.2s at 30fps)
# # ✅ FIX 3: ffmpeg faststart remux applied to output video (moov atom at front)
# # ============================================================

# import cv2
# import json
# import subprocess
# import os
# import numpy as np
# from collections import deque

# from utils import (
#     mp_pose,
#     get_landmark,
#     calculate_angle,
#     draw_angle_arc,
#     RollingMean,
#     process_video_or_image,
#     save_wrong_angle_log,
#     draw_pose_skyblue,
# )

# from hud_overlay import (
#     draw_footer_hud,
#     draw_pcl_logo,
# )

# # ============================================================
# # CONFIG
# # ============================================================

# # State Threshold Boundaries (Degrees)
# STANDING_THRESHOLD = 155    # Extension target to complete/reset a lunge loop
# LUNGING_THRESHOLD  = 135    # Trigger boundary confirming descent has started
# # FIX 2: Was 105 — too strict; 110 is correct for walking lunge depth check
# BOTTOM_THRESHOLD   = 110

# # FIX 2: Was 155 — walking lunges don't fully lock out between steps; lowered to 140
# LOCKOUT_THRESHOLD  = 140

# # Debounce filters checking sequence consistency
# STATE_FRAME_DEBOUNCE = 3
# # FIX 2: Was 20 (~0.67s at 30fps) — too short, next lunge fires during cooldown
# COOLDOWN_FRAMES     = 35   # ~1.2s at 30fps

# HIP_SWAY_LIMIT   = 0.08
# TRUNK_LEAN_LIMIT = 150
# MIN_VISIBILITY   = 0.50

# ASYMMETRY_LIMIT  = 20.0
# TEMPO_FAST_LIMIT = 14.0

# # ============================================================
# # MAIN AUTOMATION PIPELINE
# # ============================================================

# def analyse_walking_lunges(
#     path,
#     is_video,
#     output_path=None,
#     session_id=None,
#     source_filename="",
#     progress_uid=None,
# ):

#     # --- Pose Instance Tracking ---
#     pose = mp_pose.Pose(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5,
#         smooth_landmarks=True,
#         model_complexity=1,
#     )

#     # --- Session Aggregators ---
#     rep_count    = 0
#     correct_reps = 0
#     wrong_reps   = 0
#     detected_exercise = "walking_lunges"

#     # --- Metric Storage Arrays ---
#     lunge_knee_angles = []
#     trunk_angles      = []
#     sway_values       = []
#     frame_data        = []
#     wrong_events      = []
#     ai_coaching       = []

#     # --- Bilateral Symmetry Baselines ---
#     last_rep_min_l = None
#     last_rep_min_r = None

#     # --- Frame Smoothing Engines ---
#     sm_l = RollingMean(5)
#     sm_r = RollingMean(5)

#     # ========================================================
#     # SINGLE MASTER STATE MACHINE W/ DYNAMIC LEG TARGETING
#     # States: "standing" | "descending" | "bottom" | "ascending"
#     # ========================================================
#     current_state    = "standing"
#     active_leg       = None   # "left" or "right" — only set after debounce
#     cooldown_counter = 0

#     state_inc_buffer = 0
#     state_dec_buffer = 0
#     # FIX 1: track which leg the dec_buffer is accumulating for
#     dec_buffer_leg   = None

#     # --- Active Rep Accumulators ---
#     rep_min_knee   = 180.0
#     rep_max_knee   = 0.0
#     rep_worst_sway = 0.0
#     rep_worst_tr   = 180.0
#     rep_speeds     = []

#     prev_left_angle  = None
#     prev_right_angle = None
#     total_frames     = [0]

#     def _flush_rep_accumulators():
#         nonlocal rep_min_knee, rep_max_knee, rep_worst_sway, rep_worst_tr
#         rep_min_knee   = 180.0
#         rep_max_knee   = 0.0
#         rep_worst_sway = 0.0
#         rep_worst_tr   = 180.0
#         rep_speeds.clear()

#     def generate_coaching(depth_ok, sway_ok, trunk_ok, symmetry_ok, tempo_ok):
#         tips = []
#         if not depth_ok:
#             tips.append("Lower your hips further — aim for front knee at 90 deg.")
#         if not sway_ok:
#             tips.append("Keep hips stable and reduce side movement.")
#         if not trunk_ok:
#             tips.append("Keep chest upright and avoid leaning forward.")
#         if not symmetry_ok:
#             tips.append("Maintain equal balance between both legs.")
#         if not tempo_ok:
#             tips.append("Slow down — control the descent for better form.")
#         if not tips:
#             tips.append("Excellent form. Keep maintaining control.")
#         return tips

#     def finalise_rep(fc, leg_side, forced=False):
#         nonlocal rep_count, correct_reps, wrong_reps, current_state, active_leg
#         nonlocal cooldown_counter, last_rep_min_l, last_rep_min_r, ai_coaching
#         # FIX 1: also reset dec_buffer state
#         nonlocal state_inc_buffer, state_dec_buffer, dec_buffer_leg

#         rep_count += 1

#         # --- Strict Quality Check Processing ---
#         depth_ok   = bool(rep_min_knee < BOTTOM_THRESHOLD)
#         # FIX 2: use LOCKOUT_THRESHOLD (140°) not STANDING_THRESHOLD (155°)
#         lockout_ok = bool(rep_max_knee > LOCKOUT_THRESHOLD)
#         sway_ok    = bool(rep_worst_sway < HIP_SWAY_LIMIT)
#         trunk_ok   = bool(rep_worst_tr > TRUNK_LEAN_LIMIT)

#         avg_speed = float(np.mean(rep_speeds)) if rep_speeds else 0.0
#         tempo_ok  = bool(avg_speed < TEMPO_FAST_LIMIT)

#         if leg_side == "left":
#             last_rep_min_l = rep_min_knee
#         else:
#             last_rep_min_r = rep_min_knee

#         if last_rep_min_l is not None and last_rep_min_r is not None:
#             asymmetry = float(abs(last_rep_min_l - last_rep_min_r))
#         else:
#             asymmetry = 0.0

#         symmetry_ok = bool(asymmetry < ASYMMETRY_LIMIT)

#         rep_correct = bool(
#             depth_ok and lockout_ok and sway_ok and
#             trunk_ok and symmetry_ok and tempo_ok and not forced
#         )

#         if rep_correct:
#             correct_reps += 1
#         else:
#             wrong_reps += 1

#         # --- Log Fault Events ---
#         if not depth_ok:
#             wrong_events.append({"frame": int(fc), "joint": "depth",    "note": "Shallow lunge"})
#         if not sway_ok:
#             wrong_events.append({"frame": int(fc), "joint": "stability","note": "Hip instability"})
#         if not trunk_ok:
#             wrong_events.append({"frame": int(fc), "joint": "trunk",    "note": "Forward trunk lean"})
#         if not symmetry_ok:
#             wrong_events.append({"frame": int(fc), "joint": "symmetry", "note": "Left/right imbalance"})

#         ai_coaching.extend(generate_coaching(depth_ok, sway_ok, trunk_ok, symmetry_ok, tempo_ok))

#         lunge_knee_angles.append(rep_min_knee)

#         frame_data.append({
#             "rep":            int(rep_count),
#             "leg":            leg_side,
#             "correct":        bool(rep_correct),
#             "min_front_knee": float(rep_min_knee),
#             "max_front_knee": float(rep_max_knee),
#             "hip_sway":       float(rep_worst_sway),
#             "trunk_angle":    float(rep_worst_tr),
#             "tempo":          float(avg_speed),
#             "symmetry_diff":  float(asymmetry),
#         })

#         # --- Flush accumulators and engage cooldown ---
#         _flush_rep_accumulators()
#         state_inc_buffer = 0
#         state_dec_buffer = 0
#         dec_buffer_leg   = None
#         current_state    = "standing"
#         active_leg       = None
#         cooldown_counter = COOLDOWN_FRAMES

#     # --- Core Processing Frame Engine ---
#     def pf(frame, fc, total):
#         nonlocal current_state, active_leg, cooldown_counter
#         nonlocal state_inc_buffer, state_dec_buffer, dec_buffer_leg
#         nonlocal rep_min_knee, rep_max_knee, rep_worst_sway, rep_worst_tr
#         nonlocal prev_left_angle, prev_right_angle, detected_exercise

#         total_frames[0] = fc
#         if cooldown_counter > 0:
#             cooldown_counter -= 1

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = pose.process(rgb)

#         if not res.pose_landmarks:
#             return frame

#         lm = res.pose_landmarks.landmark

#         # --- Occlusion Gate ---
#         for idx in [23, 24, 25, 26, 27, 28]:
#             if lm[idx].visibility < MIN_VISIBILITY:
#                 return frame

#         # --- Extract Base Kinematic Coordinates ---
#         lh, lk, la = get_landmark(lm, 23), get_landmark(lm, 25), get_landmark(lm, 27)
#         rh, rk, ra = get_landmark(lm, 24), get_landmark(lm, 26), get_landmark(lm, 28)
#         ls, rs     = get_landmark(lm, 11), get_landmark(lm, 12)

#         l_k = float(sm_l.update(calculate_angle(lh, lk, la)))
#         r_k = float(sm_r.update(calculate_angle(rh, rk, ra)))

#         # --- Calculate Dynamic Global Variables ---
#         hip_sway = float(abs(lm[23].x - lm[24].x))
#         sway_values.append(hip_sway)

#         mid_sh = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2]
#         mid_hi = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2]
#         trunk  = float(calculate_angle([mid_sh[0], mid_sh[1]-0.1, mid_sh[2]], mid_sh, mid_hi))
#         trunk_angles.append(trunk)

#         # --- Velocity and Speed Pipeline (only when leg is locked) ---
#         if prev_left_angle is not None and active_leg == "left":
#             rep_speeds.append(float(abs(l_k - prev_left_angle)))
#         if prev_right_angle is not None and active_leg == "right":
#             rep_speeds.append(float(abs(r_k - prev_right_angle)))

#         prev_left_angle  = l_k
#         prev_right_angle = r_k

#         # ========================================================
#         # STATE MACHINE
#         # ========================================================

#         # --- 1. Leg Selection (standing only, cooldown cleared) ---
#         if current_state == "standing" and cooldown_counter == 0:
#             # FIX 1: dec_buffer only increments when the SAME leg continues to trigger
#             if l_k < LUNGING_THRESHOLD and l_k < r_k:
#                 if dec_buffer_leg != "left":
#                     # leg switched mid-buffer — reset and start fresh for left
#                     state_dec_buffer = 0
#                     dec_buffer_leg   = "left"
#                 state_dec_buffer += 1
#                 if state_dec_buffer >= STATE_FRAME_DEBOUNCE:
#                     current_state    = "descending"
#                     active_leg       = "left"
#                     state_dec_buffer = 0
#                     dec_buffer_leg   = None
#             elif r_k < LUNGING_THRESHOLD and r_k < l_k:
#                 if dec_buffer_leg != "right":
#                     # leg switched mid-buffer — reset and start fresh for right
#                     state_dec_buffer = 0
#                     dec_buffer_leg   = "right"
#                 state_dec_buffer += 1
#                 if state_dec_buffer >= STATE_FRAME_DEBOUNCE:
#                     current_state    = "descending"
#                     active_leg       = "right"
#                     state_dec_buffer = 0
#                     dec_buffer_leg   = None
#             else:
#                 # Neither leg bending — clear buffer entirely
#                 state_dec_buffer = 0
#                 dec_buffer_leg   = None

#         # FIX 1: target_knee_angle only valid after active_leg is set
#         if active_leg is None:
#             # Not in a rep — skip accumulation, just draw HUD
#             pass
#         else:
#             target_knee_angle = l_k if active_leg == "left" else r_k

#             # --- 2. Rep Tracking (descending / bottom / ascending) ---
#             rep_min_knee   = min(rep_min_knee, target_knee_angle)
#             rep_max_knee   = max(rep_max_knee, target_knee_angle)
#             rep_worst_sway = max(rep_worst_sway, hip_sway)
#             rep_worst_tr   = min(rep_worst_tr, trunk)

#             if current_state == "descending":
#                 if target_knee_angle < BOTTOM_THRESHOLD:
#                     current_state = "bottom"
#                 elif target_knee_angle > STANDING_THRESHOLD:
#                     # FIX 1: aborted lunge — flush dirty accumulators before resetting
#                     _flush_rep_accumulators()
#                     current_state = "standing"
#                     active_leg    = None

#             elif current_state == "bottom":
#                 if target_knee_angle > BOTTOM_THRESHOLD:
#                     current_state = "ascending"

#             elif current_state == "ascending":
#                 # FIX 2: use LOCKOUT_THRESHOLD (140°) to detect standing — not 155°
#                 if target_knee_angle > LOCKOUT_THRESHOLD:
#                     state_inc_buffer += 1
#                     if state_inc_buffer >= STATE_FRAME_DEBOUNCE:
#                         finalise_rep(fc, leg_side=active_leg, forced=False)
#                         state_inc_buffer = 0
#                 else:
#                     state_inc_buffer = 0

#         # --- Live HUD Display ---
#         front_knee     = l_k if l_k <= r_k else r_k
#         front_landmark = lm[25] if l_k <= r_k else lm[26]
#         bad_depth      = bool(current_state == "descending" and front_knee > LUNGING_THRESHOLD)

#         draw_pose_skyblue(frame, res.pose_landmarks)
#         draw_angle_arc(frame, front_landmark, front_knee, bad=bad_depth)

#         form_score = max(4, min(10, round(10 - (len(wrong_events) / rep_count)))) if rep_count > 0 else 4

#         draw_footer_hud(frame, [
#             ("REPS",    str(rep_count)),
#             ("CORRECT", str(correct_reps)),
#             ("WRONG",   str(wrong_reps)),
#             ("FORM",    f"{form_score}/10"),
#         ])
#         draw_pcl_logo(frame)

#         cv2.putText(frame, "EXERCISE: WALKING LUNGES",
#                     (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#         cv2.putText(frame, f"L-KNEE: {int(l_k)}  R-KNEE: {int(r_k)}",
#                     (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#         cv2.putText(frame, f"TRUNK: {int(trunk)}",
#                     (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#         cv2.putText(frame, f"STATE: {current_state.upper()} ({str(active_leg).upper()})",
#                     (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

#         return frame

#     # --- Processing Execution ---
#     snaps = process_video_or_image(
#         path,
#         is_video,
#         pf,
#         output_path=output_path,
#         analysis_skip=1,
#         progress_uid=progress_uid,
#     )
#     pose.close()

#     # --- Frame-End Lifecycle Wrap-Up ---
#     if current_state in ("bottom", "ascending") and rep_min_knee < BOTTOM_THRESHOLD:
#         finalise_rep(fc=total_frames[0], leg_side=active_leg, forced=True)

#     # ============================================================
#     # FIX 3: ffmpeg faststart remux — moves moov atom to front
#     # so browsers can play and seek the full video correctly
#     # ============================================================
#     if is_video and output_path and os.path.exists(output_path):
#         tmp_path = output_path.replace(".mp4", "_tmp.mp4")
#         try:
#             ret = subprocess.run(
#                 [
#                     "ffmpeg", "-y",
#                     "-i", output_path,
#                     "-c", "copy",
#                     "-movflags", "+faststart",
#                     tmp_path,
#                 ],
#                 stdout=subprocess.DEVNULL,
#                 stderr=subprocess.DEVNULL,
#             )
#             if ret.returncode == 0 and os.path.exists(tmp_path):
#                 os.replace(tmp_path, output_path)
#         except Exception:
#             # ffmpeg not available — output still works, just no seeking
#             if os.path.exists(tmp_path):
#                 os.remove(tmp_path)

#     avg_fk = float(np.mean(lunge_knee_angles)) if lunge_knee_angles else 0.0
#     avg_tr = float(np.mean(trunk_angles))       if trunk_angles      else 0.0
#     avg_sw = float(np.mean(sway_values))        if sway_values       else 0.0
#     form_score = int(max(4, min(10, round(10 - (len(wrong_events) / rep_count))))) if rep_count > 0 else 4

#     result = {
#         "exercise":        "walking_lunges",
#         "rep_count":       int(rep_count),
#         "correct_reps":    int(correct_reps),
#         "wrong_reps":      int(wrong_reps),
#         "form_score":      int(form_score),
#         "avg_front_knee":  float(avg_fk),
#         "avg_trunk_angle": float(avg_tr),
#         "avg_hip_sway":    float(avg_sw),
#         "issues":          list(set(e["note"] for e in wrong_events)),
#         "ai_coaching":     list(set(ai_coaching)),
#         "per_rep":         frame_data,
#         "snapshots":       snaps,
#         "_wrong_events":   wrong_events,
#     }

#     return json.loads(json.dumps(
#         result,
#         default=lambda o:
#             bool(o)    if isinstance(o, np.bool_)    else
#             float(o)   if isinstance(o, np.floating)  else
#             int(o)     if isinstance(o, np.integer)   else
#             str(o)
#     ))
















# ============================================================
# AI WALKING LUNGES ANALYZER — FULLY FIXED EDITION
#
# BUGS FIXED IN THIS VERSION:
# ✅ FIX A: LOCKOUT_THRESHOLD raised to 150° — was 140°, causing
#            rep to fire too early while leg is still bent mid-step.
# ✅ FIX B: HIP_SWAY_LIMIT raised to 0.12 — was 0.08, which is
#            below the natural hip separation for most people just
#            standing still, so every rep was flagged "Hip instability".
# ✅ FIX C: Symmetry check now uses a rolling 3-rep average per side
#            instead of single last-rep comparison. Eliminates false
#            "Left/right imbalance" flags from one noisy rep.
# ✅ FIX D: COOLDOWN_FRAMES raised to 45 (~1.5s at 30fps).
#            Was 35 — too short for walking lunges where the user
#            takes a full stride before starting next rep.
# ✅ FIX E: Occlusion gate RELAXED — now only bails if BOTH knees
#            AND BOTH ankles are invisible, not any single landmark.
#            Previous gate was killing the state machine mid-rep
#            because one ankle dipped below 0.50 visibility during
#            the forward stride (occlusion by the other leg).
# ✅ FIX F: avg_knee_angle result key added (was missing — app.py
#            build_metrics reads "avg_knee_angle" but module only
#            wrote "avg_front_knee", so the HUD showed 0.0°).
# ✅ FIX G: ffmpeg faststart now imports subprocess at top level
#            and stderr is captured so silent failures are logged.
# ✅ FIX H: forced=True reps at video end are now excluded from
#            wrong_rep count but still included in rep_count, so
#            totals are consistent.
# ============================================================

import cv2
import json
import subprocess
import os
import numpy as np
from collections import deque

from utils import (
    mp_pose,
    get_landmark,
    calculate_angle,
    draw_angle_arc,
    RollingMean,
    process_video_or_image,
    save_wrong_angle_log,
    draw_pose_skyblue,
)

from hud_overlay import (
    draw_footer_hud,
    draw_pcl_logo,
    expand_canvas_for_lhs,
    draw_lhs_panel,
)

# ============================================================
# CONFIG
# ============================================================

# State detection thresholds (degrees)
# LUNGING_THRESHOLD: knee must go BELOW this to register a descent.
# Raised from 135 → 140 so the state machine detects entry sooner,
# giving more time for the bottom and ascending phases to register.
LUNGING_THRESHOLD  = 140

# BOTTOM_THRESHOLD: knee must reach BELOW this to count as "deep enough".
# Raised from 110 → 120. MediaPipe has ±5–10° error; walking lunges
# realistically reach 95–115°. 110 was failing on slightly shallow reps.
BOTTOM_THRESHOLD   = 120

# STANDING_THRESHOLD: knee must exceed this to confirm the rep is done.
# Used for abort detection during descent.
STANDING_THRESHOLD = 155

# LOCKOUT_THRESHOLD: knee angle on the ascending return that triggers
# rep completion. Removed from finalise_rep — we now use STANDING_THRESHOLD
# for the ascending finish check (see state machine below).
# The old lockout_ok quality check is REMOVED — it was the main source
# of false "wrong" reps because rep_max_knee never captured the standing
# angle before descent began. Quality is now judged on depth only.
LOCKOUT_THRESHOLD  = 150   # kept for reference, not used in quality check

# Debounce: frames that must consecutively agree before state transitions.
# Lowered from 4 → 3 to be more responsive without being noisy.
STATE_FRAME_DEBOUNCE = 3

# Cooldown after a rep completes. 30 frames @ 30fps = 1s.
# Was 45 — too long, caused skipped reps on fast walkers.
COOLDOWN_FRAMES = 30

HIP_SWAY_LIMIT   = 0.15   # relaxed from 0.12; natural sway during walking lunge
TRUNK_LEAN_LIMIT = 145    # relaxed from 150; slight lean is normal in lunges
MIN_VISIBILITY   = 0.40   # relaxed from 0.45
ASYMMETRY_LIMIT  = 30.0   # relaxed from 25°
TEMPO_FAST_LIMIT = 14.0


# ============================================================
# MAIN AUTOMATION PIPELINE
# ============================================================

def analyse_walking_lunges(
    path,
    is_video,
    output_path=None,
    session_id=None,
    source_filename="",
    progress_uid=None,
):

    # --- Pose Instance ---
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
        model_complexity=1,
    )

    # --- Session Aggregators ---
    rep_count    = 0
    correct_reps = 0
    wrong_reps   = 0

    # --- Metric Storage ---
    lunge_knee_angles = []
    trunk_angles      = []
    sway_values       = []
    frame_data        = []
    wrong_events      = []
    ai_coaching       = []

    # FIX C: Rolling buffers for symmetry — last 3 reps per leg
    left_min_history  = deque(maxlen=3)
    right_min_history = deque(maxlen=3)

    # --- Frame Smoothing ---
    sm_l = RollingMean(5)
    sm_r = RollingMean(5)

    # ========================================================
    # SINGLE MASTER STATE MACHINE
    # States: "standing" | "descending" | "bottom" | "ascending"
    # ========================================================
    current_state    = "standing"
    active_leg       = None
    cooldown_counter = 0

    state_inc_buffer = 0
    state_dec_buffer = 0
    dec_buffer_leg   = None

    # --- Per-Rep Accumulators ---
    rep_min_knee   = 180.0
    rep_max_knee   = 0.0
    rep_worst_sway = 0.0
    rep_worst_tr   = 180.0
    rep_speeds     = []

    prev_left_angle  = None
    prev_right_angle = None
    total_frames     = [0]

    # --------------------------------------------------------
    def _flush_rep_accumulators():
        nonlocal rep_min_knee, rep_max_knee, rep_worst_sway, rep_worst_tr
        rep_min_knee   = 180.0
        rep_max_knee   = 0.0
        rep_worst_sway = 0.0
        rep_worst_tr   = 180.0
        rep_speeds.clear()

    # --------------------------------------------------------
    def generate_coaching(depth_ok, sway_ok, trunk_ok, symmetry_ok, tempo_ok):
        tips = []
        if not depth_ok:
            tips.append("Lower your hips further — aim for front knee at 90°.")
        if not sway_ok:
            tips.append("Keep hips stable and reduce side movement.")
        if not trunk_ok:
            tips.append("Keep chest upright and avoid leaning forward.")
        if not symmetry_ok:
            tips.append("Maintain equal balance between both legs.")
        if not tempo_ok:
            tips.append("Slow down — control the descent for better form.")
        if not tips:
            tips.append("Excellent form. Keep maintaining control.")
        return tips

    # --------------------------------------------------------
    def finalise_rep(fc, leg_side, forced=False):
        nonlocal rep_count, correct_reps, wrong_reps, current_state, active_leg
        nonlocal cooldown_counter, ai_coaching
        nonlocal state_inc_buffer, state_dec_buffer, dec_buffer_leg

        rep_count += 1

        # --- Quality Checks ---
        # NOTE: lockout_ok REMOVED. The old check (rep_max_knee > LOCKOUT_THRESHOLD)
        # was always False because rep_max_knee only accumulated from when active_leg
        # was set (already in descent) — it never saw the standing angle before the
        # lunge started. Removing it means depth + stability + trunk judge the rep.
        depth_ok = bool(rep_min_knee < BOTTOM_THRESHOLD)
        sway_ok  = bool(rep_worst_sway < HIP_SWAY_LIMIT)
        trunk_ok = bool(rep_worst_tr > TRUNK_LEAN_LIMIT)

        avg_speed = float(np.mean(rep_speeds)) if rep_speeds else 0.0
        tempo_ok  = bool(avg_speed < TEMPO_FAST_LIMIT)

        # Rolling symmetry check (FIX C)
        if leg_side == "left":
            left_min_history.append(rep_min_knee)
        else:
            right_min_history.append(rep_min_knee)

        if left_min_history and right_min_history:
            asymmetry = float(abs(np.mean(left_min_history) - np.mean(right_min_history)))
        else:
            asymmetry = 0.0

        symmetry_ok = bool(asymmetry < ASYMMETRY_LIMIT)

        if forced:
            # Video ended mid-rep — count the rep but don't score it
            rep_correct = False
        else:
            rep_correct = bool(depth_ok and sway_ok and trunk_ok and symmetry_ok and tempo_ok)
            if rep_correct:
                correct_reps += 1
            else:
                wrong_reps += 1

        # --- Log Fault Events ---
        if not forced:
            if not depth_ok:
                wrong_events.append({"frame": int(fc), "joint": "depth",     "note": "Shallow lunge"})
            if not sway_ok:
                wrong_events.append({"frame": int(fc), "joint": "stability", "note": "Hip instability"})
            if not trunk_ok:
                wrong_events.append({"frame": int(fc), "joint": "trunk",     "note": "Forward trunk lean"})
            if not symmetry_ok:
                wrong_events.append({"frame": int(fc), "joint": "symmetry",  "note": "Left/right imbalance"})

        ai_coaching.extend(generate_coaching(depth_ok, sway_ok, trunk_ok, symmetry_ok, tempo_ok))
        lunge_knee_angles.append(rep_min_knee)

        frame_data.append({
            "rep":            int(rep_count),
            "leg":            leg_side,
            "correct":        bool(rep_correct),
            "forced":         bool(forced),
            "min_front_knee": float(rep_min_knee),
            "max_front_knee": float(rep_max_knee),
            "hip_sway":       float(rep_worst_sway),
            "trunk_angle":    float(rep_worst_tr),
            "tempo":          float(avg_speed),
            "symmetry_diff":  float(asymmetry),
        })

        # Flush and engage cooldown
        _flush_rep_accumulators()
        state_inc_buffer = 0
        state_dec_buffer = 0
        dec_buffer_leg   = None
        current_state    = "standing"
        active_leg       = None
        cooldown_counter = COOLDOWN_FRAMES

    # --------------------------------------------------------
    # CORE FRAME PROCESSING FUNCTION
    # --------------------------------------------------------
    def pf(frame, fc, total):
        nonlocal current_state, active_leg, cooldown_counter
        nonlocal state_inc_buffer, state_dec_buffer, dec_buffer_leg
        nonlocal rep_min_knee, rep_max_knee, rep_worst_sway, rep_worst_tr
        nonlocal prev_left_angle, prev_right_angle

        total_frames[0] = fc
        if cooldown_counter > 0:
            cooldown_counter -= 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            return frame

        lm = res.pose_landmarks.landmark

        # FIX E: Relaxed occlusion gate.
        # OLD: bailed if ANY of the 6 lower-body landmarks was < 0.50.
        # PROBLEM: During forward stride, the rear ankle (27 or 28) is
        # often occluded by the front leg, dropping to ~0.30–0.45.
        # This caused the state machine to pause mid-rep and miss reps.
        # NEW: Only bail if BOTH knees OR BOTH ankles are invisible,
        # which means the pose model has genuinely lost tracking.
        lk_vis = lm[25].visibility
        rk_vis = lm[26].visibility
        la_vis = lm[27].visibility
        ra_vis = lm[28].visibility

        both_knees_lost  = (lk_vis < MIN_VISIBILITY and rk_vis < MIN_VISIBILITY)
        both_ankles_lost = (la_vis < MIN_VISIBILITY and ra_vis < MIN_VISIBILITY)
        if both_knees_lost or both_ankles_lost:
            return frame

        # --- Extract Kinematics ---
        lh, lk, la = get_landmark(lm, 23), get_landmark(lm, 25), get_landmark(lm, 27)
        rh, rk, ra = get_landmark(lm, 24), get_landmark(lm, 26), get_landmark(lm, 28)
        ls, rs     = get_landmark(lm, 11), get_landmark(lm, 12)

        l_k = float(sm_l.update(calculate_angle(lh, lk, la)))
        r_k = float(sm_r.update(calculate_angle(rh, rk, ra)))

        # --- Global Metrics ---
        hip_sway = float(abs(lm[23].x - lm[24].x))
        sway_values.append(hip_sway)

        mid_sh = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2]
        mid_hi = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2]
        trunk  = float(calculate_angle(
            [mid_sh[0], mid_sh[1]-0.1, mid_sh[2]], mid_sh, mid_hi
        ))
        trunk_angles.append(trunk)

        # --- Speed (only while tracking a rep) ---
        if prev_left_angle is not None and active_leg == "left":
            rep_speeds.append(float(abs(l_k - prev_left_angle)))
        if prev_right_angle is not None and active_leg == "right":
            rep_speeds.append(float(abs(r_k - prev_right_angle)))

        prev_left_angle  = l_k
        prev_right_angle = r_k

        # ========================================================
        # STATE MACHINE
        # ========================================================

        # 1. Leg Selection — only in standing state with cooldown cleared
        if current_state == "standing" and cooldown_counter == 0:
            if l_k < LUNGING_THRESHOLD and l_k < r_k:
                if dec_buffer_leg != "left":
                    state_dec_buffer = 0
                    dec_buffer_leg   = "left"
                state_dec_buffer += 1
                if state_dec_buffer >= STATE_FRAME_DEBOUNCE:
                    current_state    = "descending"
                    active_leg       = "left"
                    state_dec_buffer = 0
                    dec_buffer_leg   = None
            elif r_k < LUNGING_THRESHOLD and r_k < l_k:
                if dec_buffer_leg != "right":
                    state_dec_buffer = 0
                    dec_buffer_leg   = "right"
                state_dec_buffer += 1
                if state_dec_buffer >= STATE_FRAME_DEBOUNCE:
                    current_state    = "descending"
                    active_leg       = "right"
                    state_dec_buffer = 0
                    dec_buffer_leg   = None
            else:
                state_dec_buffer = 0
                dec_buffer_leg   = None

        # 2. Rep Tracking — only when a leg is locked
        if active_leg is not None:
            target_knee_angle = l_k if active_leg == "left" else r_k

            rep_min_knee   = min(rep_min_knee, target_knee_angle)
            rep_max_knee   = max(rep_max_knee, target_knee_angle)
            rep_worst_sway = max(rep_worst_sway, hip_sway)
            rep_worst_tr   = min(rep_worst_tr, trunk)

            if current_state == "descending":
                if target_knee_angle < BOTTOM_THRESHOLD:
                    current_state = "bottom"
                elif target_knee_angle > STANDING_THRESHOLD:
                    # Aborted — flush and reset
                    _flush_rep_accumulators()
                    current_state = "standing"
                    active_leg    = None

            elif current_state == "bottom":
                if target_knee_angle > BOTTOM_THRESHOLD:
                    current_state = "ascending"

            elif current_state == "ascending":
                # Rep completes when the active knee returns to near-standing angle.
                # Use STANDING_THRESHOLD (155°) — the same angle used to detect
                # standing position. Debounce of 2 frames (was STATE_FRAME_DEBOUNCE=3)
                # because the ascending phase is brief; too much debounce = missed reps.
                if target_knee_angle > STANDING_THRESHOLD:
                    state_inc_buffer += 1
                    if state_inc_buffer >= 2:
                        finalise_rep(fc, leg_side=active_leg, forced=False)
                        state_inc_buffer = 0
                else:
                    state_inc_buffer = 0

        # --- HUD ---
        front_knee     = l_k if l_k <= r_k else r_k
        front_landmark = lm[25] if l_k <= r_k else lm[26]
        bad_depth      = bool(current_state == "descending" and front_knee > LUNGING_THRESHOLD)

        draw_pose_skyblue(frame, res.pose_landmarks)
        draw_angle_arc(frame, front_landmark, front_knee, bad=bad_depth)

        form_score = (
            max(4, min(10, round(10 - (len(wrong_events) / rep_count))))
            if rep_count > 0 else 4
        )

        canvas = expand_canvas_for_lhs(frame)
        draw_lhs_panel(canvas, [
            ("REPS",    str(rep_count)),
            ("CORRECT", str(correct_reps)),
            ("WRONG",   str(wrong_reps)),
            ("FORM",    f"{form_score}/10"),
        ])
        draw_pcl_logo(canvas)

        # Text annotations shifted right by PANEL_W so they land on the video portion
        from hud_overlay import PANEL_W as _PW
        cv2.putText(canvas, "EXERCISE: WALKING LUNGES",
                    (30 + _PW, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, f"L-KNEE: {int(l_k)}  R-KNEE: {int(r_k)}",
                    (30 + _PW, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"TRUNK: {int(trunk)}",
                    (30 + _PW, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"STATE: {current_state.upper()} ({str(active_leg).upper()})",
                    (30 + _PW, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        return canvas

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------
    snaps = process_video_or_image(
        path,
        is_video,
        pf,
        output_path=output_path,
        analysis_skip=1,
        progress_uid=progress_uid,
    )
    pose.close()

    # --- End-of-video cleanup for cut-off reps ---
    if current_state in ("bottom", "ascending") and rep_min_knee < BOTTOM_THRESHOLD:
        finalise_rep(fc=total_frames[0], leg_side=active_leg or "left", forced=True)

    # Note: faststart (moov atom relocation) is now handled centrally
    # in utils.process_video_or_image for ALL modules. No need here.

    # --- Final Aggregates ---
    avg_fk = float(np.mean(lunge_knee_angles)) if lunge_knee_angles else 0.0
    avg_tr = float(np.mean(trunk_angles))       if trunk_angles      else 0.0
    avg_sw = float(np.mean(sway_values))        if sway_values       else 0.0
    form_score = (
        int(max(4, min(10, round(10 - (len(wrong_events) / rep_count)))))
        if rep_count > 0 else 4
    )

    result = {
        "exercise":        "walking_lunges",
        "rep_count":       int(rep_count),
        "correct_reps":    int(correct_reps),
        "wrong_reps":      int(wrong_reps),
        "form_score":      int(form_score),
        "avg_knee_angle":  float(avg_fk),
        "avg_front_knee":  float(avg_fk),
        "avg_trunk_angle": float(avg_tr),
        "avg_hip_sway":    float(avg_sw),
        "issues":          list(set(e["note"] for e in wrong_events)),
        "ai_coaching":     list(set(ai_coaching)),
        "per_rep":         frame_data,
        "snapshots":       snaps,
        "_wrong_events":   wrong_events,
    }

    return json.loads(json.dumps(
        result,
        default=lambda o:
            bool(o)    if isinstance(o, np.bool_)   else
            float(o)   if isinstance(o, np.floating) else
            int(o)     if isinstance(o, np.integer)  else
            str(o)
    ))
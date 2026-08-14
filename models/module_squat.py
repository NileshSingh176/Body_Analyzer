# import cv2
# import numpy as np
# from utils import (
#     mp_pose,
#     get_landmark, calculate_angle, draw_angle_arc,
#     RollingMean, process_video_or_image,
#     save_wrong_angle_log, draw_pose_skyblue,
# )
# from hud_overlay import draw_footer_hud, draw_pcl_logo

# # ════════════════════════════════════════════════════════════════
# # FIXES vs module_squat v2
# #
# #  BUG A — stage starts as None, not "up":
# #    When stage == None and avg_k < 120, stage → "down" immediately
# #    on the very first frame (person is still standing up to start).
# #    rep_max_knee is still 0 at that point, so lockout_ok is always
# #    False for the first rep → counted as WRONG even if perfect.
# #    Fix: initialise stage = "up". Add a guard: only enter "down"
# #    from "up" (not from None).
# #
# #  BUG B — lockout_ok uses rep_max_knee which resets to 0:
# #    After a rep completes, rep_max_knee resets to 0. If the next
# #    rep is completed WITHOUT avg_k ever going above 145 in the
# #    "up→down" transition phase, lockout_ok = False even if the
# #    person stood up properly.
# #    Fix: track rep_max_knee separately for the "up" phase only
# #    (i.e. only count max when stage == "up" heading into the squat).
# #    Actually cleaner fix: capture the STANDING max BEFORE entering
# #    down, and also track max AFTER the down completes.
# #    → Simplest robust fix: rep_max_knee tracks ALL frames in the rep
# #    (up phase leading into it + down + up phase out of it). The
# #    previous reset-to-0 approach missed the standing frames.
# #
# #  BUG C — stuck-in-down not handled:
# #    If person squats but never stands up (bad lockout), stage stays
# #    "down" forever and NO rep is recorded. The rep just vanishes.
# #    Fix: stuck_threshold frame counter forces rep completion as WRONG.
# #
# #  BUG D — end-of-video rep dropped:
# #    If video ends while stage == "down", the last rep is silently
# #    dropped. Fix: force-complete at end if in "down" with valid depth.
# #
# #  BUG E — stage enters "down" from wrong starting position:
# #    If the person starts the video already in a squat (avg_k < 120),
# #    stage immediately goes "down" without a valid "up" start.
# #    Fix: require at least one "up" confirmation frame (avg_k > 145)
# #    before the first "down" is accepted.
# # ════════════════════════════════════════════════════════════════

# def analyse_squat(path, is_video, output_path=None,
#                   session_id=None, source_filename="", progress_uid=None):

#     pose = mp_pose.Pose(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )

#     rep_count    = 0
#     correct_reps = 0
#     wrong_reps   = 0
#     stage        = "up"          # FIX A: start as "up", never None

#     knee_angles  = []
#     hip_angles   = []
#     ankle_angles = []
#     sway_values  = []
#     frame_data   = []
#     smoother     = RollingMean(3)
#     wrong_events = []

#     # Per-rep trackers
#     rep_min_knee   = 180
#     rep_max_knee   = 0
#     rep_worst_sway = 0.0

#     # FIX E: require confirmed standing start before first down
#     seen_standing  = False   # True once avg_k > 145 observed

#     # FIX C: stuck-in-down detector
#     frames_in_down  = 0
#     stuck_threshold = 120    # FIX: raised from 60 → 120 (~4 sec at 30fps) — prevents slow squat false forced reps

#     total_frames = [0]

#     # ── Helper: finalise a rep ────────────────────────────────────
#     def finalise_rep(fc, forced=False):
#         nonlocal rep_count, correct_reps, wrong_reps, stage
#         nonlocal rep_min_knee, rep_max_knee, rep_worst_sway
#         nonlocal frames_in_down

#         rep_count     += 1
#         stage          = "up"
#         frames_in_down = 0

#         depth_ok   = rep_min_knee < 120
#         # FIX B: lockout_ok — person must reach standing again
#         # For forced completion, lockout inherently failed
#         lockout_ok = (rep_max_knee > 145) and not forced
#         sway_ok    = rep_worst_sway < 0.10

#         rep_correct = depth_ok and lockout_ok and sway_ok

#         if rep_correct:
#             correct_reps += 1
#         else:
#             wrong_reps += 1
#             if not depth_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "knee_depth",
#                     "angle_deg": round(rep_min_knee, 1),
#                     "note": (f"Insufficient depth — min knee "
#                              f"{rep_min_knee:.0f}° (need < 120°)")
#                 })
#             if not lockout_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "knee_lockout",
#                     "angle_deg": round(rep_max_knee, 1),
#                     "note": (f"Did not stand fully — max knee "
#                              f"{rep_max_knee:.0f}° (need > 145°)"
#                              + (" [forced]" if forced else ""))
#                 })
#             if not sway_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "hip_sway",
#                     "angle_deg": round(rep_worst_sway, 3),
#                     "note": (f"Hip/trunk sway — "
#                              f"{rep_worst_sway:.3f} (limit 0.10)")
#                 })

#         frame_data.append({
#             "rep":        rep_count,
#             "min_knee":   round(rep_min_knee, 1),
#             "max_knee":   round(rep_max_knee, 1),
#             "worst_sway": round(rep_worst_sway, 3),
#             "depth_ok":   depth_ok,
#             "lockout_ok": lockout_ok,
#             "sway_ok":    sway_ok,
#             "correct":    rep_correct,
#             "forced":     forced,
#         })

#         # Reset for next rep
#         rep_min_knee   = 180
#         rep_max_knee   = 0
#         rep_worst_sway = 0.0

#     # ── Per-frame callback ────────────────────────────────────────
#     def pf(frame, fc, total):
#         nonlocal stage, frames_in_down, seen_standing
#         nonlocal rep_min_knee, rep_max_knee, rep_worst_sway
#         total_frames[0] = fc

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = pose.process(rgb)

#         if not res.pose_landmarks:
#             return frame

#         lm = res.pose_landmarks.landmark

#         # ── Landmarks ─────────────────────────────────────────────
#         lh = get_landmark(lm, 23)
#         lk = get_landmark(lm, 25)
#         la = get_landmark(lm, 27)
#         ls = get_landmark(lm, 11)
#         lf = get_landmark(lm, 31)
#         rh = get_landmark(lm, 24)
#         rk = get_landmark(lm, 26)
#         ra = get_landmark(lm, 28)

#         # ── Knee angle — visibility weighted ──────────────────────
#         l_k = calculate_angle(lh, lk, la)
#         r_k = calculate_angle(rh, rk, ra)
#         lv  = lm[25].visibility
#         rv  = lm[26].visibility
#         raw   = (l_k * lv + r_k * rv) / (lv + rv + 1e-8)
#         avg_k = smoother.update(raw)
#         knee_angles.append(avg_k)

#         hip = calculate_angle(ls, lh, lk)
#         hip_angles.append(hip)
#         ank = calculate_angle(lk, la, lf)
#         ankle_angles.append(ank)

#         # ── Hip sway ───────────────────────────────────────────────
#         hip_sway = abs(lm[23].y - lm[24].y)
#         sway_values.append(hip_sway)

#         # ── Per-rep trackers (ALL frames counted) ──────────────────
#         rep_min_knee   = min(rep_min_knee, avg_k)
#         rep_max_knee   = max(rep_max_knee, avg_k)
#         rep_worst_sway = max(rep_worst_sway, hip_sway)

#         # ── Derived flags ──────────────────────────────────────────
#         knee_asym = abs(l_k - r_k)
#         bad_k     = avg_k > 170 or knee_asym > 25
#         bad_sway  = hip_sway > 0.08

#         # ── FIX E: require confirmed standing before first squat ───
#         if avg_k > 145:
#             seen_standing = True

#         # ── Stage machine ──────────────────────────────────────────
#         if stage == "up":
#             if avg_k < 120 and seen_standing:   # FIX A + FIX E
#                 stage          = "down"
#                 frames_in_down = 0

#         elif stage == "down":
#             frames_in_down += 1

#             if avg_k > 145:
#                 # Normal rep completion — person stood back up
#                 finalise_rep(fc, forced=False)

#             elif frames_in_down >= stuck_threshold:
#                 # FIX C: stuck too long → force wrong rep
#                 finalise_rep(fc, forced=True)

#         # ── Drawing ────────────────────────────────────────────────
#         draw_pose_skyblue(frame, res.pose_landmarks)
#         draw_angle_arc(frame, lm[25], avg_k, bad=bad_k)
#         draw_angle_arc(frame, lm[23], hip_sway * 100,
#                        color=(0, 200, 255), bad=bad_sway)

#         # ── HUD — navy footer + PCL logo ───────────────────────────
#         live_min_k = round(min(knee_angles) if knee_angles else avg_k, 0)
#         live_fs = "---"
#         if rep_count > 0:
#             live_fs = f"{max(4, min(10, round(10-(len(wrong_events)/rep_count)*1.5)))}/10"

#         draw_footer_hud(frame, [
#             ("REPS",    str(rep_count)),
#             ("CORRECT", str(correct_reps)),
#             ("WRONG",   str(wrong_reps)),
#             ("FORM",    live_fs),
#         ])
#         draw_pcl_logo(frame)

#         return frame

#     # ── Run ────────────────────────────────────────────────────────
#     snaps = process_video_or_image(path, is_video, pf, output_path=output_path,
#         analysis_skip=1,
#         progress_uid=progress_uid,
#     )
#     pose.close()

#     # FIX D: force-complete any rep still in "down" at video end
#     if stage == "down" and rep_min_knee < 120:
#         finalise_rep(fc=total_frames[0], forced=True)

#     if session_id:
#         save_wrong_angle_log("squat", session_id, source_filename, wrong_events)

#     avg_k  = round(np.mean(knee_angles),  1) if knee_angles  else 0
#     min_k  = round(np.min(knee_angles),   1) if knee_angles  else 0
#     avg_h  = round(np.mean(hip_angles),   1) if hip_angles   else 0
#     avg_a  = round(np.mean(ankle_angles), 1) if ankle_angles else 0
#     avg_sw = round(np.mean(sway_values),  3) if sway_values  else 0

#     issues = []; strengths = []

#     if min_k > 130:
#         issues.append(
#             f"Very shallow squat ({min_k}° min knee) — aim for parallel (≈90°)")
#     elif min_k > 110:
#         issues.append(f"Squat depth insufficient ({min_k}° min knee)")
#     elif min_k < 70:
#         strengths.append(f"Deep squat achieved ({min_k}° min knee)")
#     else:
#         strengths.append(f"Good squat depth ({min_k}° min knee)")

#     if avg_sw > 0.10:
#         issues.append(
#             f"Hip sway / trunk lean detected (avg {avg_sw:.2f}) — brace core")
#     elif avg_h >= 80:
#         strengths.append(f"Excellent upright torso ({avg_h}° hip angle)")
#     else:
#         strengths.append(f"Good trunk position (sway {avg_sw:.2f})")

#     if avg_a < 40:
#         issues.append(f"Very restricted ankle mobility ({avg_a}°)")
#     elif avg_a < 55:
#         issues.append(f"Ankle mobility limiting depth ({avg_a}°)")
#     else:
#         strengths.append(f"Sufficient ankle dorsiflexion ({avg_a}°)")

#     if correct_reps == rep_count and rep_count > 0:
#         strengths.append("All reps completed with correct form!")
#     elif correct_reps > 0:
#         strengths.append(f"{correct_reps}/{rep_count} reps with good form")

#     if not issues:
#         issues = ["No major form issues detected"]

#     if rep_count > 0:
#         issue_rate = len(wrong_events) / rep_count
#         form_score = max(4, min(10, round(10 - issue_rate * 1.5)))
#     else:
#         form_score = 4

#     return {
#         "exercise"         : "Squat",
#         "rep_count"        : rep_count,
#         "correct_reps"     : correct_reps,
#         "wrong_reps"       : wrong_reps,
#         "avg_knee_angle"   : avg_k,
#         "min_knee_angle"   : min_k,
#         "avg_hip_angle"    : avg_h,
#         "avg_ankle_angle"  : avg_a,
#         "form_score"       : form_score,
#         "issues"           : issues,
#         "strengths"        : strengths,
#         "per_rep"          : frame_data,
#         "snapshots"        : snaps,
#         "wrong_angle_count": len(wrong_events),
#         "_wrong_events"    : wrong_events,
#         "metrics": [
#             {"label": "Total Reps",       "value": str(rep_count)},
#             {"label": "Correct Reps",     "value": str(correct_reps)},
#             {"label": "Wrong Reps",       "value": str(wrong_reps)},
#             {"label": "Avg Knee Angle",   "value": f"{avg_k}°"},
#             {"label": "Min Knee (Depth)", "value": f"{min_k}°"},
#             {"label": "Hip Angle",        "value": f"{avg_h}°"},
#             {"label": "Ankle Angle",      "value": f"{avg_a}°"},
#             {"label": "Avg Hip Sway",     "value": f"{avg_sw:.3f}"},
#             {"label": "Form Score",       "value": f"{form_score}/10"},
#         ],
#     }








# import copy
# import cv2
# import numpy as np
# from utils import (
#     mp_pose,
#     get_landmark, calculate_angle, draw_angle_arc,
#     RollingMean, LandmarkSmoother, process_video_or_image,
#     save_wrong_angle_log, draw_pose_skyblue,
# )
# from hud_overlay import draw_footer_hud, draw_pcl_logo, expand_canvas_for_lhs, draw_lhs_panel

# # ════════════════════════════════════════════════════════════════
# # FIXES vs module_squat v2  (original bugs A-E preserved above)
# #
# #  FIX 1 — Front-view rep detection (0 reps bug):
# #    In a front-facing video hip→knee→ankle are nearly collinear
# #    horizontally, so calculate_angle returns ~170-180° always.
# #    avg_k < 120 never fires → 0 reps.
# #    Fix: detect front-view by measuring left/right hip X-spread.
# #    Use the KNEE-HIP vertical gap (knee_y - avg_hip_y in normalised
# #    coords) as depth signal.  When squatting this gap SHRINKS because
# #    knees rise toward hips.  Threshold: gap < 85% of standing gap.
# #    also fix depth_ok for front-view using hip Y drop instead of angle.
# #
# #  FIX 2 — Remove palm/finger skeleton:
# #    draw_pose_skyblue uses pts array; setting pts[17..22] to (0,0)
# #    makes the function skip those joints (it checks for zero coords).
# #    Indices 17-22 = left/right pinky, index, thumb.
# #
# #  FIX 3 — First of two similar reps skipped:
# #    RollingMean(3) delays transitions ~3 frames.  After finalise_rep
# #    stage resets to "up" but smoother still reports low angle for
# #    ~3 frames so "seen_standing" never fires before the next descent.
# #    Also rep_max_knee resets to 0 so lockout_ok always fails rep 2.
# #    Fix A: use RAW (unsmoothed) knee angle for stage transitions;
# #           keep smoother only for display.
# #    Fix B: track rep_max_knee in a separate "up-phase only" variable
# #           that captures the peak angle AFTER a rep completes.
# # ════════════════════════════════════════════════════════════════

# # MediaPipe hand landmark indices in the FULL BODY pose model
# # 17=left pinky, 18=right pinky, 19=left index, 20=right index,
# # 21=left thumb, 22=right thumb
# _HAND_LANDMARK_INDICES = set(range(17, 23))


# def analyse_squat(path, is_video, output_path=None,
#                   session_id=None, source_filename="", progress_uid=None):

#     pose = mp_pose.Pose(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )

#     rep_count    = 0
#     correct_reps = 0
#     wrong_reps   = 0
#     stage        = "up"          # FIX A: start as "up", never None

#     knee_angles  = []
#     hip_angles   = []
#     ankle_angles = []
#     sway_values  = []
#     frame_data   = []
#     smoother     = RollingMean(3)
#     lm_smoother  = LandmarkSmoother(alpha=0.80)   # high alpha = low lag
#     wrong_events = []

#     # Per-rep trackers
#     rep_min_knee   = 180
#     rep_max_knee   = 0
#     rep_worst_sway = 0.0

#     # FIX E: require confirmed standing start before first down
#     seen_standing  = False

#     # FIX C: stuck-in-down detector
#     frames_in_down  = 0
#     stuck_threshold = 120

#     total_frames = [0]

#     # ── FIX 1: front-view state ────────────────────────────────────
#     # knee_hip_gap_ref: knee-to-hip Y distance when person is standing.
#     # In front-view this gap SHRINKS during a squat.
#     # hip_y_ref: standing hip Y for absolute-drop fallback.
#     knee_hip_gap_ref = None   # set once when seen_standing fires
#     hip_y_ref        = None

#     # ── FIX 3: separate up-phase max knee tracker ──────────────────
#     # Tracks the maximum (most-extended) knee angle seen while stage=="up".
#     # Used for lockout_ok so it's unaffected by rep_max_knee reset.
#     up_phase_max_knee = 0.0

#     # ── Helper: finalise a rep ────────────────────────────────────
#     def finalise_rep(fc, forced=False, is_front=False, hip_drop_at_bottom=0.0):
#         nonlocal rep_count, correct_reps, wrong_reps, stage
#         nonlocal rep_min_knee, rep_max_knee, rep_worst_sway
#         nonlocal frames_in_down, up_phase_max_knee

#         rep_count     += 1
#         stage          = "up"
#         frames_in_down = 0

#         # FIX 1: for front-view depth is judged by hip drop, not knee angle
#         if is_front:
#             depth_ok = hip_drop_at_bottom >= 0.04   # hip dropped at least 4%
#         else:
#             depth_ok = rep_min_knee < 120

#         # FIX 3B: use up_phase_max_knee (captured BEFORE this rep's descent)
#         lockout_ok = (up_phase_max_knee > 145) and not forced
#         sway_ok    = rep_worst_sway < 0.10

#         rep_correct = depth_ok and lockout_ok and sway_ok

#         if rep_correct:
#             correct_reps += 1
#         else:
#             wrong_reps += 1
#             if not depth_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "knee_depth",
#                     "angle_deg": round(rep_min_knee, 1),
#                     "note": (f"Insufficient depth — min knee "
#                              f"{rep_min_knee:.0f}° (need < 120°)")
#                 })
#             if not lockout_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "knee_lockout",
#                     "angle_deg": round(up_phase_max_knee, 1),
#                     "note": (f"Did not stand fully — max knee "
#                              f"{up_phase_max_knee:.0f}° (need > 145°)"
#                              + (" [forced]" if forced else ""))
#                 })
#             if not sway_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "hip_sway",
#                     "angle_deg": round(rep_worst_sway, 3),
#                     "note": (f"Hip/trunk sway — "
#                              f"{rep_worst_sway:.3f} (limit 0.10)")
#                 })

#         frame_data.append({
#             "rep":        rep_count,
#             "min_knee":   round(rep_min_knee, 1),
#             "max_knee":   round(rep_max_knee, 1),
#             "worst_sway": round(rep_worst_sway, 3),
#             "depth_ok":   depth_ok,
#             "lockout_ok": lockout_ok,
#             "sway_ok":    sway_ok,
#             "correct":    rep_correct,
#             "forced":     forced,
#         })

#         # Reset per-rep trackers
#         rep_min_knee      = 180
#         rep_max_knee      = 0
#         rep_worst_sway    = 0.0
#         up_phase_max_knee = 0.0   # FIX 3B: reset — will be re-populated in next up phase

#     # ── Per-frame callback ────────────────────────────────────────
#     def pf(frame, fc, total):
#         nonlocal stage, frames_in_down, seen_standing
#         nonlocal rep_min_knee, rep_max_knee, rep_worst_sway
#         nonlocal knee_hip_gap_ref, hip_y_ref, up_phase_max_knee
#         total_frames[0] = fc

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = pose.process(rgb)

#         if not res.pose_landmarks:
#             return frame

#         lm = res.pose_landmarks.landmark

#         # ── Landmarks ─────────────────────────────────────────────
#         lh = get_landmark(lm, 23)
#         lk = get_landmark(lm, 25)
#         la = get_landmark(lm, 27)
#         ls = get_landmark(lm, 11)
#         lf = get_landmark(lm, 31)
#         rh = get_landmark(lm, 24)
#         rk = get_landmark(lm, 26)
#         ra = get_landmark(lm, 28)

#         # ── Knee angle — visibility weighted ──────────────────────
#         l_k = calculate_angle(lh, lk, la)
#         r_k = calculate_angle(rh, rk, ra)
#         lv  = lm[25].visibility
#         rv  = lm[26].visibility
#         # FIX 3A: raw_k = unsmoothed, used for stage transitions
#         raw_k = (l_k * lv + r_k * rv) / (lv + rv + 1e-8)
#         avg_k = smoother.update(raw_k)   # smoothed, used for display only
#         knee_angles.append(avg_k)

#         hip = calculate_angle(ls, lh, lk)
#         hip_angles.append(hip)
#         ank = calculate_angle(lk, la, lf)
#         ankle_angles.append(ank)

#         # ── Hip sway ───────────────────────────────────────────────
#         hip_sway = abs(lm[23].y - lm[24].y)
#         sway_values.append(hip_sway)

#         # ── Per-rep trackers (ALL frames) ──────────────────────────
#         rep_min_knee   = min(rep_min_knee,   raw_k)
#         rep_max_knee   = max(rep_max_knee,   raw_k)
#         rep_worst_sway = max(rep_worst_sway, hip_sway)

#         # ── FIX 3B: accumulate up-phase max while in "up" stage ───
#         if stage == "up":
#             up_phase_max_knee = max(up_phase_max_knee, raw_k)

#         # ── Derived flags ──────────────────────────────────────────
#         knee_asym = abs(l_k - r_k)
#         bad_k     = avg_k > 170 or knee_asym > 25
#         bad_sway  = hip_sway > 0.08

#         # ── FIX 1: front-view detection ───────────────────────────
#         # Left-hip vs right-hip horizontal separation > 0.10 = front-view.
#         # In side-view both hips overlap (spread < 0.05).
#         hip_x_spread  = abs(lm[23].x - lm[24].x)
#         is_front_view = hip_x_spread > 0.10

#         # Hip midpoint Y (increases downward in MediaPipe normalised coords)
#         avg_hip_y  = (lm[23].y + lm[24].y) / 2.0
#         avg_knee_y = (lm[25].y + lm[26].y) / 2.0

#         # ── FIX E + FIX 1: mark seen_standing + calibrate ref ─────
#         if raw_k > 145:
#             seen_standing = True

#         if is_front_view and stage == "up":
#             # In front-view also mark seen_standing when person is upright
#             # (knee-hip Y gap is large = legs extended = standing)
#             current_gap = avg_knee_y - avg_hip_y
#             if knee_hip_gap_ref is None or current_gap > knee_hip_gap_ref:
#                 # Larger gap = more extended = update standing reference
#                 knee_hip_gap_ref = current_gap
#                 hip_y_ref        = avg_hip_y
#                 seen_standing    = True

#         # ── FIX 1: front-view depth signals ───────────────────────
#         if is_front_view and knee_hip_gap_ref is not None and knee_hip_gap_ref > 0:
#             current_gap   = avg_knee_y - avg_hip_y
#             gap_ratio     = current_gap / knee_hip_gap_ref   # <1 when squatting
#             hip_drop      = avg_hip_y - hip_y_ref            # >0 when squatting

#             # "Down": gap has shrunk to <85% of standing gap
#             #         OR hip has dropped by >4% of frame
#             front_in_down = (gap_ratio < 0.85) or (hip_drop > 0.04)
#             # "Up": gap is back to >95% of standing gap
#             #       AND hip is within 2% of standing hip Y
#             front_in_up   = (gap_ratio > 0.95) and (hip_drop < 0.02)
#         else:
#             front_in_down = False
#             front_in_up   = False

#         # ── Combine signals ────────────────────────────────────────
#         if is_front_view:
#             # Use front-view signals primarily.
#             # Also accept side-view angle if it moves meaningfully
#             # (handles angled / diagonal camera positions).
#             side_also_moving = raw_k < 165
#             in_down = front_in_down or (side_also_moving and raw_k < 120)
#             in_up   = front_in_up   or (side_also_moving and raw_k > 145)
#         else:
#             # Pure side-view — use raw angle (FIX 3A: no smoother delay)
#             in_down = raw_k < 120
#             in_up   = raw_k > 145

#         # ── Stage machine ──────────────────────────────────────────
#         if stage == "up":
#             if in_down and seen_standing:
#                 stage          = "down"
#                 frames_in_down = 0

#         elif stage == "down":
#             frames_in_down += 1
#             # Capture hip drop at bottom for front-view depth check
#             if is_front_view and hip_y_ref is not None:
#                 _hip_drop_now = avg_hip_y - hip_y_ref
#             else:
#                 _hip_drop_now = 0.0

#             if in_up:
#                 finalise_rep(fc, forced=False,
#                              is_front=is_front_view,
#                              hip_drop_at_bottom=rep_worst_sway)   # reuse slot

#             elif frames_in_down >= stuck_threshold:
#                 finalise_rep(fc, forced=True,
#                              is_front=is_front_view,
#                              hip_drop_at_bottom=_hip_drop_now)

#         # ── Drawing ────────────────────────────────────────────────
#         h, w = frame.shape[:2]
#         pts = lm_smoother.smooth(lm, w, h)

#         # FIX 2: zero-out hand landmark coords in pts so draw_pose_skyblue
#         # skips drawing them (it checks for (0,0) and skips).
#         # Indices 17-22: left/right pinky, index, thumb.
#         for hand_idx in _HAND_LANDMARK_INDICES:
#             pts[hand_idx] = (0, 0)

#         draw_pose_skyblue(frame, res.pose_landmarks, pts)
#         draw_angle_arc(frame, lm[25], avg_k, bad=bad_k)
#         draw_angle_arc(frame, lm[23], hip_sway * 100,
#                        color=(0, 200, 255), bad=bad_sway)

#         # ── HUD ────────────────────────────────────────────────────
#         live_min_k = round(min(knee_angles) if knee_angles else avg_k, 0)
#         live_fs = "---"
#         if rep_count > 0:
#             live_fs = f"{max(4, min(10, round(10-(len(wrong_events)/rep_count)*1.5)))}/10"

#         canvas = expand_canvas_for_lhs(frame)
#         draw_lhs_panel(canvas, [
#             ("REPS",    str(rep_count)),
#             ("CORRECT", str(correct_reps)),
#             ("WRONG",   str(wrong_reps)),
#             ("FORM",    live_fs),
#         ])
#         draw_pcl_logo(canvas)

#         return canvas

#     # ── Run ────────────────────────────────────────────────────────
#     snaps = process_video_or_image(path, is_video, pf, output_path=output_path,
#         analysis_skip=1,
#         progress_uid=progress_uid,
#     )
#     pose.close()

#     # FIX D: force-complete any rep still in "down" at video end
#     if stage == "down" and rep_min_knee < 120:
#         finalise_rep(fc=total_frames[0], forced=True)

#     if session_id:
#         save_wrong_angle_log("squat", session_id, source_filename, wrong_events)

#     avg_k  = round(np.mean(knee_angles),  1) if knee_angles  else 0
#     min_k  = round(np.min(knee_angles),   1) if knee_angles  else 0
#     avg_h  = round(np.mean(hip_angles),   1) if hip_angles   else 0
#     avg_a  = round(np.mean(ankle_angles), 1) if ankle_angles else 0
#     avg_sw = round(np.mean(sway_values),  3) if sway_values  else 0

#     issues = []; strengths = []

#     if min_k > 130:
#         issues.append(
#             f"Very shallow squat ({min_k}° min knee) — aim for parallel (≈90°)")
#     elif min_k > 110:
#         issues.append(f"Squat depth insufficient ({min_k}° min knee)")
#     elif min_k < 70:
#         strengths.append(f"Deep squat achieved ({min_k}° min knee)")
#     else:
#         strengths.append(f"Good squat depth ({min_k}° min knee)")

#     if avg_sw > 0.10:
#         issues.append(
#             f"Hip sway / trunk lean detected (avg {avg_sw:.2f}) — brace core")
#     elif avg_h >= 80:
#         strengths.append(f"Excellent upright torso ({avg_h}° hip angle)")
#     else:
#         strengths.append(f"Good trunk position (sway {avg_sw:.2f})")

#     if avg_a < 40:
#         issues.append(f"Very restricted ankle mobility ({avg_a}°)")
#     elif avg_a < 55:
#         issues.append(f"Ankle mobility limiting depth ({avg_a}°)")
#     else:
#         strengths.append(f"Sufficient ankle dorsiflexion ({avg_a}°)")

#     if correct_reps == rep_count and rep_count > 0:
#         strengths.append("All reps completed with correct form!")
#     elif correct_reps > 0:
#         strengths.append(f"{correct_reps}/{rep_count} reps with good form")

#     if not issues:
#         issues = ["No major form issues detected"]

#     if rep_count > 0:
#         issue_rate = len(wrong_events) / rep_count
#         form_score = max(4, min(10, round(10 - issue_rate * 1.5)))
#     else:
#         form_score = 4

#     return {
#         "exercise"         : "Squat",
#         "rep_count"        : rep_count,
#         "correct_reps"     : correct_reps,
#         "wrong_reps"       : wrong_reps,
#         "avg_knee_angle"   : avg_k,
#         "min_knee_angle"   : min_k,
#         "avg_hip_angle"    : avg_h,
#         "avg_ankle_angle"  : avg_a,
#         "form_score"       : form_score,
#         "issues"           : issues,
#         "strengths"        : strengths,
#         "per_rep"          : frame_data,
#         "snapshots"        : snaps,
#         "wrong_angle_count": len(wrong_events),
#         "_wrong_events"    : wrong_events,
#         "metrics": [
#             {"label": "Total Reps",       "value": str(rep_count)},
#             {"label": "Correct Reps",     "value": str(correct_reps)},
#             {"label": "Wrong Reps",       "value": str(wrong_reps)},
#             {"label": "Avg Knee Angle",   "value": f"{avg_k}°"},
#             {"label": "Min Knee (Depth)", "value": f"{min_k}°"},
#             {"label": "Hip Angle",        "value": f"{avg_h}°"},
#             {"label": "Ankle Angle",      "value": f"{avg_a}°"},
#             {"label": "Avg Hip Sway",     "value": f"{avg_sw:.3f}"},
#             {"label": "Form Score",       "value": f"{form_score}/10"},
#         ],
#     }























import cv2
import numpy as np
from utils import (
    mp_pose,
    get_landmark, calculate_angle, draw_angle_arc,
    RollingMean, LandmarkSmoother, process_video_or_image,
    save_wrong_angle_log, draw_pose_skyblue,
)
from hud_overlay import draw_footer_hud, draw_pcl_logo

# ════════════════════════════════════════════════════════════════
# FIXES vs module_squat v2 (carried over, still valid)
#
#  BUG A — stage starts as None, not "up": fixed by stage = "up" init.
#  BUG B — lockout_ok uses rep_max_knee which resets to 0: fixed by
#    tracking rep_max_knee across the whole rep window.
#  BUG C — stuck-in-down not handled: fixed by stuck_threshold force.
#  BUG D — end-of-video rep dropped: fixed by force-complete at end.
#  BUG E — stage enters "down" from wrong starting position: fixed by
#    requiring a standing confirmation before the first "down".
#
# ════════════════════════════════════════════════════════════════
# NEW FIXES (this revision)
#
#  FIX F — FRONT VIEW RETURNS 0 REPS:
#    Root cause: rep detection was 100% dependent on avg_k, the
#    sagittal hip→knee→ankle angle. In a front-facing video this
#    angle barely changes between standing and squatting (the flexion
#    happens mostly toward/away from the camera, not across it), so
#    avg_k never crosses the 120°/145° thresholds and stage never
#    leaves "up".
#    Fix: classify the camera view ONCE during a calibration window
#    (shoulder/hip width vs torso-height ratio). For FRONT VIEW,
#    switch the rep-driving signal from avg_k to a combined hip-drop +
#    shoulder-drop percentage (vertical displacement from a calibrated
#    standing baseline, normalized by standing torso length). SIDE
#    VIEW keeps using avg_k exactly as before — no change to existing
#    side-view behaviour.
#
#  FIX G — FIRST OF TWO NEAR-IDENTICAL REPS MISSED:
#    Root cause: `seen_standing` was a one-shot latch set the instant
#    the SMOOTHED avg_k first crossed 145. With a 3-frame rolling
#    mean, if the very first squat's descent begins before the
#    smoother has settled above 145 for even one frame, `seen_standing`
#    flips True on the same frame (or later) that avg_k is already
#    falling — so the up→down transition for rep 1 is skipped entirely
#    (stage never leaves "up"). By rep 2, seen_standing is already
#    True, so rep 2 triggers normally. Net effect: only the SECOND of
#    two closely-spaced/identical reps gets counted, no matter how
#    correct rep 1 was.
#    Fix: replace the reactive "wait for one good frame" gate with an
#    explicit CALIBRATION PHASE. The first N valid-pose frames are
#    used only to establish a standing baseline signal (and view
#    classification) — no rep counting happens during calibration.
#    Once calibration completes, the standing baseline is known
#    upfront (not dependent on a lucky smoothed-value spike), so the
#    very first descent — even one that starts immediately after
#    calibration — is detected correctly. If the person is already
#    mid-squat when calibration ends, Phase 3 below handles that case
#    explicitly instead of silently eating the rep.
#
#  FIX H — PALM/FINGER TRACKING:
#    No hand landmark model was actually being invoked (mp_pose.Pose
#    only ever computes the 33 standard BlazePose body landmarks).
#    However landmarks 17–22 (pinky/index/thumb tips) ARE part of that
#    33-point body model and WERE being drawn by draw_pose_skyblue.
#    Fix applied in utils.py: draw_pose_skyblue now excludes indices
#    17–22 (finger/palm tips) from both the connection lines and the
#    joint dots, in addition to the existing face exclusion (0–10).
#    Wrists (15/16) are kept — tracking now visibly stops at the
#    wrist. No change needed here in module_squat.py for this part,
#    since this module never read 17–22 in its own angle math.
# ════════════════════════════════════════════════════════════════

CALIBRATION_FRAMES = 15   # frames used to classify view + set standing baseline
MAX_CALIBRATION_FRAMES = 90  # hard cap (~3s at 30fps) if no clear standing point is found


def _classify_view(shoulder_w, hip_w, torso_h):
    """
    Heuristic view classifier using the ratio of horizontal body width
    to torso height. Front view: shoulders/hips are seen near full
    width (ratio close to typical anatomical proportions, ~0.35-0.55).
    Side view: shoulders/hips are foreshortened by perspective, so the
    visible width collapses to a much smaller fraction of torso height.
    """
    if torso_h < 1e-6:
        return "side"  # safe fallback — side-view logic is the proven path
    width_ratio = max(shoulder_w, hip_w) / torso_h
    return "front" if width_ratio > 0.30 else "side"


def analyse_squat(path, is_video, output_path=None,
                  session_id=None, source_filename="", progress_uid=None):

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    rep_count    = 0
    correct_reps = 0
    wrong_reps   = 0
    stage        = "up"          # FIX A: start as "up", never None

    knee_angles  = []
    hip_angles   = []
    ankle_angles = []
    sway_values  = []
    frame_data   = []
    smoother     = RollingMean(3)
    depth_smoother = RollingMean(3)   # for front-view hip/shoulder-drop signal
    lm_smoother  = LandmarkSmoother(alpha=0.40)  # XY position EMA smoother
    wrong_events = []

    # Per-rep trackers — generalized to "depth_signal" so the same
    # finalise_rep logic works for both side view (knee angle, where
    # LOWER = deeper) and front view (drop %, where HIGHER = deeper).
    rep_min_knee   = 180
    rep_max_knee   = 0
    rep_worst_sway = 0.0
    rep_max_drop   = 0.0   # front-view: deepest hip/shoulder drop % reached this rep

    # FIX G: explicit calibration phase replaces the reactive seen_standing latch
    calib_frames_seen   = 0
    calib_shoulder_w    = []
    calib_hip_w         = []
    calib_torso_h       = []
    calib_hip_y         = []
    calib_shoulder_y    = []
    calibrated          = False
    view_mode           = "side"     # default/fallback until classified
    standing_knee_baseline = 145.0   # side view default, refined after calibration
    squat_knee_baseline     = 120.0  # side view default, refined after calibration
    standing_hip_y      = None       # front view baseline
    standing_shoulder_y = None       # front view baseline
    standing_torso_h    = None       # front view normalization

    # Phase 3 (module-level): if the person is ALREADY mid-squat when
    # calibration ends, we cannot get a clean standing baseline from
    # frame 1. In that case we still finish calibration on schedule but
    # mark `started_mid_rep` so the first rep is allowed to be counted
    # as soon as a clear ascent to standing is observed, without
    # requiring a separate descent to have been seen first.
    started_mid_rep = False

    # FIX C: stuck-in-down detector
    frames_in_down  = 0
    stuck_threshold = 120    # ~4 sec at 30fps — prevents slow squat false forced reps

    total_frames = [0]
    last_drop_pct = [0.0]  # front-view: most recent drop_pct, for end-of-video force-complete

    # ── Helper: finalise a rep ────────────────────────────────────
    def finalise_rep(fc, forced=False, end_drop=0.0):
        nonlocal rep_count, correct_reps, wrong_reps, stage
        nonlocal rep_min_knee, rep_max_knee, rep_worst_sway, rep_max_drop
        nonlocal frames_in_down

        rep_count     += 1
        stage          = "up"
        frames_in_down = 0

        if view_mode == "side":
            depth_ok   = rep_min_knee < squat_knee_baseline
            lockout_ok = (rep_max_knee > standing_knee_baseline) and not forced
            depth_note_val = round(rep_min_knee, 1)
        else:
            # front view: depth is "good" when the deepest point reached
            # a large enough drop %; lockout is "good" when the drop %
            # AT THE MOMENT THE REP ENDED is back near the standing
            # baseline (NOT the max over the rep, which is always deep
            # by definition).
            depth_ok   = rep_max_drop > 0.18   # ≥18% normalized drop ≈ a real squat
            lockout_ok = (end_drop < 0.06) and not forced  # back near standing baseline
            depth_note_val = round(rep_max_drop * 100, 1)

        sway_ok = rep_worst_sway < 0.10
        rep_correct = depth_ok and lockout_ok and sway_ok

        if rep_correct:
            correct_reps += 1
        else:
            wrong_reps += 1
            if not depth_ok:
                if view_mode == "side":
                    note = (f"Insufficient depth — min knee "
                             f"{rep_min_knee:.0f}° (need < {squat_knee_baseline:.0f}°)")
                else:
                    note = (f"Insufficient depth — hip/shoulder drop "
                             f"{depth_note_val:.1f}% (need > 18%)")
                wrong_events.append({
                    "frame": fc, "joint": "knee_depth",
                    "angle_deg": depth_note_val,
                    "note": note,
                })
            if not lockout_ok:
                if view_mode == "side":
                    note = (f"Did not stand fully — max knee "
                             f"{rep_max_knee:.0f}° (need > {standing_knee_baseline:.0f}°)"
                             + (" [forced]" if forced else ""))
                    angle_val = round(rep_max_knee, 1)
                else:
                    note = (f"Did not stand fully — residual drop "
                             f"{end_drop * 100:.1f}% (need < 6%)"
                             + (" [forced]" if forced else ""))
                    angle_val = round(end_drop * 100, 1)
                wrong_events.append({
                    "frame": fc, "joint": "knee_lockout",
                    "angle_deg": angle_val,
                    "note": note,
                })
            if not sway_ok:
                wrong_events.append({
                    "frame": fc, "joint": "hip_sway",
                    "angle_deg": round(rep_worst_sway, 3),
                    "note": (f"Hip/trunk sway — "
                             f"{rep_worst_sway:.3f} (limit 0.10)")
                })

        frame_data.append({
            "rep":        rep_count,
            "min_knee":   round(rep_min_knee, 1),
            "max_knee":   round(rep_max_knee, 1),
            "max_drop_pct": round(rep_max_drop * 100, 1),
            "worst_sway": round(rep_worst_sway, 3),
            "depth_ok":   depth_ok,
            "lockout_ok": lockout_ok,
            "sway_ok":    sway_ok,
            "correct":    rep_correct,
            "forced":     forced,
            "view_mode":  view_mode,
        })

        # Reset for next rep
        rep_min_knee   = 180
        rep_max_knee   = 0
        rep_worst_sway = 0.0
        rep_max_drop   = 0.0

    # ── Per-frame callback ────────────────────────────────────────
    def pf(frame, fc, total):
        nonlocal stage, frames_in_down
        nonlocal rep_min_knee, rep_max_knee, rep_worst_sway, rep_max_drop
        nonlocal calib_frames_seen, calibrated, view_mode
        nonlocal standing_knee_baseline, squat_knee_baseline
        nonlocal standing_hip_y, standing_shoulder_y, standing_torso_h
        nonlocal started_mid_rep
        total_frames[0] = fc

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            return frame

        lm = res.pose_landmarks.landmark

        # ── Landmarks ─────────────────────────────────────────────
        lh = get_landmark(lm, 23)
        lk = get_landmark(lm, 25)
        la = get_landmark(lm, 27)
        ls = get_landmark(lm, 11)
        lf = get_landmark(lm, 31)
        rh = get_landmark(lm, 24)
        rk = get_landmark(lm, 26)
        ra = get_landmark(lm, 28)
        rs = get_landmark(lm, 12)

        # ── Knee angle — visibility weighted (side-view signal) ────
        l_k = calculate_angle(lh, lk, la)
        r_k = calculate_angle(rh, rk, ra)
        lv  = lm[25].visibility
        rv  = lm[26].visibility
        raw   = (l_k * lv + r_k * rv) / (lv + rv + 1e-8)
        avg_k = smoother.update(raw)
        knee_angles.append(avg_k)

        hip = calculate_angle(ls, lh, lk)
        hip_angles.append(hip)
        ank = calculate_angle(lk, la, lf)
        ankle_angles.append(ank)

        # ── Hip sway ───────────────────────────────────────────────
        hip_sway = abs(lm[23].y - lm[24].y)
        sway_values.append(hip_sway)

        # ── Front-view signal: mean hip-y / shoulder-y + torso height ─
        mean_hip_y      = (lm[23].y + lm[24].y) / 2.0
        mean_shoulder_y = (lm[11].y + lm[12].y) / 2.0
        shoulder_w      = abs(lm[11].x - lm[12].x)
        hip_w           = abs(lm[23].x - lm[24].x)
        torso_h         = abs(mean_hip_y - mean_shoulder_y)

        # ── FIX G: CALIBRATION PHASE (replaces reactive seen_standing) ─
        if not calibrated:
            calib_frames_seen += 1
            calib_shoulder_w.append(shoulder_w)
            calib_hip_w.append(hip_w)
            calib_torso_h.append(torso_h)
            calib_hip_y.append(mean_hip_y)
            calib_shoulder_y.append(mean_shoulder_y)

            window_long_enough = calib_frames_seen >= CALIBRATION_FRAMES
            # Cap how long calibration can run by the fixed max AND by at
            # most half the video's length — a short or held-pose-heavy
            # clip must not let calibration consume nearly the whole
            # video, leaving no frames for actual rep detection.
            effective_max_calib = min(MAX_CALIBRATION_FRAMES,
                                       max(CALIBRATION_FRAMES, total // 2))
            window_maxed_out   = calib_frames_seen >= effective_max_calib

            # View classification only needs the first CALIBRATION_FRAMES —
            # determine it as soon as that fixed window is available, since
            # everything downstream (whether the "looks_standing" extension
            # is even meaningful) depends on knowing the view first.
            provisional_view = None
            if window_long_enough:
                shoulder_w_so_far = calib_shoulder_w[:CALIBRATION_FRAMES]
                hip_w_so_far      = calib_hip_w[:CALIBRATION_FRAMES]
                torso_h_so_far    = calib_torso_h[:CALIBRATION_FRAMES]
                provisional_view = _classify_view(
                    float(np.median(shoulder_w_so_far)),
                    float(np.median(hip_w_so_far)),
                    float(np.median(torso_h_so_far)) or 1e-6,
                )

            # "looks_standing" early-exit extension is ONLY meaningful for
            # side view, where knee angle is an absolute per-frame
            # reference. For front view, a held pose can sit at any
            # constant (and coincidentally high) knee angle without
            # actually being standing, so this signal must not be used
            # to finalize front-view calibration early.
            calib_max_knee_so_far = max(knee_angles[-len(calib_hip_y):]) if knee_angles else 0.0
            looks_standing = (provisional_view == "side") and (calib_max_knee_so_far > 150.0)

            # Front view has no early-exit signal, so it must default to
            # finalizing at the normal fixed window (NOT extending every
            # time, which would eat real rep data). It only extends past
            # the fixed window in the genuinely ambiguous case where the
            # window shows essentially zero hip-y variance (can't tell
            # standing from a held squat) — that ambiguous case alone
            # warrants borrowing extra frames to look for a clearer signal.
            front_window_is_ambiguous = False
            if provisional_view == "front" and window_long_enough:
                hip_y_so_far = calib_hip_y[:max(len(calib_hip_y), CALIBRATION_FRAMES)]
                torso_h_so_far_val = float(np.median(calib_torso_h[:CALIBRATION_FRAMES])) or 1e-6
                hip_y_range = max(hip_y_so_far) - min(hip_y_so_far)
                front_window_is_ambiguous = (hip_y_range / torso_h_so_far_val) < 0.04

            ready_to_finalize = window_long_enough and (
                looks_standing
                or window_maxed_out
                or (provisional_view == "front" and not front_window_is_ambiguous)
            )

            if ready_to_finalize:
                avg_shoulder_w = float(np.median(calib_shoulder_w))
                avg_hip_w      = float(np.median(calib_hip_w))
                avg_torso_h    = float(np.median(calib_torso_h)) or 1e-6
                view_mode      = _classify_view(avg_shoulder_w, avg_hip_w, avg_torso_h)

                # Use the HIGHEST (least-flexed) frame seen during the
                # (possibly extended) calibration window as the standing
                # baseline reference.
                topmost_idx = int(np.argmin(calib_hip_y))  # smallest y = highest = most standing
                standing_hip_y      = calib_hip_y[topmost_idx]
                standing_shoulder_y = calib_shoulder_y[topmost_idx]
                standing_torso_h    = avg_torso_h if avg_torso_h > 1e-6 else 1e-6

                # Refine side-view knee thresholds from the calibration
                # window's own max observed knee angle, with the original
                # 145/120 as sane floors/ceilings (keeps existing side-view
                # behaviour intact when calibration sees a normal stance).
                calib_max_knee = calib_max_knee_so_far if calib_max_knee_so_far > 0 else 145.0
                standing_knee_baseline = min(145.0, max(135.0, calib_max_knee - 5.0))
                squat_knee_baseline    = 120.0  # unchanged — depth criterion, not a gate

                # If we maxed out the window WITHOUT ever finding a frame
                # that looks like standing, the person was likely already
                # mid-rep when calibration started — the first subsequent
                # return to baseline should count as a completed rep.
                # Side view confirms this directly (no frame crossed the
                # absolute 150° standing reference). Front view has no
                # such absolute reference, so it falls through to the
                # relative drop-from-topmost-frame check below in all
                # cases — which also correctly yields "no" when the
                # window shows no real movement (flat pose throughout).
                if view_mode == "side" and window_maxed_out and not looks_standing:
                    started_mid_rep = True
                elif view_mode == "side":
                    started_mid_rep = calib_max_knee < (standing_knee_baseline - 10)
                else:
                    max_calib_drop = max(
                        (y_h - standing_hip_y) / standing_torso_h for y_h in calib_hip_y
                    )
                    started_mid_rep = max_calib_drop > 0.18

                calibrated = True

            if not calibrated:
                # Still mid-calibration — no rep counting/depth tracking yet.
                h, w = frame.shape[:2]
                pts = lm_smoother.smooth(lm, w, h)
                draw_pose_skyblue(frame, res.pose_landmarks, pts)
                draw_footer_hud(frame, [
                    ("REPS", "0"), ("CORRECT", "0"), ("WRONG", "0"), ("FORM", "---"),
                ])
                draw_pcl_logo(frame)
                return frame
            # else: calibration just completed on THIS frame — fall
            # through below so this frame's own motion isn't lost (this
            # matters for short videos or videos that start mid-squat,
            # where the rep signal may begin on the very frame
            # calibration finishes).

        # ── Compute front-view drop % (normalized, smoothed) ───────
        hip_drop_pct      = max(0.0, (mean_hip_y - standing_hip_y) / standing_torso_h)
        shoulder_drop_pct = max(0.0, (mean_shoulder_y - standing_shoulder_y) / standing_torso_h)
        combined_drop_raw = (hip_drop_pct + shoulder_drop_pct) / 2.0
        drop_pct = depth_smoother.update(combined_drop_raw)
        last_drop_pct[0] = drop_pct

        # ── Per-rep trackers (ALL frames counted) ───────────────────
        rep_min_knee   = min(rep_min_knee, avg_k)
        rep_max_knee   = max(rep_max_knee, avg_k)
        rep_worst_sway = max(rep_worst_sway, hip_sway)
        rep_max_drop   = max(rep_max_drop, drop_pct)

        # ── Derived flags ──────────────────────────────────────────
        knee_asym = abs(l_k - r_k)
        bad_k     = avg_k > 170 or knee_asym > 25
        bad_sway  = hip_sway > 0.08

        # ── Stage machine — view-dependent signal, same hysteresis ──
        if view_mode == "side":
            entering_down = avg_k < squat_knee_baseline
            entering_up   = avg_k > standing_knee_baseline
        else:
            entering_down = drop_pct > 0.18
            entering_up   = drop_pct < 0.06

        if started_mid_rep and stage == "up" and rep_count == 0:
            # Person was already squatting when calibration ended.
            # Allow the FIRST ascent observed to complete a rep even
            # though we never saw a clean standing→down transition.
            if entering_up:
                finalise_rep(fc, forced=False, end_drop=drop_pct)
                started_mid_rep = False  # only applies to the very first rep

        if stage == "up":
            if entering_down:
                stage          = "down"
                frames_in_down = 0

        elif stage == "down":
            frames_in_down += 1

            if entering_up:
                # Normal rep completion — person stood back up
                finalise_rep(fc, forced=False, end_drop=drop_pct)

            elif frames_in_down >= stuck_threshold:
                # FIX C: stuck too long → force wrong rep
                finalise_rep(fc, forced=True, end_drop=drop_pct)

        # ── Drawing ────────────────────────────────────────────────
        h, w = frame.shape[:2]
        pts = lm_smoother.smooth(lm, w, h)
        draw_pose_skyblue(frame, res.pose_landmarks, pts)
        draw_angle_arc(frame, lm[25], avg_k, bad=bad_k)
        draw_angle_arc(frame, lm[23], hip_sway * 100,
                       color=(0, 200, 255), bad=bad_sway)

        # ── HUD — navy footer + PCL logo ───────────────────────────
        live_fs = "---"
        if rep_count > 0:
            live_fs = f"{max(4, min(10, round(10-(len(wrong_events)/rep_count)*1.5)))}/10"

        draw_footer_hud(frame, [
            ("REPS",    str(rep_count)),
            ("CORRECT", str(correct_reps)),
            ("WRONG",   str(wrong_reps)),
            ("FORM",    live_fs),
        ])
        draw_pcl_logo(frame)

        return frame

    # ── Run ────────────────────────────────────────────────────────
    snaps = process_video_or_image(path, is_video, pf, output_path=output_path,
        analysis_skip=1,
        progress_uid=progress_uid,
    )
    pose.close()

    # FIX D: force-complete any rep still in "down" at video end
    if stage == "down":
        if view_mode == "side" and rep_min_knee < squat_knee_baseline:
            finalise_rep(fc=total_frames[0], forced=True, end_drop=last_drop_pct[0])
        elif view_mode != "side" and rep_max_drop > 0.18:
            finalise_rep(fc=total_frames[0], forced=True, end_drop=last_drop_pct[0])

    if session_id:
        save_wrong_angle_log("squat", session_id, source_filename, wrong_events)

    avg_k  = round(np.mean(knee_angles),  1) if knee_angles  else 0
    min_k  = round(np.min(knee_angles),   1) if knee_angles  else 0
    avg_h  = round(np.mean(hip_angles),   1) if hip_angles   else 0
    avg_a  = round(np.mean(ankle_angles), 1) if ankle_angles else 0
    avg_sw = round(np.mean(sway_values),  3) if sway_values  else 0

    issues = []; strengths = []

    if view_mode == "side":
        if min_k > 130:
            issues.append(
                f"Very shallow squat ({min_k}° min knee) — aim for parallel (≈90°)")
        elif min_k > 110:
            issues.append(f"Squat depth insufficient ({min_k}° min knee)")
        elif min_k < 70:
            strengths.append(f"Deep squat achieved ({min_k}° min knee)")
        else:
            strengths.append(f"Good squat depth ({min_k}° min knee)")
    else:
        best_depth_pct = round(max((r["max_drop_pct"] for r in frame_data), default=0.0), 1)
        if best_depth_pct < 12:
            issues.append(f"Very shallow squat ({best_depth_pct}% drop) — aim for deeper hip descent")
        elif best_depth_pct < 18:
            issues.append(f"Squat depth insufficient ({best_depth_pct}% drop)")
        else:
            strengths.append(f"Good squat depth ({best_depth_pct}% hip/shoulder drop)")

    if avg_sw > 0.10:
        issues.append(
            f"Hip sway / trunk lean detected (avg {avg_sw:.2f}) — brace core")
    elif avg_h >= 80:
        strengths.append(f"Excellent upright torso ({avg_h}° hip angle)")
    else:
        strengths.append(f"Good trunk position (sway {avg_sw:.2f})")

    if view_mode == "side":
        if avg_a < 40:
            issues.append(f"Very restricted ankle mobility ({avg_a}°)")
        elif avg_a < 55:
            issues.append(f"Ankle mobility limiting depth ({avg_a}°)")
        else:
            strengths.append(f"Sufficient ankle dorsiflexion ({avg_a}°)")

    if correct_reps == rep_count and rep_count > 0:
        strengths.append("All reps completed with correct form!")
    elif correct_reps > 0:
        strengths.append(f"{correct_reps}/{rep_count} reps with good form")

    if not issues:
        issues = ["No major form issues detected"]

    if rep_count > 0:
        issue_rate = len(wrong_events) / rep_count
        form_score = max(4, min(10, round(10 - issue_rate * 1.5)))
    else:
        form_score = 4

    metrics = [
        {"label": "Total Reps",       "value": str(rep_count)},
        {"label": "Correct Reps",     "value": str(correct_reps)},
        {"label": "Wrong Reps",       "value": str(wrong_reps)},
        {"label": "View Detected",    "value": view_mode.capitalize()},
    ]
    if view_mode == "side":
        metrics += [
            {"label": "Avg Knee Angle",   "value": f"{avg_k}°"},
            {"label": "Min Knee (Depth)", "value": f"{min_k}°"},
            {"label": "Ankle Angle",      "value": f"{avg_a}°"},
        ]
    else:
        best_depth_pct = round(max((r["max_drop_pct"] for r in frame_data), default=0.0), 1)
        metrics += [
            {"label": "Best Depth (Hip/Shoulder Drop)", "value": f"{best_depth_pct}%"},
        ]
    metrics += [
        {"label": "Hip Angle",        "value": f"{avg_h}°"},
        {"label": "Avg Hip Sway",     "value": f"{avg_sw:.3f}"},
        {"label": "Form Score",       "value": f"{form_score}/10"},
    ]

    return {
        "exercise"         : "Squat",
        "rep_count"        : rep_count,
        "correct_reps"     : correct_reps,
        "wrong_reps"       : wrong_reps,
        "view_mode"        : view_mode,
        "avg_knee_angle"   : avg_k,
        "min_knee_angle"   : min_k,
        "avg_hip_angle"    : avg_h,
        "avg_ankle_angle"  : avg_a,
        "form_score"       : form_score,
        "issues"           : issues,
        "strengths"        : strengths,
        "per_rep"          : frame_data,
        "snapshots"        : snaps,
        "wrong_angle_count": len(wrong_events),
        "_wrong_events"    : wrong_events,
        "metrics": metrics,
    }
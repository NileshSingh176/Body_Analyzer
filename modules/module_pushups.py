# # import cv2
# # import mediapipe as mp
# # import numpy as np

# # from utils import (
# #     mp_pose,
# #     get_landmark, calculate_angle, draw_angle_arc,
# #     frame_to_b64, RollingMean, process_video_or_image,
# #     save_wrong_angle_log, draw_pose_skyblue,
# # )
# # from hud_overlay import draw_footer_hud, draw_pcl_logo

# # # ════════════════════════════════════════════════════════════════
# # # PUSHUPS — v6  DEFINITIVE FIX
# # #
# # # STANDARD ANGLES:
# # #   UP   (arms extended):    elbow ≥ 145°
# # #   DOWN (chest near floor): elbow ≤ 90°
# # #   Hip sway limit: 0.10 (relaxed — body is horizontal in pushup)
# # #
# # # ── ROOT CAUSES OF PREVIOUS MISCOUNTING ─────────────────────────
# # #
# # #  BUG 1 — seen_up threshold too strict
# # #  ─────────────────────────────────────
# # #  Previous code required avg_e ≥ 150° for 3 consecutive frames
# # #  before accepting the first "down". MediaPipe + RollingMean(3)
# # #  smoothing means the peak smoothed value at arms-extended is
# # #  often 148-158°. If it never hits 150° cleanly for 3 frames,
# # #  seen_up stays False → stage never enters "down" → zero reps.
# # #  FIX: Use ELBOW_UP_SEEN = 155° as a SEPARATE "person is up"
# # #  confirmation threshold. UP_CONFIRM_NEEDED reduced to 2 frames.
# # #
# # #  BUG 2 — ELBOW_DOWN_ENTRY = 100° misses many real pushups
# # #  ──────────────────────────────────────────────────────────
# # #  If someone's range is 110°-168° (common for beginners),
# # #  they never reach 100° → stage never goes "down" → no reps.
# # #  FIX: ELBOW_DOWN_ENTRY = 110°. This detects the bend early.
# # #  Correctness still requires ≤ 90° (ELBOW_DOWN_DEPTH).
# # #
# # #  BUG 3 — ELBOW_UP_LOCKOUT = 150° too strict with smoother lag
# # #  ─────────────────────────────────────────────────────────────
# # #  RollingMean(3) causes lag at peaks. If MediaPipe reads
# # #  160° at peak but the smoothed value is 148°, the rep never
# # #  "completes" (avg_e never reaches 150°) → stage stays "down"
# # #  → stuck_threshold eventually fires → forced wrong rep.
# # #  FIX: ELBOW_UP_LOCKOUT = 145°. Still biomechanically "extended".
# # #
# # #  BUG 4 — Hip sway threshold 0.08 too tight for horizontal body
# # #  ─────────────────────────────────────────────────────────────
# # #  In a pushup, the body is horizontal. Any small hip rotation or
# # #  camera angle causes abs(hip_L.y - hip_R.y) > 0.08 even with
# # #  perfect form. This marks every rep as wrong due to "sway".
# # #  FIX: HIP_SWAY_LIMIT = 0.10 for pushups.
# # #
# # # ── HOW lockout_ok IS CAPTURED CORRECTLY ────────────────────────
# # #
# # #  rep_max_elbow is updated EVERY frame (even during "up" stage).
# # #  When stage == "down" and avg_e rises back to 145°, finalise_rep
# # #  is called. AT THAT MOMENT rep_max_elbow already contains 145°
# # #  (from the current frame's update, which runs BEFORE the stage
# # #  check). So lockout_ok = (145 >= 145) = True. ✓
# # #
# # #  Between reps: stage = "up", rep_max_elbow resets to 0.
# # #  Person stays at ~160°, rep_max_elbow accumulates 160°.
# # #  When they go down and come back up, rep_max_elbow = 160° when
# # #  finalise_rep fires → lockout_ok = True. ✓
# # # ════════════════════════════════════════════════════════════════

# # # ── Angle / threshold constants ─────────────────────────────────
# # ELBOW_UP_SEEN     = 155   # deg — relaxed "person is up" for start guard
# # ELBOW_DOWN_ENTRY  = 110   # deg — entering down phase trigger
# # ELBOW_DOWN_DEPTH  = 90    # deg — correct depth (bottom of pushup)
# # ELBOW_UP_LOCKOUT  = 145   # deg — correct lockout (top of pushup)
# # HIP_SWAY_LIMIT    = 0.10  # normalised (relaxed for horizontal body)
# # STUCK_THRESHOLD   = 90    # frames (~3 sec at 30fps)
# # UP_CONFIRM_NEEDED = 2     # frames to confirm "up" position at start


# # def analyse_pushups(path, is_video, output_path=None,
# #                     session_id=None, source_filename="", progress_uid=None):

# #     pose = mp_pose.Pose(
# #         min_detection_confidence=0.5,
# #         min_tracking_confidence=0.5,
# #     )

# #     rep_count    = 0
# #     correct_reps = 0
# #     wrong_reps   = 0
# #     stage        = "up"   # always start as "up"

# #     elbow_angles = []
# #     sway_values  = []
# #     frame_data   = []
# #     smoother     = RollingMean(3)
# #     wrong_events = []

# #     # ── Per-rep trackers ─────────────────────────────────────────
# #     # Updated EVERY frame regardless of stage.
# #     # Reset only at the END of finalise_rep(), AFTER all checks.
# #     rep_min_elbow  = 180.0
# #     rep_max_elbow  = 0.0
# #     rep_worst_sway = 0.0

# #     # ── Start guard ──────────────────────────────────────────────
# #     seen_up          = False
# #     up_confirm_count = 0

# #     # ── Stuck guard ──────────────────────────────────────────────
# #     frames_in_down = 0

# #     total_frames = [0]

# #     # ────────────────────────────────────────────────────────────
# #     def finalise_rep(fc, forced=False):
# #         nonlocal rep_count, correct_reps, wrong_reps, stage
# #         nonlocal rep_min_elbow, rep_max_elbow, rep_worst_sway
# #         nonlocal frames_in_down

# #         rep_count     += 1
# #         stage          = "up"
# #         frames_in_down = 0

# #         depth_ok   = rep_min_elbow <= ELBOW_DOWN_DEPTH
# #         lockout_ok = (rep_max_elbow >= ELBOW_UP_LOCKOUT) and not forced
# #         sway_ok    = rep_worst_sway <= HIP_SWAY_LIMIT

# #         rep_correct = depth_ok and lockout_ok and sway_ok

# #         if rep_correct:
# #             correct_reps += 1
# #         else:
# #             wrong_reps += 1
# #             if not depth_ok:
# #                 wrong_events.append({
# #                     "frame": fc, "joint": "elbow_depth",
# #                     "angle_deg": round(rep_min_elbow, 1),
# #                     "note": (f"Insufficient depth — min elbow {rep_min_elbow:.0f}°"
# #                              f" (need ≤ {ELBOW_DOWN_DEPTH}°)")
# #                 })
# #             if not lockout_ok:
# #                 wrong_events.append({
# #                     "frame": fc, "joint": "elbow_lockout",
# #                     "angle_deg": round(rep_max_elbow, 1),
# #                     "note": (f"Incomplete lockout — max elbow {rep_max_elbow:.0f}°"
# #                              f" (need ≥ {ELBOW_UP_LOCKOUT}°)"
# #                              + (" [forced]" if forced else ""))
# #                 })
# #             if not sway_ok:
# #                 wrong_events.append({
# #                     "frame": fc, "joint": "hip_sway",
# #                     "angle_deg": round(rep_worst_sway, 3),
# #                     "note": (f"Hip sway / spine sag — "
# #                              f"{rep_worst_sway:.3f} (limit {HIP_SWAY_LIMIT})")
# #                 })

# #         frame_data.append({
# #             "rep":        rep_count,
# #             "min_elbow":  round(rep_min_elbow, 1),
# #             "max_elbow":  round(rep_max_elbow, 1),
# #             "worst_sway": round(rep_worst_sway, 3),
# #             "depth_ok":   depth_ok,
# #             "lockout_ok": lockout_ok,
# #             "sway_ok":    sway_ok,
# #             "correct":    rep_correct,
# #             "forced":     forced,
# #         })

# #         # ── Reset AFTER all checks ───────────────────────────
# #         rep_min_elbow  = 180.0
# #         rep_max_elbow  = 0.0
# #         rep_worst_sway = 0.0

# #     # ────────────────────────────────────────────────────────────
# #     def pf(frame, fc, total):
# #         nonlocal stage, frames_in_down, seen_up, up_confirm_count
# #         nonlocal rep_min_elbow, rep_max_elbow, rep_worst_sway
# #         total_frames[0] = fc

# #         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         res = pose.process(rgb)
# #         if not res.pose_landmarks:
# #             return frame

# #         lm = res.pose_landmarks.landmark

# #         # ── Elbow angle — visibility weighted ─────────────────
# #         ls = get_landmark(lm, 11); le = get_landmark(lm, 13); lw = get_landmark(lm, 15)
# #         rs = get_landmark(lm, 12); re = get_landmark(lm, 14); rw = get_landmark(lm, 16)

# #         l_e = calculate_angle(ls, le, lw)
# #         r_e = calculate_angle(rs, re, rw)
# #         lv  = lm[13].visibility
# #         rv  = lm[14].visibility
# #         raw_e = (l_e * lv + r_e * rv) / (lv + rv + 1e-8)
# #         avg_e = smoother.update(raw_e)
# #         elbow_angles.append(avg_e)

# #         # ── Hip sway ───────────────────────────────────────────
# #         lh = lm[23]; rh = lm[24]
# #         hip_sway = abs(lh.y - rh.y)
# #         sway_values.append(hip_sway)

# #         # ── Per-rep trackers: update EVERY frame ───────────────
# #         # Must happen BEFORE the stage check so that the frame
# #         # which triggers finalise_rep() is already included.
# #         rep_min_elbow  = min(rep_min_elbow, avg_e)
# #         rep_max_elbow  = max(rep_max_elbow, avg_e)
# #         rep_worst_sway = max(rep_worst_sway, hip_sway)

# #         # ── Seen-up guard ──────────────────────────────────────
# #         # ELBOW_UP_SEEN (155°) is intentionally LOOSER than
# #         # ELBOW_UP_LOCKOUT (145°) — it only confirms start posture.
# #         if avg_e >= ELBOW_UP_SEEN:
# #             up_confirm_count += 1
# #             if up_confirm_count >= UP_CONFIRM_NEEDED:
# #                 seen_up = True
# #         else:
# #             up_confirm_count = 0

# #         # ── Stage machine ──────────────────────────────────────
# #         if stage == "up":
# #             if avg_e <= ELBOW_DOWN_ENTRY and seen_up:
# #                 stage          = "down"
# #                 frames_in_down = 0

# #         elif stage == "down":
# #             frames_in_down += 1

# #             if avg_e >= ELBOW_UP_LOCKOUT:
# #                 # Rep complete — person extended arms back up.
# #                 # rep_max_elbow already contains this frame's value
# #                 # (updated above before this check).
# #                 finalise_rep(fc, forced=False)

# #             elif frames_in_down >= STUCK_THRESHOLD:
# #                 finalise_rep(fc, forced=True)

# #         # ── Drawing ────────────────────────────────────────────
# #         draw_pose_skyblue(frame, res.pose_landmarks)
# #         elbow_bad = (stage == "down" and avg_e > ELBOW_DOWN_DEPTH)
# #         draw_angle_arc(frame, lm[13], avg_e, bad=elbow_bad)

# #         sway_bad   = hip_sway > HIP_SWAY_LIMIT
# #         sway_color = (0, 60, 255) if sway_bad else (0, 220, 100)

# #         if stage == "down":
# #             status_txt   = "GOOD DEPTH" if avg_e <= ELBOW_DOWN_DEPTH else "GO LOWER"
# #             status_color = (0, 255, 0)  if avg_e <= ELBOW_DOWN_DEPTH else (0, 165, 255)
# #         else:
# #             status_txt   = "UP / READY"
# #             status_color = (0, 255, 255)

# #         # ── HUD — navy footer + PCL logo ───────────────────────
# #         # Live form score
# #         live_fs = "---"
# #         if rep_count > 0:
# #             live_fs = f"{max(4, min(10, round(10 - (len(wrong_events)/rep_count)*1.5)))}/10"

# #         draw_footer_hud(frame, [
# #             ("REPS",    str(rep_count)),
# #             ("CORRECT", str(correct_reps)),
# #             ("WRONG",   str(wrong_reps)),
# #             ("FORM",    live_fs),
# #         ])
# #         draw_pcl_logo(frame)

# #         return frame

# #     # ── Run ────────────────────────────────────────────────────
# #     snaps = process_video_or_image(path, is_video, pf, output_path=output_path,
# #         analysis_skip=1,
# #         progress_uid=progress_uid,
# #     )
# #     pose.close()

# #     # Force-complete any unfinished rep at video end
# #     if stage == "down" and rep_min_elbow <= ELBOW_DOWN_ENTRY:
# #         finalise_rep(fc=total_frames[0], forced=True)

# #     if session_id:
# #         save_wrong_angle_log("pushups", session_id, source_filename, wrong_events)

# #     avg_e  = round(np.mean(elbow_angles), 1) if elbow_angles else 0
# #     min_e  = round(np.min(elbow_angles),  1) if elbow_angles else 0
# #     avg_sw = round(np.mean(sway_values),  3) if sway_values  else 0

# #     if rep_count > 0:
# #         issue_rate = len(wrong_events) / rep_count
# #         form_score = max(4, min(10, round(10 - issue_rate * 1.5)))
# #     else:
# #         form_score = 4

# #     issues    = []
# #     strengths = []

# #     if min_e > ELBOW_DOWN_DEPTH:
# #         issues.append(
# #             f"Depth insufficient — min elbow {min_e}° (target ≤ {ELBOW_DOWN_DEPTH}°)")
# #     elif min_e < 70:
# #         strengths.append(f"Excellent push-up depth — elbows reach {min_e}°")
# #     else:
# #         strengths.append(f"Good push-up depth ({min_e}°)")

# #     if avg_sw > HIP_SWAY_LIMIT:
# #         issues.append(
# #             f"Hip sway / spine sag detected (avg {avg_sw:.3f}) — brace your core")
# #     else:
# #         strengths.append(f"Stable plank alignment (avg sway {avg_sw:.3f})")

# #     if correct_reps == rep_count and rep_count > 0:
# #         strengths.append("All reps completed with correct form!")
# #     elif correct_reps > 0:
# #         strengths.append(f"{correct_reps}/{rep_count} reps with good form")

# #     if not issues:
# #         issues = ["No major form issues detected"]

# #     return {
# #         "exercise"         : "Pushups",
# #         "rep_count"        : rep_count,
# #         "correct_reps"     : correct_reps,
# #         "wrong_reps"       : wrong_reps,
# #         "avg_elbow_angle"  : avg_e,
# #         "min_elbow_angle"  : min_e,
# #         "spine_alignment"  : avg_sw,
# #         "form_score"       : form_score,
# #         "per_rep"          : frame_data,
# #         "snapshots"        : snaps,
# #         "issues"           : issues,
# #         "strengths"        : strengths,
# #         "wrong_angle_count": len(wrong_events),
# #         "_wrong_events"    : wrong_events,
# #         "metrics": [
# #             {"label": "Total Reps",   "value": str(rep_count)},
# #             {"label": "Correct Reps", "value": str(correct_reps)},
# #             {"label": "Wrong Reps",   "value": str(wrong_reps)},
# #             {"label": "Min Elbow",    "value": f"{min_e}°"},
# #             {"label": "Avg Elbow",    "value": f"{avg_e}°"},
# #             {"label": "Avg Hip Sway", "value": f"{avg_sw:.3f}"},
# #             {"label": "Form Score",   "value": f"{form_score}/10"},
# #         ],
# #     }







# import cv2
# import mediapipe as mp
# import numpy as np

# from utils import (
#     mp_pose,
#     get_landmark, calculate_angle, draw_angle_arc,
#     frame_to_b64, RollingMean, LandmarkSmoother, process_video_or_image,
#     save_wrong_angle_log, draw_pose_skyblue,
# )
# from hud_overlay import draw_footer_hud, draw_pcl_logo, expand_canvas_for_lhs, draw_lhs_panel

# # ════════════════════════════════════════════════════════════════
# # PUSHUPS — v6  DEFINITIVE FIX
# #
# # STANDARD ANGLES:
# #   UP   (arms extended):    elbow ≥ 145°
# #   DOWN (chest near floor): elbow ≤ 90°
# #   Hip sway limit: 0.10 (relaxed — body is horizontal in pushup)
# #
# # ── ROOT CAUSES OF PREVIOUS MISCOUNTING ─────────────────────────
# #
# #  BUG 1 — seen_up threshold too strict
# #  ─────────────────────────────────────
# #  Previous code required avg_e ≥ 150° for 3 consecutive frames
# #  before accepting the first "down". MediaPipe + RollingMean(3)
# #  smoothing means the peak smoothed value at arms-extended is
# #  often 148-158°. If it never hits 150° cleanly for 3 frames,
# #  seen_up stays False → stage never enters "down" → zero reps.
# #  FIX: Use ELBOW_UP_SEEN = 155° as a SEPARATE "person is up"
# #  confirmation threshold. UP_CONFIRM_NEEDED reduced to 2 frames.
# #
# #  BUG 2 — ELBOW_DOWN_ENTRY = 100° misses many real pushups
# #  ──────────────────────────────────────────────────────────
# #  If someone's range is 110°-168° (common for beginners),
# #  they never reach 100° → stage never goes "down" → no reps.
# #  FIX: ELBOW_DOWN_ENTRY = 110°. This detects the bend early.
# #  Correctness still requires ≤ 90° (ELBOW_DOWN_DEPTH).
# #
# #  BUG 3 — ELBOW_UP_LOCKOUT = 150° too strict with smoother lag
# #  ─────────────────────────────────────────────────────────────
# #  RollingMean(3) causes lag at peaks. If MediaPipe reads
# #  160° at peak but the smoothed value is 148°, the rep never
# #  "completes" (avg_e never reaches 150°) → stage stays "down"
# #  → stuck_threshold eventually fires → forced wrong rep.
# #  FIX: ELBOW_UP_LOCKOUT = 145°. Still biomechanically "extended".
# #
# #  BUG 4 — Hip sway threshold 0.08 too tight for horizontal body
# #  ─────────────────────────────────────────────────────────────
# #  In a pushup, the body is horizontal. Any small hip rotation or
# #  camera angle causes abs(hip_L.y - hip_R.y) > 0.08 even with
# #  perfect form. This marks every rep as wrong due to "sway".
# #  FIX: HIP_SWAY_LIMIT = 0.10 for pushups.
# #
# # ── HOW lockout_ok IS CAPTURED CORRECTLY ────────────────────────
# #
# #  rep_max_elbow is updated EVERY frame (even during "up" stage).
# #  When stage == "down" and avg_e rises back to 145°, finalise_rep
# #  is called. AT THAT MOMENT rep_max_elbow already contains 145°
# #  (from the current frame's update, which runs BEFORE the stage
# #  check). So lockout_ok = (145 >= 145) = True. ✓
# #
# #  Between reps: stage = "up", rep_max_elbow resets to 0.
# #  Person stays at ~160°, rep_max_elbow accumulates 160°.
# #  When they go down and come back up, rep_max_elbow = 160° when
# #  finalise_rep fires → lockout_ok = True. ✓
# # ════════════════════════════════════════════════════════════════

# # ── Angle / threshold constants ─────────────────────────────────
# ELBOW_UP_SEEN     = 155   # deg — relaxed "person is up" for start guard
# ELBOW_DOWN_ENTRY  = 110   # deg — entering down phase trigger
# ELBOW_DOWN_DEPTH  = 90    # deg — correct depth (bottom of pushup)
# ELBOW_UP_LOCKOUT  = 145   # deg — correct lockout (top of pushup)
# HIP_SWAY_LIMIT    = 0.10  # normalised (relaxed for horizontal body)
# STUCK_THRESHOLD   = 90    # frames (~3 sec at 30fps)
# UP_CONFIRM_NEEDED = 2     # frames to confirm "up" position at start


# def analyse_pushups(path, is_video, output_path=None,
#                     session_id=None, source_filename="", progress_uid=None):

#     pose = mp_pose.Pose(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5,
#     )

#     rep_count    = 0
#     correct_reps = 0
#     wrong_reps   = 0
#     stage        = "up"   # always start as "up"

#     elbow_angles = []
#     sway_values  = []
#     frame_data   = []
#     smoother     = RollingMean(3)
#     lm_smoother  = LandmarkSmoother(alpha=0.40)
#     wrong_events = []

#     # ── Per-rep trackers ─────────────────────────────────────────
#     # Updated EVERY frame regardless of stage.
#     # Reset only at the END of finalise_rep(), AFTER all checks.
#     rep_min_elbow  = 180.0
#     rep_max_elbow  = 0.0
#     rep_worst_sway = 0.0

#     # ── Start guard ──────────────────────────────────────────────
#     seen_up          = False
#     up_confirm_count = 0

#     # ── Stuck guard ──────────────────────────────────────────────
#     frames_in_down = 0

#     total_frames = [0]

#     # ────────────────────────────────────────────────────────────
#     def finalise_rep(fc, forced=False):
#         nonlocal rep_count, correct_reps, wrong_reps, stage
#         nonlocal rep_min_elbow, rep_max_elbow, rep_worst_sway
#         nonlocal frames_in_down

#         rep_count     += 1
#         stage          = "up"
#         frames_in_down = 0

#         depth_ok   = rep_min_elbow <= ELBOW_DOWN_DEPTH
#         lockout_ok = (rep_max_elbow >= ELBOW_UP_LOCKOUT) and not forced
#         sway_ok    = rep_worst_sway <= HIP_SWAY_LIMIT

#         rep_correct = depth_ok and lockout_ok and sway_ok

#         if rep_correct:
#             correct_reps += 1
#         else:
#             wrong_reps += 1
#             if not depth_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "elbow_depth",
#                     "angle_deg": round(rep_min_elbow, 1),
#                     "note": (f"Insufficient depth — min elbow {rep_min_elbow:.0f}°"
#                              f" (need ≤ {ELBOW_DOWN_DEPTH}°)")
#                 })
#             if not lockout_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "elbow_lockout",
#                     "angle_deg": round(rep_max_elbow, 1),
#                     "note": (f"Incomplete lockout — max elbow {rep_max_elbow:.0f}°"
#                              f" (need ≥ {ELBOW_UP_LOCKOUT}°)"
#                              + (" [forced]" if forced else ""))
#                 })
#             if not sway_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "hip_sway",
#                     "angle_deg": round(rep_worst_sway, 3),
#                     "note": (f"Hip sway / spine sag — "
#                              f"{rep_worst_sway:.3f} (limit {HIP_SWAY_LIMIT})")
#                 })

#         frame_data.append({
#             "rep":        rep_count,
#             "min_elbow":  round(rep_min_elbow, 1),
#             "max_elbow":  round(rep_max_elbow, 1),
#             "worst_sway": round(rep_worst_sway, 3),
#             "depth_ok":   depth_ok,
#             "lockout_ok": lockout_ok,
#             "sway_ok":    sway_ok,
#             "correct":    rep_correct,
#             "forced":     forced,
#         })

#         # ── Reset AFTER all checks ───────────────────────────
#         rep_min_elbow  = 180.0
#         rep_max_elbow  = 0.0
#         rep_worst_sway = 0.0

#     # ────────────────────────────────────────────────────────────
#     def pf(frame, fc, total):
#         nonlocal stage, frames_in_down, seen_up, up_confirm_count
#         nonlocal rep_min_elbow, rep_max_elbow, rep_worst_sway
#         total_frames[0] = fc

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = pose.process(rgb)
#         if not res.pose_landmarks:
#             return frame

#         lm = res.pose_landmarks.landmark

#         # ── Elbow angle — visibility weighted ─────────────────
#         ls = get_landmark(lm, 11); le = get_landmark(lm, 13); lw = get_landmark(lm, 15)
#         rs = get_landmark(lm, 12); re = get_landmark(lm, 14); rw = get_landmark(lm, 16)

#         l_e = calculate_angle(ls, le, lw)
#         r_e = calculate_angle(rs, re, rw)
#         lv  = lm[13].visibility
#         rv  = lm[14].visibility
#         raw_e = (l_e * lv + r_e * rv) / (lv + rv + 1e-8)
#         avg_e = smoother.update(raw_e)
#         elbow_angles.append(avg_e)

#         # ── Hip sway ───────────────────────────────────────────
#         lh = lm[23]; rh = lm[24]
#         hip_sway = abs(lh.y - rh.y)
#         sway_values.append(hip_sway)

#         # ── Per-rep trackers: update EVERY frame ───────────────
#         # Must happen BEFORE the stage check so that the frame
#         # which triggers finalise_rep() is already included.
#         rep_min_elbow  = min(rep_min_elbow, avg_e)
#         rep_max_elbow  = max(rep_max_elbow, avg_e)
#         rep_worst_sway = max(rep_worst_sway, hip_sway)

#         # ── Seen-up guard ──────────────────────────────────────
#         # ELBOW_UP_SEEN (155°) is intentionally LOOSER than
#         # ELBOW_UP_LOCKOUT (145°) — it only confirms start posture.
#         if avg_e >= ELBOW_UP_SEEN:
#             up_confirm_count += 1
#             if up_confirm_count >= UP_CONFIRM_NEEDED:
#                 seen_up = True
#         else:
#             up_confirm_count = 0

#         # ── Stage machine ──────────────────────────────────────
#         if stage == "up":
#             if avg_e <= ELBOW_DOWN_ENTRY and seen_up:
#                 stage          = "down"
#                 frames_in_down = 0

#         elif stage == "down":
#             frames_in_down += 1

#             if avg_e >= ELBOW_UP_LOCKOUT:
#                 # Rep complete — person extended arms back up.
#                 # rep_max_elbow already contains this frame's value
#                 # (updated above before this check).
#                 finalise_rep(fc, forced=False)

#             elif frames_in_down >= STUCK_THRESHOLD:
#                 finalise_rep(fc, forced=True)

#         # ── Drawing ────────────────────────────────────────────
#         h, w = frame.shape[:2]
#         pts = lm_smoother.smooth(lm, w, h)
#         draw_pose_skyblue(frame, res.pose_landmarks, pts)
#         elbow_bad = (stage == "down" and avg_e > ELBOW_DOWN_DEPTH)
#         draw_angle_arc(frame, lm[13], avg_e, bad=elbow_bad)

#         sway_bad   = hip_sway > HIP_SWAY_LIMIT
#         sway_color = (0, 60, 255) if sway_bad else (0, 220, 100)

#         if stage == "down":
#             status_txt   = "GOOD DEPTH" if avg_e <= ELBOW_DOWN_DEPTH else "GO LOWER"
#             status_color = (0, 255, 0)  if avg_e <= ELBOW_DOWN_DEPTH else (0, 165, 255)
#         else:
#             status_txt   = "UP / READY"
#             status_color = (0, 255, 255)

#         # ── HUD — navy footer + PCL logo ───────────────────────
#         # Live form score
#         live_fs = "---"
#         if rep_count > 0:
#             live_fs = f"{max(4, min(10, round(10 - (len(wrong_events)/rep_count)*1.5)))}/10"

#         canvas = expand_canvas_for_lhs(frame)
#         draw_lhs_panel(canvas, [
#             ("REPS",    str(rep_count)),
#             ("CORRECT", str(correct_reps)),
#             ("WRONG",   str(wrong_reps)),
#             ("FORM",    live_fs),
#         ])
#         draw_pcl_logo(canvas)

#         return canvas

#     # ── Run ────────────────────────────────────────────────────
#     snaps = process_video_or_image(path, is_video, pf, output_path=output_path,
#         analysis_skip=1,
#         progress_uid=progress_uid,
#     )
#     pose.close()

#     # Force-complete any unfinished rep at video end
#     if stage == "down" and rep_min_elbow <= ELBOW_DOWN_ENTRY:
#         finalise_rep(fc=total_frames[0], forced=True)

#     if session_id:
#         save_wrong_angle_log("pushups", session_id, source_filename, wrong_events)

#     avg_e  = round(np.mean(elbow_angles), 1) if elbow_angles else 0
#     min_e  = round(np.min(elbow_angles),  1) if elbow_angles else 0
#     avg_sw = round(np.mean(sway_values),  3) if sway_values  else 0

#     if rep_count > 0:
#         issue_rate = len(wrong_events) / rep_count
#         form_score = max(4, min(10, round(10 - issue_rate * 1.5)))
#     else:
#         form_score = 4

#     issues    = []
#     strengths = []

#     if min_e > ELBOW_DOWN_DEPTH:
#         issues.append(
#             f"Depth insufficient — min elbow {min_e}° (target ≤ {ELBOW_DOWN_DEPTH}°)")
#     elif min_e < 70:
#         strengths.append(f"Excellent push-up depth — elbows reach {min_e}°")
#     else:
#         strengths.append(f"Good push-up depth ({min_e}°)")

#     if avg_sw > HIP_SWAY_LIMIT:
#         issues.append(
#             f"Hip sway / spine sag detected (avg {avg_sw:.3f}) — brace your core")
#     else:
#         strengths.append(f"Stable plank alignment (avg sway {avg_sw:.3f})")

#     if correct_reps == rep_count and rep_count > 0:
#         strengths.append("All reps completed with correct form!")
#     elif correct_reps > 0:
#         strengths.append(f"{correct_reps}/{rep_count} reps with good form")

#     if not issues:
#         issues = ["No major form issues detected"]

#     return {
#         "exercise"         : "Pushups",
#         "rep_count"        : rep_count,
#         "correct_reps"     : correct_reps,
#         "wrong_reps"       : wrong_reps,
#         "avg_elbow_angle"  : avg_e,
#         "min_elbow_angle"  : min_e,
#         "spine_alignment"  : avg_sw,
#         "form_score"       : form_score,
#         "per_rep"          : frame_data,
#         "snapshots"        : snaps,
#         "issues"           : issues,
#         "strengths"        : strengths,
#         "wrong_angle_count": len(wrong_events),
#         "_wrong_events"    : wrong_events,
#         "metrics": [
#             {"label": "Total Reps",   "value": str(rep_count)},
#             {"label": "Correct Reps", "value": str(correct_reps)},
#             {"label": "Wrong Reps",   "value": str(wrong_reps)},
#             {"label": "Min Elbow",    "value": f"{min_e}°"},
#             {"label": "Avg Elbow",    "value": f"{avg_e}°"},
#             {"label": "Avg Hip Sway", "value": f"{avg_sw:.3f}"},
#             {"label": "Form Score",   "value": f"{form_score}/10"},
#         ],
#     }

















import cv2
import mediapipe as mp
import numpy as np

from utils import (
    mp_pose,
    get_landmark, calculate_angle, draw_angle_arc,
    frame_to_b64, RollingMean, LandmarkSmoother, process_video_or_image,
    save_wrong_angle_log, draw_pose_skyblue,
)
from hud_overlay import draw_footer_hud, draw_pcl_logo, expand_canvas_for_lhs, draw_lhs_panel

# ════════════════════════════════════════════════════════════════
# PUSHUPS — v7
#
# ROOT CAUSE OF "every rep marked wrong":
#
#   The old metric  hip_sway = abs(hip_L.y - hip_R.y)
#   measures LEFT-RIGHT hip tilt in image Y coords.
#   In a pushup the body is HORIZONTAL, so both hips
#   have nearly the same Y — this value is ~0.00–0.03
#   in perfect form.  BUT any slight camera rotation,
#   lateral lean, or MediaPipe jitter sends it above
#   HIP_SWAY_LIMIT = 0.10, marking the rep wrong even
#   when the athlete has perfect form.
#
# FIX 1 — Replace hip_sway with spine_sag
# ─────────────────────────────────────────
#   spine_sag = perpendicular distance of hip midpoint
#   from the shoulder-midpoint → ankle-midpoint line.
#   This catches ACTUAL faults (hips dropping or piking)
#   and is robust to camera rotation.
#   Threshold: SPINE_SAG_LIMIT = 0.09 (normalised).
#   Falls back to shoulder-hip vertical gap if ankles
#   are occluded.
#
# FIX 2 — Use rolling average sag, not worst-frame
# ──────────────────────────────────────────────────
#   Old code used rep_worst_sag = max(...) so a SINGLE
#   jittery frame invalidated the entire rep.
#   New code uses the MEAN sag across the rep's down
#   phase — one bad frame no longer ruins the rep.
#
# FIX 3 — ELBOW_UP_SEEN lowered 155° → 140°
# ────────────────────────────────────────────
#   Athletes starting mid-set or not fully locking out
#   never reached 155°, so seen_up stayed False → 0 reps.
#   140° is still clearly "arms extended".
#
# FIX 4 — Hysteresis band ELBOW_UP_RESET = 130°
# ───────────────────────────────────────────────
#   After each rep, require avg_e to re-cross 130°
#   before the next "down" can be entered.
#   Prevents RollingMean lag at the top from double-
#   counting a single physical rep as two reps.
#
# FIX 5 — MIN_DOWN_FRAMES = 5 guard
# ───────────────────────────────────
#   Natural finalise_rep only fires after ≥ 5 frames
#   in "down". Stops a momentary dip from triggering
#   an instant phantom rep.
#
# FIX 6 — STUCK_THRESHOLD 90 → 60 frames
# ────────────────────────────────────────
#   3 s was too long; 2 s is still generous.
#
# THRESHOLDS:
#   ELBOW_UP_SEEN     = 140°   (was 155°)
#   ELBOW_DOWN_ENTRY  = 110°   (unchanged)
#   ELBOW_DOWN_DEPTH  =  90°   (unchanged)
#   ELBOW_UP_LOCKOUT  = 145°   (unchanged)
#   ELBOW_UP_RESET    = 130°   (new)
#   SPINE_SAG_LIMIT   =  0.09  (replaces HIP_SWAY_LIMIT)
#   STUCK_THRESHOLD   =  60    (was 90)
#   MIN_DOWN_FRAMES   =   5    (new)
#   UP_CONFIRM_NEEDED =   2    (unchanged)
# ════════════════════════════════════════════════════════════════

ELBOW_UP_SEEN     = 140
ELBOW_DOWN_ENTRY  = 110
ELBOW_DOWN_DEPTH  = 90
ELBOW_UP_LOCKOUT  = 145
ELBOW_UP_RESET    = 130
SPINE_SAG_LIMIT   = 0.09
STUCK_THRESHOLD   = 60
MIN_DOWN_FRAMES   = 5
UP_CONFIRM_NEEDED = 2


# ── Spine sag helper ─────────────────────────────────────────────
def _spine_sag(lm):
    """
    Perpendicular distance of hip midpoint from the
    shoulder-midpoint → ankle-midpoint line (normalised coords).

    0.00 = perfect straight-line alignment
    0.09+ = significant sag or pike (wrong form)

    Falls back to shoulder-hip vertical deviation if ankles
    are not visible (body cropped above ankles).
    """
    ls = lm[11]; rs = lm[12]
    if min(ls.visibility, rs.visibility) < 0.3:
        return 0.0
    sx = (ls.x + rs.x) / 2
    sy = (ls.y + rs.y) / 2

    lh = lm[23]; rh = lm[24]
    if min(lh.visibility, rh.visibility) < 0.3:
        return 0.0
    hx = (lh.x + rh.x) / 2
    hy = (lh.y + rh.y) / 2

    la = lm[27]; ra = lm[28]
    if min(la.visibility, ra.visibility) >= 0.3:
        ax = (la.x + ra.x) / 2
        ay = (la.y + ra.y) / 2
        dx = ax - sx
        dy = ay - sy
        length = np.hypot(dx, dy)
        if length < 1e-6:
            return 0.0
        cross = abs(dx * (sy - hy) - dy * (sx - hx))
        return round(cross / length, 4)
    else:
        # Fallback: vertical gap between shoulder and hip
        return round(abs(hy - sy), 4)


def analyse_pushups(path, is_video, output_path=None,
                    session_id=None, source_filename="", progress_uid=None):

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rep_count    = 0
    correct_reps = 0
    wrong_reps   = 0
    stage        = "up"

    elbow_angles = []
    sag_values   = []
    frame_data   = []
    smoother     = RollingMean(3)
    lm_smoother  = LandmarkSmoother(alpha=0.40)
    wrong_events = []

    # ── Per-rep trackers ─────────────────────────────────────────
    rep_min_elbow  = 180.0
    rep_max_elbow  = 0.0
    # Use a list to accumulate sag values during the DOWN phase
    # so we can take the MEAN (not the worst single frame).
    rep_sag_frames = []

    # ── Start guard ──────────────────────────────────────────────
    seen_up          = False
    up_confirm_count = 0

    # ── Hysteresis guard ─────────────────────────────────────────
    above_reset = False

    # ── Stuck guard ──────────────────────────────────────────────
    frames_in_down = 0

    total_frames = [0]

    # ────────────────────────────────────────────────────────────
    def finalise_rep(fc, forced=False):
        nonlocal rep_count, correct_reps, wrong_reps, stage
        nonlocal rep_min_elbow, rep_max_elbow, rep_sag_frames
        nonlocal frames_in_down, above_reset

        rep_count     += 1
        stage          = "up"
        frames_in_down = 0
        above_reset    = False

        depth_ok   = rep_min_elbow <= ELBOW_DOWN_DEPTH
        lockout_ok = (rep_max_elbow >= ELBOW_UP_LOCKOUT) and not forced

        # Mean sag across the down phase — single jittery frames
        # no longer invalidate the entire rep.
        mean_sag = float(np.mean(rep_sag_frames)) if rep_sag_frames else 0.0
        sag_ok   = mean_sag <= SPINE_SAG_LIMIT

        rep_correct = depth_ok and lockout_ok and sag_ok

        if rep_correct:
            correct_reps += 1
        else:
            wrong_reps += 1
            if not depth_ok:
                wrong_events.append({
                    "frame": fc, "joint": "elbow_depth",
                    "angle_deg": round(rep_min_elbow, 1),
                    "note": (f"Insufficient depth — min elbow {rep_min_elbow:.0f}°"
                             f" (need ≤ {ELBOW_DOWN_DEPTH}°)")
                })
            if not lockout_ok:
                wrong_events.append({
                    "frame": fc, "joint": "elbow_lockout",
                    "angle_deg": round(rep_max_elbow, 1),
                    "note": (f"Incomplete lockout — max elbow {rep_max_elbow:.0f}°"
                             f" (need ≥ {ELBOW_UP_LOCKOUT}°)"
                             + (" [forced]" if forced else ""))
                })
            if not sag_ok:
                wrong_events.append({
                    "frame": fc, "joint": "spine_sag",
                    "angle_deg": round(mean_sag, 4),
                    "note": (f"Spine sag / hip deviation — "
                             f"mean {mean_sag:.4f} (limit {SPINE_SAG_LIMIT})")
                })

        frame_data.append({
            "rep":        rep_count,
            "min_elbow":  round(rep_min_elbow, 1),
            "max_elbow":  round(rep_max_elbow, 1),
            "mean_sag":   round(mean_sag, 4),
            "depth_ok":   bool(depth_ok),
            "lockout_ok": bool(lockout_ok),
            "sag_ok":     bool(sag_ok),
            "correct":    bool(rep_correct),
            "forced":     bool(forced),
        })

        # Reset AFTER all checks
        rep_min_elbow  = 180.0
        rep_max_elbow  = 0.0
        rep_sag_frames = []

    # ────────────────────────────────────────────────────────────
    def pf(frame, fc, total):
        nonlocal stage, frames_in_down, seen_up, up_confirm_count
        nonlocal rep_min_elbow, rep_max_elbow, rep_sag_frames
        nonlocal above_reset
        total_frames[0] = fc

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            canvas = expand_canvas_for_lhs(frame)
            draw_lhs_panel(canvas, [
                ("REPS",    str(rep_count)),
                ("CORRECT", str(correct_reps)),
                ("WRONG",   str(wrong_reps)),
                ("FORM",    "---"),
            ])
            draw_pcl_logo(canvas)
            return canvas

        lm = res.pose_landmarks.landmark

        # ── Elbow angle — visibility weighted ─────────────────
        ls = get_landmark(lm, 11); le = get_landmark(lm, 13); lw = get_landmark(lm, 15)
        rs = get_landmark(lm, 12); re = get_landmark(lm, 14); rw = get_landmark(lm, 16)

        l_e = calculate_angle(ls, le, lw)
        r_e = calculate_angle(rs, re, rw)
        lv  = lm[13].visibility
        rv  = lm[14].visibility
        raw_e = (l_e * lv + r_e * rv) / (lv + rv + 1e-8)
        avg_e = smoother.update(raw_e)
        elbow_angles.append(avg_e)

        # ── Spine sag (replaces hip_sway) ─────────────────────
        spine_sag = _spine_sag(lm)
        sag_values.append(spine_sag)

        # ── Per-rep trackers: update EVERY frame ───────────────
        rep_min_elbow = min(rep_min_elbow, avg_e)
        rep_max_elbow = max(rep_max_elbow, avg_e)
        # Only accumulate sag during the DOWN phase so that the
        # mean reflects the actual pushup movement, not idle standing.
        if stage == "down":
            rep_sag_frames.append(spine_sag)

        # ── Seen-up guard ──────────────────────────────────────
        if avg_e >= ELBOW_UP_SEEN:
            up_confirm_count += 1
            if up_confirm_count >= UP_CONFIRM_NEEDED:
                seen_up = True
        else:
            up_confirm_count = 0

        # ── Hysteresis: track whether avg_e has re-extended ────
        if avg_e >= ELBOW_UP_RESET:
            above_reset = True

        # ── Stage machine ──────────────────────────────────────
        if stage == "up":
            if avg_e <= ELBOW_DOWN_ENTRY and seen_up and above_reset:
                stage          = "down"
                frames_in_down = 0
                above_reset    = False

        elif stage == "down":
            frames_in_down += 1

            if avg_e >= ELBOW_UP_LOCKOUT and frames_in_down >= MIN_DOWN_FRAMES:
                finalise_rep(fc, forced=False)

            elif frames_in_down >= STUCK_THRESHOLD:
                finalise_rep(fc, forced=True)

        # ── Drawing ────────────────────────────────────────────
        h, w = frame.shape[:2]
        pts = lm_smoother.smooth(lm, w, h)
        draw_pose_skyblue(frame, res.pose_landmarks, pts)
        elbow_bad = (stage == "down" and avg_e > ELBOW_DOWN_DEPTH)
        draw_angle_arc(frame, lm[13], avg_e, bad=elbow_bad)

        if stage == "down":
            status_txt   = "GOOD DEPTH" if avg_e <= ELBOW_DOWN_DEPTH else "GO LOWER"
            status_color = (0, 255, 0)  if avg_e <= ELBOW_DOWN_DEPTH else (0, 165, 255)
        else:
            status_txt   = "UP / READY"
            status_color = (0, 255, 255)

        # ── HUD ────────────────────────────────────────────────
        live_fs = "---"
        if rep_count > 0:
            live_fs = f"{max(4, min(10, round(10 - (len(wrong_events)/rep_count)*1.5)))}/10"

        canvas = expand_canvas_for_lhs(frame)
        draw_lhs_panel(canvas, [
            ("REPS",    str(rep_count)),
            ("CORRECT", str(correct_reps)),
            ("WRONG",   str(wrong_reps)),
            ("FORM",    live_fs),
        ])
        draw_pcl_logo(canvas)
        return canvas

    # ── Run ────────────────────────────────────────────────────
    snaps = process_video_or_image(path, is_video, pf, output_path=output_path,
        analysis_skip=1,
        progress_uid=progress_uid,
    )
    pose.close()

    # Force-complete any unfinished rep at video end
    if stage == "down" and rep_min_elbow <= ELBOW_DOWN_ENTRY:
        finalise_rep(fc=total_frames[0], forced=True)

    if session_id:
        save_wrong_angle_log("pushups", session_id, source_filename, wrong_events)

    avg_e   = round(np.mean(elbow_angles), 1) if elbow_angles else 0
    min_e   = round(np.min(elbow_angles),  1) if elbow_angles else 0
    avg_sag = round(np.mean(sag_values),   4) if sag_values   else 0

    if rep_count > 0:
        issue_rate = len(wrong_events) / rep_count
        form_score = max(4, min(10, round(10 - issue_rate * 1.5)))
    else:
        form_score = 4

    issues    = []
    strengths = []

    if min_e > ELBOW_DOWN_DEPTH:
        issues.append(
            f"Depth insufficient — min elbow {min_e}° (target ≤ {ELBOW_DOWN_DEPTH}°)")
    elif min_e < 70:
        strengths.append(f"Excellent push-up depth — elbows reach {min_e}°")
    else:
        strengths.append(f"Good push-up depth ({min_e}°)")

    if avg_sag > SPINE_SAG_LIMIT:
        issues.append(
            f"Spine sag / hip deviation detected (avg {avg_sag:.4f}) — brace your core")
    else:
        strengths.append(f"Stable plank alignment (avg deviation {avg_sag:.4f})")

    if correct_reps == rep_count and rep_count > 0:
        strengths.append("All reps completed with correct form!")
    elif correct_reps > 0:
        strengths.append(f"{correct_reps}/{rep_count} reps with good form")

    if not issues:
        issues = ["No major form issues detected"]

    return {
        "exercise"         : "Pushups",
        "rep_count"        : rep_count,
        "correct_reps"     : correct_reps,
        "wrong_reps"       : wrong_reps,
        "avg_elbow_angle"  : avg_e,
        "min_elbow_angle"  : min_e,
        "spine_alignment"  : avg_sag,
        "form_score"       : form_score,
        "per_rep"          : frame_data,
        "snapshots"        : snaps,
        "issues"           : issues,
        "strengths"        : strengths,
        "wrong_angle_count": len(wrong_events),
        "_wrong_events"    : wrong_events,
        "metrics": [
            {"label": "Total Reps",   "value": str(rep_count)},
            {"label": "Correct Reps", "value": str(correct_reps)},
            {"label": "Wrong Reps",   "value": str(wrong_reps)},
            {"label": "Min Elbow",    "value": f"{min_e}°"},
            {"label": "Avg Elbow",    "value": f"{avg_e}°"},
            {"label": "Avg Sag",      "value": f"{avg_sag:.4f}"},
            {"label": "Form Score",   "value": f"{form_score}/10"},
        ],
    }
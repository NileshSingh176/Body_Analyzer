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
# # module_squat_with_stick.py — v2  DEFINITIVE FIX
# #
# # FIXES vs v1:
# #
# #  BUG A — stage = None (phantom first rep)
# #  ────────────────────────────────────────
# #  stage started as None. The condition `avg_k < 120 and stage != "down"`
# #  is True on the very first frame if person is still crouching to start.
# #  Then `avg_k > 145 and stage == "down"` fires almost immediately →
# #  phantom rep counted with rep_max_knee = 0 → lockout_ok = False →
# #  counted as WRONG even before person moved.
# #  FIX: stage = "up" from start (same as squat/pushup/single_leg_squat).
# #
# #  BUG B — No seen_standing guard
# #  ────────────────────────────────
# #  Without confirming the person is standing first, any video that
# #  starts mid-squat immediately enters "down" → phantom reps.
# #  FIX: seen_standing flag, requires avg_k > (UP_THRESH - 10) for
# #  3 consecutive frames before first "down" is accepted.
# #
# #  BUG C — No stuck_threshold
# #  ────────────────────────────
# #  If person squats but never stands fully (bad lockout), stage stays
# #  "down" forever — rep is silently lost.
# #  FIX: stuck_threshold = 120 frames (~4 sec at 30fps). Forces rep
# #  completion as WRONG if down phase lasts too long.
# #
# #  BUG D — End-of-video rep dropped
# #  ──────────────────────────────────
# #  If video ends while stage == "down", last rep vanishes.
# #  FIX: force-complete after process_video_or_image returns.
# #
# #  BUG E — Inline rep logic → extracted to finalise_rep()
# #  ────────────────────────────────────────────────────────
# #  Rep validation was inline inside pf(). Extracted to finalise_rep()
# #  so it can be called from both normal completion and forced paths.
# #  Resets happen AFTER all checks (same pattern as pushup/squat v6).
# #
# # All prior v1 fixes retained:
# #  - bad_k lower bound removed (deep knees are GOOD)
# #  - hip_sway instead of hip angle (camera-robust)
# #  - bad_wrist threshold relaxed to > 0.05
# #  - RollingMean(3)
# #  - Lockout threshold 145°, down entry 120°
# # ════════════════════════════════════════════════════════════════

# DOWN_THRESH    = 120    # enter "down" when avg_k < this
# UP_THRESH      = 145    # complete rep when avg_k > this
# STUCK_THRESH   = 120    # frames — force-complete if stuck in down (~4 sec at 30fps)
# HIP_SWAY_LIMIT = 0.10
# WRIST_FWD_LIM  = 0.08   # wrist_rel < this = arms at/above shoulder (good)


# def analyse_squat_with_stick(path, is_video, output_path=None,
#                               session_id=None, source_filename="", progress_uid=None):

#     pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

#     rep_count    = 0
#     correct_reps = 0
#     wrong_reps   = 0
#     stage        = "up"    # FIX A: never None

#     knee_angles   = []
#     hip_angles    = []
#     ankle_angles  = []
#     wrist_heights = []
#     sway_values   = []
#     frame_data    = []
#     smoother      = RollingMean(3)
#     wrong_events  = []

#     # Per-rep trackers
#     rep_min_knee    = 180
#     rep_max_knee    = 0
#     rep_worst_sway  = 0.0
#     rep_worst_wrist = -999.0

#     # FIX B: seen_standing guard
#     seen_standing        = False
#     standing_frame_count = 0

#     # FIX C: stuck-in-down
#     frames_in_down = 0

#     total_frames = [0]

#     # ── Helper: finalise a rep ────────────────────────────────────
#     def finalise_rep(fc, forced=False):
#         nonlocal rep_count, correct_reps, wrong_reps, stage
#         nonlocal rep_min_knee, rep_max_knee, rep_worst_sway, rep_worst_wrist
#         nonlocal frames_in_down

#         rep_count     += 1
#         stage          = "up"
#         frames_in_down = 0

#         depth_ok   = rep_min_knee < DOWN_THRESH
#         lockout_ok = (rep_max_knee > UP_THRESH) and not forced
#         sway_ok    = rep_worst_sway < HIP_SWAY_LIMIT
#         wrist_ok   = rep_worst_wrist < WRIST_FWD_LIM

#         rep_correct = depth_ok and lockout_ok and sway_ok and wrist_ok

#         if rep_correct:
#             correct_reps += 1
#         else:
#             wrong_reps += 1
#             if not depth_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "knee_depth",
#                     "angle_deg": round(rep_min_knee, 1),
#                     "note": f"Insufficient depth — {rep_min_knee:.0f}° (need < {DOWN_THRESH}°)"
#                 })
#             if not lockout_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "knee_lockout",
#                     "angle_deg": round(rep_max_knee, 1),
#                     "note": (f"Did not stand fully — {rep_max_knee:.0f}° (need > {UP_THRESH}°)"
#                              + (" [forced]" if forced else ""))
#                 })
#             if not sway_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "hip_sway",
#                     "angle_deg": round(rep_worst_sway, 3),
#                     "note": f"Hip sway {rep_worst_sway:.3f} (limit {HIP_SWAY_LIMIT})"
#                 })
#             if not wrist_ok:
#                 wrong_events.append({
#                     "frame": fc, "joint": "wrist_overhead",
#                     "angle_deg": round(rep_worst_wrist, 3),
#                     "note": f"Arms drifting forward ({rep_worst_wrist:.3f})"
#                 })

#         frame_data.append({
#             "rep":          rep_count,
#             "min_knee":     round(rep_min_knee, 1),
#             "max_knee":     round(rep_max_knee, 1),
#             "worst_sway":   round(rep_worst_sway, 3),
#             "worst_wrist":  round(rep_worst_wrist, 3),
#             "depth_ok":     depth_ok,
#             "lockout_ok":   lockout_ok,
#             "sway_ok":      sway_ok,
#             "wrist_ok":     wrist_ok,
#             "correct":      rep_correct,
#             "forced":       forced,
#         })

#         # Reset AFTER all checks
#         rep_min_knee    = 180
#         rep_max_knee    = 0
#         rep_worst_sway  = 0.0
#         rep_worst_wrist = -999.0

#     # ── Per-frame callback ────────────────────────────────────────
#     def pf(frame, fc, total):
#         nonlocal stage, frames_in_down, seen_standing, standing_frame_count
#         nonlocal rep_min_knee, rep_max_knee, rep_worst_sway, rep_worst_wrist
#         total_frames[0] = fc

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = pose.process(rgb)
#         if not res.pose_landmarks:
#             return frame

#         lm = res.pose_landmarks.landmark
#         lh = get_landmark(lm, 23); lk = get_landmark(lm, 25); la = get_landmark(lm, 27)
#         ls = get_landmark(lm, 11); lf = get_landmark(lm, 31)
#         rh = get_landmark(lm, 24); rk = get_landmark(lm, 26); ra = get_landmark(lm, 28)

#         l_k   = calculate_angle(lh, lk, la)
#         r_k   = calculate_angle(rh, rk, ra)
#         avg_k = smoother.update((l_k + r_k) / 2)
#         knee_angles.append(avg_k)

#         hip = calculate_angle(ls, lh, lk)
#         hip_angles.append(hip)
#         ank = calculate_angle(lk, la, lf)
#         ankle_angles.append(ank)

#         wrist_rel = lm[15].y - lm[11].y   # negative = arms overhead (good)
#         wrist_heights.append(wrist_rel)

#         hip_sway = abs(lm[23].y - lm[24].y)
#         sway_values.append(hip_sway)

#         # Per-rep trackers: update EVERY frame BEFORE stage check
#         rep_min_knee    = min(rep_min_knee, avg_k)
#         rep_max_knee    = max(rep_max_knee, avg_k)
#         rep_worst_sway  = max(rep_worst_sway, hip_sway)
#         rep_worst_wrist = max(rep_worst_wrist, wrist_rel)

#         # Display flags
#         bad_k     = avg_k > 170 or abs(l_k - r_k) > 25
#         bad_sway  = hip_sway > 0.08
#         bad_wrist = wrist_rel > 0.05

#         # FIX B: seen_standing guard (3 consecutive frames)
#         if avg_k > (UP_THRESH - 10):
#             standing_frame_count += 1
#             if standing_frame_count >= 3:
#                 seen_standing = True
#         else:
#             standing_frame_count = 0

#         # Stage machine
#         if stage == "up":
#             if avg_k < DOWN_THRESH and seen_standing:    # FIX A + FIX B
#                 stage          = "down"
#                 frames_in_down = 0

#         elif stage == "down":
#             frames_in_down += 1

#             if avg_k > UP_THRESH:
#                 finalise_rep(fc, forced=False)

#             elif frames_in_down >= STUCK_THRESH:         # FIX C
#                 finalise_rep(fc, forced=True)

#         # Drawing
#         draw_pose_skyblue(frame, res.pose_landmarks)
#         draw_angle_arc(frame, lm[25], avg_k, bad=bad_k)

#         live_min_k = round(min(knee_angles) if knee_angles else avg_k, 0)
#         live_fs = "---"
#         if rep_count > 0:
#             live_fs = f"{max(4, min(10, round(10-(len(wrong_events)/rep_count)*1.5)))}/10"
#         arm_val = "OK" if not bad_wrist else "FWD"

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
#     if stage == "down" and rep_min_knee < DOWN_THRESH:
#         finalise_rep(fc=total_frames[0], forced=True)

#     if session_id:
#         save_wrong_angle_log("squat_with_stick", session_id, source_filename, wrong_events)

#     if is_video and len(knee_angles) < 10:
#         raise ValueError("No reliable overhead squat motion detected.")

#     avg_k  = round(np.mean(knee_angles),   1) if knee_angles   else 0
#     min_k  = round(np.min(knee_angles),    1) if knee_angles   else 0
#     avg_h  = round(np.mean(hip_angles),    1) if hip_angles    else 0
#     avg_a  = round(np.mean(ankle_angles),  1) if ankle_angles  else 0
#     avg_w  = round(np.mean(wrist_heights), 3) if wrist_heights else 0
#     avg_sw = round(np.mean(sway_values),   3) if sway_values   else 0

#     issues = []; strengths = []
#     if min_k > 130:   issues.append(f"Very shallow overhead squat ({min_k}° min knee)")
#     elif min_k > 110: issues.append(f"Squat depth insufficient ({min_k}°)")
#     elif min_k < 75:  strengths.append(f"Excellent overhead squat depth ({min_k}°)")
#     else:             strengths.append(f"Good overhead squat depth ({min_k}°)")

#     if avg_sw > 0.10:  issues.append(f"Hip sway / trunk lean ({avg_sw:.2f}) — brace core")
#     elif avg_h >= 75:  strengths.append(f"Excellent upright trunk ({avg_h}° hip)")
#     else:              strengths.append(f"Good trunk position (sway {avg_sw:.2f})")

#     if avg_a < 40:    issues.append(f"Severely restricted ankle mobility ({avg_a}°)")
#     elif avg_a < 55:  issues.append(f"Ankle dorsiflexion limiting depth ({avg_a}°)")
#     else:             strengths.append(f"Sufficient ankle mobility ({avg_a}°)")

#     if avg_w > 0.08:  issues.append(f"Arms drifting forward (avg {avg_w:.3f}) — keep overhead")
#     else:             strengths.append(f"Overhead arm position good ({avg_w:.3f})")

#     if correct_reps == rep_count and rep_count > 0:
#         strengths.append("All reps correct!")
#     if not issues:
#         issues = ["No major form issues detected"]

#     if rep_count > 0:
#         form_score = max(4, min(10, round(10 - (len(wrong_events) / rep_count) * 1.5)))
#     else:
#         form_score = 4

#     return {
#         "exercise"         : "Overhead Squat with Stick",
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
from hud_overlay import draw_footer_hud, draw_pcl_logo, expand_canvas_for_lhs, draw_lhs_panel

# ════════════════════════════════════════════════════════════════
# module_squat_with_stick.py — v2  DEFINITIVE FIX
#
# FIXES vs v1:
#
#  BUG A — stage = None (phantom first rep)
#  ────────────────────────────────────────
#  stage started as None. The condition `avg_k < 120 and stage != "down"`
#  is True on the very first frame if person is still crouching to start.
#  Then `avg_k > 145 and stage == "down"` fires almost immediately →
#  phantom rep counted with rep_max_knee = 0 → lockout_ok = False →
#  counted as WRONG even before person moved.
#  FIX: stage = "up" from start (same as squat/pushup/single_leg_squat).
#
#  BUG B — No seen_standing guard
#  ────────────────────────────────
#  Without confirming the person is standing first, any video that
#  starts mid-squat immediately enters "down" → phantom reps.
#  FIX: seen_standing flag, requires avg_k > (UP_THRESH - 10) for
#  3 consecutive frames before first "down" is accepted.
#
#  BUG C — No stuck_threshold
#  ────────────────────────────
#  If person squats but never stands fully (bad lockout), stage stays
#  "down" forever — rep is silently lost.
#  FIX: stuck_threshold = 120 frames (~4 sec at 30fps). Forces rep
#  completion as WRONG if down phase lasts too long.
#
#  BUG D — End-of-video rep dropped
#  ──────────────────────────────────
#  If video ends while stage == "down", last rep vanishes.
#  FIX: force-complete after process_video_or_image returns.
#
#  BUG E — Inline rep logic → extracted to finalise_rep()
#  ────────────────────────────────────────────────────────
#  Rep validation was inline inside pf(). Extracted to finalise_rep()
#  so it can be called from both normal completion and forced paths.
#  Resets happen AFTER all checks (same pattern as pushup/squat v6).
#
# All prior v1 fixes retained:
#  - bad_k lower bound removed (deep knees are GOOD)
#  - hip_sway instead of hip angle (camera-robust)
#  - bad_wrist threshold relaxed to > 0.05
#  - RollingMean(3)
#  - Lockout threshold 145°, down entry 120°
# ════════════════════════════════════════════════════════════════

DOWN_THRESH    = 120    # enter "down" when avg_k < this
UP_THRESH      = 145    # complete rep when avg_k > this
STUCK_THRESH   = 120    # frames — force-complete if stuck in down (~4 sec at 30fps)
HIP_SWAY_LIMIT = 0.10
WRIST_FWD_LIM  = 0.08   # wrist_rel < this = arms at/above shoulder (good)


def analyse_squat_with_stick(path, is_video, output_path=None,
                              session_id=None, source_filename="", progress_uid=None):

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    rep_count    = 0
    correct_reps = 0
    wrong_reps   = 0
    stage        = "up"    # FIX A: never None

    knee_angles   = []
    hip_angles    = []
    ankle_angles  = []
    wrist_heights = []
    sway_values   = []
    frame_data    = []
    smoother      = RollingMean(3)
    lm_smoother   = LandmarkSmoother(alpha=0.40)
    wrong_events  = []

    # Per-rep trackers
    rep_min_knee    = 180
    rep_max_knee    = 0
    rep_worst_sway  = 0.0
    rep_worst_wrist = -999.0

    # FIX B: seen_standing guard
    seen_standing        = False
    standing_frame_count = 0

    # FIX C: stuck-in-down
    frames_in_down = 0

    total_frames = [0]

    # ── Helper: finalise a rep ────────────────────────────────────
    def finalise_rep(fc, forced=False):
        nonlocal rep_count, correct_reps, wrong_reps, stage
        nonlocal rep_min_knee, rep_max_knee, rep_worst_sway, rep_worst_wrist
        nonlocal frames_in_down

        rep_count     += 1
        stage          = "up"
        frames_in_down = 0

        depth_ok   = rep_min_knee < DOWN_THRESH
        lockout_ok = (rep_max_knee > UP_THRESH) and not forced
        sway_ok    = rep_worst_sway < HIP_SWAY_LIMIT
        wrist_ok   = rep_worst_wrist < WRIST_FWD_LIM

        rep_correct = depth_ok and lockout_ok and sway_ok and wrist_ok

        if rep_correct:
            correct_reps += 1
        else:
            wrong_reps += 1
            if not depth_ok:
                wrong_events.append({
                    "frame": fc, "joint": "knee_depth",
                    "angle_deg": round(rep_min_knee, 1),
                    "note": f"Insufficient depth — {rep_min_knee:.0f}° (need < {DOWN_THRESH}°)"
                })
            if not lockout_ok:
                wrong_events.append({
                    "frame": fc, "joint": "knee_lockout",
                    "angle_deg": round(rep_max_knee, 1),
                    "note": (f"Did not stand fully — {rep_max_knee:.0f}° (need > {UP_THRESH}°)"
                             + (" [forced]" if forced else ""))
                })
            if not sway_ok:
                wrong_events.append({
                    "frame": fc, "joint": "hip_sway",
                    "angle_deg": round(rep_worst_sway, 3),
                    "note": f"Hip sway {rep_worst_sway:.3f} (limit {HIP_SWAY_LIMIT})"
                })
            if not wrist_ok:
                wrong_events.append({
                    "frame": fc, "joint": "wrist_overhead",
                    "angle_deg": round(rep_worst_wrist, 3),
                    "note": f"Arms drifting forward ({rep_worst_wrist:.3f})"
                })

        frame_data.append({
            "rep":          rep_count,
            "min_knee":     round(rep_min_knee, 1),
            "max_knee":     round(rep_max_knee, 1),
            "worst_sway":   round(rep_worst_sway, 3),
            "worst_wrist":  round(rep_worst_wrist, 3),
            "depth_ok":     depth_ok,
            "lockout_ok":   lockout_ok,
            "sway_ok":      sway_ok,
            "wrist_ok":     wrist_ok,
            "correct":      rep_correct,
            "forced":       forced,
        })

        # Reset AFTER all checks
        rep_min_knee    = 180
        rep_max_knee    = 0
        rep_worst_sway  = 0.0
        rep_worst_wrist = -999.0

    # ── Per-frame callback ────────────────────────────────────────
    def pf(frame, fc, total):
        nonlocal stage, frames_in_down, seen_standing, standing_frame_count
        nonlocal rep_min_knee, rep_max_knee, rep_worst_sway, rep_worst_wrist
        total_frames[0] = fc

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            return frame

        lm = res.pose_landmarks.landmark
        lh = get_landmark(lm, 23); lk = get_landmark(lm, 25); la = get_landmark(lm, 27)
        ls = get_landmark(lm, 11); lf = get_landmark(lm, 31)
        rh = get_landmark(lm, 24); rk = get_landmark(lm, 26); ra = get_landmark(lm, 28)

        l_k   = calculate_angle(lh, lk, la)
        r_k   = calculate_angle(rh, rk, ra)
        avg_k = smoother.update((l_k + r_k) / 2)
        knee_angles.append(avg_k)

        hip = calculate_angle(ls, lh, lk)
        hip_angles.append(hip)
        ank = calculate_angle(lk, la, lf)
        ankle_angles.append(ank)

        wrist_rel = lm[15].y - lm[11].y   # negative = arms overhead (good)
        wrist_heights.append(wrist_rel)

        hip_sway = abs(lm[23].y - lm[24].y)
        sway_values.append(hip_sway)

        # Per-rep trackers: update EVERY frame BEFORE stage check
        rep_min_knee    = min(rep_min_knee, avg_k)
        rep_max_knee    = max(rep_max_knee, avg_k)
        rep_worst_sway  = max(rep_worst_sway, hip_sway)
        rep_worst_wrist = max(rep_worst_wrist, wrist_rel)

        # Display flags
        bad_k     = avg_k > 170 or abs(l_k - r_k) > 25
        bad_sway  = hip_sway > 0.08
        bad_wrist = wrist_rel > 0.05

        # FIX B: seen_standing guard (3 consecutive frames)
        if avg_k > (UP_THRESH - 10):
            standing_frame_count += 1
            if standing_frame_count >= 3:
                seen_standing = True
        else:
            standing_frame_count = 0

        # Stage machine
        if stage == "up":
            if avg_k < DOWN_THRESH and seen_standing:    # FIX A + FIX B
                stage          = "down"
                frames_in_down = 0

        elif stage == "down":
            frames_in_down += 1

            if avg_k > UP_THRESH:
                finalise_rep(fc, forced=False)

            elif frames_in_down >= STUCK_THRESH:         # FIX C
                finalise_rep(fc, forced=True)

        # Drawing
        h, w = frame.shape[:2]
        pts = lm_smoother.smooth(lm, w, h)
        draw_pose_skyblue(frame, res.pose_landmarks, pts)
        draw_angle_arc(frame, lm[25], avg_k, bad=bad_k)

        live_min_k = round(min(knee_angles) if knee_angles else avg_k, 0)
        live_fs = "---"
        if rep_count > 0:
            live_fs = f"{max(4, min(10, round(10-(len(wrong_events)/rep_count)*1.5)))}/10"
        arm_val = "OK" if not bad_wrist else "FWD"

        canvas = expand_canvas_for_lhs(frame)
        draw_lhs_panel(canvas, [
            ("REPS",    str(rep_count)),
            ("CORRECT", str(correct_reps)),
            ("WRONG",   str(wrong_reps)),
            ("FORM",    live_fs),
        ])
        draw_pcl_logo(canvas)
        return canvas

    # ── Run ────────────────────────────────────────────────────────
    snaps = process_video_or_image(path, is_video, pf, output_path=output_path,
        analysis_skip=1,
        progress_uid=progress_uid,
    )
    pose.close()

    # FIX D: force-complete any rep still in "down" at video end
    if stage == "down" and rep_min_knee < DOWN_THRESH:
        finalise_rep(fc=total_frames[0], forced=True)

    if session_id:
        save_wrong_angle_log("squat_with_stick", session_id, source_filename, wrong_events)

    if is_video and len(knee_angles) < 10:
        raise ValueError("No reliable overhead squat motion detected.")

    avg_k  = round(np.mean(knee_angles),   1) if knee_angles   else 0
    min_k  = round(np.min(knee_angles),    1) if knee_angles   else 0
    avg_h  = round(np.mean(hip_angles),    1) if hip_angles    else 0
    avg_a  = round(np.mean(ankle_angles),  1) if ankle_angles  else 0
    avg_w  = round(np.mean(wrist_heights), 3) if wrist_heights else 0
    avg_sw = round(np.mean(sway_values),   3) if sway_values   else 0

    issues = []; strengths = []
    if min_k > 130:   issues.append(f"Very shallow overhead squat ({min_k}° min knee)")
    elif min_k > 110: issues.append(f"Squat depth insufficient ({min_k}°)")
    elif min_k < 75:  strengths.append(f"Excellent overhead squat depth ({min_k}°)")
    else:             strengths.append(f"Good overhead squat depth ({min_k}°)")

    if avg_sw > 0.10:  issues.append(f"Hip sway / trunk lean ({avg_sw:.2f}) — brace core")
    elif avg_h >= 75:  strengths.append(f"Excellent upright trunk ({avg_h}° hip)")
    else:              strengths.append(f"Good trunk position (sway {avg_sw:.2f})")

    if avg_a < 40:    issues.append(f"Severely restricted ankle mobility ({avg_a}°)")
    elif avg_a < 55:  issues.append(f"Ankle dorsiflexion limiting depth ({avg_a}°)")
    else:             strengths.append(f"Sufficient ankle mobility ({avg_a}°)")

    if avg_w > 0.08:  issues.append(f"Arms drifting forward (avg {avg_w:.3f}) — keep overhead")
    else:             strengths.append(f"Overhead arm position good ({avg_w:.3f})")

    if correct_reps == rep_count and rep_count > 0:
        strengths.append("All reps correct!")
    if not issues:
        issues = ["No major form issues detected"]

    if rep_count > 0:
        form_score = max(4, min(10, round(10 - (len(wrong_events) / rep_count) * 1.5)))
    else:
        form_score = 4

    return {
        "exercise"         : "Overhead Squat with Stick",
        "rep_count"        : rep_count,
        "correct_reps"     : correct_reps,
        "wrong_reps"       : wrong_reps,
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
        "metrics": [
            {"label": "Total Reps",       "value": str(rep_count)},
            {"label": "Correct Reps",     "value": str(correct_reps)},
            {"label": "Wrong Reps",       "value": str(wrong_reps)},
            {"label": "Avg Knee Angle",   "value": f"{avg_k}°"},
            {"label": "Min Knee (Depth)", "value": f"{min_k}°"},
            {"label": "Hip Angle",        "value": f"{avg_h}°"},
            {"label": "Ankle Angle",      "value": f"{avg_a}°"},
            {"label": "Avg Hip Sway",     "value": f"{avg_sw:.3f}"},
            {"label": "Form Score",       "value": f"{form_score}/10"},
        ],
    }
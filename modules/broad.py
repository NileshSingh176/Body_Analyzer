import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH   = "broad_jump.mp4"
OUTPUT_PATH  = "output_broad_jump9.mp4"
MODEL_PATH   = "yolov8n-pose.pt"
CONFIDENCE   = 0.3

# ============================================================
# PIXEL TO METER SCALE
# ============================================================
PIXEL_TO_METER = 0.00933

# ============================================================
# DETECTION PARAMETERS
# ============================================================

AIRBORNE_THRESHOLD_PX = 28
LANDING_CONFIRM_FRAMES = 4
MIN_GROUND_FRAMES = 10
MIN_AIRBORNE_FRAMES = 5
MIN_FRAMES_BEFORE_LANDING_CHECK = 4

SMOOTHING = 4

# ============================================================
# COLORS
# ============================================================

SKELETON_COLOR    = (0, 255, 0)
JOINT_COLOR       = (0, 0, 255)
TEXT_COLOR        = (255, 255, 255)
DIST_LINE_COLOR   = (0, 255, 255)
TAKEOFF_COLOR     = (0, 165, 255)
LANDING_COLOR     = (255, 100, 255)

# ============================================================
# COCO SKELETON CONNECTIONS
# ============================================================

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# ============================================================
# LOAD MODEL AND VIDEO
# ============================================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Video nahi mila: {VIDEO_PATH}")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fps    = fps if fps > 0 else 30.0

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

# ============================================================
# STATE MACHINE
# ============================================================

STATE_GROUNDED = "GROUNDED"
STATE_AIRBORNE = "AIRBORNE"
STATE_LANDED   = "LANDED"

state = STATE_GROUNDED

ground_ankle_y        = None
ground_ankle_y_locked = False
ground_frames         = 0

takeoff_ankle_x = None
airborne_frames = 0
peak_airborne_y = None

landing_buf_x  = []
landing_stable = 0

jump_distance = 0.0
jump_count    = 0
all_jumps     = []

ax_hist = deque(maxlen=SMOOTHING)
ay_hist = deque(maxlen=SMOOTHING)

# ============================================================
# MAIN LOOP
# ============================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONFIDENCE, verbose=False)

    best_pts  = None
    best_area = 0

    for result in results:
        if result.keypoints is None:
            continue
        for kpts in result.keypoints.xy.cpu().numpy():
            pts   = kpts.astype(int)
            valid = pts[(pts[:, 0] > 0) & (pts[:, 1] > 0)]
            if len(valid) < 5:
                continue
            area = (valid[:, 0].max() - valid[:, 0].min()) * \
                   (valid[:, 1].max() - valid[:, 1].min())
            if area > best_area:
                best_area = area
                best_pts  = pts

    if best_pts is not None:
        pts = best_pts

        for pair in SKELETON:
            p1, p2 = pts[pair[0]], pts[pair[1]]
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(frame, tuple(p1), tuple(p2), SKELETON_COLOR, 2)

        for p in pts:
            if p[0] > 0 and p[1] > 0:
                cv2.circle(frame, tuple(p), 4, JOINT_COLOR, -1)

        left_ankle  = pts[15]
        right_ankle = pts[16]
        valid_ankles = [a for a in [left_ankle, right_ankle]
                        if a[0] > 0 and a[1] > 0]

        if valid_ankles:
            raw_ax = int(np.mean([a[0] for a in valid_ankles]))
            raw_ay = int(np.mean([a[1] for a in valid_ankles]))
            ax_hist.append(raw_ax)
            ay_hist.append(raw_ay)

    if not ax_hist:
        out.write(frame)
        continue

    ankle_x = int(np.mean(ax_hist))
    ankle_y = int(np.mean(ay_hist))

    # --------------------------------------------------------
    # STATE TRANSITIONS
    # --------------------------------------------------------

    if state == STATE_GROUNDED:

        if ground_ankle_y is None:
            ground_ankle_y = ankle_y
        elif not ground_ankle_y_locked:
            ground_ankle_y = max(ground_ankle_y, ankle_y)

        ground_frames   += 1
        airborne_frames  = 0

        if ground_frames == MIN_GROUND_FRAMES:
            ground_ankle_y_locked = True
            print(f"GROUND LOCKED at y={ground_ankle_y} (frame {ground_frames})")

        if (ground_frames >= MIN_GROUND_FRAMES and
                ankle_y < ground_ankle_y - AIRBORNE_THRESHOLD_PX):
            state           = STATE_AIRBORNE
            takeoff_ankle_x = ankle_x
            peak_airborne_y = None
            landing_buf_x   = []
            landing_stable  = 0
            airborne_frames = 0
            print(f"TAKEOFF  ankle_x={takeoff_ankle_x}  "
                  f"ground_y={ground_ankle_y}  ankle_y={ankle_y}")
        else:
            takeoff_ankle_x = ankle_x

    elif state == STATE_AIRBORNE:

        if peak_airborne_y is None:
            peak_airborne_y = ankle_y
        else:
            peak_airborne_y = min(peak_airborne_y, ankle_y)

        print(f"AIRBORNE f={airborne_frames} ankle_y={ankle_y} "
              f"ground_y={ground_ankle_y} peak_y={peak_airborne_y} "
              f"stable={landing_stable}")

        if airborne_frames < MIN_FRAMES_BEFORE_LANDING_CHECK:
            airborne_frames += 1
            landing_stable = 0
            landing_buf_x  = []
        else:
            abs_near_ground = (ground_ankle_y is not None and
                               ankle_y >= ground_ankle_y - AIRBORNE_THRESHOLD_PX * 2)

            rel_near_ground = (peak_airborne_y is not None and
                               ankle_y >= peak_airborne_y + 30)

            near_ground = abs_near_ground or rel_near_ground

            if near_ground:
                landing_stable += 1
                landing_buf_x.append(ankle_x)
            else:
                airborne_frames += 1
                if landing_stable < 3:
                    landing_stable = 0
                    landing_buf_x  = []

        if landing_stable >= LANDING_CONFIRM_FRAMES:
            if airborne_frames >= MIN_AIRBORNE_FRAMES:
                landing_ankle_x = int(np.median(landing_buf_x))
                pixel_dist      = abs(landing_ankle_x - takeoff_ankle_x)
                jump_distance   = pixel_dist * PIXEL_TO_METER
                jump_count     += 1
                all_jumps.append(jump_distance)
                print(f"LANDING  takeoff_x={takeoff_ankle_x}  "
                      f"landing_x={landing_ankle_x}  "
                      f"px={pixel_dist}  dist={jump_distance:.2f}m  "
                      f"(airborne {airborne_frames} frames)")
                state = STATE_LANDED
            else:
                print(f"SKIP hop  airborne={airborne_frames} frames < {MIN_AIRBORNE_FRAMES}")
                state         = STATE_GROUNDED
                ground_frames = 0

            landing_stable = 0
            landing_buf_x  = []

    elif state == STATE_LANDED:
        state           = STATE_GROUNDED
        ground_frames   = MIN_GROUND_FRAMES
        airborne_frames = 0
        peak_airborne_y = None
        landing_stable  = 0
        landing_buf_x   = []
        print(f"RESET → GROUNDED  (jump {jump_count} complete)")

    # --------------------------------------------------------
    # DRAW DISTANCE LINE + MARKERS
    # --------------------------------------------------------

    line_y = height - 55

    if takeoff_ankle_x is not None and state in (STATE_AIRBORNE, STATE_LANDED):

        cv2.circle(frame, (takeoff_ankle_x, line_y), 9, TAKEOFF_COLOR, -1)
        cv2.putText(frame, "T", (takeoff_ankle_x - 5, line_y - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TAKEOFF_COLOR, 2)

        cv2.circle(frame, (ankle_x, line_y), 7, LANDING_COLOR, -1)

        cv2.line(frame,
                 (takeoff_ankle_x, line_y),
                 (ankle_x, line_y),
                 DIST_LINE_COLOR, 4)

        live_dist = abs(ankle_x - takeoff_ankle_x) * PIXEL_TO_METER
        mid_x = (takeoff_ankle_x + ankle_x) // 2
        label = f"{live_dist:.2f} m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(frame, label,
                    (mid_x - tw // 2, line_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    DIST_LINE_COLOR, 2)

    # --------------------------------------------------------
    # HUD OVERLAY — Top bar + Right side jump log
    # --------------------------------------------------------

    STATE_HUD_COLOR = {
        STATE_GROUNDED: (220, 220, 220),
        STATE_AIRBORNE: (50, 255, 50),
        STATE_LANDED:   (50, 200, 255),
    }

    best_dist = max(all_jumps) if all_jumps else 0.0
    hud_color = STATE_HUD_COLOR[state]

    # Top bar (State, Live, Jumps, Best)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"State : {state}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)

    # Live distance — only when airborne
    if state == STATE_AIRBORNE and takeoff_ankle_x is not None:
        live_m = abs(ankle_x - takeoff_ankle_x) * PIXEL_TO_METER
        cv2.putText(frame, f"Live  : {live_m:.2f} m",
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Live  : --",
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 2)

    cv2.putText(frame, f"Jumps : {jump_count}",
                (width // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
    cv2.putText(frame, f"Best  : {best_dist:.2f} m",
                (width // 2, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 215, 255) if best_dist > 0 else TEXT_COLOR, 2)

    # Right side panel — jump log (grows as jumps complete)
    if all_jumps:
        log_x     = width - 210
        log_top   = 115
        log_h     = 30 + len(all_jumps) * 30
        overlay2  = frame.copy()
        cv2.rectangle(overlay2, (log_x - 8, log_top),
                      (width - 4, log_top + log_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, "-- Jump Log --",
                    (log_x, log_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 50), 2)

        for idx, jd in enumerate(all_jumps):
            is_best = (jd == best_dist and best_dist > 0)
            y_pos   = log_top + 50 + idx * 30
            color   = (0, 255, 120) if is_best else TEXT_COLOR
            star    = " *" if is_best else ""
            cv2.putText(frame, f"Jump {idx+1}: {jd:.2f} m{star}",
                        (log_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

# ============================================================
# END-OF-VIDEO: Force-confirm landing if still AIRBORNE
# ============================================================

if state == STATE_AIRBORNE and airborne_frames >= MIN_AIRBORNE_FRAMES and ax_hist:
    landing_ankle_x = int(np.median(list(ax_hist)))
    pixel_dist      = abs(landing_ankle_x - takeoff_ankle_x)
    jump_distance   = pixel_dist * PIXEL_TO_METER
    jump_count     += 1
    all_jumps.append(jump_distance)
    print(f"END-OF-VIDEO forced landing: takeoff_x={takeoff_ankle_x}  "
          f"landing_x={landing_ankle_x}  px={pixel_dist}  dist={jump_distance:.2f}m")

# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()

print("\n======================================")
print("VIDEO SAVED:", OUTPUT_PATH)
print("======================================")
for i, d in enumerate(all_jumps):
    print(f"Jump {i+1}: {d:.2f} meters")
if all_jumps:
    print(f"Best  : {max(all_jumps):.2f} meters")
print("======================================")
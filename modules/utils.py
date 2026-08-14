"""
utils.py — PCL Body Analyser
Shared utilities for all 8 exercise analysis modules.

Provides:
    mp_pose, mp_drawing, mp_drawing_styles  — MediaPipe shortcuts
    get_landmark(lm, idx)                   — (x, y, z) tuple
    calculate_angle(a, b, c)                — joint angle in degrees
    draw_angle_arc(frame, landmark, angle)  — overlay arc on frame
    frame_to_b64(frame)                     — JPEG → base64 string
    RollingMean(window)                     — smoothing helper
    process_video_or_image(...)             — unified video/image runner
    save_wrong_angle_log(...)               — save JSON log to disk
"""

import os
import cv2
import base64
import subprocess
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Callable, List, Optional


# ════════════════════════════════════════════════════════════════
# Global: last processed video frames for browser playback
# ════════════════════════════════════════════════════════════════
_last_video_frames: dict = {"frames": [], "fps": 6.0}

# ════════════════════════════════════════════════════════════════
# Global: real-time progress tracking per session
# Key: session_uid (str)  Value: {"pct": 0-100, "label": str, "done": bool}
# ════════════════════════════════════════════════════════════════
_progress_store: dict = {}

def set_progress(uid: str, pct: int, label: str = "", done: bool = False, jump_event: dict = None):
    entry = {"pct": pct, "label": label, "done": done}
    if jump_event is not None:
        entry["jump_event"] = jump_event
    _progress_store[uid] = entry

def get_progress(uid: str) -> dict:
    return _progress_store.get(uid, {"pct": 0, "label": "Starting…", "done": False})

def clear_progress(uid: str):
    _progress_store.pop(uid, None)

# ════════════════════════════════════════════════════════════════
# MediaPipe shortcuts  (imported by every module)
# ════════════════════════════════════════════════════════════════
mp_pose           = mp.solutions.pose
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ════════════════════════════════════════════════════════════════
# Landmark helper
# ════════════════════════════════════════════════════════════════
def get_landmark(landmarks, idx: int):
    lm = landmarks[idx]
    return [lm.x, lm.y, lm.z]


# ════════════════════════════════════════════════════════════════
# Angle calculation
# ════════════════════════════════════════════════════════════════
def calculate_angle(a, b, c) -> float:
    a = np.array(a[:2], dtype=float)
    b = np.array(b[:2], dtype=float)
    c = np.array(c[:2], dtype=float)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle  = np.degrees(np.arccos(cosine))
    return float(angle)


# ════════════════════════════════════════════════════════════════
# Angle color helper  (BGR)
# ════════════════════════════════════════════════════════════════
COLOR_GOOD = (0, 196, 122)   # Green
COLOR_BAD  = (68, 68, 255)   # Red

def angle_color(bad: bool) -> tuple:
    return COLOR_BAD if bad else COLOR_GOOD


# ════════════════════════════════════════════════════════════════
# Draw angle arc overlay
# ════════════════════════════════════════════════════════════════
def draw_angle_arc(
    frame,
    landmark,
    angle: float,
    color=None,          # ignored — always green/red based on bad
    bad: bool = False,
    radius: int = 20,
):
    h, w = frame.shape[:2]
    cx = int(landmark.x * w)
    cy = int(landmark.y * h)

    draw_color = COLOR_BAD if bad else COLOR_GOOD

    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 0, int(angle), draw_color, 2)

    text = f"{angle:.0f}"
    cv2.putText(frame, text, (cx + radius + 4, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2)


# ════════════════════════════════════════════════════════════════
# Face landmark indices (0–10 = face mesh points in BlazePose)
# ════════════════════════════════════════════════════════════════
_FACE_LANDMARK_INDICES = set(range(0, 11))

# ════════════════════════════════════════════════════════════════
# Hand landmark indices (17–22 = pinky/index/thumb tips in BlazePose).
# Wrists themselves are 15 (left) / 16 (right) and are KEPT — only the
# finger/palm points beyond the wrist are excluded.
# ════════════════════════════════════════════════════════════════
_HAND_LANDMARK_INDICES = {17, 18, 19, 20, 21, 22}

# Combined set of indices never drawn (face + finger/palm points)
_EXCLUDED_LANDMARK_INDICES = _FACE_LANDMARK_INDICES | _HAND_LANDMARK_INDICES


# ════════════════════════════════════════════════════════════════
# Sky-blue pose drawing — NO face lines, NO palm/finger lines
# (tracking stops at the wrist: landmarks 15/16 are the last point
#  drawn down each arm)
#
# pts: optional (33, 2) array of already-smoothed pixel (x, y)
# coords (e.g. from LandmarkSmoother.smooth()). If provided, these
# are used for drawing positions instead of raw landmark.x/y, while
# `landmarks` is still used for visibility checks.
# ════════════════════════════════════════════════════════════════
def draw_pose_skyblue(frame, pose_landmarks, pts=None):
    if pose_landmarks is None:
        return

    WHITE     = (255, 255, 255)
    JOINT_CLR = (0, 0, 0)
    THICKNESS = 2
    JOINT_R   = 5

    h, w = frame.shape[:2]
    lm   = pose_landmarks.landmark

    def _xy(idx):
        if pts is not None:
            return int(pts[idx][0]), int(pts[idx][1])
        return int(lm[idx].x * w), int(lm[idx].y * h)

    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx in _EXCLUDED_LANDMARK_INDICES or end_idx in _EXCLUDED_LANDMARK_INDICES:
            continue
        s = lm[start_idx]
        e = lm[end_idx]
        if s.visibility > 0.3 and e.visibility > 0.3:
            sx, sy = _xy(start_idx)
            ex, ey = _xy(end_idx)
            cv2.line(frame, (sx, sy), (ex, ey), WHITE, THICKNESS, cv2.LINE_AA)

    for idx, lmk in enumerate(lm):
        if idx in _EXCLUDED_LANDMARK_INDICES:
            continue
        if lmk.visibility > 0.3:
            cx, cy = _xy(idx)
            cv2.circle(frame, (cx, cy), JOINT_R,     JOINT_CLR,      -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), JOINT_R + 1, (180, 180, 180), 1,  cv2.LINE_AA)


draw_pose_green = draw_pose_skyblue


# ════════════════════════════════════════════════════════════════
# Frame → Base64  (JPEG quality 65 — faster transfer)
# ════════════════════════════════════════════════════════════════
def frame_to_b64(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    return base64.b64encode(buf).decode("utf-8")


# ════════════════════════════════════════════════════════════════
# Rolling mean smoother
# ════════════════════════════════════════════════════════════════
class RollingMean:
    def __init__(self, window: int = 5):
        self._buf = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._buf.append(value)
        return float(np.mean(self._buf))

    def reset(self):
        self._buf.clear()


# ════════════════════════════════════════════════════════════════
# LandmarkSmoother — per-landmark EMA on XY pixel positions
# ════════════════════════════════════════════════════════════════
class LandmarkSmoother:
    """
    Exponential Moving Average smoothing on raw MediaPipe landmark
    pixel coordinates (x, y).  One instance per pose session.

    alpha=0.40 → moderate smoothing (lower = smoother but more lag)
    """

    def __init__(self, n_landmarks: int = 33, alpha: float = 0.40):
        self._alpha  = alpha
        self._buf    = np.full((n_landmarks, 2), -1.0, dtype=np.float32)
        self._init   = np.zeros(n_landmarks, dtype=bool)

    def smooth(self, landmarks, frame_w: int, frame_h: int):
        """
        Returns a (33, 2) float32 array of smoothed pixel (x, y) coords.
        Landmarks with visibility < 0.25 are passed through without
        updating the EMA buffer (avoids poisoning the buffer with
        occluded-landmark noise).
        """
        out = np.empty((len(landmarks), 2), dtype=np.float32)
        for i, lmk in enumerate(landmarks):
            px = lmk.x * frame_w
            py = lmk.y * frame_h
            if lmk.visibility >= 0.25:
                if not self._init[i]:
                    self._buf[i] = [px, py]
                    self._init[i] = True
                else:
                    self._buf[i] = (1 - self._alpha) * self._buf[i] + self._alpha * np.array([px, py])
                out[i] = self._buf[i]
            else:
                out[i] = self._buf[i] if self._init[i] else [px, py]
        return out

    def reset(self):
        self._buf[:] = -1.0
        self._init[:] = False


# ════════════════════════════════════════════════════════════════
# Browser-compatible video writer
# Priority: avc1 (H.264) -> H264 -> mp4v + FFmpeg re-encode
# ════════════════════════════════════════════════════════════════
def _ensure_mp4_path(output_path: str) -> str:
    base, ext = os.path.splitext(output_path)
    return base + ".mp4" if ext.lower() != ".mp4" else output_path


def _make_writer(output_path: str, fps: float, width: int, height: int):
    output_path = _ensure_mp4_path(output_path)

    # 1st: avc1 — direct H.264, browser-ready, no re-encode needed
    for codec in ("avc1", "H264"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"  [CODEC]  {codec} (H.264 direct) — browser-ready: {output_path}")
            return writer, codec
        writer.release()

    # Fallback: mp4v — FFmpeg re-encode needed after writing
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"  [CODEC]  mp4v fallback — FFmpeg re-encode will run after: {output_path}")
    return writer, "mp4v"


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5, check=True)
        return True
    except Exception:
        return False


def _reencode_h264(src: str, dst: str) -> bool:
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src,
                "-vcodec", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                dst,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=600,   # FIX: was 300 — long videos (5+ min) need more time
        )
        ok = (result.returncode == 0
              and os.path.exists(dst)
              and os.path.getsize(dst) > 0)
        if not ok:
            err = result.stderr.decode(errors="replace").strip()[-300:]
            print(f"  [FFMPEG] re-encode failed: {err}")
        return ok
    except FileNotFoundError:
        print("  [FFMPEG] ffmpeg not found — install ffmpeg for browser-compatible video")
        return False
    except subprocess.TimeoutExpired:
        print("  [FFMPEG] re-encode TIMEOUT — video is very long, consider shorter clips")
        return False


def _faststart_inplace(path: str) -> bool:
    """
    Move the moov atom to the front of an existing mp4 file (faststart).
    This is the CRITICAL step that lets browsers show the correct video
    duration and seek before the entire file is downloaded.
    Works in-place: writes to a temp file then replaces the original.
    Returns True on success.
    """
    tmp = path + ".faststart_tmp.mp4"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", path,
                "-c", "copy",
                "-movflags", "+faststart",
                tmp,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=600,
        )
        ok = (result.returncode == 0
              and os.path.exists(tmp)
              and os.path.getsize(tmp) > 0)
        if ok:
            os.replace(tmp, path)
            print(f"  [FFMPEG] faststart OK → {path}")
            return True
        else:
            err = result.stderr.decode(errors="replace").strip()[-300:]
            print(f"  [FFMPEG] faststart failed: {err}")
            if os.path.exists(tmp):
                os.remove(tmp)
            return False
    except FileNotFoundError:
        print("  [FFMPEG] ffmpeg not found — browser video may not show full duration")
        return False
    except subprocess.TimeoutExpired:
        print("  [FFMPEG] faststart TIMEOUT")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    except Exception as e:
        print(f"  [FFMPEG] faststart error: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


# ════════════════════════════════════════════════════════════════
# Unified video / image processor
# ════════════════════════════════════════════════════════════════
def process_video_or_image(
    path: str,
    is_video: bool,
    frame_processor: Callable,
    output_path: Optional[str] = None,
    snap_pcts: Optional[List[float]] = None,
    analysis_skip: int = 2,
    progress_uid: Optional[str] = None,
) -> List[str]:
    if snap_pcts is None:
        snap_pcts = [0.1, 0.3, 0.5, 0.7, 0.9]

    snapshots: List[str] = []

    # ── IMAGE ────────────────────────────────────────────────────
    if not is_video:
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Could not read image: {path}")
        h, w = frame.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        processed = frame_processor(frame, 0, 1)
        if processed is not None:
            frame = processed

        if output_path:
            cv2.imwrite(output_path, frame)

        snapshots.append(frame_to_b64(frame))
        return snapshots

    # ── VIDEO ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = 1.0
    if max(width, height) > 1280:
        scale  = 1280 / max(width, height)
        width  = int(width  * scale)
        height = int(height * scale)

    # ── Output video setup ──────────────────────────────────────
    writer     = None
    used_codec = None
    temp_path  = None

    if output_path:
        output_path = _ensure_mp4_path(output_path)
        writer, used_codec = _make_writer(output_path, fps, width, height)

        # mp4v fallback: write to temp, re-encode to H.264 via FFmpeg after
        if used_codec == "mp4v" and _ffmpeg_available():
            base, _ = os.path.splitext(output_path)
            temp_path = base + "_tmp.mp4"
            writer.release()
            writer, _ = _make_writer(temp_path, fps, width, height)
            print("  [FFMPEG] Available — will re-encode after writing")

    snap_indices = set(int(p * max(total - 1, 1)) for p in snap_pcts)

    # ── 6fps canvas + analysis skip for speed ───────────────────
    video_fps_target = 6.0
    frame_skip       = max(1, int(fps / video_fps_target))
    MAX_VIDEO_FRAMES = 200
    ANALYSIS_SKIP    = analysis_skip  # MediaPipe har Nth frame pe (1=har frame, 2=har 2nd frame)
    video_frames_b64 = []
    last_frame       = None     # skipped frames ke liye last annotated frame reuse

    fc = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scale < 1.0:
            frame = cv2.resize(frame, (width, height))

        if fc % ANALYSIS_SKIP == 0:
            processed = frame_processor(frame, fc, total)
            if processed is not None:
                frame = processed
            last_frame = frame
        else:
            if last_frame is not None:
                frame = last_frame

        # ── Real progress emit ────────────────────────────────────
        if progress_uid and total > 0 and fc % max(1, total // 100) == 0:
            pct = max(5, min(95, int(fc / total * 90) + 5))
            set_progress(progress_uid, pct, f"Processing frame {fc}/{total}")

        if writer:
            writer.write(frame)

        if fc in snap_indices:
            snapshots.append(frame_to_b64(frame))

        # Browser video frames — skip + max cap
        if fc % frame_skip == 0 and len(video_frames_b64) < MAX_VIDEO_FRAMES:
            video_frames_b64.append(frame_to_b64(frame))

        fc += 1

    cap.release()
    if writer:
        writer.release()

    # ── Re-encode mp4v → H.264 (only when FFmpeg was available) ──
    if output_path and temp_path and os.path.exists(temp_path):
        print("  [FFMPEG] Re-encoding mp4v → H.264 for browser playback …")
        ok = _reencode_h264(temp_path, output_path)
        if ok:
            print("  [FFMPEG] H.264 re-encode successful ✓")
            # Only remove temp AFTER successful re-encode
            try:
                os.remove(temp_path)
            except OSError:
                pass
        else:
            print("  [FFMPEG] Re-encode failed — using mp4v output as fallback")
            # FIX: was deleting temp BEFORE checking fallback. Now we keep
            # the temp file and rename it as the output if output is missing.
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                try:
                    os.replace(temp_path, output_path)
                    print("  [FFMPEG] Copied mp4v temp as fallback output")
                except OSError:
                    pass
            else:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    # ── ALWAYS run faststart so moov atom is at front of file ──────
    # This is what makes the browser show the correct full video
    # duration immediately, instead of 0:00 or a truncated seekbar.
    # Needed even when avc1/H264 was written directly (no re-encode),
    # because OpenCV's VideoWriter puts moov at the END of the file.
    if output_path and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        _faststart_inplace(output_path)

    while len(snapshots) < len(snap_pcts):
        snapshots.append(snapshots[-1] if snapshots else "")

    _last_video_frames["frames"] = video_frames_b64
    _last_video_frames["fps"]    = video_fps_target

    return snapshots


# ════════════════════════════════════════════════════════════════
# Wrong-angle event logger
# ════════════════════════════════════════════════════════════════
def save_wrong_angle_log(
    exercise: str,
    session_id: str,
    source_filename: str,
    wrong_events: list,
    log_dir: str = "logs",
):
    if not wrong_events:
        return
    import json, os
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{exercise}_{session_id}_wrong.json")
    payload = {
        "exercise":        exercise,
        "session_id":      session_id,
        "source_filename": source_filename,
        "wrong_events":    wrong_events,
    }
    try:
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    except OSError:
        pass
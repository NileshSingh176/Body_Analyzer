
"""
PCL Body Analyser — Flask Backend
Sabhi 8 exercise modules handle karta hai.
Single DB: pcl_body_analyser | 8 fixed tables
"""

import os
import re
import json
import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context
import time

# ── Exercise modules ─────────────────────────────────────────────
from module_pushups          import analyse_pushups
from module_squat            import analyse_squat
from module_single_leg_squat import analyse_single_leg_squat
from module_broad_jump       import analyse_broad_jump
from module_walking_lunges   import analyse_walking_lunges
from module_squat_with_stick import analyse_squat_with_stick
from module_vertical_jump    import analyse_vertical_jump
from module_speed_20m        import analyse_speed_20m

# ── Database ─────────────────────────────────────────────────────
from database import (
    init_db,
    save_analysis_result,
    get_session,
    list_sessions,
    get_exercise_stats,
    start_session_timer,
    stop_session_timer,
)

# ── Utils progress tracking ──────────────────────────────────────
from utils import set_progress, get_progress, clear_progress

# ════════════════════════════════════════════════════════════════
# PATH RESOLUTION
# ════════════════════════════════════════════════════════════════
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

def _find_dir(name):
    candidates = [BASE_DIR, PROJECT_ROOT, os.path.dirname(PROJECT_ROOT)]
    for base in candidates:
        p = os.path.join(base, name)
        if os.path.isdir(p):
            return p
    return os.path.join(PROJECT_ROOT, name)

TEMPLATE_DIR = _find_dir("templates")
STATIC_DIR   = _find_dir("static")
VIDEO_DIR    = os.path.join(PROJECT_ROOT, "Video")
if not os.path.isdir(VIDEO_DIR):
    VIDEO_DIR = _find_dir("Video")

INPUTS_ROOT  = os.path.join(PROJECT_ROOT, "inputs")
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, "outputs")

# ── Flask app ────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB

VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}
ALLOWED_EXTS = VIDEO_EXTS | IMAGE_EXTS

EXERCISE_MAP = {
    "pushups":          analyse_pushups,
    "squat":            analyse_squat,
    "single_leg_squat": analyse_single_leg_squat,
    "broad_jump":       analyse_broad_jump,
    "walking_lunges":   analyse_walking_lunges,
    "squat_with_stick": analyse_squat_with_stick,
    "vertical_jump":    analyse_vertical_jump,
    "speed_20m":        analyse_speed_20m,
}

# ── Helpers ──────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTS

def is_video_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in VIDEO_EXTS

def make_exercise_dirs(exercise: str):
    in_dir  = os.path.join(INPUTS_ROOT,  exercise)
    out_dir = os.path.join(OUTPUTS_ROOT, exercise)
    os.makedirs(in_dir,  exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    return in_dir, out_dir

def save_report(out_dir: str, session_id: str, exercise: str, result: dict):
    """Save a lean JSON report (no snapshots) to disk."""
    report = {
        "session_id": session_id,
        "exercise":   exercise,
        "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result":     {k: v for k, v in result.items() if k != "snapshots"},
    }
    report_filename = f"{exercise}_{session_id}_report.json"
    report_path     = os.path.join(out_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report_filename


def build_metrics(exercise: str, result: dict) -> list:
    """Frontend-ready metrics array."""
    r = result

    if exercise == "speed_20m":
        return [
            {"label": "Duration",   "value": f"{r.get('duration_sec', 0):.1f}s"},
            {"label": "Avg Speed",  "value": f"{r.get('avg_speed_kph', 0):.1f} km/h"},
            {"label": "Peak Speed", "value": f"{r.get('peak_speed_kph', 0):.1f} km/h"},
            {"label": "Form Speed", "value": f"{r.get('form_speed_kph', 0):.1f} km/h"},
            {"label": "Form Score", "value": f"{r.get('form_score', 0)}/10"},
        ]
    if exercise == "pushups":
        return [
            {"label": "Total Reps",   "value": str(r.get('rep_count', 0))},
            {"label": "Correct Reps", "value": str(r.get('correct_reps', 0))},
            {"label": "Wrong Reps",   "value": str(r.get('wrong_reps', 0))},
            {"label": "Min Elbow",    "value": f"{r.get('min_elbow_angle', 0):.1f}°"},
            {"label": "Avg Elbow",    "value": f"{r.get('avg_elbow_angle', 0):.1f}°"},
            {"label": "Form Score",   "value": f"{r.get('form_score', 0)}/10"},
        ]
    if exercise == "squat":
        return [
            {"label": "Rep Count",      "value": str(r.get('rep_count', 0))},
            {"label": "Correct Reps",   "value": str(r.get('correct_reps', 0))},
            {"label": "Avg Knee Angle", "value": f"{r.get('avg_knee_angle', 0):.1f}°"},
            {"label": "Avg Hip Angle",  "value": f"{r.get('avg_hip_angle', 0):.1f}°"},
            {"label": "Form Score",     "value": f"{r.get('form_score', 0)}/10"},
        ]
    if exercise == "single_leg_squat":
        return [
            {"label": "Rep Count",      "value": str(r.get('rep_count', 0))},
            {"label": "Correct Reps",   "value": str(r.get('correct_reps', 0))},
            {"label": "Wrong Reps",     "value": str(r.get('wrong_reps', 0))},
            {"label": "Avg Knee Angle", "value": f"{r.get('avg_knee_angle', 0):.1f}°"},
            {"label": "Pelvic Drop",    "value": f"{r.get('pelvic_drop', 0):.1f}°"},
            {"label": "Form Score",     "value": f"{r.get('form_score', 0)}/10"},
        ]
    if exercise == "broad_jump":
        return [
            {"label": "Jump Distance",  "value": f"{r.get('jump_distance_cm', 0):.1f}cm"},
            {"label": "Takeoff Angle",  "value": f"{r.get('avg_takeoff_angle', 0):.1f}°"},
            {"label": "Landing Angle",  "value": f"{r.get('avg_landing_angle', 0):.1f}°"},
            {"label": "Landing Score",  "value": f"{r.get('landing_score', 0)}/10"},
            {"label": "Form Score",     "value": f"{r.get('form_score', 0)}/10"},
        ]
    if exercise == "walking_lunges":
        return [
            {"label": "Rep Count",    "value": str(r.get('rep_count', 0))},
            {"label": "Correct Reps", "value": str(r.get('correct_reps', 0))},
            {"label": "Avg Knee",     "value": f"{r.get('avg_knee_angle', 0):.1f}°"},
            {"label": "Trunk Angle",  "value": f"{r.get('avg_trunk_angle', 0):.1f}°"},
            {"label": "Form Score",   "value": f"{r.get('form_score', 0)}/10"},
        ]
    if exercise == "squat_with_stick":
        return [
            {"label": "Rep Count",      "value": str(r.get('rep_count', 0))},
            {"label": "Correct Reps",   "value": str(r.get('correct_reps', 0))},
            {"label": "Avg Knee Angle", "value": f"{r.get('avg_knee_angle', 0):.1f}°"},
            {"label": "Avg Hip Angle",  "value": f"{r.get('avg_hip_angle', 0):.1f}°"},
            {"label": "Form Score",     "value": f"{r.get('form_score', 0)}/10"},
        ]
    if exercise == "vertical_jump":
        return [
            {"label": "Jump Height",   "value": f"{r.get('jump_height_cm', 0):.1f}cm"},
            {"label": "Flight Time",   "value": f"{r.get('flight_time_ms', 0)}ms"},
            {"label": "Landing Score", "value": f"{r.get('landing_score', 0)}/10"},
            {"label": "Form Score",    "value": f"{r.get('form_score', 0)}/10"},
        ]
    return []


# ════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ── Demo video streaming ─────────────────────────────────────────
@app.route("/video/<path:filename>")
def video(filename):
    file_path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": f"Video '{filename}' not found."}), 404

    file_size = os.path.getsize(file_path)
    mime      = "video/mp4" if filename.lower().endswith(".mp4") else "video/webm"

    range_header = request.headers.get("Range")
    if range_header:
        byte_start, byte_end = 0, file_size - 1
        m = re.search(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            byte_start = int(m.group(1))
            if m.group(2):
                byte_end = int(m.group(2))
        length = byte_end - byte_start + 1

        def generate():
            with open(file_path, "rb") as fh:
                fh.seek(byte_start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        resp = Response(generate(), status=206, mimetype=mime,
                        content_type=mime, direct_passthrough=True)
        resp.headers["Content-Range"]  = f"bytes {byte_start}-{byte_end}/{file_size}"
        resp.headers["Accept-Ranges"]  = "bytes"
        resp.headers["Content-Length"] = str(length)
        return resp

    return send_from_directory(VIDEO_DIR, filename, mimetype=mime)


# ── Main analysis route ──────────────────────────────────────────
@app.route("/progress/<uid>")
def progress_stream(uid):
    """SSE endpoint — streams real frame-level progress for a running analysis."""
    def generate():
        sent_done = False
        timeout = 300  # max 5 minutes
        start = time.time()
        while not sent_done and (time.time() - start) < timeout:
            p = get_progress(uid)
            data = f"data: {json.dumps(p)}\n\n"
            yield data
            if p.get("done"):
                sent_done = True
                break
            time.sleep(0.4)
        clear_progress(uid)
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ── Per-job result store ───────────────────────────────────────────
# ── Per-job result store (uid → result dict or error dict) ───────
_job_results: dict = {}


def _run_analysis_job(
    uid, progress_uid, exercise, input_path, output_path,
    is_video, source_filename, timer_ctx, out_dir,
    input_filename, output_filename, extra_kwargs,
):
    """Runs in a background thread so SSE can stream real progress."""
    try:
        set_progress(progress_uid, 2, "Initialising AI models…")

        result = EXERCISE_MAP[exercise](
            path=input_path,
            is_video=is_video,
            output_path=output_path,
            session_id=uid,
            source_filename=source_filename,
            progress_uid=progress_uid,
            **extra_kwargs,
        )
        set_progress(progress_uid, 98, "Finalising results…")

        session_id = stop_session_timer(timer_ctx)
        result.pop("_wrong_events", None)

        if "metrics" not in result or not result["metrics"]:
            result["metrics"] = build_metrics(exercise, result)

        report_file = save_report(out_dir, session_id, exercise, result)
        print(f"  [OUTPUT] video  → outputs/{exercise}/{output_filename}")
        print(f"  [OUTPUT] report → outputs/{exercise}/{report_file}")
        print(f"  [SESSION] {session_id}")

        try:
            save_analysis_result(
                exercise=exercise,
                result=result,
                session_id=session_id,
                input_file=input_filename,
                output_file=output_filename,
            )
        except Exception as db_err:
            app.logger.error("DB save failed for session %s: %s", session_id, db_err)

        result["session_id"]  = session_id
        result["input_file"]  = input_filename
        result["output_file"] = output_filename
        result["report_file"] = report_file
        result["output_url"]  = f"/outputs/{exercise}/{output_filename}"
        result["media_type"]  = "video" if is_video else "image"

        if is_video:
            from utils import _last_video_frames
            result["video_frames"] = _last_video_frames.get("frames", [])
            result["video_fps"]    = _last_video_frames.get("fps", 8.0)

        _job_results[uid] = {"ok": True, "result": result}
        set_progress(progress_uid, 100, "Complete", done=True)

    except ValueError as ve:
        _job_results[uid] = {
            "ok": False, "error": str(ve),
            "remark": "Ensure the full exercise movement is clearly visible in the video."
        }
        set_progress(progress_uid, 100, "Error", done=True)

    except Exception as e:
        app.logger.exception("Analysis failed for '%s'", exercise)
        _job_results[uid] = {
            "ok": False, "error": f"Analysis failed: {str(e)}",
            "remark": "Please try again with a clearer video or different angle."
        }
        set_progress(progress_uid, 100, "Error", done=True)


@app.route("/analyse", methods=["POST"])
def analyse():
    import threading

    exercise = request.form.get("exercise", "").strip().lower()
    if not exercise:
        return jsonify({"error": "Exercise field missing in request."}), 400
    if exercise not in EXERCISE_MAP:
        return jsonify({
            "error": f"Unknown exercise '{exercise}'.",
            "valid_exercises": list(EXERCISE_MAP.keys()),
        }), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(f.filename):
        return jsonify({
            "error": f"Unsupported file type '{os.path.splitext(f.filename)[1]}'.",
            "remark": "Upload MP4, MOV, AVI, WEBM, JPG, or PNG.",
        }), 400

    ext      = os.path.splitext(f.filename)[1].lower()
    is_video = is_video_file(f.filename)
    out_ext  = ".mp4" if is_video else ext

    in_dir, out_dir = make_exercise_dirs(exercise)

    timer_ctx = start_session_timer(exercise)
    uid       = timer_ctx["uid"]

    input_filename  = f"{exercise}_{uid}{ext}"
    input_path      = os.path.join(in_dir, input_filename)
    output_filename = f"{exercise}_{uid}_output{out_ext}"
    output_path     = os.path.join(out_dir, output_filename)

    f.save(input_path)
    print(f"  [INPUT]  saved → inputs/{exercise}/{input_filename}")

    progress_uid = request.form.get("progress_uid", uid)

    try:
        user_height_cm = float(request.form.get("user_height_cm", 0))
    except (TypeError, ValueError):
        user_height_cm = 0.0

    extra_kwargs = {}
    if exercise == "broad_jump" and user_height_cm > 0:
        extra_kwargs["user_height_cm"] = user_height_cm

    # Launch in background thread — SSE can now stream freely
    t = threading.Thread(
        target=_run_analysis_job,
        args=(uid, progress_uid, exercise, input_path, output_path,
              is_video, f.filename, timer_ctx, out_dir,
              input_filename, output_filename, extra_kwargs),
        daemon=True,
    )
    t.start()

    # Return uid immediately — frontend polls /result/<uid> after SSE done
    return jsonify({"uid": uid, "progress_uid": progress_uid, "status": "processing"}), 202


@app.route("/result/<uid>", methods=["GET"])
def get_result(uid):
    """Poll this after SSE signals done=True to get the full analysis result."""
    job = _job_results.pop(uid, None)
    if job is None:
        return jsonify({"status": "pending"}), 202
    if job["ok"]:
        return jsonify(job["result"]), 200
    return jsonify({"error": job["error"], "remark": job.get("remark", "")}), 422


# ── Output file serving ──────────────────────────────────────────
@app.route("/outputs/<exercise>/<path:filename>")
def serve_output(exercise, filename):
    out_dir   = os.path.join(OUTPUTS_ROOT, exercise)
    file_path = os.path.join(out_dir, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    file_size = os.path.getsize(file_path)
    mime = "video/mp4" if filename.lower().endswith(".mp4") else "application/octet-stream"

    range_header = request.headers.get("Range")
    if range_header:
        byte_start, byte_end = 0, file_size - 1
        m = re.search(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            byte_start = int(m.group(1))
            if m.group(2):
                byte_end = int(m.group(2))
        length = byte_end - byte_start + 1

        def generate():
            with open(file_path, "rb") as fh:
                fh.seek(byte_start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        resp = Response(generate(), status=206, mimetype=mime,
                        content_type=mime, direct_passthrough=True)
        resp.headers["Content-Range"]  = f"bytes {byte_start}-{byte_end}/{file_size}"
        resp.headers["Accept-Ranges"]  = "bytes"
        resp.headers["Content-Length"] = str(length)
        return resp

    return send_from_directory(out_dir, filename, mimetype=mime)


# ── Utility routes ───────────────────────────────────────────────
@app.route("/exercises", methods=["GET"])
def list_exercises():
    return jsonify({"exercises": list(EXERCISE_MAP.keys())}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modules_loaded": len(EXERCISE_MAP)}), 200


@app.route("/files/<exercise>", methods=["GET"])
def list_files(exercise):
    if exercise not in EXERCISE_MAP:
        return jsonify({"error": f"Unknown exercise '{exercise}'"}), 400
    in_dir  = os.path.join(INPUTS_ROOT,  exercise)
    out_dir = os.path.join(OUTPUTS_ROOT, exercise)
    inputs  = sorted(os.listdir(in_dir))  if os.path.isdir(in_dir)  else []
    outputs = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
    return jsonify({"exercise": exercise, "inputs": inputs, "outputs": outputs}), 200


# ════════════════════════════════════════════════════════════════
# DB-BACKED API ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/sessions", methods=["GET"])
def api_list_sessions():
    exercise = request.args.get("exercise")
    limit    = int(request.args.get("limit", 50))
    if exercise and exercise not in EXERCISE_MAP:
        return jsonify({"error": f"Unknown exercise '{exercise}'"}), 400
    try:
        rows = list_sessions(exercise=exercise, limit=limit)
        return jsonify({"sessions": rows, "count": len(rows)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sessions/<session_id>", methods=["GET"])
def api_get_session(session_id):
    try:
        data = get_session(session_id)
        if not data:
            return jsonify({"error": f"Session '{session_id}' not found"}), 404
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stats/<exercise>", methods=["GET"])
def api_exercise_stats(exercise):
    if exercise not in EXERCISE_MAP:
        return jsonify({"error": f"Unknown exercise '{exercise}'"}), 400
    try:
        stats = get_exercise_stats(exercise)
        return jsonify({"exercise": exercise, "stats": stats}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  PCL Body Analyser — Starting Flask Server")
    print("=" * 55)
    print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"  templates    : {TEMPLATE_DIR}")
    print(f"  static       : {STATIC_DIR}")
    print(f"  inputs/      : {INPUTS_ROOT}")
    print(f"  outputs/     : {OUTPUTS_ROOT}")
    print(f"  Video/       : {VIDEO_DIR}")
    print()

    # ── Initialise DB ─────────────────────────────────────────────
    try:
        init_db()
    except Exception as db_err:
        print(f"  WARN  DB init failed: {db_err}")

    # ── Create exercise dirs ──────────────────────────────────────
    for ex in EXERCISE_MAP:
        os.makedirs(os.path.join(INPUTS_ROOT,  ex), exist_ok=True)
        os.makedirs(os.path.join(OUTPUTS_ROOT, ex), exist_ok=True)

    # ── Check demo videos ─────────────────────────────────────────
    for ex in EXERCISE_MAP:
        for prefix in ("input", "output"):
            demo = f"{prefix}-{ex}.mp4"
            vp = os.path.join(VIDEO_DIR, demo)
            status = "OK  " if os.path.exists(vp) else "WARN"
            print(f"  {status}  Video/{demo}")

    print(f"\n  Loaded {len(EXERCISE_MAP)} exercise modules:")
    for name in EXERCISE_MAP:
        print(f"    OK  {name}")

    print()
    print("  DB: pcl_body_analyser | 8 fixed tables")
    print("  API routes:")
    print("    GET  /sessions                   — all sessions")
    print("    GET  /sessions?exercise=squat    — filter by exercise")
    print("    GET  /sessions/<id>              — session detail")
    print("    GET  /stats/<exercise>           — aggregate stats")
    print("-" * 55)
    print("  Open : http://localhost:5001")
    print("=" * 55)

    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
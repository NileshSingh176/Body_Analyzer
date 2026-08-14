# PCL Body Analyser

An AI-powered sports movement analysis platform that analyses 8 different exercises using **MediaPipe Pose** and **YOLOv8** detection. Upload a video or image and get real-time form feedback, rep counting, speed metrics, and annotated output — all via a Flask web interface.

---

## Project Structure

```
pcl_project/
├── modules/
│   ├── app.py                      # Flask backend — all routes & API
│   ├── database.py                 # SQLite storage (auto-creates pcl_analyser.db)
│   ├── utils.py                    # Shared utilities (pose, drawing, video I/O)
│   ├── module_pushups.py           # Push-up analysis
│   ├── module_squat.py             # Squat analysis
│   ├── module_single_leg_squat.py  # Single-leg squat analysis
│   ├── module_broad_jump.py        # Broad jump analysis
│   ├── module_walking_lunges.py    # Walking lunges analysis
│   ├── module_squat_with_stick.py  # Overhead squat with stick analysis
│   ├── module_vertical_jump.py     # Vertical jump analysis
│   ├── module_speed_20m.py         # 20m sprint speed analysis
│   ├── yolov8n.pt                  # YOLOv8 nano model weights
│   └── __init__.py
├── inputs/                         # Uploaded files (auto-created per exercise)
│   ├── squat/
│   ├── pushups/
│   └── ...
├── outputs/                        # Annotated outputs + JSON reports (auto-created)
│   ├── squat/
│   ├── pushups/
│   └── ...
├── templates/
│   └── index.html                  # Frontend UI
├── static/
│   └── styles.css
├── Video/                          # Demo videos for homepage
└── pcl_analyser.db                 # SQLite database (auto-created on first run)
```

---

## Supported Exercises

| Exercise ID | Module | What It Measures |
|---|---|---|
| `pushups` | `module_pushups.py` | Elbow angle, back alignment, neck angle, rep count |
| `squat` | `module_squat.py` | Knee depth, hip angle, ankle mobility, hip sway, rep count |
| `single_leg_squat` | `module_single_leg_squat.py` | Knee angle, pelvic drop, rep count |
| `broad_jump` | `module_broad_jump.py` | Jump distance, takeoff/landing angle, landing score |
| `walking_lunges` | `module_walking_lunges.py` | Front knee depth, trunk angle, hip sway, rep count |
| `squat_with_stick` | `module_squat_with_stick.py` | Overhead squat depth, wrist position, ankle mobility |
| `vertical_jump` | `module_vertical_jump.py` | Jump height, flight time, landing score |
| `speed_20m` | `module_speed_20m.py` | Sprint speed (km/h), knee drive, stride asymmetry |

---

## Features

- **Pose Estimation** — MediaPipe BlazePose extracts 33 body landmarks per frame
- **Person Detection** — YOLOv8n tracks the athlete for speed/distance calculation
- **Rep Counting** — State-machine logic with per-rep pass/fail validation
- **Form Scoring** — 0–10 score based on form error rate
- **Annotated Output** — Angle arcs and live metrics overlaid on the output video/image
- **JSON Reports** — Per-session report saved to `outputs/<exercise>/`
- **SQLite Database** — All results persisted automatically; no server setup needed
- **Browser Video Streaming** — HTTP range requests for smooth in-browser playback
- **REST API** — Query sessions and aggregate stats programmatically

---

## Tech Stack

| Component | Library |
|---|---|
| Web framework | Flask 3.x |
| Person detection | Ultralytics YOLOv8 (`yolov8n.pt`) |
| Pose estimation | MediaPipe Pose |
| Video / image processing | OpenCV |
| Numerical computation | NumPy |
| Database | SQLite (stdlib `sqlite3`) |
| H.264 re-encoding | FFmpeg (optional, auto-detected) |

See `requirements.txt` for the full dependency list.

---

## Installation

```bash
pip install -r requirements.txt
```

FFmpeg is optional but recommended for browser-compatible H.264 output.

```bash
# Windows (via Chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

---

## Running the Server

```bash
python -u modules/app.py
```

On first run the server will:
1. Create `pcl_analyser.db` in the project root automatically
2. Create all `inputs/<exercise>/` and `outputs/<exercise>/` directories
3. Check for demo videos in `Video/` and warn if any are missing

Then open [http://localhost:5000](http://localhost:5000)

---

## API Reference

### Core

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Frontend UI |
| `GET` | `/health` | Server health check |
| `GET` | `/exercises` | List all supported exercise IDs |

### Analysis

#### `POST /analyse`

Upload a video or image for analysis.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | Video or image to analyse |
| `exercise` | String | Exercise ID (e.g. `squat`, `pushups`) |

Supported formats: `.mp4`, `.avi`, `.mov`, `.webm`, `.mkv`, `.jpg`, `.jpeg`, `.png`, `.bmp`
Max file size: **200 MB**

**Example response (squat):**
```json
{
  "exercise": "Squat",
  "session_id": "a1b2c3d4",
  "rep_count": 10,
  "correct_reps": 8,
  "wrong_reps": 2,
  "avg_knee_angle": 112.3,
  "min_knee_angle": 88.5,
  "avg_hip_angle": 95.1,
  "avg_ankle_angle": 62.4,
  "form_score": 8,
  "issues": ["Hip sway detected on rep 3"],
  "strengths": ["Good squat depth (88.5°)", "Sufficient ankle dorsiflexion (62°)"],
  "output_url": "/outputs/squat/<filename>_annotated.mp4",
  "media_type": "video",
  "metrics": [...]
}
```

### File Serving

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/outputs/<exercise>/<filename>` | Serve annotated video/image or JSON report |
| `GET` | `/video/<filename>` | Serve demo video from `Video/` folder |
| `GET` | `/files/<exercise>` | List input and output files for an exercise |

### Database API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/sessions` | List recent sessions (all exercises) |
| `GET` | `/sessions?exercise=squat&limit=20` | Filter sessions by exercise |
| `GET` | `/sessions/<session_id>` | Full session detail (metrics, reps, errors) |
| `GET` | `/sessions/<session_id>/raw` | Raw JSON result stored at analysis time |
| `GET` | `/stats/<exercise>` | Aggregate stats across all sessions |

**Example `/stats/squat` response:**
```json
{
  "exercise": "squat",
  "stats": {
    "total_sessions": 42,
    "avg_form_score": 7.6,
    "best_form_score": 10.0,
    "avg_reps": 9.3
  }
}
```

---

## Database Schema

SQLite database at `pcl_analyser.db` — auto-created on first run. No configuration needed.

| Table | Description |
|---|---|
| `sessions` | One row per analysis run — all metrics, angles, and exercise-specific fields |
| `metrics` | Frontend display cards (label/value pairs) per session |
| `issues` | Form problems detected per session |
| `strengths` | Positive form observations per session |
| `rep_data` | Per-rep breakdown for squat, lunges, and squat_with_stick |
| `wrong_angle_events` | Every bad-angle frame event with joint name, angle value, and frame number |

---

## Form Scoring — How It Works

Each module validates form per-rep using joint angle thresholds:

| Check | Threshold | Flag |
|---|---|---|
| Squat depth | Knee angle < 120° | `depth_ok` |
| Lockout | Knee angle > 145° | `lockout_ok` |
| Hip sway | Sway value < 0.10 | `sway_ok` |
| Lunge depth | Front knee < 120° | `depth_ok` |
| Overhead arms | Wrist relative position < 0.08 | `wrist_ok` |

A rep is **correct** only when all applicable checks pass. `form_score` is computed as:

```
form_score = max(4, min(10, 10 - (wrong_events / rep_count) × 1.5))
```

---

## Key Modules

### `app.py`
Flask server — handles upload, routes to the right exercise module, saves to DB, and returns enriched JSON. DB errors are caught silently so analysis always returns a result even if the database is unavailable.

### `database.py`
SQLite persistence layer. Uses `ON CONFLICT DO UPDATE` upsert so re-running the same `session_id` overwrites the previous result cleanly. All connections use WAL journal mode for safe concurrent writes.

### `utils.py`
Shared helpers used by all 8 modules: `get_landmark()`, `calculate_angle()`, `draw_angle_arc()`, `draw_pose_skyblue()`, `process_video_or_image()`, `RollingMean`, `frame_to_b64()`. Also handles video writing with automatic H.264 codec detection and optional FFmpeg re-encoding fallback.

### Exercise Modules
Each module exports a single `analyse_<exercise>(path, is_video, output_path, session_id, source_filename)` function. It runs a per-frame callback with MediaPipe and (for speed) YOLO, accumulates angle data, validates form per-rep, and returns a standardised result dict that `app.py` saves and forwards to the frontend.

---

## Notes

- **Speed calibration** (`module_speed_20m.py`): assumes the 20m track spans ~600 pixels horizontally. Adjust `CALIBRATION_PIXEL_DISTANCE` to match your camera setup for accurate km/h readings.
- **Stuck-rep detection** (`module_squat.py`): if an athlete squats but never stands up within 60 frames (~2 sec at 30fps), the rep is force-completed and marked as wrong. Adjust `stuck_threshold` for slower cameras.
- **Browser video frames**: `process_video_or_image()` sub-samples annotated frames to 6fps (max 200 frames) for efficient browser playback, stored in `_last_video_frames`.
- **YOLOv8 weights**: `yolov8n.pt` must be in the `modules/` folder. It downloads automatically via Ultralytics on first run if not found locally.
- **DB failure is non-fatal**: if `init_db()` or `save_analysis_result()` fails (e.g. disk permission issue), the server logs a warning and continues — the analysis API response is unaffected.

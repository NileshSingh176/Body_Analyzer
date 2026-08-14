"""
database.py — PCL Body Analyser (MySQL Edition)
================================================
Single database  : pcl_body_analyser
8 fixed tables   : pushup, squat, single_leg, broad_jump,
                   walking_lunges, squat_with_stick, vertical_jump, speed_20m

Har row = ek video session ka important summary data + session_id

Requirements:
    pip install mysql-connector-python
"""

import json
import uuid
import datetime
from contextlib import contextmanager

import mysql.connector
from mysql.connector import Error as MySQLError

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

MYSQL_BASE_CONFIG = {
    "host":               "localhost",
    "port":               3306,
    "user":               "root",
    "password":           "Dhananjay@123",
    "charset":            "utf8mb4",
    "collation":          "utf8mb4_unicode_ci",
    "autocommit":         False,
    "connection_timeout": 10,
    "use_unicode":        True,
}

DATABASE_NAME = "pcl_body_analyser"

# exercise key → table name
EXERCISE_TABLE_MAP = {
    "pushups":          "pushup",
    "squat":            "squat",
    "single_leg_squat": "single_leg",
    "broad_jump":       "broad_jump",
    "walking_lunges":   "walking_lunges",
    "squat_with_stick": "squat_with_stick",
    "vertical_jump":    "vertical_jump",
    "speed_20m":        "speed_20m",
}


def _table_name(exercise: str) -> str:
    tbl = EXERCISE_TABLE_MAP.get(exercise)
    if not tbl:
        raise ValueError(
            f"Unknown exercise '{exercise}'. Valid: {list(EXERCISE_TABLE_MAP.keys())}"
        )
    return tbl


# ════════════════════════════════════════════════════════════════
# CONNECTION
# ════════════════════════════════════════════════════════════════

@contextmanager
def get_connection():
    """pcl_body_analyser database ka connection do."""
    conn = None
    try:
        conn = mysql.connector.connect(**MYSQL_BASE_CONFIG, database=DATABASE_NAME)
        yield conn
        conn.commit()
    except MySQLError as e:
        if conn:
            conn.rollback()
        raise RuntimeError(f"MySQL error [{DATABASE_NAME}]: {e}") from e
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()


@contextmanager
def _root_connection():
    """Root connection — database create karne ke liye."""
    conn = None
    try:
        conn = mysql.connector.connect(**MYSQL_BASE_CONFIG)
        yield conn
        conn.commit()
    except MySQLError as e:
        if conn:
            conn.rollback()
        raise RuntimeError(f"MySQL root error: {e}") from e
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()


# ════════════════════════════════════════════════════════════════
# TABLE DDL — 8 tables, sirf important columns
# ════════════════════════════════════════════════════════════════

TABLE_DDL = {

    "pushup": """
        CREATE TABLE IF NOT EXISTS `pushup` (
            id              INT         AUTO_INCREMENT PRIMARY KEY,
            session_id      VARCHAR(64) NOT NULL UNIQUE,
            timestamp       DATETIME,
            input_file      VARCHAR(512),
            output_file     VARCHAR(512),
            rep_count       INT,
            correct_reps    INT,
            wrong_reps      INT,
            avg_elbow_angle FLOAT,
            min_elbow_angle FLOAT,
            avg_hip_sway    FLOAT,
            avg_back_angle  FLOAT,
            avg_neck_angle  FLOAT,
            form_score      FLOAT,
            issues          JSON,
            strengths       JSON
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    "squat": """
        CREATE TABLE IF NOT EXISTS `squat` (
            id              INT         AUTO_INCREMENT PRIMARY KEY,
            session_id      VARCHAR(64) NOT NULL UNIQUE,
            timestamp       DATETIME,
            input_file      VARCHAR(512),
            output_file     VARCHAR(512),
            rep_count       INT,
            correct_reps    INT,
            wrong_reps      INT,
            avg_knee_angle  FLOAT,
            min_knee_angle  FLOAT,
            avg_hip_angle   FLOAT,
            avg_ankle_angle FLOAT,
            avg_hip_sway    FLOAT,
            form_score      FLOAT,
            issues          JSON,
            strengths       JSON
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    "single_leg": """
        CREATE TABLE IF NOT EXISTS `single_leg` (
            id              INT         AUTO_INCREMENT PRIMARY KEY,
            session_id      VARCHAR(64) NOT NULL UNIQUE,
            timestamp       DATETIME,
            input_file      VARCHAR(512),
            output_file     VARCHAR(512),
            rep_count       INT,
            correct_reps    INT,
            wrong_reps      INT,
            avg_knee_angle  FLOAT,
            min_knee_angle  FLOAT,
            avg_hip_angle   FLOAT,
            pelvic_drop     FLOAT,
            avg_hip_sway    FLOAT,
            form_score      FLOAT,
            issues          JSON,
            strengths       JSON
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    "broad_jump": """
        CREATE TABLE IF NOT EXISTS `broad_jump` (
            id                  INT         AUTO_INCREMENT PRIMARY KEY,
            session_id          VARCHAR(64) NOT NULL UNIQUE,
            timestamp           DATETIME,
            input_file          VARCHAR(512),
            output_file         VARCHAR(512),
            jump_count          INT,
            correct_jumps       INT,
            wrong_jumps         INT,
            jump_distance_cm    FLOAT,
            avg_takeoff_angle   FLOAT,
            avg_landing_angle   FLOAT,
            landing_score       FLOAT,
            form_score          FLOAT,
            issues              JSON,
            strengths           JSON
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    "walking_lunges": """
        CREATE TABLE IF NOT EXISTS `walking_lunges` (
            id              INT         AUTO_INCREMENT PRIMARY KEY,
            session_id      VARCHAR(64) NOT NULL UNIQUE,
            timestamp       DATETIME,
            input_file      VARCHAR(512),
            output_file     VARCHAR(512),
            rep_count       INT,
            correct_reps    INT,
            wrong_reps      INT,
            avg_knee_angle  FLOAT,
            min_knee_angle  FLOAT,
            avg_trunk_angle FLOAT,
            stride_asymmetry FLOAT,
            avg_hip_sway    FLOAT,
            form_score      FLOAT,
            issues          JSON,
            strengths       JSON
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    "squat_with_stick": """
        CREATE TABLE IF NOT EXISTS `squat_with_stick` (
            id              INT         AUTO_INCREMENT PRIMARY KEY,
            session_id      VARCHAR(64) NOT NULL UNIQUE,
            timestamp       DATETIME,
            input_file      VARCHAR(512),
            output_file     VARCHAR(512),
            rep_count       INT,
            correct_reps    INT,
            wrong_reps      INT,
            avg_knee_angle  FLOAT,
            min_knee_angle  FLOAT,
            avg_hip_angle   FLOAT,
            avg_trunk_angle FLOAT,
            avg_hip_sway    FLOAT,
            form_score      FLOAT,
            issues          JSON,
            strengths       JSON
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    "vertical_jump": """
        CREATE TABLE IF NOT EXISTS `vertical_jump` (
            id                  INT         AUTO_INCREMENT PRIMARY KEY,
            session_id          VARCHAR(64) NOT NULL UNIQUE,
            timestamp           DATETIME,
            input_file          VARCHAR(512),
            output_file         VARCHAR(512),
            jump_count          INT,
            correct_jumps       INT,
            wrong_jumps         INT,
            jump_height_cm      FLOAT,
            flight_time_ms      INT,
            flight_height_index FLOAT,
            avg_takeoff_angle   FLOAT,
            avg_landing_angle   FLOAT,
            landing_score       FLOAT,
            form_score          FLOAT,
            issues              JSON,
            strengths           JSON
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    "speed_20m": """
        CREATE TABLE IF NOT EXISTS `speed_20m` (
            id              INT         AUTO_INCREMENT PRIMARY KEY,
            session_id      VARCHAR(64) NOT NULL UNIQUE,
            timestamp       DATETIME,
            input_file      VARCHAR(512),
            output_file     VARCHAR(512),
            duration_sec    FLOAT,
            avg_speed_kph   FLOAT,
            peak_speed_kph  FLOAT,
            form_speed_kph  FLOAT,
            avg_knee_drive  FLOAT,
            total_strides   INT,
            correct_strides INT,
            wrong_strides   INT,
            stride_asymmetry FLOAT,
            form_score      FLOAT,
            issues          JSON,
            strengths       JSON
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
}


# ════════════════════════════════════════════════════════════════
# INIT — ek database, 8 tables banao
# ════════════════════════════════════════════════════════════════

def init_db():
    """
    'pcl_body_analyser' database create karo.
    Uske andar 8 fixed tables banao.
    """
    # Step 1: Database create karo
    with _root_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{DATABASE_NAME}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cur.close()
    print(f"  [DB] Database ready: {DATABASE_NAME}")

    # Step 2: 8 tables create karo
    with get_connection() as conn:
        cur = conn.cursor()
        for tbl_name, ddl in TABLE_DDL.items():
            cur.execute(ddl)
            print(f"  [DB] Table ready  : {tbl_name}")
        cur.close()

    print(f"  [DB] All 8 tables initialised in '{DATABASE_NAME}' ✓")


# ════════════════════════════════════════════════════════════════
# SESSION ID GENERATOR
# ════════════════════════════════════════════════════════════════

def start_session_timer(exercise: str) -> dict:
    return {
        "exercise":   exercise,
        "start_time": datetime.datetime.now(),
        "uid":        uuid.uuid4().hex[:6],
    }


def stop_session_timer(ctx: dict) -> str:
    """
    Format:  <exercise>_<uid>_<HHMM>
    Example: speed_20m_a3f9c1_1735
    """
    t = ctx["start_time"].strftime("%H%M")
    return f"{ctx['exercise']}_{ctx['uid']}_{t}"


# ════════════════════════════════════════════════════════════════
# SAVE — exercise ke hisaab se sahi columns insert karo
# ════════════════════════════════════════════════════════════════

def save_analysis_result(
    exercise: str,
    result: dict,
    session_id: str = None,
    input_file: str = "",
    output_file: str = "",
    **kwargs,   # extra args ignore karo (is_video, report_file, etc.)
) -> str:
    """
    Ek video ka result sahi exercise table mein save karo.
    Session_id UNIQUE hai — duplicate pe UPDATE hoga.
    Returns: session_id
    """
    if not session_id:
        ts  = datetime.datetime.now().strftime("%H%M")
        uid = uuid.uuid4().hex[:6]
        session_id = f"{exercise}_{uid}_{ts}"

    timestamp = datetime.datetime.now()
    table     = _table_name(exercise)

    # JSON fields safe banao
    def _j(key):
        val = result.get(key, [])
        if isinstance(val, str):
            return val
        return json.dumps(val or [], ensure_ascii=False)

    with get_connection() as conn:
        cur = conn.cursor()

        # ── Exercise ke hisaab se columns aur values ─────────────

        if exercise == "pushups":
            cols = [
                "session_id", "timestamp", "input_file", "output_file",
                "rep_count", "correct_reps", "wrong_reps",
                "avg_elbow_angle", "min_elbow_angle",
                "avg_hip_sway", "avg_back_angle", "avg_neck_angle",
                "form_score", "issues", "strengths",
            ]
            vals = [
                session_id, timestamp, input_file, output_file,
                result.get("rep_count"), result.get("correct_reps"), result.get("wrong_reps"),
                result.get("avg_elbow_angle"), result.get("min_elbow_angle"),
                result.get("spine_alignment"), None, None,  # module has spine_alignment only
                result.get("form_score"), _j("issues"), _j("strengths"),
            ]

        elif exercise == "squat":
            cols = [
                "session_id", "timestamp", "input_file", "output_file",
                "rep_count", "correct_reps", "wrong_reps",
                "avg_knee_angle", "min_knee_angle", "avg_hip_angle",
                "avg_ankle_angle", "avg_hip_sway",
                "form_score", "issues", "strengths",
            ]
            vals = [
                session_id, timestamp, input_file, output_file,
                result.get("rep_count"), result.get("correct_reps"), result.get("wrong_reps"),
                result.get("avg_knee_angle"), result.get("min_knee_angle"),
                result.get("avg_hip_angle"), result.get("avg_ankle_angle"), result.get("avg_hip_sway"),
                result.get("form_score"), _j("issues"), _j("strengths"),
            ]

        elif exercise == "single_leg_squat":
            cols = [
                "session_id", "timestamp", "input_file", "output_file",
                "rep_count", "correct_reps", "wrong_reps",
                "avg_knee_angle", "min_knee_angle", "avg_hip_angle",
                "pelvic_drop", "avg_hip_sway",
                "form_score", "issues", "strengths",
            ]
            vals = [
                session_id, timestamp, input_file, output_file,
                result.get("rep_count"), result.get("correct_reps"), result.get("wrong_reps"),
                result.get("avg_knee_angle"), result.get("min_knee_angle"),
                result.get("avg_hip_angle"), result.get("avg_pelvic_drop"), result.get("avg_hip_sway"),
                result.get("form_score"), _j("issues"), _j("strengths"),
            ]

        elif exercise == "broad_jump":
            cols = [
                "session_id", "timestamp", "input_file", "output_file",
                "jump_count", "correct_jumps", "wrong_jumps",
                "jump_distance_cm", "avg_takeoff_angle", "avg_landing_angle",
                "landing_score", "form_score", "issues", "strengths",
            ]
            vals = [
                session_id, timestamp, input_file, output_file,
                result.get("jump_count"), result.get("correct_jumps"), result.get("wrong_jumps"),
                result.get("jump_distance_cm"), result.get("avg_takeoff_angle"),
                result.get("avg_landing_angle"), result.get("landing_score"),
                result.get("form_score"), _j("issues"), _j("strengths"),
            ]

        elif exercise == "walking_lunges":
            cols = [
                "session_id", "timestamp", "input_file", "output_file",
                "rep_count", "correct_reps", "wrong_reps",
                "avg_knee_angle", "min_knee_angle", "avg_trunk_angle",
                "stride_asymmetry", "avg_hip_sway",
                "form_score", "issues", "strengths",
            ]
            vals = [
                session_id, timestamp, input_file, output_file,
                result.get("rep_count"), result.get("correct_reps"), result.get("wrong_reps"),
                result.get("avg_front_knee"),   # module key
                result.get("min_front_knee"),   # module key
                result.get("avg_trunk_angle"), None,  # stride_asymmetry not in module
                result.get("avg_hip_sway"),
                result.get("form_score"), _j("issues"), _j("strengths"),
            ]

        elif exercise == "squat_with_stick":
            cols = [
                "session_id", "timestamp", "input_file", "output_file",
                "rep_count", "correct_reps", "wrong_reps",
                "avg_knee_angle", "min_knee_angle", "avg_hip_angle",
                "avg_trunk_angle", "avg_hip_sway",
                "form_score", "issues", "strengths",
            ]
            vals = [
                session_id, timestamp, input_file, output_file,
                result.get("rep_count"), result.get("correct_reps"), result.get("wrong_reps"),
                result.get("avg_knee_angle"), result.get("min_knee_angle"),
                result.get("avg_hip_angle"), None,  # avg_trunk_angle not in module
                result.get("avg_hip_sway"),
                result.get("form_score"), _j("issues"), _j("strengths"),
            ]

        elif exercise == "vertical_jump":
            cols = [
                "session_id", "timestamp", "input_file", "output_file",
                "jump_count", "correct_jumps", "wrong_jumps",
                "jump_height_cm", "flight_time_ms", "flight_height_index",
                "avg_takeoff_angle", "avg_landing_angle",
                "landing_score", "form_score", "issues", "strengths",
            ]
            vals = [
                session_id, timestamp, input_file, output_file,
                result.get("jump_count"), result.get("correct_jumps"), result.get("wrong_jumps"),
                result.get("jump_height_cm"), result.get("flight_time_ms"),
                None,  # flight_height_index not in module
                result.get("avg_takeoff_angle"), result.get("avg_landing_angle"),
                result.get("landing_score"),
                result.get("form_score"), _j("issues"), _j("strengths"),
            ]

        elif exercise == "speed_20m":
            cols = [
                "session_id", "timestamp", "input_file", "output_file",
                "duration_sec", "avg_speed_kph", "peak_speed_kph", "form_speed_kph",
                "avg_knee_drive", "total_strides", "correct_strides", "wrong_strides",
                "stride_asymmetry", "form_score", "issues", "strengths",
            ]
            vals = [
                session_id, timestamp, input_file, output_file,
                result.get("duration_sec"), result.get("avg_speed_kph"),
                result.get("peak_speed_kph"), result.get("form_speed_kph"),
                None, None, None, None, None,  # not computed by module
                result.get("form_score"), _j("issues"), _j("strengths"),
            ]

        else:
            cur.close()
            raise ValueError(f"Unknown exercise: {exercise}")

        # ── INSERT … ON DUPLICATE KEY UPDATE (session_id unique hai) ──
        col_str    = ", ".join(f"`{c}`" for c in cols)
        ph_str     = ", ".join(["%s"] * len(cols))
        update_str = ", ".join(
            f"`{c}` = VALUES(`{c}`)"
            for c in cols if c != "session_id"
        )

        cur.execute(
            f"""INSERT INTO `{table}` ({col_str})
                VALUES ({ph_str})
                ON DUPLICATE KEY UPDATE {update_str}""",
            vals,
        )
        cur.close()

    print(f"  [DB] Saved → {DATABASE_NAME}.`{table}` | session: {session_id} ✓")
    return session_id


# ════════════════════════════════════════════════════════════════
# READ HELPERS
# ════════════════════════════════════════════════════════════════

def _as_dict(cursor, row) -> dict:
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


def _parse_json_fields(data: dict) -> dict:
    for field in ("issues", "strengths"):
        val = data.get(field)
        if isinstance(val, str):
            try:
                data[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                data[field] = []
        elif val is None:
            data[field] = []
    return data


def get_session(session_id: str) -> dict:
    """Session_id se us session ka data fetch karo."""
    # session_id se exercise nikalo
    exercise = None
    for ex in EXERCISE_TABLE_MAP:
        if session_id.startswith(ex + "_"):
            exercise = ex
            break
    if not exercise:
        return {}

    table = _table_name(exercise)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM `{table}` WHERE session_id = %s LIMIT 1", (session_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            return {}
        result = _parse_json_fields(_as_dict(cur, row))
        cur.close()
        return result


def list_sessions(exercise: str = None, limit: int = 50) -> list:
    """
    Saare sessions list karo.
    exercise diya  → sirf us table se
    exercise nahi  → saare 8 tables se
    """
    rows = []

    def _fetch(ex: str):
        tbl = _table_name(ex)
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"SELECT * FROM `{tbl}` ORDER BY timestamp DESC LIMIT %s",
                    (limit,)
                )
                for row in cur.fetchall():
                    rows.append(_parse_json_fields(_as_dict(cur, row)))
                cur.close()
        except Exception as e:
            print(f"  [DB] WARN list_sessions({ex}): {e}")

    if exercise:
        _fetch(exercise)
    else:
        for ex in EXERCISE_TABLE_MAP:
            _fetch(ex)

    rows.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return rows[:limit]


def get_exercise_stats(exercise: str) -> dict:
    """Ek exercise ke saare sessions ka aggregate stats."""
    table = _table_name(exercise)
    stats = {}

    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""SELECT
                        COUNT(*)           AS total_sessions,
                        AVG(form_score)    AS avg_form_score,
                        MAX(form_score)    AS best_form_score
                    FROM `{table}`"""
            )
            row = cur.fetchone()
            if row:
                stats["total_sessions"] = row[0]
                stats["avg_form_score"] = round(row[1], 2) if row[1] else None
                stats["best_form_score"] = row[2]

            # Exercise-specific aggregates
            if exercise == "speed_20m":
                cur.execute(
                    f"SELECT AVG(avg_speed_kph), MAX(peak_speed_kph), MIN(duration_sec) FROM `{table}`"
                )
                r = cur.fetchone()
                if r:
                    stats["avg_speed_kph"]   = round(r[0], 2) if r[0] else None
                    stats["peak_speed_kph"]  = r[1]
                    stats["best_time_sec"]   = r[2]

            elif exercise in ("pushups", "squat", "single_leg_squat",
                              "walking_lunges", "squat_with_stick"):
                cur.execute(
                    f"SELECT AVG(rep_count), AVG(correct_reps) FROM `{table}`"
                )
                r = cur.fetchone()
                if r:
                    stats["avg_rep_count"]    = round(r[0], 1) if r[0] else None
                    stats["avg_correct_reps"] = round(r[1], 1) if r[1] else None

            elif exercise in ("broad_jump", "vertical_jump"):
                col = "jump_distance_cm" if exercise == "broad_jump" else "jump_height_cm"
                cur.execute(f"SELECT AVG({col}), MAX({col}) FROM `{table}`")
                r = cur.fetchone()
                if r:
                    stats[f"avg_{col}"]  = round(r[0], 2) if r[0] else None
                    stats[f"best_{col}"] = r[1]

            cur.close()
    except Exception as e:
        print(f"  [DB] WARN stats({exercise}): {e}")

    return stats


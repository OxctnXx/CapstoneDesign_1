import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import mysql.connector
from mysql.connector import IntegrityError
from werkzeug.security import generate_password_hash


BASE_DIR = Path(__file__).resolve().parent
MYSQL_CONFIG_PATH = BASE_DIR / "mysql_config.json"
_DATABASE_READY = False


def _load_mysql_config():
    config = {
        "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
        "port": int(os.environ.get("MYSQL_PORT", "3306")),
        "user": os.environ.get("MYSQL_USER", "root"),
        "password": os.environ.get("MYSQL_PASSWORD", ""),
        "database": os.environ.get("MYSQL_DATABASE", "web1_v3"),
    }

    if MYSQL_CONFIG_PATH.exists():
        with open(MYSQL_CONFIG_PATH, "r", encoding="utf-8") as handle:
            local_config = json.load(handle)
        config.update({key: value for key, value in local_config.items() if value is not None})

    for env_key, config_key in [
        ("MYSQL_HOST", "host"),
        ("MYSQL_PORT", "port"),
        ("MYSQL_USER", "user"),
        ("MYSQL_PASSWORD", "password"),
        ("MYSQL_DATABASE", "database"),
    ]:
        if os.environ.get(env_key):
            config[config_key] = os.environ[env_key]

    config["port"] = int(config["port"])
    return config


MYSQL_CONFIG = _load_mysql_config()


def _quote_identifier(identifier):
    if not re.fullmatch(r"[A-Za-z0-9_]+", identifier or ""):
        raise ValueError(f"Unsafe MySQL identifier: {identifier}")
    return f"`{identifier}`"


def _mysql_query(query):
    return query.replace("?", "%s")


def _normalize_params(params):
    if params is None:
        return None
    if isinstance(params, list):
        return tuple(params)
    return params


class MySQLCursor:
    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, query, params=None):
        self._cursor.execute(_mysql_query(query), _normalize_params(params))
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    @property
    def rowcount(self):
        return self._cursor.rowcount

    def close(self):
        self._cursor.close()

    def __iter__(self):
        return iter(self._cursor)


class MySQLConnection:
    def __init__(self, connection):
        self._connection = connection

    def cursor(self):
        return MySQLCursor(self._connection.cursor(dictionary=True, buffered=True))

    def execute(self, query, params=None):
        cursor = self.cursor()
        return cursor.execute(query, params)

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()


def _connect(database=True):
    options = {
        "host": MYSQL_CONFIG["host"],
        "port": MYSQL_CONFIG["port"],
        "user": MYSQL_CONFIG["user"],
        "password": MYSQL_CONFIG["password"],
        "charset": "utf8mb4",
        "collation": "utf8mb4_unicode_ci",
        "use_unicode": True,
        "autocommit": False,
    }
    if database:
        options["database"] = MYSQL_CONFIG["database"]
    return mysql.connector.connect(**options)


def _ensure_database():
    global _DATABASE_READY
    if _DATABASE_READY:
        return

    database_name = _quote_identifier(MYSQL_CONFIG["database"])
    conn = _connect(database=False)
    cursor = conn.cursor()
    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS {database_name} "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    conn.commit()
    cursor.close()
    conn.close()
    _DATABASE_READY = True


def get_db_connection():
    _ensure_database()
    return MySQLConnection(_connect(database=True))


def _normalize_student_id(value):
    text = unicodedata.normalize("NFKC", (value or "").strip())
    cleaned = "".join(re.findall(r"[A-Za-z0-9]", text)).upper()
    digit_count = len(re.findall(r"\d", cleaned))
    if digit_count >= 3 and digit_count >= len(cleaned) * 0.55:
        cleaned = cleaned.translate(str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"}))
    return cleaned


def _normalize_grade(value):
    text = unicodedata.normalize("NFKC", (value or "").strip())
    text = text.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5"}))
    digits = "".join(re.findall(r"\d", text))
    return digits


def _normalize_profile_text(value):
    text = unicodedata.normalize("NFKC", (value or "").strip())
    text = re.sub(r"\s+", "", text)
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _digit_distance(left, right):
    if not left or not right:
        return None

    rows = len(left) + 1
    cols = len(right) + 1
    matrix = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        matrix[row][0] = row
    for col in range(cols):
        matrix[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            cost = 0 if left[row - 1] == right[col - 1] else 1
            matrix[row][col] = min(
                matrix[row - 1][col] + 1,
                matrix[row][col - 1] + 1,
                matrix[row - 1][col - 1] + cost,
            )
    return matrix[-1][-1]


def _canonical_student_fields(user, fallback_fields):
    fallback_fields = fallback_fields or {}
    if not user:
        return {
            "student_id": fallback_fields.get("student_id", ""),
            "name": fallback_fields.get("name", ""),
            "grade": fallback_fields.get("grade", ""),
            "department": fallback_fields.get("department", ""),
        }

    return {
        "student_id": user["student_id"] or fallback_fields.get("student_id", ""),
        "name": user["name"] or fallback_fields.get("name", ""),
        "grade": user["grade"] or fallback_fields.get("grade", ""),
        "department": user["department"] or fallback_fields.get("department", ""),
    }


def _merge_account_value(existing_value, incoming_value, prefer_incoming):
    incoming_value = (incoming_value or "").strip()
    existing_value = (existing_value or "").strip()
    if prefer_incoming and incoming_value:
        return incoming_value
    return existing_value or incoming_value


def _match_existing_student(conn, student_id="", name="", grade="", department=""):
    normalized_student_id = _normalize_student_id(student_id)
    normalized_name = _normalize_profile_text(name)
    normalized_grade = _normalize_grade(grade)
    normalized_department = _normalize_profile_text(department)

    if normalized_student_id:
        exact = conn.execute(
            "SELECT * FROM users WHERE role = 'user' AND UPPER(student_id) = ?",
            (normalized_student_id.upper(),),
        ).fetchone()
        if exact:
            return exact, "student_id_exact", {"distance": 0}

    if not normalized_name:
        return None, "unresolved", {}

    candidates = []
    rows = conn.execute("SELECT * FROM users WHERE role = 'user' ORDER BY id DESC").fetchall()
    for row in rows:
        name_key = _normalize_profile_text(row["name"])
        if name_key != normalized_name:
            continue

        department_key = _normalize_profile_text(row["department"])
        grade_key = _normalize_grade(row["grade"])
        student_id_key = _normalize_student_id(row["student_id"])

        department_exact = bool(normalized_department and department_key == normalized_department)
        grade_exact = bool(normalized_grade and grade_key == normalized_grade)
        student_distance = _digit_distance(normalized_student_id, student_id_key)

        score = 4.0
        reasons = ["name_exact"]

        if department_exact:
            score += 2.5
            reasons.append("department_exact")
        if grade_exact:
            score += 0.8
            reasons.append("grade_exact")
        if student_distance == 1:
            score += 2.2
            reasons.append("student_id_near")
        elif student_distance == 2 and normalized_student_id and len(normalized_student_id) >= 8:
            score += 0.8
            reasons.append("student_id_close")

        if department_exact or "student_id_near" in reasons:
            candidates.append(
                {
                    "row": row,
                    "score": score,
                    "reasons": reasons,
                    "distance": student_distance,
                }
            )

    if not candidates:
        return None, "unresolved", {}

    candidates.sort(key=lambda item: (item["score"], -(item["distance"] or 99), item["row"]["id"]), reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    score_gap = best["score"] - second["score"] if second else best["score"]

    if best["score"] >= 6.0 and score_gap >= 1.0:
        return best["row"], "profile_match", {"reasons": best["reasons"], "distance": best["distance"]}

    return None, "unresolved", {}


def _table_columns(conn, table_name):
    rows = conn.execute(f"SHOW COLUMNS FROM {_quote_identifier(table_name)}").fetchall()
    return {row["Field"] for row in rows}


def _ensure_column(conn, table_name, column_name, definition):
    if column_name not in _table_columns(conn, table_name):
        conn.execute(
            f"ALTER TABLE {_quote_identifier(table_name)} "
            f"ADD COLUMN {_quote_identifier(column_name)} {definition}"
        )


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT PRIMARY KEY AUTO_INCREMENT,
            student_id VARCHAR(64) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            name VARCHAR(255) NOT NULL,
            grade VARCHAR(64),
            gender VARCHAR(64),
            email VARCHAR(255),
            role VARCHAR(32) DEFAULT 'user',
            department VARCHAR(255),
            auto_created TINYINT DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS omr_templates (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(255) UNIQUE NOT NULL,
            source_filename VARCHAR(512) NOT NULL,
            preview_image VARCHAR(512) NOT NULL,
            page_width INT NOT NULL,
            page_height INT NOT NULL,
            config_json LONGTEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS exams (
            id INT PRIMARY KEY AUTO_INCREMENT,
            template_id INT NOT NULL,
            title VARCHAR(255) NOT NULL,
            description TEXT,
            question_count INT NOT NULL,
            answer_key_json LONGTEXT NOT NULL,
            active TINYINT DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY(template_id) REFERENCES omr_templates(id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS recognition_records (
            id INT PRIMARY KEY AUTO_INCREMENT,
            user_id INT,
            subject VARCHAR(255),
            name VARCHAR(255),
            student_id VARCHAR(64),
            grade VARCHAR(64),
            class_name VARCHAR(128),
            total INT,
            correct INT,
            wrong INT,
            unanswered INT,
            score DOUBLE,
            objective_score DOUBLE DEFAULT 0,
            total_score DOUBLE DEFAULT 0,
            original_image VARCHAR(512),
            annotated_image VARCHAR(512),
            cropped_choice_image VARCHAR(512),
            sa_images LONGTEXT,
            sa_result LONGTEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            exam_id INT,
            template_id INT,
            student_user_id INT,
            department VARCHAR(255),
            answers_json LONGTEXT,
            ocr_json LONGTEXT,
            source_filename VARCHAR(512),
            source_kind VARCHAR(32) DEFAULT 'image',
            source_hash VARCHAR(128),
            exam_title VARCHAR(255),
            template_name VARCHAR(255),
            INDEX idx_records_exam_student (exam_id, student_id),
            INDEX idx_records_student_user (student_user_id),
            INDEX idx_records_source_hash (source_hash)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )

    _ensure_column(conn, "users", "department", "VARCHAR(255)")
    _ensure_column(conn, "users", "auto_created", "TINYINT DEFAULT 0")
    _ensure_column(conn, "users", "created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP")

    _ensure_column(conn, "recognition_records", "exam_id", "INT")
    _ensure_column(conn, "recognition_records", "template_id", "INT")
    _ensure_column(conn, "recognition_records", "student_user_id", "INT")
    _ensure_column(conn, "recognition_records", "department", "VARCHAR(255)")
    _ensure_column(conn, "recognition_records", "answers_json", "LONGTEXT")
    _ensure_column(conn, "recognition_records", "ocr_json", "LONGTEXT")
    _ensure_column(conn, "recognition_records", "source_filename", "VARCHAR(512)")
    _ensure_column(conn, "recognition_records", "source_kind", "VARCHAR(32) DEFAULT 'image'")
    _ensure_column(conn, "recognition_records", "source_hash", "VARCHAR(128)")
    _ensure_column(conn, "recognition_records", "exam_title", "VARCHAR(255)")
    _ensure_column(conn, "recognition_records", "template_name", "VARCHAR(255)")
    cursor.execute('SELECT 1 FROM users WHERE student_id = ?', ("admin001",))
    if not cursor.fetchone():
        cursor.execute(
            """
            INSERT INTO users (student_id, password, name, role, auto_created)
            VALUES (?, ?, ?, ?, 0)
            """,
            ("admin001", generate_password_hash("admin123"), "Administrator", "admin"),
        )

    cursor.execute('SELECT 1 FROM users WHERE student_id = ?', ("20210001",))
    if not cursor.fetchone():
        cursor.execute(
            """
            INSERT INTO users
            (student_id, password, name, grade, gender, email, department, role, auto_created)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                "20210001",
                generate_password_hash("123456"),
                "Test Student",
                "4학년",
                "M",
                "test@school.local",
                "종합설계학과",
                "user",
            ),
        )

    conn.commit()
    conn.execute(
        "UPDATE users SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP)"
    )
    conn.commit()
    conn.close()


def create_user(
    student_id,
    password,
    name,
    grade="",
    gender="",
    email="",
    role="user",
    department="",
    auto_created=0,
):
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO users
            (student_id, password, name, grade, gender, email, department, role, auto_created, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                student_id,
                generate_password_hash(password),
                name,
                grade,
                gender,
                email,
                department,
                role,
                auto_created,
            ),
        )
        conn.commit()
        return True
    except IntegrityError:
        return False
    finally:
        conn.close()


def get_user_by_student_id(student_id):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE UPPER(student_id) = UPPER(?)", (student_id,)).fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return user


def update_user_info(user_id, name, grade, gender, email, department=""):
    conn = get_db_connection()
    conn.execute(
        """
        UPDATE users
        SET name = ?, grade = ?, gender = ?, email = ?, department = ?
        WHERE id = ?
        """,
        (name, grade, gender, email, department, user_id),
    )
    conn.commit()
    conn.close()


def update_user_password(user_id, raw_password):
    conn = get_db_connection()
    conn.execute(
        "UPDATE users SET password = ? WHERE id = ?",
        (generate_password_hash(raw_password), user_id),
    )
    conn.commit()
    conn.close()


def get_all_users(search=""):
    conn = get_db_connection()
    like_value = f"%{search}%"
    users = conn.execute(
        """
        SELECT * FROM users
        WHERE student_id LIKE ? OR name LIKE ? OR department LIKE ?
        ORDER BY id DESC
        """,
        (like_value, like_value, like_value),
    ).fetchall()
    conn.close()
    return users


def delete_user(user_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def admin_update_user(user_id, student_id, name, grade, gender, email, role, department=""):
    conn = get_db_connection()
    conn.execute(
        """
        UPDATE users
        SET student_id = ?, name = ?, grade = ?, gender = ?, email = ?, role = ?, department = ?
        WHERE id = ?
        """,
        (student_id, name, grade, gender, email, role, department, user_id),
    )
    conn.commit()
    conn.close()


def upsert_student_account(student_id, name="", grade="", department=""):
    incoming_fields = {
        "student_id": _normalize_student_id(student_id),
        "name": (name or "").strip(),
        "grade": _normalize_grade(grade) or (grade or "").strip(),
        "department": (department or "").strip(),
    }

    conn = get_db_connection()
    try:
        matched_user, matched_by, match_meta = _match_existing_student(
            conn,
            student_id=incoming_fields["student_id"],
            name=incoming_fields["name"],
            grade=incoming_fields["grade"],
            department=incoming_fields["department"],
        )

        created = False
        user_id = None

        if matched_user:
            prefer_incoming = bool(matched_user["auto_created"])
            merged_name = _merge_account_value(
                matched_user["name"],
                incoming_fields["name"],
                prefer_incoming,
            )
            merged_grade = _merge_account_value(
                matched_user["grade"],
                incoming_fields["grade"],
                prefer_incoming,
            )
            merged_department = _merge_account_value(
                matched_user["department"],
                incoming_fields["department"],
                prefer_incoming,
            )
            conn.execute(
                """
                UPDATE users
                SET name = ?, grade = ?, department = ?
                WHERE id = ?
                """,
                (merged_name, merged_grade, merged_department, matched_user["id"]),
            )
            conn.commit()
            user_id = matched_user["id"]
        elif incoming_fields["student_id"]:
            conn.execute(
                """
                INSERT INTO users
                (student_id, password, name, grade, department, role, auto_created, created_at)
                VALUES (?, ?, ?, ?, ?, 'user', 1, CURRENT_TIMESTAMP)
                """,
                (
                    incoming_fields["student_id"],
                    generate_password_hash(incoming_fields["student_id"]),
                    incoming_fields["name"] or incoming_fields["student_id"],
                    incoming_fields["grade"],
                    incoming_fields["department"],
                ),
            )
            conn.commit()
            user_id = conn.execute(
                "SELECT id FROM users WHERE student_id = ?",
                (incoming_fields["student_id"],),
            ).fetchone()["id"]
            created = True
            matched_by = "auto_created"
        else:
            matched_by = "unresolved"

        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone() if user_id else None
        return {
            "user": user,
            "created": created,
            "matched_by": matched_by,
            "linked_existing_user": bool(user and not created),
            "auto_created": bool(user and user["auto_created"]),
            "match_meta": match_meta,
            "resolved_fields": _canonical_student_fields(user, incoming_fields),
            "input_fields": incoming_fields,
        }
    finally:
        conn.close()


def list_templates():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM omr_templates ORDER BY updated_at DESC, id DESC").fetchall()
    conn.close()
    return rows


def get_template(template_id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM omr_templates WHERE id = ?", (template_id,)).fetchone()
    conn.close()
    return row


def get_template_by_name(name):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM omr_templates WHERE name = ?", (name,)).fetchone()
    conn.close()
    return row


def create_template(name, source_filename, preview_image, page_width, page_height, config_json):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO omr_templates
        (name, source_filename, preview_image, page_width, page_height, config_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (name, source_filename, preview_image, page_width, page_height, config_json),
    )
    conn.commit()
    template_id = cursor.lastrowid
    conn.close()
    return template_id


def update_template(template_id, name, source_filename, preview_image, page_width, page_height, config_json):
    conn = get_db_connection()
    conn.execute(
        """
        UPDATE omr_templates
        SET name = ?, source_filename = ?, preview_image = ?, page_width = ?, page_height = ?,
            config_json = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (name, source_filename, preview_image, page_width, page_height, config_json, template_id),
    )
    conn.commit()
    conn.close()


def list_exams(active_only=False):
    conn = get_db_connection()
    query = """
        SELECT exams.*, omr_templates.name AS template_label, omr_templates.preview_image,
               omr_templates.config_json
        FROM exams
        JOIN omr_templates ON omr_templates.id = exams.template_id
    """
    params = ()
    if active_only:
        query += " WHERE exams.active = 1"
    query += " ORDER BY exams.updated_at DESC, exams.id DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def get_exam(exam_id):
    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT exams.*, omr_templates.name AS template_label, omr_templates.preview_image,
               omr_templates.source_filename, omr_templates.page_width, omr_templates.page_height,
               omr_templates.config_json
        FROM exams
        JOIN omr_templates ON omr_templates.id = exams.template_id
        WHERE exams.id = ?
        """,
        (exam_id,),
    ).fetchone()
    conn.close()
    return row


def create_exam(template_id, title, description, question_count, answer_key_json, active=1):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO exams
        (template_id, title, description, question_count, answer_key_json, active)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (template_id, title, description, question_count, answer_key_json, active),
    )
    conn.commit()
    exam_id = cursor.lastrowid
    conn.close()
    return exam_id


def update_exam(exam_id, template_id, title, description, question_count, answer_key_json, active=1):
    conn = get_db_connection()
    conn.execute(
        """
        UPDATE exams
        SET template_id = ?, title = ?, description = ?, question_count = ?,
            answer_key_json = ?, active = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (template_id, title, description, question_count, answer_key_json, active, exam_id),
    )
    conn.commit()
    conn.close()


def save_recognition_record(
    uploader_user_id,
    exam_id,
    template_id,
    exam_title,
    template_name,
    student_user_id,
    student_name,
    student_id,
    grade,
    department,
    evaluation,
    original_image,
    annotated_image,
    cropped_choice_image,
    answers_json,
    ocr_json,
    source_filename,
    source_kind,
    source_hash="",
    subjective_images=None,
    subjective_results=None,
):
    objective_score = 0
    total_score = round(float(evaluation["percentage"]) + float(objective_score), 1)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO recognition_records
        (
            user_id, exam_id, template_id, student_user_id, subject, exam_title, template_name,
            name, student_id, grade, class_name, department,
            total, correct, wrong, unanswered, score, objective_score, total_score,
            original_image, annotated_image, cropped_choice_image, sa_images, sa_result,
            answers_json, ocr_json, source_filename, source_kind, source_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uploader_user_id,
            exam_id,
            template_id,
            student_user_id,
            exam_title,
            exam_title,
            template_name,
            student_name,
            student_id,
            grade,
            "",
            department,
            evaluation["total"],
            evaluation["correct"],
            evaluation["wrong"],
            evaluation["unanswered"],
            evaluation["percentage"],
            objective_score,
            total_score,
            original_image,
            annotated_image,
            cropped_choice_image,
            json.dumps(subjective_images or [], ensure_ascii=False),
            json.dumps(subjective_results or {}, ensure_ascii=False),
            answers_json,
            ocr_json,
            source_filename,
            source_kind,
            source_hash,
        ),
    )
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id


def get_duplicate_records(exam_id, student_id, exclude_record_id=None):
    student_id = _normalize_student_id(student_id)
    if not exam_id or not student_id:
        return []

    conn = get_db_connection()
    params = [exam_id, student_id]
    query = """
        SELECT * FROM recognition_records
        WHERE exam_id = ? AND UPPER(student_id) = ?
    """
    if exclude_record_id:
        query += " AND id <> ?"
        params.append(exclude_record_id)
    query += " ORDER BY created_at DESC, id DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def get_duplicate_records_by_hash(exam_id, student_id, source_hash, exclude_record_id=None):
    student_id = _normalize_student_id(student_id)
    source_hash = (source_hash or "").strip()
    if not exam_id or not student_id or not source_hash:
        return []

    conn = get_db_connection()
    params = [exam_id, student_id, source_hash]
    query = """
        SELECT * FROM recognition_records
        WHERE exam_id = ? AND UPPER(student_id) = ? AND source_hash = ?
    """
    if exclude_record_id:
        query += " AND id <> ?"
        params.append(exclude_record_id)
    query += " ORDER BY created_at DESC, id DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def delete_records(record_ids):
    ids = [int(record_id) for record_id in (record_ids or []) if str(record_id).strip()]
    if not ids:
        return 0

    conn = get_db_connection()
    placeholders = ",".join("?" for _ in ids)
    cursor = conn.execute(
        f"DELETE FROM recognition_records WHERE id IN ({placeholders})",
        ids,
    )
    conn.commit()
    deleted = cursor.rowcount
    conn.close()
    return deleted


def update_objective_score(record_id, objective_score):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT score FROM recognition_records WHERE id = ?",
        (record_id,),
    ).fetchone()
    if not row:
        conn.close()
        return False

    total_score = round(float(row["score"]) + float(objective_score), 1)
    conn.execute(
        """
        UPDATE recognition_records
        SET objective_score = ?, total_score = ?
        WHERE id = ?
        """,
        (objective_score, total_score, record_id),
    )
    conn.commit()
    conn.close()
    return True


def update_subjective_grading(record_id, subjective_results):
    subjective_results = subjective_results or {}
    manual_score = 0.0
    normalized = {}
    for question_number, item in subjective_results.items():
        question_key = str(question_number)
        status = (item.get("status") or "ungraded").strip()
        try:
            score_value = float(item.get("score") or 0)
        except (TypeError, ValueError):
            score_value = 0.0
        if status == "ungraded":
            score_value = 0.0
        manual_score += score_value
        normalized[question_key] = {
            "status": status,
            "score": score_value,
            "comment": (item.get("comment") or "").strip(),
        }

    conn = get_db_connection()
    row = conn.execute(
        "SELECT score FROM recognition_records WHERE id = ?",
        (record_id,),
    ).fetchone()
    if not row:
        conn.close()
        return False

    total_score = round(float(row["score"] or 0) + manual_score, 1)
    conn.execute(
        """
        UPDATE recognition_records
        SET sa_result = ?, objective_score = ?, total_score = ?
        WHERE id = ?
        """,
        (json.dumps(normalized, ensure_ascii=False), round(manual_score, 1), total_score, record_id),
    )
    conn.commit()
    conn.close()
    return True


def get_record_by_id(record_id):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM recognition_records WHERE id = ?", (record_id,)).fetchone()
    conn.close()
    return row


def get_all_records_admin(search=""):
    conn = get_db_connection()
    like_value = f"%{search}%"
    rows = conn.execute(
        """
        SELECT * FROM recognition_records
        WHERE COALESCE(name, '') LIKE ?
           OR COALESCE(student_id, '') LIKE ?
           OR COALESCE(department, '') LIKE ?
           OR COALESCE(exam_title, subject, '') LIKE ?
        ORDER BY created_at DESC, id DESC
        """,
        (like_value, like_value, like_value, like_value),
    ).fetchall()
    conn.close()
    return rows


def delete_record(record_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM recognition_records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


def update_record_student_info(record_id, name, student_id, grade, department):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT ocr_json FROM recognition_records WHERE id = ?",
        (record_id,),
    ).fetchone()
    ocr_payload = {}
    if row and row["ocr_json"]:
        try:
            ocr_payload = json.loads(row["ocr_json"])
        except json.JSONDecodeError:
            ocr_payload = {}
    ocr_payload["fields"] = {
        **(ocr_payload.get("fields") or {}),
        "name": name,
        "student_id": student_id,
        "grade": grade,
        "department": department,
    }
    conn.execute(
        """
        UPDATE recognition_records
        SET name = ?, student_id = ?, grade = ?, department = ?, ocr_json = ?
        WHERE id = ?
        """,
        (name, student_id, grade, department, json.dumps(ocr_payload, ensure_ascii=False), record_id),
    )
    conn.commit()
    conn.close()


def get_user_records_by_student_id(student_id):
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT * FROM recognition_records
        WHERE student_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (student_id,),
    ).fetchall()
    conn.close()
    return rows


def get_user_records_for_account(user_id, student_id):
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT * FROM recognition_records
        WHERE student_user_id = ? OR student_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (user_id, student_id),
    ).fetchall()
    conn.close()
    return rows


def get_exam_statistics():
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT exam_id,
               COALESCE(exam_title, subject, 'Unknown Exam') AS label,
               COUNT(*) AS item_count,
               COUNT(DISTINCT NULLIF(student_id, '')) AS student_count,
               AVG(total_score) AS avg_score,
               MAX(total_score) AS max_score,
               MIN(total_score) AS min_score,
               AVG(unanswered) AS avg_unanswered,
               MAX(created_at) AS last_graded_at
        FROM recognition_records
        GROUP BY exam_id, COALESCE(exam_title, subject, 'Unknown Exam')
        ORDER BY last_graded_at DESC, item_count DESC
        """
    ).fetchall()
    conn.close()

    return [
        {
            "exam_id": row["exam_id"],
            "exam_title": row["label"],
            "count": row["item_count"],
            "student_count": row["student_count"] or 0,
            "avg_score": round(row["avg_score"] or 0, 1),
            "max_score": round(row["max_score"] or 0, 1),
            "min_score": round(row["min_score"] or 0, 1),
            "avg_unanswered": round(row["avg_unanswered"] or 0, 1),
            "last_graded_at": row["last_graded_at"],
        }
        for row in rows
    ]


def get_exam_score_distribution(exam_id):
    conn = get_db_connection()
    ranges = [
        (0, 20, "0-20"),
        (20, 40, "20-40"),
        (40, 60, "40-60"),
        (60, 80, "60-80"),
        (80, 101, "80-100"),
    ]
    distribution = {}
    for lower, upper, label in ranges:
        if exam_id is None:
            distribution[label] = 0
        else:
            distribution[label] = conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM recognition_records
                WHERE exam_id = ? AND total_score >= ? AND total_score < ?
                """,
                (exam_id, lower, upper),
            ).fetchone()["cnt"]
    conn.close()
    return distribution


def get_exam_summary(exam_id):
    if exam_id is None:
        return {
            "count": 0,
            "student_count": 0,
            "avg_score": 0,
            "max_score": 0,
            "min_score": 0,
            "avg_mc_score": 0,
            "avg_manual_score": 0,
            "avg_unanswered": 0,
        }

    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT COUNT(*) AS item_count,
               COUNT(DISTINCT NULLIF(student_id, '')) AS student_count,
               AVG(total_score) AS avg_score,
               MAX(total_score) AS max_score,
               MIN(total_score) AS min_score,
               AVG(score) AS avg_mc_score,
               AVG(objective_score) AS avg_manual_score,
               AVG(unanswered) AS avg_unanswered
        FROM recognition_records
        WHERE exam_id = ?
        """,
        (exam_id,),
    ).fetchone()
    conn.close()
    return {
        "count": row["item_count"] or 0,
        "student_count": row["student_count"] or 0,
        "avg_score": round(row["avg_score"] or 0, 1),
        "max_score": round(row["max_score"] or 0, 1),
        "min_score": round(row["min_score"] or 0, 1),
        "avg_mc_score": round(row["avg_mc_score"] or 0, 1),
        "avg_manual_score": round(row["avg_manual_score"] or 0, 1),
        "avg_unanswered": round(row["avg_unanswered"] or 0, 1),
    }


def serialize_rows(rows):
    serialized = []
    for row in rows:
        item = dict(row)
        for key, value in item.items():
            if isinstance(value, datetime):
                item[key] = value.strftime("%Y-%m-%d %H:%M:%S")
        serialized.append(item)
    return serialized


def parse_json_field(row, key, default):
    value = row[key]
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def format_record_timestamp(row):
    raw = row["created_at"]
    if not raw:
        return ""
    if isinstance(raw, datetime):
        return raw.strftime("%Y-%m-%d %H:%M")
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return raw

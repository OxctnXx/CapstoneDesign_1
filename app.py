import json
import hashlib
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from werkzeug.security import check_password_hash
from functools import wraps

from database import (
    admin_update_user,
    create_exam,
    create_template,
    create_user,
    delete_record,
    delete_records,
    delete_user,
    format_record_timestamp,
    get_all_records_admin,
    get_all_users,
    get_duplicate_records,
    get_duplicate_records_by_hash,
    get_db_connection,
    get_exam,
    get_record_by_id,
    get_exam_score_distribution,
    get_exam_statistics,
    get_exam_summary,
    get_template,
    get_template_by_name,
    get_user_by_id,
    get_user_by_student_id,
    get_user_records_for_account,
    init_db,
    list_exams,
    list_templates,
    parse_json_field,
    save_recognition_record,
    update_exam,
    update_objective_score,
    update_record_student_info,
    update_subjective_grading,
    update_template,
    update_user_info,
    update_user_password,
    upsert_student_account,
)
from omr_engine import (
    build_default_template_config,
    load_file_as_page,
    process_submission,
    save_image_path,
)
from template_builder import build_omr_template, normalize_layout_config, normalize_option_labels


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
STATIC_DIR = BASE_DIR / "static"
GENERATED_DIR = STATIC_DIR / "generated"
DEFAULT_TEMPLATE_NAME = "OMR 종합설계 답안지"
DEFAULT_TEMPLATE_FILE = BASE_DIR / "OMR 종합설계 답안지.pdf"
DEFAULT_TEMPLATE_PREVIEW_REL = "generated/default_template_preview.png"
SUPER_ADMIN_STUDENT_ID = "admin001"

for directory in [UPLOAD_DIR, RESULT_DIR, GENERATED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


app = Flask(__name__)
app.config["SECRET_KEY"] = "secure-key-2026-template-upgrade"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["RESULTS_FOLDER"] = str(RESULT_DIR)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "로그인이 필요합니다."
login_manager.login_message_category = "error"


class User(UserMixin):
    def __init__(self, row):
        self.id = str(row["id"])
        self.student_id = row["student_id"]
        self.name = row["name"]
        self.role = row["role"]
        self.department = row["department"] if "department" in row.keys() else ""


@login_manager.user_loader
def load_user(user_id):
    row = get_user_by_id(user_id)
    return User(row) if row else None


def admin_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            flash("관리자만 접근할 수 있습니다.", "error")
            return redirect(url_for("dashboard"))
        return view_func(*args, **kwargs)

    return wrapper


def student_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "user":
            flash("학생 계정으로 로그인해 주세요.", "error")
            return redirect(url_for("dashboard"))
        return view_func(*args, **kwargs)

    return wrapper


def allowed_file(filename):
    return Path(filename).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".pdf"}


def relative_from_base(path_obj):
    return str(path_obj.relative_to(BASE_DIR)).replace("\\", "/")


def parse_answer_key_input(raw_text, question_count, option_labels=None):
    normalized = (raw_text or "").strip().upper()
    normalized = normalized.replace(" ", "").replace("\n", "").replace("\r", "")
    if not normalized:
        raise ValueError("정답을 입력해 주세요.")

    if any(separator in normalized for separator in [",", ";", "|", "/"]):
        tokens = [token for token in re_split_answers(normalized) if token]
    else:
        tokens = list(normalized)

    option_labels = [str(label).strip().upper() for label in (option_labels or ["1", "2", "3", "4", "5"])]
    option_lookup = {label: label for label in option_labels}
    alpha_lookup = {chr(ord("A") + index): label for index, label in enumerate(option_labels)}
    numeric_lookup = {str(index + 1): label for index, label in enumerate(option_labels)}

    converted = []
    for token in tokens:
        token = token.strip().upper()
        converted.append(option_lookup.get(token) or alpha_lookup.get(token) or numeric_lookup.get(token) or token)

    if len(converted) != question_count:
        raise ValueError(f"객관식 문항 수({question_count})와 정답 개수({len(converted)})가 일치하지 않습니다.")

    invalid = [token for token in converted if token not in option_labels]
    if invalid:
        raise ValueError(f"정답은 설정한 선택항({', '.join(option_labels)}) 안에서만 입력할 수 있습니다.")

    return {index + 1: token for index, token in enumerate(converted)}


def re_split_answers(text):
    for separator in [",", ";", "|", "/"]:
        text = text.replace(separator, " ")
    return text.split()


def parse_answer_key_json(raw_json):
    data = json.loads(raw_json)
    return {int(key): str(value) for key, value in data.items()}


def format_answer_key_preview(answer_key_json, question_count=12):
    answer_key = parse_answer_key_json(answer_key_json)
    ordered = [answer_key[index] for index in sorted(answer_key.keys())]
    preview = "".join(ordered[:question_count])
    if len(ordered) > question_count:
        preview += "..."
    return preview


def exam_view_model(row):
    data = dict(row)
    answer_key = parse_answer_key_json(row["answer_key_json"])
    template_config = {}
    if "config_json" in row.keys() and row["config_json"]:
        try:
            template_config = json.loads(row["config_json"])
        except (TypeError, json.JSONDecodeError):
            template_config = {}
    settings = template_config.get("template_settings", {})
    option_labels = settings.get("option_labels") or ["1", "2", "3", "4", "5"]
    data["answer_key_preview"] = format_answer_key_preview(row["answer_key_json"])
    data["answer_key_length"] = len(answer_key)
    data["answer_key_text"] = ",".join(answer_key[index] for index in sorted(answer_key.keys()))
    data["objective_count"] = int(settings.get("objective_count") or row["question_count"])
    data["subjective_count"] = int(settings.get("subjective_count") or 0)
    data["option_labels"] = option_labels
    data["option_labels_text"] = ",".join(option_labels)
    data["option_count"] = int(settings.get("option_count") or len(option_labels))
    data["option_mode"] = settings.get("option_mode") or "numeric"
    data["page_count"] = int(settings.get("page_count") or len(template_config.get("pages", [])) or 1)
    data["layout"] = normalize_layout_config(settings.get("layout", {}))
    return data


def template_view_model(row):
    data = dict(row)
    data["config"] = json.loads(row["config_json"])
    data["preview_url"] = url_for("static", filename=row["preview_image"])
    preview_dir = str(Path(row["preview_image"]).parent).replace("\\", "/")
    data["preview_urls"] = []
    for page_image in data["config"].get("preview_images", []):
        filename = f"{preview_dir}/{page_image}" if preview_dir != "." else page_image
        data["preview_urls"].append(url_for("static", filename=filename))
    if not data["preview_urls"]:
        data["preview_urls"] = [data["preview_url"]]
    source_path = BASE_DIR / row["source_filename"]
    data["pdf_url"] = url_for("download_template_pdf", template_id=row["id"]) if source_path.exists() else ""
    return data


def file_sha256(path_obj):
    digest = hashlib.sha256()
    with open(path_obj, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_generated_template(title, objective_count, subjective_count, option_labels, layout_config=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_title = "".join(ch if ch.isalnum() else "_" for ch in (title or "omr"))[:40].strip("_") or "omr"
    basename = f"omr_{timestamp}_{safe_title}"
    artifact = build_omr_template(
        GENERATED_DIR,
        title,
        objective_count,
        subjective_count,
        option_labels,
        basename,
        layout_config=layout_config,
    )
    source_rel = relative_from_base(artifact["pdf_path"])
    preview_rel = relative_from_base(artifact["preview_path"]).replace("static/", "", 1)
    template_name = f"{title or 'OMR'} 양식 {timestamp}"
    template_id = create_template(
        template_name,
        source_rel,
        preview_rel,
        artifact["page_width"],
        artifact["page_height"],
        json.dumps(artifact["config"], ensure_ascii=False),
    )
    return get_template(template_id)


def _form_float(name, default):
    value = request.form.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def layout_config_from_form():
    return normalize_layout_config(
        {
            "objective_area": {
                "x": _form_float("objective_x", 6),
                "y": _form_float("objective_y", 25),
                "w": _form_float("objective_w", 88),
                "h": _form_float("objective_h", 49),
            },
            "subjective_area": {
                "x": _form_float("subjective_x", 6),
                "y": _form_float("subjective_y", 76),
                "w": _form_float("subjective_w", 88),
                "h": _form_float("subjective_h", 17),
            },
            "subjective_box_height": request.form.get("subjective_box_height", 44),
        }
    )


def duplicate_record_payload(row):
    view = record_view_model(row)
    fields = view["ocr"].get("fields", {})
    return {
        "record_id": view["id"],
        "exam_id": view.get("exam_id"),
        "exam_title": view.get("exam_title") or view.get("subject"),
        "student_id": fields.get("student_id") or view.get("student_id") or "",
        "student_name": fields.get("name") or view.get("name") or "",
        "department": fields.get("department") or view.get("department") or "",
        "grade": fields.get("grade") or view.get("grade") or "",
        "score": view.get("score"),
        "total_score": view.get("total_score"),
        "created_at": view.get("created_at_display"),
        "original_image": view.get("original_url"),
        "annotated_image": view.get("annotated_url"),
        "choice_image": view.get("choice_url"),
        "sheet_image": (view.get("sheet_images") or [{}])[0].get("url", ""),
    }


def record_view_model(row):
    data = dict(row)
    data["created_at_display"] = format_record_timestamp(row)
    data["answers"] = parse_json_field(row, "answers_json", {})
    data["ocr"] = parse_json_field(row, "ocr_json", {})
    data["subjective_images"] = parse_json_field(row, "sa_images", [])
    for item in data["subjective_images"]:
        if item.get("file"):
            item["url"] = url_for("result_file", filename=item["file"])
    data["subjective_results"] = parse_json_field(row, "sa_result", {})
    data["field_images"] = data["ocr"].get("field_images", [])
    data["sheet_images"] = data["ocr"].get("sheet_images", [])
    for item in data["sheet_images"]:
        if item.get("file"):
            item["url"] = url_for("result_file", filename=item["file"])
    data["annotated_url"] = url_for("result_file", filename=row["annotated_image"]) if row["annotated_image"] else ""
    data["original_url"] = url_for("uploaded_file", filename=row["original_image"]) if row["original_image"] else ""
    data["source_url"] = url_for("uploaded_file", filename=row["source_filename"]) if row["source_filename"] else ""
    data["choice_url"] = url_for("result_file", filename=row["cropped_choice_image"]) if row["cropped_choice_image"] else ""
    return data


def prefer_ocr_fields(raw_fields, account_fields=None):
    raw_fields = raw_fields or {}
    account_fields = account_fields or {}
    merged = {}
    for key in ["department", "student_id", "grade", "name"]:
        raw_value = (raw_fields.get(key) or "").strip()
        account_value = (account_fields.get(key) or "").strip()
        merged[key] = raw_value or account_value
    return merged


def ensure_default_template():
    if not DEFAULT_TEMPLATE_FILE.exists():
        return None

    preview_abs = STATIC_DIR / DEFAULT_TEMPLATE_PREVIEW_REL
    template_image, _ = load_file_as_page(str(DEFAULT_TEMPLATE_FILE))
    save_image_path(str(preview_abs), template_image)
    config = build_default_template_config(str(preview_abs))

    existing = get_template_by_name(DEFAULT_TEMPLATE_NAME)
    source_filename = DEFAULT_TEMPLATE_FILE.name
    preview_rel = DEFAULT_TEMPLATE_PREVIEW_REL.replace("\\", "/")
    config_json = json.dumps(config, ensure_ascii=False)

    if existing:
        update_template(
            existing["id"],
            DEFAULT_TEMPLATE_NAME,
            source_filename,
            preview_rel,
            config["page_size"]["width"],
            config["page_size"]["height"],
            config_json,
        )
        return get_template(existing["id"])

    template_id = create_template(
        DEFAULT_TEMPLATE_NAME,
        source_filename,
        preview_rel,
        config["page_size"]["width"],
        config["page_size"]["height"],
        config_json,
    )
    return get_template(template_id)


def current_user_row():
    return get_user_by_id(current_user.id)


def is_super_admin():
    return (
        current_user.is_authenticated
        and current_user.role == "admin"
        and (current_user.student_id or "").strip().lower() == SUPER_ADMIN_STUDENT_ID
    )


def can_access_record(record):
    if current_user.role == "admin":
        return True
    return str(record["student_user_id"] or "") == str(current_user.id) or record["student_id"] == current_user.student_id


@app.route("/")
def index():
    if current_user.is_authenticated:
        if current_user.role == "admin":
            return redirect(url_for("dashboard"))
        return redirect(url_for("student_scores"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        account = request.form.get("account", "").strip()
        password = request.form.get("password", "").strip()
        user = get_user_by_student_id(account)

        if not user or not check_password_hash(user["password"], password):
            flash("학번 또는 비밀번호가 올바르지 않습니다.", "error")
            return render_template("login.html")

        login_user(User(user))
        if user["role"] == "admin":
            return redirect(url_for("dashboard"))
        return redirect(url_for("student_scores"))

    return render_template("login.html")


@app.route("/register", methods=["POST"])
def register():
    student_id = request.form.get("student_id", "").strip()
    password = request.form.get("password", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()
    name = request.form.get("name", "").strip()
    grade = request.form.get("grade", "").strip()
    gender = request.form.get("gender", "").strip()
    email = request.form.get("email", "").strip()
    department = request.form.get("department", "").strip()

    if password != confirm_password:
        flash("비밀번호 확인이 일치하지 않습니다.", "error")
        return redirect(url_for("login"))

    if not student_id or not password or not name:
        flash("학번, 비밀번호, 이름은 필수입니다.", "error")
        return redirect(url_for("login"))

    created = create_user(
        student_id=student_id,
        password=password,
        name=name,
        grade=grade,
        gender=gender,
        email=email,
        role="user",
        department=department,
    )
    if created:
        flash("회원가입이 완료되었습니다. 로그인해 주세요.", "success")
    else:
        flash("이미 존재하는 학번입니다.", "error")
    return redirect(url_for("login"))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    if current_user.role != "admin":
        return redirect(url_for("student_scores"))

    templates = [template_view_model(row) for row in list_templates()]
    exams = [exam_view_model(row) for row in list_exams()]
    edit_exam_id = request.args.get("edit_exam", type=int)
    edit_row = get_exam(edit_exam_id) if edit_exam_id else None
    edit_exam = exam_view_model(edit_row) if edit_row else None
    return render_template(
        "dashboard.html",
        user=current_user,
        templates=templates,
        exams=exams,
        edit_exam=edit_exam,
    )


@app.route("/admin/exams/save", methods=["POST"])
@login_required
@admin_required
def save_exam():
    exam_id = request.form.get("exam_id", "").strip()
    template_id = request.form.get("template_id", type=int)
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    objective_count = request.form.get("objective_count", type=int)
    subjective_count = request.form.get("subjective_count", type=int) or 0
    option_count = request.form.get("option_count", type=int) or 5
    option_mode = request.form.get("option_mode", "numeric")
    custom_options = request.form.get("custom_options", "")
    question_count = objective_count or request.form.get("question_count", type=int)
    answer_key_text = request.form.get("answer_key", "").strip()
    active = 1 if request.form.get("active") == "on" else 0
    layout_config = layout_config_from_form()

    if not title or not question_count:
        flash("시험명과 객관식 문항 수는 필수입니다.", "error")
        return redirect(url_for("dashboard"))

    if question_count < 1 or question_count > 300:
        flash("객관식 문항 수는 1~300 사이로 설정해 주세요.", "error")
        return redirect(url_for("dashboard", edit_exam=exam_id) if exam_id else url_for("dashboard"))

    if subjective_count < 0 or subjective_count > 200:
        flash("주관식 문항 수는 0~200 사이로 설정해 주세요.", "error")
        return redirect(url_for("dashboard", edit_exam=exam_id) if exam_id else url_for("dashboard"))

    try:
        option_labels = normalize_option_labels(option_mode, option_count, custom_options)
        answer_key = parse_answer_key_input(answer_key_text, question_count, option_labels)
    except ValueError as exc:
        flash(str(exc), "error")
        if exam_id:
            return redirect(url_for("dashboard", edit_exam=exam_id))
        return redirect(url_for("dashboard"))

    generated_template = create_generated_template(title, question_count, subjective_count, option_labels, layout_config)
    template_id = generated_template["id"]

    answer_key_json = json.dumps({str(k): v for k, v in answer_key.items()}, ensure_ascii=False)
    if exam_id:
        update_exam(int(exam_id), template_id, title, description, question_count, answer_key_json, active)
        flash("시험과 OMR 양식이 수정되었습니다. 새 PDF를 바로 출력할 수 있습니다.", "success")
    else:
        create_exam(template_id, title, description, question_count, answer_key_json, active)
        flash("새 시험과 OMR 양식이 생성되었습니다. 미리보기 확인 후 PDF를 배포하세요.", "success")
    return redirect(url_for("dashboard"))


@app.route("/exam/<int:exam_id>")
@login_required
@admin_required
def exam_page(exam_id):
    exam = get_exam(exam_id)
    if not exam:
        flash("시험을 찾을 수 없습니다.", "error")
        return redirect(url_for("dashboard"))

    exam_data = exam_view_model(exam)
    template_data = template_view_model(get_template(exam["template_id"]))
    return render_template("exam.html", user=current_user, exam=exam_data, template=template_data)


@app.route("/upload/exam/<int:exam_id>", methods=["POST"])
@login_required
@admin_required
def upload_exam_file(exam_id):
    exam = get_exam(exam_id)
    if not exam:
        return jsonify({"success": False, "error": "시험 정보를 찾을 수 없습니다."}), 404

    if "file" not in request.files:
        return jsonify({"success": False, "error": "업로드할 파일이 없습니다."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "파일명을 확인해 주세요."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "PDF 또는 이미지 파일만 지원합니다."}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = Path(file.filename).suffix.lower()
    source_filename = f"{timestamp}{suffix}"
    source_abs = UPLOAD_DIR / source_filename
    file.save(source_abs)
    source_hash = file_sha256(source_abs)

    template_row = get_template(exam["template_id"])
    template_config = json.loads(exam["config_json"])
    preview_abs = STATIC_DIR / template_row["preview_image"]
    answer_key = parse_answer_key_json(exam["answer_key_json"])

    try:
        result = process_submission(
            str(source_abs),
            str(preview_abs),
            template_config,
            answer_key,
            filename_hint=file.filename,
        )
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

    original_preview_filename = f"{timestamp}_original.png"
    annotated_filename = f"{timestamp}_annotated.png"
    answer_area_filename = f"{timestamp}_answer_area.png"
    sheet_filename = f"{timestamp}_sheet.png"

    original_preview_image, _ = load_file_as_page(str(source_abs))
    save_image_path(str(UPLOAD_DIR / original_preview_filename), original_preview_image)
    save_image_path(str(RESULT_DIR / annotated_filename), result["annotated_image"])
    save_image_path(str(RESULT_DIR / answer_area_filename), result["answer_area_image"])
    save_image_path(str(RESULT_DIR / sheet_filename), result["warped_image"])

    field_images = []
    for field_name, field_image in result["field_crops"].items():
        field_filename = f"{timestamp}_{field_name}.png"
        save_image_path(str(RESULT_DIR / field_filename), field_image)
        field_images.append(
            {
                "field": field_name,
                "file": field_filename,
                "url": url_for("result_file", filename=field_filename),
            }
        )

    subjective_images = []
    for item in result.get("subjective_images", []):
        question_number = int(item["question_number"])
        subjective_filename = f"{timestamp}_subjective_q{question_number}.png"
        save_image_path(str(RESULT_DIR / subjective_filename), item["image"])
        subjective_images.append(
            {
                "question_number": question_number,
                "page_number": int(item.get("page_number") or 1),
                "region": item.get("region", []),
                "file": subjective_filename,
                "url": url_for("result_file", filename=subjective_filename),
            }
        )

    raw_recognized = result["recognized_fields"]
    account_result = upsert_student_account(
        raw_recognized.get("student_id", ""),
        raw_recognized.get("name", ""),
        raw_recognized.get("grade", ""),
        raw_recognized.get("department", ""),
    )
    student_user = account_result["user"] if account_result else None
    recognized = prefer_ocr_fields(
        raw_recognized,
        account_result["resolved_fields"] if account_result else {},
    )
    if student_user:
        student_user = get_user_by_id(student_user["id"])

    sheet_images = [
        {
            "title": "답안 작성 화면",
            "file": sheet_filename,
            "url": url_for("result_file", filename=sheet_filename),
        }
    ]
    ocr_payload = {
        "fields": recognized,
        "raw_fields": raw_recognized,
        "details": result["ocr_debug"],
        "field_images": field_images,
        "sheet_images": sheet_images,
        "subjective_images": subjective_images,
        "account_match": {
            "matched_by": account_result["matched_by"] if account_result else "unresolved",
            "linked_existing_user": bool(account_result and account_result["linked_existing_user"]),
            "created": bool(account_result and account_result["created"]),
            "auto_created": bool(account_result and account_result["auto_created"]),
            "meta": account_result["match_meta"] if account_result else {},
        },
    }
    record_id = save_recognition_record(
        uploader_user_id=int(current_user.id),
        exam_id=exam["id"],
        template_id=template_row["id"],
        exam_title=exam["title"],
        template_name=template_row["name"],
        student_user_id=student_user["id"] if student_user else None,
        student_name=recognized.get("name", ""),
        student_id=recognized.get("student_id", ""),
        grade=recognized.get("grade", ""),
        department=recognized.get("department", ""),
        evaluation=result["evaluation"],
        original_image=original_preview_filename,
        annotated_image=annotated_filename,
        cropped_choice_image=answer_area_filename,
        answers_json=json.dumps(result["student_answers"], ensure_ascii=False),
        ocr_json=json.dumps(ocr_payload, ensure_ascii=False),
        source_filename=source_filename,
        source_kind=result["source_kind"],
        source_hash=source_hash,
        subjective_images=subjective_images,
        subjective_results={},
    )

    auto_replaced_record_ids = []
    duplicate_candidates = []
    student_id_for_duplicate = recognized.get("student_id", "")
    exact_duplicates = get_duplicate_records_by_hash(
        exam["id"],
        student_id_for_duplicate,
        source_hash,
        exclude_record_id=record_id,
    )
    if exact_duplicates:
        auto_replaced_record_ids = [row["id"] for row in exact_duplicates]
        delete_records(auto_replaced_record_ids)
    duplicate_candidates = get_duplicate_records(
        exam["id"],
        student_id_for_duplicate,
        exclude_record_id=record_id,
    )

    current_record = get_record_by_id(record_id)

    return jsonify(
        {
            "success": True,
            "record_id": record_id,
            "evaluation": result["evaluation"],
            "student_answers": result["student_answers"],
            "recognized_fields": recognized,
            "raw_recognized_fields": raw_recognized,
            "field_images": field_images,
            "sheet_images": sheet_images,
            "subjective_images": subjective_images,
            "subjective_results": {},
            "annotated_image": url_for("result_file", filename=annotated_filename),
            "cropped_choice_image": url_for("result_file", filename=answer_area_filename),
            "original_image": url_for("uploaded_file", filename=original_preview_filename),
            "account_match": ocr_payload["account_match"],
            "auto_account_created": bool(account_result and account_result["created"]),
            "auto_replaced_record_ids": auto_replaced_record_ids,
            "duplicate_conflict": bool(duplicate_candidates),
            "duplicate_candidates": [duplicate_record_payload(row) for row in duplicate_candidates],
            "current_record": duplicate_record_payload(current_record) if current_record else None,
        }
    )


@app.route("/admin/duplicates/resolve", methods=["POST"])
@login_required
@admin_required
def resolve_duplicate_records():
    payload = request.get_json(force=True)
    keep_record_id = int(payload.get("keep_record_id") or 0)
    duplicate_record_ids = [
        int(record_id)
        for record_id in payload.get("duplicate_record_ids", [])
        if str(record_id).strip()
    ]
    all_ids = sorted(set(duplicate_record_ids + [keep_record_id]))
    if not keep_record_id or len(all_ids) < 2:
        return jsonify({"success": False, "error": "중복 처리할 기록을 확인해 주세요."}), 400

    records = [get_record_by_id(record_id) for record_id in all_ids]
    records = [record for record in records if record]
    if len(records) != len(all_ids):
        return jsonify({"success": False, "error": "일부 기록을 찾을 수 없습니다."}), 404

    exam_ids = {record["exam_id"] for record in records}
    student_ids = {record["student_id"] for record in records}
    if len(exam_ids) != 1 or len(student_ids) != 1:
        return jsonify({"success": False, "error": "같은 시험과 같은 학생의 기록만 중복 처리할 수 있습니다."}), 400

    delete_ids = [record_id for record_id in all_ids if record_id != keep_record_id]
    deleted = delete_records(delete_ids)
    return jsonify({"success": True, "deleted": deleted, "kept": keep_record_id})


@app.route("/recognition-history")
@login_required
@admin_required
def recognition_history():
    search = request.args.get("search", "").strip()
    records = [record_view_model(row) for row in get_all_records_admin(search)]
    return render_template("recognition_history.html", user=current_user, records=records, search=search)


@app.route("/admin/update", methods=["POST"])
@login_required
@admin_required
def admin_update_record():
    record_id = request.form.get("id", type=int)
    name = request.form.get("name", "").strip()
    student_id = request.form.get("student_id", "").strip()
    grade = request.form.get("grade", "").strip()
    department = request.form.get("department", "").strip()
    update_record_student_info(record_id, name, student_id, grade, department)
    return jsonify({"success": True})


@app.route("/admin/delete/<int:record_id>")
@login_required
@admin_required
def admin_delete_record(record_id):
    delete_record(record_id)
    return jsonify({"success": True})


@app.route("/save-objective-score", methods=["POST"])
@login_required
@admin_required
def save_objective_score_route():
    record_id = request.form.get("record_id", type=int)
    objective_score = request.form.get("objective_score", type=float)
    success = update_objective_score(record_id, objective_score)
    return jsonify({"success": success})


@app.route("/save-subjective-grading", methods=["POST"])
@login_required
@admin_required
def save_subjective_grading_route():
    payload = request.get_json(force=True)
    record_id = int(payload.get("record_id") or 0)
    subjective_results = payload.get("subjective_results") or {}
    success = update_subjective_grading(record_id, subjective_results)
    record = record_view_model(get_record_by_id(record_id)) if success else None
    return jsonify(
        {
            "success": success,
            "objective_score": record["objective_score"] if record else 0,
            "total_score": record["total_score"] if record else 0,
            "subjective_results": record["subjective_results"] if record else {},
        }
    )


@app.route("/student/scores")
@login_required
@student_required
def student_scores():
    records = [
        record_view_model(row)
        for row in get_user_records_for_account(int(current_user.id), current_user.student_id)
    ]
    return render_template("student_scores.html", user=current_user_row(), records=records)


@app.route("/admin/stats")
@login_required
@admin_required
def admin_stats():
    stats = get_exam_statistics()
    selected_exam_id = request.args.get("exam_id", type=int)
    if selected_exam_id is None and stats:
        selected_exam_id = stats[0]["exam_id"]
    selected_exam = next((item for item in stats if item["exam_id"] == selected_exam_id), None)
    dist = get_exam_score_distribution(selected_exam_id)
    summary = get_exam_summary(selected_exam_id)
    return render_template(
        "admin_stats.html",
        user=current_user,
        stats=stats,
        dist=dist,
        summary=summary,
        selected_exam=selected_exam,
        selected_exam_id=selected_exam_id,
    )


@app.route("/admin/users")
@login_required
@admin_required
def admin_users():
    search = request.args.get("search", "").strip()
    users = [dict(row) for row in get_all_users(search)]
    return render_template(
        "admin_users.html",
        user=current_user,
        users=users,
        search=search,
        can_grant_admin=is_super_admin(),
        super_admin_student_id=SUPER_ADMIN_STUDENT_ID,
    )


@app.route("/get-user/<int:user_id>")
@login_required
@admin_required
def get_user(user_id):
    row = get_user_by_id(user_id)
    return jsonify(dict(row) if row else {})


@app.route("/admin/user/update", methods=["POST"])
@login_required
@admin_required
def admin_user_update():
    user_id = request.form.get("id", type=int)
    target_user = get_user_by_id(user_id)
    if not target_user:
        return jsonify({"success": False, "msg": "사용자를 찾을 수 없습니다."}), 404

    requested_role = request.form.get("role", "").strip() or target_user["role"]
    if requested_role != target_user["role"] and not is_super_admin():
        return jsonify({"success": False, "msg": "최고 관리자만 권한을 변경할 수 있습니다."}), 403
    if (
        int(current_user.id) == int(user_id)
        and (target_user["student_id"] or "").strip().lower() == SUPER_ADMIN_STUDENT_ID
    ):
        requested_role = "admin"

    admin_update_user(
        user_id,
        request.form.get("student_id", "").strip(),
        request.form.get("name", "").strip(),
        request.form.get("grade", "").strip(),
        request.form.get("gender", "").strip(),
        request.form.get("email", "").strip(),
        requested_role,
        request.form.get("department", "").strip(),
    )
    return jsonify({"success": True})


@app.route("/admin/user/delete/<int:user_id>")
@login_required
@admin_required
def admin_user_delete(user_id):
    if int(current_user.id) == user_id:
        return jsonify({"success": False, "msg": "본인 계정은 삭제할 수 없습니다."})
    target_user = get_user_by_id(user_id)
    if target_user and target_user["role"] == "admin" and not is_super_admin():
        return jsonify({"success": False, "msg": "최고 관리자만 관리자 계정을 삭제할 수 있습니다."}), 403
    delete_user(user_id)
    return jsonify({"success": True})


@app.route("/admin/user/reset-pwd", methods=["POST"])
@login_required
@admin_required
def admin_user_reset_pwd():
    user_id = request.form.get("user_id", type=int)
    new_password = request.form.get("new_password", "").strip()
    update_user_password(user_id, new_password)
    return jsonify({"success": True})


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user_row = current_user_row()
    if request.method == "POST":
        action = request.form.get("action")
        if action == "update_profile":
            update_user_info(
                user_row["id"],
                request.form.get("name", "").strip(),
                request.form.get("grade", "").strip(),
                request.form.get("gender", "").strip(),
                request.form.get("email", "").strip(),
                request.form.get("department", "").strip(),
            )
            flash("프로필이 수정되었습니다.", "success")
        elif action == "change_password":
            current_password = request.form.get("current_password", "").strip()
            new_password = request.form.get("new_password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()
            if new_password != confirm_password:
                flash("새 비밀번호 확인이 일치하지 않습니다.", "error")
                return redirect(url_for("profile"))
            if not check_password_hash(user_row["password"], current_password):
                flash("현재 비밀번호가 올바르지 않습니다.", "error")
                return redirect(url_for("profile"))
            update_user_password(user_row["id"], new_password)
            flash("비밀번호가 변경되었습니다.", "success")
        return redirect(url_for("profile"))

    template_name = "profile1.html" if current_user.role == "admin" else "profile.html"
    return render_template(template_name, user=user_row)


@app.route("/profile1", methods=["GET", "POST"])
@login_required
@admin_required
def profile_admin():
    return profile()


@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results/<filename>")
@login_required
def result_file(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)


@app.route("/admin/templates/<int:template_id>/pdf")
@login_required
@admin_required
def download_template_pdf(template_id):
    template = get_template(template_id)
    if not template:
        return "Template not found", 404

    source_path = (BASE_DIR / template["source_filename"]).resolve()
    base_path = BASE_DIR.resolve()
    if base_path not in source_path.parents and source_path != base_path:
        return "Invalid template path", 400
    if not source_path.exists():
        return "Template PDF not found", 404

    safe_name = "".join(ch if ch.isalnum() else "_" for ch in template["name"])[:80] or "omr_template"
    return send_file(source_path, as_attachment=True, download_name=f"{safe_name}.pdf")


@app.route("/export-pdf/<int:record_id>")
@login_required
def export_pdf(record_id):
    record = get_record_by_id(record_id)
    if not record or not can_access_record(record):
        return "Record not found", 404

    view = record_view_model(record)
    answers = view["answers"]
    fields = view["ocr"].get("fields", {})

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 48

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "OMR Grading Report")
    y -= 28

    pdf.setFont("Helvetica", 11)
    lines = [
        f"Exam: {view.get('exam_title') or view.get('subject')}",
        f"Template: {view.get('template_name') or ''}",
        f"Name: {fields.get('name') or view.get('name') or ''}",
        f"Student ID: {fields.get('student_id') or view.get('student_id') or ''}",
        f"Department: {fields.get('department') or view.get('department') or ''}",
        f"Grade: {fields.get('grade') or view.get('grade') or ''}",
        f"Score: {view['total_score']} (MC {view['score']} + Manual {view['objective_score']})",
    ]
    for line in lines:
        pdf.drawString(50, y, line)
        y -= 18

    y -= 8
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Recognized Answers")
    y -= 20
    pdf.setFont("Helvetica", 10)

    items = sorted(((int(key), value) for key, value in answers.items()), key=lambda item: item[0])
    for question_number, answer in items:
        pdf.drawString(60, y, f"Q{question_number:02d}: {answer or '-'}")
        y -= 14
        if y < 50:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 10)

    pdf.save()
    payload = buffer.getvalue()
    buffer.close()

    response = make_response(payload)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename=record_{record_id}.pdf"
    return response


@app.route("/export-single-xlsx/<int:record_id>")
@login_required
def export_single_xlsx(record_id):
    record = get_record_by_id(record_id)
    if not record or not can_access_record(record):
        return "Record not found", 404

    view = record_view_model(record)
    answers = view["answers"]
    fields = view["ocr"].get("fields", {})

    summary = [
        {
            "Exam": view.get("exam_title") or view.get("subject"),
            "Template": view.get("template_name"),
            "Name": fields.get("name") or view.get("name"),
            "Student ID": fields.get("student_id") or view.get("student_id"),
            "Department": fields.get("department") or view.get("department"),
            "Grade": fields.get("grade") or view.get("grade"),
            "MC Score": view["score"],
            "Manual Score": view["objective_score"],
            "Total Score": view["total_score"],
            "Created At": view["created_at_display"],
        }
    ]
    answer_rows = [
        {"Question": int(question_number), "Answer": answer}
        for question_number, answer in sorted(answers.items(), key=lambda item: int(item[0]))
    ]

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(answer_rows).to_excel(writer, sheet_name="Answers", index=False)

    output.seek(0)
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        download_name=f"record_{record_id}.xlsx",
    )


@app.route("/export-all-xlsx")
@login_required
@admin_required
def export_all_xlsx():
    rows = [record_view_model(row) for row in get_all_records_admin("")]
    summary = []
    answers = []
    for row in rows:
        fields = row["ocr"].get("fields", {})
        summary.append(
            {
                "Record ID": row["id"],
                "Exam": row.get("exam_title") or row.get("subject"),
                "Template": row.get("template_name"),
                "Name": fields.get("name") or row.get("name"),
                "Student ID": fields.get("student_id") or row.get("student_id"),
                "Department": fields.get("department") or row.get("department"),
                "Grade": fields.get("grade") or row.get("grade"),
                "MC Score": row["score"],
                "Manual Score": row["objective_score"],
                "Total Score": row["total_score"],
                "Created At": row["created_at_display"],
            }
        )
        for question_number, answer in sorted(row["answers"].items(), key=lambda item: int(item[0])):
            answers.append(
                {
                    "Record ID": row["id"],
                    "Student ID": fields.get("student_id") or row.get("student_id"),
                    "Question": int(question_number),
                    "Answer": answer,
                }
            )

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="Records", index=False)
        pd.DataFrame(answers).to_excel(writer, sheet_name="Answers", index=False)

    output.seek(0)
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        download_name="all_records.xlsx",
    )


@app.route("/export-batch-results", methods=["POST"])
@login_required
@admin_required
def export_batch_results():
    payload = request.get_json(force=True)
    raw_results = payload.get("raw_results", [])

    summary_rows = []
    answer_rows = []
    for index, item in enumerate(raw_results, start=1):
        fields = item.get("recognized_fields", {})
        evaluation = item.get("evaluation", {})
        summary_rows.append(
            {
                "No": index,
                "File Name": item.get("fileName"),
                "Exam": payload.get("exam_title", ""),
                "Name": fields.get("name", ""),
                "Student ID": fields.get("student_id", ""),
                "Department": fields.get("department", ""),
                "Grade": fields.get("grade", ""),
                "Total Questions": evaluation.get("total", 0),
                "Correct": evaluation.get("correct", 0),
                "Wrong": evaluation.get("wrong", 0),
                "Unanswered": evaluation.get("unanswered", 0),
                "Score (%)": evaluation.get("percentage", 0),
            }
        )
        for question_number, answer in sorted(item.get("student_answers", {}).items(), key=lambda row: int(row[0])):
            answer_rows.append(
                {
                    "File Name": item.get("fileName"),
                    "Student ID": fields.get("student_id", ""),
                    "Question": int(question_number),
                    "Answer": answer,
                }
            )

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(answer_rows).to_excel(writer, sheet_name="Answers", index=False)

    output.seek(0)
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        download_name="batch_results.xlsx",
    )


with app.app_context():
    init_db()
    ensure_default_template()


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002)

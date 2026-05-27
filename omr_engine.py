import json
import os
import re
import unicodedata
from pathlib import Path

import cv2
import fitz
import numpy as np

try:
    import easyocr
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover - optional dependency
    RapidOCR = None


BASE_PAGE_SIZE = (1191, 1684)
BASE_FIELD_BOXES = {
    "department": [194, 228, 361, 325],
    "student_id": [419, 228, 587, 325],
    "grade": [645, 228, 714, 325],
    "name": [769, 228, 1037, 325],
}
BASE_GROUPS = [
    {"question_numbers": list(range(1, 11)), "region": [200, 480, 500, 980]},
    {"question_numbers": list(range(11, 21)), "region": [650, 480, 960, 980]},
    {"question_numbers": list(range(21, 31)), "region": [200, 1040, 500, 1540]},
    {"question_numbers": list(range(31, 41)), "region": [650, 1040, 960, 1540]},
]
BASE_ANSWER_AREA = [120, 450, 960, 1520]
DEFAULT_OPTIONS = ["1", "2", "3", "4", "5"]


def _scale_box(box, sx, sy):
    x1, y1, x2, y2 = box
    return [int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy))]


def build_default_template_config(preview_image_path):
    preview = load_image_path(preview_image_path)
    if preview is None:
        raise ValueError("Template preview image could not be loaded.")

    height, width = preview.shape[:2]
    sx = width / BASE_PAGE_SIZE[0]
    sy = height / BASE_PAGE_SIZE[1]

    groups = []
    for group in BASE_GROUPS:
        groups.append(
            {
                "question_numbers": group["question_numbers"],
                "region": _scale_box(group["region"], sx, sy),
                "options": DEFAULT_OPTIONS,
            }
        )

    config = {
        "page_size": {"width": width, "height": height},
        "student_fields": {
            key: _scale_box(value, sx, sy) for key, value in BASE_FIELD_BOXES.items()
        },
        "question_groups": groups,
        "answer_area": _scale_box(BASE_ANSWER_AREA, sx, sy),
        "fill_threshold": 0.11,
        "fill_gap_threshold": 0.03,
        "bubble_radius": max(10, int(round(10 * min(sx, sy)))),
    }
    config["bubble_map"] = detect_bubble_map(preview, config["question_groups"])
    return config


def load_file_as_page(path):
    pages, source_kind = load_file_pages(path)
    return (pages[0] if pages else None), source_kind


def load_file_pages(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        doc = fitz.open(path)
        pages = []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
            pages.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        doc.close()
        return pages, "pdf"

    image = load_image_path(path)
    return ([image] if image is not None else []), "image"


def load_image_path(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_image_path(path, image):
    ext = Path(path).suffix.lower() or ".png"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        raise ValueError(f"Failed to encode image for {path}")
    encoded.tofile(path)


def order_points(points):
    rect = np.zeros((4, 2), dtype=np.float32)
    summed = points.sum(axis=1)
    rect[0] = points[np.argmin(summed)]
    rect[2] = points[np.argmax(summed)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def _corners_from_min_area_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.array(box, dtype=np.float32)


def find_document_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = image.shape[0] * image.shape[1]

    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.2:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

        return _corners_from_min_area_rect(contour)

    height, width = image.shape[:2]
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def warp_to_template(image, page_width, page_height):
    corners = find_document_corners(image)
    ordered = order_points(corners)
    destination = np.array(
        [[0, 0], [page_width - 1, 0], [page_width - 1, page_height - 1], [0, page_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(image, matrix, (page_width, page_height))
    return warped


def detect_bubble_map(template_image, groups):
    bubble_map = {}
    for group in groups:
        x1, y1, x2, y2 = group["region"]
        crop = cv2.cvtColor(template_image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            crop,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=100,
            param2=12,
            minRadius=8,
            maxRadius=14,
        )
        if circles is None:
            raise ValueError("Could not detect answer bubbles from the template preview.")

        circles = np.round(circles[0]).astype(int)
        circles[:, 0] += x1
        circles[:, 1] += y1
        circles = sorted(circles.tolist(), key=lambda item: (item[1], item[0]))

        row_centers = _cluster_axis([circle[1] for circle in circles], tolerance=16)
        if len(row_centers) != len(group["question_numbers"]):
            raise ValueError("The detected bubble rows do not match the configured question count.")

        for question_number, row_y in zip(group["question_numbers"], row_centers):
            row_items = [circle for circle in circles if abs(circle[1] - row_y) <= 12]
            row_items = sorted(row_items, key=lambda item: item[0])[: len(group["options"])]
            if len(row_items) != len(group["options"]):
                raise ValueError("The detected bubble columns do not match the configured option count.")

            bubble_map[str(question_number)] = {
                option: {"x": int(circle[0]), "y": int(circle[1]), "r": int(circle[2])}
                for option, circle in zip(group["options"], row_items)
            }

    return bubble_map


def _cluster_axis(values, tolerance=16):
    sorted_values = sorted(values)
    buckets = []
    for value in sorted_values:
        if not buckets or abs(value - buckets[-1][-1]) > tolerance:
            buckets.append([value])
        else:
            buckets[-1].append(value)
    return [int(round(sum(bucket) / len(bucket))) for bucket in buckets]


def crop_box(image, box):
    x1, y1, x2, y2 = [int(value) for value in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    return image[y1:y2, x1:x2].copy()


def _mask_circle(radius):
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _bubble_score(binary_image, x, y, radius):
    radius = max(6, int(radius))
    mask = _mask_circle(radius)
    y1 = max(0, y - radius)
    y2 = min(binary_image.shape[0], y + radius + 1)
    x1 = max(0, x - radius)
    x2 = min(binary_image.shape[1], x + radius + 1)
    roi = binary_image[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    roi_mask = mask[: roi.shape[0], : roi.shape[1]]
    return float((roi[roi_mask] > 0).mean())


def preprocess_for_grading(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def recognize_answers(warped_image, template_image, config, answer_key):
    bubble_map = config["bubble_map"]
    blank_binary = preprocess_for_grading(template_image)
    scan_binary = preprocess_for_grading(warped_image)
    fill_threshold = float(config.get("fill_threshold", 0.11))
    gap_threshold = float(config.get("fill_gap_threshold", 0.03))
    radius_override = int(config.get("bubble_radius", 10))

    student_answers = {}
    debug_scores = {}
    for question_number in sorted(answer_key.keys()):
        options = bubble_map.get(str(question_number), {})
        scores = {}
        for option, bubble in options.items():
            radius = max(radius_override, int(bubble["r"] * 0.75))
            blank_score = _bubble_score(blank_binary, bubble["x"], bubble["y"], radius)
            scan_score = _bubble_score(scan_binary, bubble["x"], bubble["y"], radius)
            scores[option] = round(scan_score - blank_score, 4)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        debug_scores[str(question_number)] = scores

        if not ranked:
            student_answers[question_number] = None
            continue

        best_option, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0
        if best_score >= fill_threshold and (best_score - second_score) >= gap_threshold:
            student_answers[question_number] = best_option
        else:
            student_answers[question_number] = None

    return student_answers, debug_scores


def evaluate_answers(student_answers, answer_key):
    correct_count = 0
    wrong_count = 0
    unanswered_count = 0

    for question_number, correct_answer in answer_key.items():
        given_answer = student_answers.get(question_number)
        if given_answer is None:
            unanswered_count += 1
        elif str(given_answer) == str(correct_answer):
            correct_count += 1
        else:
            wrong_count += 1

    total = len(answer_key)
    percentage = round((correct_count / total) * 100, 1) if total else 0.0
    return {
        "total": total,
        "correct": correct_count,
        "wrong": wrong_count,
        "unanswered": unanswered_count,
        "percentage": percentage,
    }


def annotate_answers(warped_image, config, student_answers, answer_key):
    annotated = warped_image.copy()
    bubble_map = config["bubble_map"]

    for question_number, options in bubble_map.items():
        q_num = int(question_number)
        expected = str(answer_key.get(q_num, ""))
        selected = student_answers.get(q_num)
        for option, bubble in options.items():
            center = (bubble["x"], bubble["y"])
            radius = max(11, int(bubble["r"]))

            if option == str(selected) and option == expected:
                color = (0, 200, 0)
                thickness = 3
            elif option == str(selected):
                color = (0, 140, 255)
                thickness = 3
            elif option == expected:
                color = (0, 200, 0)
                thickness = 2
            else:
                color = (70, 70, 220)
                thickness = 1

            cv2.circle(annotated, center, radius + 3, color, thickness)

    return annotated


def stack_images_vertically(images, background=(255, 255, 255), gap=18):
    images = [image for image in images if image is not None and image.size]
    if not images:
        return None

    width = max(image.shape[1] for image in images)
    height = sum(image.shape[0] for image in images) + gap * (len(images) - 1)
    stacked = np.full((height, width, 3), background, dtype=np.uint8)
    y = 0
    for image in images:
        x = (width - image.shape[1]) // 2
        stacked[y : y + image.shape[0], x : x + image.shape[1]] = image
        y += image.shape[0] + gap
    return stacked


class OCRService:
    _easy_reader = None
    _rapid_reader = None
    _easy_error = ""
    _rapid_error = ""

    def __init__(self):
        self.tesseract_cmd = self._find_tesseract()
        self.easy = self._get_easy_reader()
        self.rapid = self._get_rapid_reader()

        if pytesseract and self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    @classmethod
    def _get_easy_reader(cls):
        if not easyocr:
            return None
        if cls._easy_reader is None:
            try:
                cls._easy_reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
                cls._easy_error = ""
            except Exception as exc:
                cls._easy_error = str(exc)
                return None
        return cls._easy_reader

    @classmethod
    def _get_rapid_reader(cls):
        if not RapidOCR:
            return None
        if cls._rapid_reader is None:
            try:
                cls._rapid_reader = RapidOCR()
                cls._rapid_error = ""
            except Exception as exc:
                cls._rapid_error = str(exc)
                return None
        return cls._rapid_reader

    def _find_tesseract(self):
        candidates = []
        if os.environ.get("TESSERACT_CMD"):
            candidates.append(os.environ["TESSERACT_CMD"])

        candidates.extend(
            [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\Public\Tesseract-OCR\tesseract.exe",
            ]
        )

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def _pad_field(self, image, padding=14):
        return cv2.copyMakeBorder(
            image,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    def _engine_status(self):
        return {
            "easyocr": bool(self.easy),
            "easyocr_error": self._easy_error if not self.easy else "",
            "rapidocr": bool(self.rapid),
            "rapidocr_error": self._rapid_error if not self.rapid else "",
            "tesseract": bool(pytesseract and self.tesseract_cmd),
            "tesseract_cmd": self.tesseract_cmd or "",
        }

    def _erase_box_frame(self, image):
        cleaned = image.copy()
        height, width = cleaned.shape[:2]
        border = max(1, int(round(min(width, height) * 0.025)))
        cleaned[:border, :] = 255
        cleaned[height - border :, :] = 255
        cleaned[:, :border] = 255
        cleaned[:, width - border :] = 255
        return cleaned

    def _erase_field_label(self, image, field_type):
        cleaned = image.copy()
        height, width = cleaned.shape[:2]
        label_width_ratio = {
            "department": 0.27,
            "student_id": 0.25,
            "grade": 0.45,
            "name": 0.20,
        }.get(field_type, 0.24)
        label_width = min(width, max(32, int(round(width * label_width_ratio))))
        label_height = min(height, max(18, int(round(height * 0.42))))
        cleaned[:label_height, :label_width] = 255
        return cleaned

    def _removed_ink_ratio(self, before, after):
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        before_ink = before_gray < 210
        after_ink = after_gray < 210
        before_count = int(before_ink.sum())
        if before_count == 0:
            return 0.0
        removed_count = int(np.logical_and(before_ink, np.logical_not(after_ink)).sum())
        return removed_count / before_count

    def _append_crop(self, crops, name, crop):
        if crop is None or crop.size == 0:
            return
        height, width = crop.shape[:2]
        if height < 10 or width < 12:
            return
        crops.append((name, crop.copy()))

    def _field_content_crops(self, image, field_type):
        height, width = image.shape[:2]
        margin_x = max(2, int(width * 0.025))
        margin_y = max(2, int(height * 0.05))
        top_cut = max(margin_y, int(height * 0.30))
        frame_clean = self._erase_box_frame(image)
        label_clean = self._erase_field_label(frame_clean, field_type)
        label_removed_allowed = self._removed_ink_ratio(frame_clean, label_clean) <= 0.20
        content_base = label_clean if label_removed_allowed else frame_clean
        crops = []
        self._append_crop(crops, "full", frame_clean)
        if label_removed_allowed:
            self._append_crop(crops, "label_removed", label_clean)
        if height - top_cut - margin_y >= 12 and width - margin_x * 2 >= 16:
            self._append_crop(
                crops,
                "content",
                content_base[top_cut : height - margin_y, margin_x : width - margin_x],
            )
        soft_top_cut = max(margin_y, int(height * 0.18))
        if height - soft_top_cut - margin_y >= 12 and width - margin_x * 2 >= 16:
            self._append_crop(
                crops,
                "content_wide",
                content_base[soft_top_cut : height - margin_y, margin_x : width - margin_x],
            )
        right_cut = int(round(width * (0.10 if field_type == "student_id" else 0.14)))
        if height - margin_y * 2 >= 12 and width - right_cut - margin_x >= 16:
            self._append_crop(
                crops,
                "handwriting",
                content_base[margin_y : height - margin_y, right_cut : width - margin_x],
            )
        return crops

    def _build_variants(self, image, field_type):
        variants = {}
        for crop_name, crop_image in self._field_content_crops(image, field_type):
            variants.update(self._build_crop_variants(crop_image, field_type, crop_name))
        return variants

    def _build_crop_variants(self, image, field_type, crop_name):
        padded = self._pad_field(image)
        scale = 4 if field_type == "student_id" else 3
        enlarged = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 60, 60)
        sharpen = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0)
        binary = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        adaptive = cv2.adaptiveThreshold(
            sharpen,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        inverted = cv2.bitwise_not(binary)

        variants = {
            f"{crop_name}_raw": enlarged,
            f"{crop_name}_gray": gray,
            f"{crop_name}_binary": binary,
            f"{crop_name}_adaptive": adaptive,
        }
        if field_type in {"name", "department"}:
            variants[f"{crop_name}_sharpen"] = sharpen
        else:
            variants[f"{crop_name}_inverted"] = inverted
        return variants

    def _read_tesseract(self, image, field_type):
        if not (pytesseract and self.tesseract_cmd):
            return []

        whitelist = ""
        lang = "kor+eng"
        if field_type == "student_id":
            lang = "eng"
            whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

        config = "--oem 3 --psm 7"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"

        try:
            text = pytesseract.image_to_string(image, lang=lang, config=config).strip()
        except Exception:
            return []

        if not text:
            return []
        return [{"engine": "tesseract", "text": text, "confidence": 0.5}]

    def _read_rapidocr(self, image):
        if not self.rapid:
            return []
        try:
            result, _ = self.rapid(image)
        except Exception:
            return []
        if not result:
            return []

        texts = []
        confidences = []
        for part in result:
            if len(part) < 3:
                continue
            text = str(part[1]).strip()
            if not text:
                continue
            texts.append(text)
            try:
                confidences.append(float(part[2]))
            except (TypeError, ValueError):
                confidences.append(0.0)

        joined = " ".join(texts).strip()
        if not joined:
            return []
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return [{"engine": "rapidocr", "text": joined, "confidence": confidence}]

    def _read_easyocr(self, image, field_type):
        if not self.easy:
            return []

        allowlist = None
        if field_type == "student_id":
            allowlist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

        try:
            result = self.easy.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist=allowlist,
                batch_size=1,
            )
        except Exception:
            return []

        if not result:
            return []

        texts = []
        confidences = []
        for part in result:
            if len(part) < 3:
                continue
            text = str(part[1]).strip()
            if not text:
                continue
            texts.append(text)
            try:
                confidences.append(float(part[2]))
            except (TypeError, ValueError):
                confidences.append(0.0)

        joined = " ".join(texts).strip()
        if not joined:
            return []
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return [{"engine": "easyocr", "text": joined, "confidence": confidence}]

    def read_field(self, image, field_type):
        candidates = []
        for variant_name, variant_image in self._build_variants(image, field_type).items():
            candidate_batches = [
                self._read_easyocr(variant_image, field_type),
                self._read_tesseract(variant_image, field_type),
                self._read_rapidocr(variant_image),
            ]
            for batch in candidate_batches:
                for candidate in batch:
                    candidates.append(
                        {
                            **candidate,
                            "variant": variant_name,
                            "normalized": normalize_field_text(field_type, candidate["text"]),
                        }
                    )

        text, best = self._choose_best(field_type, candidates)
        return text, {
            "engine": best["engine"] if best else "",
            "raw_text": best["text"] if best else "",
            "normalized": text,
            "candidates": candidates,
            "engine_status": self._engine_status(),
        }

    def _choose_best(self, field_type, candidates):
        if not candidates:
            return "", None

        ranked = []
        for candidate in candidates:
            normalized = candidate.get("normalized", "")
            raw_text = candidate.get("text", "")
            confidence = float(candidate.get("confidence", 0.0) or 0.0)
            score = confidence
            if not normalized:
                continue
            if candidate.get("variant", "").startswith("content"):
                score += 0.45
            elif candidate.get("variant", "").startswith("handwriting"):
                score += 0.35
            elif candidate.get("variant", "").startswith("label_removed"):
                score += 0.25

            if field_type == "student_id":
                alnum_count = len(re.findall(r"[A-Z0-9]", normalized))
                digit_count = len(re.findall(r"\d", normalized))
                alpha_count = len(re.findall(r"[A-Z]", normalized))
                score += min(alnum_count, 14) * 0.18
                if 6 <= alnum_count <= 12:
                    score += 2.0
                elif alnum_count >= 4:
                    score += 1.0
                if digit_count >= 4:
                    score += 0.5
                if alpha_count == 0 and digit_count >= 4:
                    score += 0.9
                elif alpha_count and digit_count >= 4:
                    score -= alpha_count * 0.55
                raw_tokens = re.findall(r"[A-Za-z0-9]+", raw_text)
                if len(raw_tokens) > 1 and len(normalized) > max(len(token) for token in raw_tokens):
                    score -= 0.8
            elif field_type == "grade":
                if normalized.isdigit():
                    score += 2.0
                    if 1 <= int(normalized) <= 12:
                        score += 1.0
                elif re.search(r"(학년|grade|year|freshman|sophomore|junior|senior)", normalized, re.IGNORECASE):
                    score += 1.5
                score += len(normalized) * 0.08
            elif field_type == "name":
                hangul_count = len(re.findall(r"[가-힣]", normalized))
                latin_count = len(re.findall(r"[A-Za-z]", normalized))
                if 2 <= hangul_count <= 6:
                    score += 2.0 + hangul_count * 0.2
                    if hangul_count == 3:
                        score += 0.45
                elif 2 <= latin_count <= 20:
                    score += 1.0 + latin_count * 0.05
            elif field_type == "department":
                hangul_count = len(re.findall(r"[가-힣]", normalized))
                latin_count = len(re.findall(r"[A-Za-z]", normalized))
                if hangul_count >= 2:
                    score += 1.6 + hangul_count * 0.1
                if any(keyword in normalized for keyword in ["학과", "학부", "전공"]):
                    score += 1.5
                if 2 <= latin_count <= 40:
                    score += 1.0 + min(latin_count, 20) * 0.04
            if field_type in {"department", "name"} and confidence < 0.15:
                score -= 0.5
            score += len(raw_text.strip()) * 0.02
            ranked.append((score, normalized, candidate))

        if not ranked:
            return "", None

        grouped = {}
        for score, normalized, candidate in ranked:
            group = grouped.setdefault(
                normalized,
                {
                    "best_score": score,
                    "best_candidate": candidate,
                    "count": 0,
                    "confidence_sum": 0.0,
                },
            )
            group["count"] += 1
            group["confidence_sum"] += float(candidate.get("confidence", 0.0) or 0.0)
            if score > group["best_score"]:
                group["best_score"] = score
                group["best_candidate"] = candidate

        consensus = []
        for normalized, group in grouped.items():
            average_confidence = group["confidence_sum"] / max(group["count"], 1)
            vote_weight = 0.35 if field_type in {"student_id", "grade"} else 0.10
            score = group["best_score"] + min(group["count"], 6) * vote_weight + average_confidence * 0.2
            consensus.append((score, normalized, group["best_candidate"]))

        consensus.sort(key=lambda item: item[0], reverse=True)
        return consensus[0][1], consensus[0][2]


def normalize_field_text(field_type, text):
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    text = _strip_field_label(field_type, text)
    if field_type == "student_id":
        cleaned = "".join(re.findall(r"[A-Za-z0-9]", text)).upper()
        digit_count = len(re.findall(r"\d", cleaned))
        if digit_count >= 3 and digit_count >= len(cleaned) * 0.55:
            cleaned = cleaned.translate(str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "U": "1"}))
        return cleaned
    if field_type == "grade":
        digit_map = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5"})
        text = text.translate(digit_map)
        digits = "".join(re.findall(r"\d", text))
        if digits:
            return digits
        return re.sub(r"[^가-힣A-Za-z0-9 ]", "", text).strip()
    if field_type == "name":
        cleaned = re.sub(r"[^가-힣A-Za-z ]", "", text).strip()
        if re.search(r"[가-힣]", cleaned):
            cleaned = re.sub(r"\s+", "", cleaned)
        return cleaned
    if field_type == "department":
        cleaned = re.sub(r"[^가-힣A-Za-z0-9 ]", "", text).strip()
        cleaned = re.sub(r"\s+", "", cleaned)
        if re.fullmatch(r"[A-Za-z0-9]+", cleaned or ""):
            return cleaned.upper()
        return cleaned
    return text


def _strip_field_label(field_type, text):
    label_patterns = {
        "department": r"^(학과|학부|전공|악파|학파|확과|하과|department|dept|major)\s*[:：\-]?\s*",
        "student_id": r"^(학번|수험번호|학생번호|학본|악번|student\s*id|id|number|no\.?)\s*[:：\-]?\s*",
        "grade": r"^(학년|학넌|악년|grade|year)\s*[:：\-]?\s*",
        "name": r"^(이름|성명|니름|01름|0름|name)\s*[:：\-]?\s*",
    }
    cleaned = re.sub(label_patterns.get(field_type, r"^$"), "", text, flags=re.IGNORECASE).strip()
    if field_type == "student_id":
        cleaned = re.sub(r"^(ND|NO|NB|N5|ID)\s*(?=[A-Za-z0-9])", "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned.lower() in {"학과", "학부", "전공", "학번", "학년", "이름", "성명", "name", "department", "grade"}:
        return ""
    return cleaned


def parse_filename_hints(filename):
    stem = unicodedata.normalize("NFKC", Path(filename).stem)
    parts = [part.strip() for part in stem.split("_") if part.strip()]
    empty = {"name": "", "student_id": "", "grade": "", "department": ""}
    if len(parts) != 4:
        return empty

    hints = {
        "name": parts[0],
        "student_id": parts[1],
        "grade": parts[2],
        "department": parts[3],
    }
    joined = " ".join(parts).lower()
    if any(keyword in joined for keyword in ["양식", "template", "시험", "테스트", "omr", "page", "preview"]):
        return empty

    student_id_hint = normalize_field_text("student_id", hints["student_id"])
    grade_hint = normalize_field_text("grade", hints["grade"])
    department_hint = normalize_field_text("department", hints["department"])
    name_hint = normalize_field_text("name", hints["name"])

    digit_count = len(re.findall(r"\d", student_id_hint))
    if len(student_id_hint) < 4 or digit_count < 3:
        return empty
    if not grade_hint.isdigit() or not 1 <= int(grade_hint) <= 12:
        return empty
    if not name_hint or not department_hint:
        return empty
    if re.fullmatch(r"\d{6,}", name_hint) or re.fullmatch(r"\d{6,}", department_hint):
        return empty

    return {
        "name": name_hint,
        "student_id": student_id_hint,
        "grade": grade_hint,
        "department": department_hint,
    }


def merge_fields(ocr_fields, filename_hints):
    merged = {}
    for key in ["department", "student_id", "grade", "name"]:
        merged[key] = ocr_fields.get(key) or filename_hints.get(key, "")
    return merged


def _resolve_template_page_path(template_preview_path, page_config):
    preview_path = Path(template_preview_path)
    page_preview = page_config.get("preview_image")
    if not page_preview:
        return preview_path

    page_path = Path(page_preview)
    if page_path.is_absolute():
        return page_path

    same_dir_path = preview_path.parent / page_path.name
    if same_dir_path.exists():
        return same_dir_path

    sibling_path = preview_path.parent / page_path
    if sibling_path.exists():
        return sibling_path

    return preview_path


def _page_answer_key(answer_key, page_config):
    page_questions = {int(key) for key in page_config.get("bubble_map", {}).keys()}
    return {
        question_number: answer
        for question_number, answer in answer_key.items()
        if question_number in page_questions
    }


def _process_single_page(
    source_image,
    template_image,
    page_config,
    answer_key,
    filename_hint="",
    read_student_fields=True,
):
    page_width = int(page_config["page_size"]["width"])
    page_height = int(page_config["page_size"]["height"])
    warped = warp_to_template(source_image, page_width, page_height)

    field_crops = {}
    ocr_fields = {}
    ocr_debug = {}
    if read_student_fields:
        ocr_service = OCRService()
        field_crops = {
            field_name: crop_box(warped, box)
            for field_name, box in page_config.get("student_fields", {}).items()
        }
        for field_name, field_image in field_crops.items():
            text, debug = ocr_service.read_field(field_image, field_name)
            ocr_fields[field_name] = text
            ocr_debug[field_name] = debug

    page_answer_key = _page_answer_key(answer_key, page_config)
    student_answers, answer_debug = recognize_answers(warped, template_image, page_config, page_answer_key)
    annotated = annotate_answers(warped, page_config, student_answers, page_answer_key)
    answer_area = crop_box(warped, page_config.get("answer_area", [0, 0, page_width, page_height]))
    subjective_images = []
    for item in page_config.get("subjective_questions", []):
        question_number = item.get("question_number")
        region = item.get("region")
        if not question_number or not region:
            continue
        subjective_images.append(
            {
                "question_number": int(question_number),
                "region": [int(value) for value in region],
                "page_number": int(page_config.get("page_number") or 1),
                "image": crop_box(warped, region),
            }
        )

    return {
        "warped_image": warped,
        "annotated_image": annotated,
        "answer_area_image": answer_area,
        "student_answers": student_answers,
        "subjective_images": subjective_images,
        "field_crops": field_crops,
        "ocr_fields": merge_fields(ocr_fields, parse_filename_hints(filename_hint)) if read_student_fields else {},
        "ocr_debug": {**ocr_debug, "answer_scores": answer_debug},
    }


def process_submission(input_path, template_preview_path, template_config, answer_key, filename_hint=""):
    source_pages, source_kind = load_file_pages(input_path)
    if not source_pages:
        raise ValueError("The uploaded file could not be opened.")

    page_configs = template_config.get("pages") or [template_config]
    if len(source_pages) < len(page_configs):
        raise ValueError(
            f"Uploaded answer sheet has {len(source_pages)} page(s), but this OMR template requires {len(page_configs)} page(s)."
        )

    page_results = []
    all_answers = {}
    all_debug = {}
    field_crops = {}
    merged_fields = {}
    subjective_images = []
    for page_index, page_config in enumerate(page_configs):
        template_image = load_image_path(_resolve_template_page_path(template_preview_path, page_config))
        if template_image is None:
            raise ValueError("The template preview image could not be loaded.")

        normalized_page_config = {
            **template_config,
            **page_config,
            "page_size": page_config.get("page_size") or template_config["page_size"],
            "student_fields": page_config.get("student_fields") or template_config.get("student_fields", {}),
            "bubble_map": page_config.get("bubble_map", {}),
            "answer_area": page_config.get("answer_area") or template_config.get("answer_area"),
            "fill_threshold": template_config.get("fill_threshold", 0.11),
            "fill_gap_threshold": template_config.get("fill_gap_threshold", 0.03),
            "bubble_radius": page_config.get("bubble_radius", template_config.get("bubble_radius", 10)),
        }
        page_result = _process_single_page(
            source_pages[page_index],
            template_image,
            normalized_page_config,
            answer_key,
            filename_hint=filename_hint or input_path,
            read_student_fields=(page_index == 0),
        )
        page_results.append(page_result)
        all_answers.update(page_result["student_answers"])
        subjective_images.extend(page_result["subjective_images"])
        all_debug[f"page_{page_index + 1}"] = page_result["ocr_debug"]
        if page_index == 0:
            field_crops = page_result["field_crops"]
            merged_fields = page_result["ocr_fields"]

    evaluation = evaluate_answers(all_answers, answer_key)
    annotated = stack_images_vertically([item["annotated_image"] for item in page_results])
    answer_area = stack_images_vertically([item["answer_area_image"] for item in page_results])
    warped = stack_images_vertically([item["warped_image"] for item in page_results])

    return {
        "source_kind": source_kind,
        "warped_image": warped,
        "annotated_image": annotated,
        "answer_area_image": answer_area,
        "student_answers": all_answers,
        "evaluation": evaluation,
        "recognized_fields": merged_fields,
        "ocr_debug": all_debug,
        "field_crops": field_crops,
        "subjective_images": subjective_images,
        "page_count": len(page_configs),
    }


def answer_key_to_json(answer_key):
    return json.dumps({str(key): str(value) for key, value in answer_key.items()}, ensure_ascii=False)

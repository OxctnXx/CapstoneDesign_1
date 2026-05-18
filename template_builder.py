import math
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

from omr_engine import load_file_pages, save_image_path


PAGE_WIDTH, PAGE_HEIGHT = A4
FONT_NAME = "HYGothic-Medium"
DEFAULT_LAYOUT = {
    "objective_area": {"x": 6, "y": 25, "w": 88, "h": 49},
    "subjective_area": {"x": 6, "y": 76, "w": 88, "h": 17},
    "subjective_box_height": 44,
}


def _register_fonts():
    try:
        pdfmetrics.getFont(FONT_NAME)
    except KeyError:
        pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))


def normalize_option_labels(option_mode, option_count, custom_options=""):
    option_count = max(2, min(10, int(option_count or 5)))
    option_mode = (option_mode or "numeric").strip().lower()

    if option_mode == "alpha":
        return [chr(ord("A") + index) for index in range(option_count)]

    if option_mode == "custom":
        tokens = [
            token.strip().upper()
            for token in re.split(r"[,;/|\s]+", custom_options or "")
            if token.strip()
        ]
        if len(tokens) >= 2:
            return tokens[:10]

    return [str(index) for index in range(1, option_count + 1)]


def option_mode_for_labels(labels):
    if labels == [str(index) for index in range(1, len(labels) + 1)]:
        return "numeric"
    if labels == [chr(ord("A") + index) for index in range(len(labels))]:
        return "alpha"
    return "custom"


def _pt_to_px(value, scale):
    return int(round(value * scale))


def _rect_to_px(rect, sx, sy):
    x1, y_top, x2, y_bottom = rect
    return [
        _pt_to_px(x1, sx),
        _pt_to_px(PAGE_HEIGHT - y_top, sy),
        _pt_to_px(x2, sx),
        _pt_to_px(PAGE_HEIGHT - y_bottom, sy),
    ]


def _point_to_px(x, y, sx, sy):
    return _pt_to_px(x, sx), _pt_to_px(PAGE_HEIGHT - y, sy)


def _normalize_area(area, default):
    area = area or {}
    x = max(0, min(96, float(area.get("x", default["x"]))))
    y = max(12, min(94, float(area.get("y", default["y"]))))
    w = max(18, min(100 - x, float(area.get("w", default["w"]))))
    h = max(8, min(98 - y, float(area.get("h", default["h"]))))
    return {"x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)}


def normalize_layout_config(layout_config=None):
    layout_config = layout_config or {}
    subjective_box_height = int(layout_config.get("subjective_box_height", DEFAULT_LAYOUT["subjective_box_height"]))
    subjective_box_height = max(24, min(96, subjective_box_height))
    return {
        "objective_area": _normalize_area(
            layout_config.get("objective_area"),
            DEFAULT_LAYOUT["objective_area"],
        ),
        "subjective_area": _normalize_area(
            layout_config.get("subjective_area"),
            DEFAULT_LAYOUT["subjective_area"],
        ),
        "subjective_box_height": subjective_box_height,
    }


def _area_to_rect(area):
    x1 = PAGE_WIDTH * area["x"] / 100
    y_top = PAGE_HEIGHT * (1 - area["y"] / 100)
    x2 = x1 + PAGE_WIDTH * area["w"] / 100
    y_bottom = y_top - PAGE_HEIGHT * area["h"] / 100
    return [x1, y_top, x2, y_bottom]


def _rect_size(rect):
    x1, y_top, x2, y_bottom = rect
    return x2 - x1, y_top - y_bottom


def _union_rects(rects):
    rects = [rect for rect in rects if rect]
    if not rects:
        return [36, 690, PAGE_WIDTH - 36, 54]
    return [
        min(rect[0] for rect in rects),
        max(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        min(rect[3] for rect in rects),
    ]


def _draw_field(pdf, label, rect):
    x1, y_top, x2, y_bottom = rect
    pdf.setStrokeColor(colors.HexColor("#1F2937"))
    pdf.setLineWidth(0.8)
    pdf.rect(x1, y_bottom, x2 - x1, y_top - y_bottom, stroke=1, fill=0)
    pdf.setFont(FONT_NAME, 8.5)
    pdf.setFillColor(colors.HexColor("#4B5563"))
    pdf.drawString(x1 + 6, y_top - 13, label)
    pdf.setFillColor(colors.black)


def _draw_page_header(pdf, title, page_number, page_count):
    margin_x = 38
    usable_width = PAGE_WIDTH - margin_x * 2
    pdf.setFillColor(colors.black)
    pdf.setFont(FONT_NAME, 18)
    pdf.drawCentredString(PAGE_WIDTH / 2, 804, title or "OMR 답안지")
    pdf.setFont(FONT_NAME, 8.5)
    pdf.setFillColor(colors.HexColor("#4B5563"))
    pdf.drawCentredString(PAGE_WIDTH / 2, 784, f"{page_number} / {page_count} 페이지")
    pdf.setFillColor(colors.black)

    field_top = 760
    field_bottom = 716
    gap = 8
    field_widths = [130, 132, 70, usable_width - 130 - 132 - 70 - gap * 3]
    student_fields = {}
    x = margin_x
    for key, label, width in zip(["department", "student_id", "grade", "name"], ["학과", "학번", "학년", "이름"], field_widths):
        rect = [x, field_top, x + width, field_bottom]
        _draw_field(pdf, label, rect)
        student_fields[key] = rect
        x += width + gap
    return student_fields


def _draw_section_boundary(pdf, label, rect):
    x1, y_top, x2, y_bottom = rect
    pdf.setStrokeColor(colors.HexColor("#CBD5E1"))
    pdf.setLineWidth(0.5)
    pdf.rect(x1, y_bottom, x2 - x1, y_top - y_bottom, stroke=1, fill=0)
    pdf.setFont(FONT_NAME, 10)
    pdf.setFillColor(colors.HexColor("#111827"))
    pdf.drawString(x1, min(y_top + 8, PAGE_HEIGHT - 22), label)
    pdf.setFillColor(colors.black)


def _objective_capacity(rect, option_count):
    width, height = _rect_size(rect)
    columns = 2 if width >= 360 and option_count <= 10 else 1
    row_height = 17.5
    usable_height = max(1, height - 30)
    rows = max(1, int(usable_height // row_height))
    return rows * columns, columns, row_height


def _subjective_capacity(rect, box_height):
    width, height = _rect_size(rect)
    columns = 2 if width >= 360 else 1
    gap = 8
    usable_height = max(1, height - 26)
    rows = max(1, int((usable_height + gap) // (box_height + gap)))
    return rows * columns, columns


def _draw_objective_chunk(pdf, rect, option_labels, start_question, end_question, page_index):
    if start_question > end_question:
        return {}, []

    _draw_section_boundary(pdf, "객관식 답란", rect)
    x1, y_top, x2, y_bottom = rect
    _, columns, row_height = _objective_capacity(rect, len(option_labels))
    count = end_question - start_question + 1
    rows_per_col = math.ceil(count / columns)
    col_gap = 18
    col_width = ((x2 - x1) - col_gap * (columns - 1)) / columns
    option_gap = min(22, max(13, (col_width - 42) / max(len(option_labels), 1)))
    bubble_radius = max(3.6, min(5.2, row_height * 0.27))
    bubble_points = {}
    regions = []

    for col in range(columns):
        col_start = start_question + col * rows_per_col
        col_end = min(end_question, col_start + rows_per_col - 1)
        if col_start > end_question:
            continue
        col_x = x1 + col * (col_width + col_gap)
        header_y = y_top - 16
        pdf.setFont(FONT_NAME, 7.4)
        pdf.setFillColor(colors.HexColor("#6B7280"))
        for option_index, label in enumerate(option_labels):
            pdf.drawCentredString(col_x + 36 + option_index * option_gap, header_y, label)
        pdf.setFillColor(colors.black)

        region_top = y_top - 8
        region_bottom = y_top - 30 - (col_end - col_start + 1) * row_height
        regions.append([col_x - 4, region_top, col_x + col_width, max(region_bottom, y_bottom + 3)])
        for row_index, question_number in enumerate(range(col_start, col_end + 1)):
            y = y_top - 30 - row_index * row_height
            if y < y_bottom + 8:
                break
            pdf.setFont(FONT_NAME, 7.6)
            pdf.drawRightString(col_x + 22, y - 2.4, str(question_number))
            bubble_points[str(question_number)] = {}
            for option_index, label in enumerate(option_labels):
                cx = col_x + 36 + option_index * option_gap
                pdf.circle(cx, y, bubble_radius, stroke=1, fill=0)
                bubble_points[str(question_number)][label] = {
                    "x_pt": cx,
                    "y_pt": y,
                    "r_pt": bubble_radius,
                    "page": page_index,
                }

    return bubble_points, regions


def _draw_subjective_chunk(pdf, rect, objective_count, start_index, end_index, box_height):
    if start_index > end_index:
        return [], []

    _draw_section_boundary(pdf, "주관식 답란", rect)
    x1, y_top, x2, y_bottom = rect
    _, columns = _subjective_capacity(rect, box_height)
    count = end_index - start_index + 1
    rows_per_col = math.ceil(count / columns)
    gap = 8
    col_width = ((x2 - x1) - gap * (columns - 1)) / columns
    boxes = []
    regions = []

    for col in range(columns):
        col_start = start_index + col * rows_per_col
        col_end = min(end_index, col_start + rows_per_col - 1)
        if col_start > end_index:
            continue
        col_x = x1 + col * (col_width + gap)
        for row_index, subject_index in enumerate(range(col_start, col_end + 1)):
            q_number = objective_count + subject_index + 1
            top = y_top - 24 - row_index * (box_height + gap)
            bottom = top - box_height
            if bottom < y_bottom:
                break
            pdf.setStrokeColor(colors.HexColor("#9CA3AF"))
            pdf.setLineWidth(0.8)
            pdf.rect(col_x, bottom, col_width, box_height, stroke=1, fill=0)
            pdf.setFillColor(colors.HexColor("#4B5563"))
            pdf.setFont(FONT_NAME, 8)
            pdf.drawString(col_x + 6, top - 13, f"{q_number}번")
            pdf.setFillColor(colors.black)
            box_rect = [col_x, top, col_x + col_width, bottom]
            boxes.append({"question_number": q_number, "rect_pt": box_rect})
            regions.append(box_rect)

    return boxes, regions


def _render_pdf_previews(pdf_path, output_dir, basename):
    pages, _ = load_file_pages(str(pdf_path))
    preview_paths = []
    for index, image in enumerate(pages, start=1):
        preview_name = f"{basename}.png" if index == 1 else f"{basename}_page_{index}.png"
        preview_path = output_dir / preview_name
        save_image_path(str(preview_path), image)
        preview_paths.append(preview_path)
    return pages, preview_paths


def build_omr_template(
    output_dir,
    title,
    objective_count,
    subjective_count,
    option_labels,
    basename,
    layout_config=None,
):
    _register_fonts()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{basename}.pdf"

    objective_count = max(1, min(300, int(objective_count or 1)))
    subjective_count = max(0, min(200, int(subjective_count or 0)))
    option_labels = [str(label).strip().upper() for label in option_labels if str(label).strip()]
    if len(option_labels) < 2:
        option_labels = ["1", "2", "3", "4", "5"]
    option_labels = option_labels[:10]
    layout = normalize_layout_config(layout_config)
    objective_rect = _area_to_rect(layout["objective_area"])
    subjective_rect = _area_to_rect(layout["subjective_area"])
    objective_capacity, _, _ = _objective_capacity(objective_rect, len(option_labels))
    subjective_capacity, _ = _subjective_capacity(subjective_rect, layout["subjective_box_height"])
    objective_capacity = max(1, objective_capacity)
    subjective_capacity = max(1, subjective_capacity)
    objective_pages = math.ceil(objective_count / objective_capacity) if objective_count else 0
    subjective_pages = math.ceil(subjective_count / subjective_capacity) if subjective_count else 0
    page_count = max(1, objective_pages, subjective_pages)

    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    pdf.setTitle(title or "OMR Answer Sheet")

    page_configs = []
    aggregate_bubble_map = {}
    aggregate_subjective = []
    first_student_fields = None
    for page_index in range(page_count):
        student_fields = _draw_page_header(pdf, title, page_index + 1, page_count)
        if first_student_fields is None:
            first_student_fields = student_fields

        objective_start = page_index * objective_capacity + 1
        objective_end = min(objective_count, (page_index + 1) * objective_capacity)
        page_bubbles, objective_regions = _draw_objective_chunk(
            pdf,
            objective_rect,
            option_labels,
            objective_start,
            objective_end,
            page_index + 1,
        )

        subjective_start = page_index * subjective_capacity
        subjective_end = min(subjective_count - 1, (page_index + 1) * subjective_capacity - 1)
        subjective_boxes, subjective_regions = _draw_subjective_chunk(
            pdf,
            subjective_rect,
            objective_count,
            subjective_start,
            subjective_end,
            layout["subjective_box_height"],
        )

        pdf.setFont(FONT_NAME, 7.5)
        pdf.setFillColor(colors.HexColor("#6B7280"))
        pdf.drawString(38, 28, "검은색 펜으로 빈 원 안을 진하게 표시하세요. 주관식은 지정된 답란 안에 작성하세요.")
        pdf.setFillColor(colors.black)

        aggregate_bubble_map.update(page_bubbles)
        aggregate_subjective.extend(subjective_boxes)
        page_configs.append(
            {
                "page_number": page_index + 1,
                "page_size": {},
                "student_fields_pt": student_fields,
                "bubble_points": page_bubbles,
                "subjective_boxes": subjective_boxes,
                "answer_rect_pt": _union_rects([*objective_regions, *subjective_regions]),
            }
        )
        if page_index < page_count - 1:
            pdf.showPage()
    pdf.save()

    preview_images, preview_paths = _render_pdf_previews(pdf_path, output_dir, basename)
    if not preview_images:
        raise ValueError("Failed to render OMR template preview pages.")

    page_height, page_width = preview_images[0].shape[:2]
    sx = page_width / PAGE_WIDTH
    sy = page_height / PAGE_HEIGHT
    page_outputs = []
    top_level_bubble_map = {}
    top_level_subjective = []

    for index, page_config in enumerate(page_configs):
        bubble_map = {}
        for question_number, options in page_config["bubble_points"].items():
            bubble_map[question_number] = {}
            for label, bubble in options.items():
                x_px, y_px = _point_to_px(bubble["x_pt"], bubble["y_pt"], sx, sy)
                bubble_map[question_number][label] = {
                    "x": x_px,
                    "y": y_px,
                    "r": max(5, _pt_to_px(bubble["r_pt"], min(sx, sy))),
                }
                top_level_bubble_map.setdefault(question_number, {})[label] = {
                    **bubble_map[question_number][label],
                    "page": index + 1,
                }

        subjective_questions = [
            {
                "question_number": item["question_number"],
                "region": _rect_to_px(item["rect_pt"], sx, sy),
            }
            for item in page_config["subjective_boxes"]
        ]
        top_level_subjective.extend(subjective_questions)
        page_outputs.append(
            {
                "page_number": index + 1,
                "preview_image": preview_paths[index].name,
                "page_size": {"width": page_width, "height": page_height},
                "student_fields": {
                    key: _rect_to_px(rect, sx, sy)
                    for key, rect in page_config["student_fields_pt"].items()
                },
                "bubble_map": bubble_map,
                "answer_area": _rect_to_px(page_config["answer_rect_pt"], sx, sy),
                "subjective_questions": subjective_questions,
                "bubble_radius": 6,
            }
        )

    config = {
        "page_size": {"width": page_width, "height": page_height},
        "student_fields": {
            key: _rect_to_px(rect, sx, sy) for key, rect in first_student_fields.items()
        },
        "question_groups": [],
        "answer_area": page_outputs[0]["answer_area"],
        "bubble_map": top_level_bubble_map,
        "pages": page_outputs,
        "preview_images": [path.name for path in preview_paths],
        "fill_threshold": 0.11,
        "fill_gap_threshold": 0.03,
        "bubble_radius": 6,
        "subjective_questions": top_level_subjective,
        "template_settings": {
            "title": title or "OMR 답안지",
            "objective_count": objective_count,
            "subjective_count": subjective_count,
            "option_labels": option_labels,
            "option_count": len(option_labels),
            "option_mode": option_mode_for_labels(option_labels),
            "page_count": page_count,
            "layout": layout,
            "objective_capacity": objective_capacity,
            "subjective_capacity": subjective_capacity,
        },
    }

    return {
        "pdf_path": pdf_path,
        "preview_path": preview_paths[0],
        "preview_paths": preview_paths,
        "page_width": page_width,
        "page_height": page_height,
        "config": config,
    }

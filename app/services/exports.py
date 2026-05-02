from __future__ import annotations

import csv
import io

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.models import ComparisonRow, PlanVersion


def memo_to_pdf_bytes(plan: PlanVersion) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    for line in plan.memo_markdown.splitlines():
        if not line:
            y -= 14
            continue
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(40, y, line[:130])
        y -= 14

    c.save()
    buffer.seek(0)
    return buffer.read()


def comparison_to_csv_bytes(rows: list[ComparisonRow]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "department",
            "mayor_directional",
            "proposed_directional",
            "delta_directional",
            "mayor_fy_2025_26",
            "mayor_fy_2026_27",
            "proposed_fy_2025_26",
            "proposed_fy_2026_27",
            "rationale",
            "citations",
        ]
    )

    for row in rows:
        writer.writerow(
            [
                row.department,
                row.mayor_directional,
                row.proposed_directional,
                row.delta_directional,
                row.mayor_fy_2025_26,
                row.mayor_fy_2026_27,
                row.proposed_fy_2025_26,
                row.proposed_fy_2026_27,
                row.rationale,
                row.citations,
            ]
        )

    return buf.getvalue().encode("utf-8")

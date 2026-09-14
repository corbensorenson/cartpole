#!/usr/bin/env python
"""Render the consolidated seven-through-ten-link paper as a polished PDF."""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#071525")
INK = colors.HexColor("#152536")
MUTED = colors.HexColor("#5D6B7A")
TEAL = colors.HexColor("#00A6A6")
ORANGE = colors.HexColor("#FF8A3D")
PALE = colors.HexColor("#EEF4F7")
GREEN = colors.HexColor("#167B5B")

EVIDENCE_FRAMES = {
    7: (
        ROOT / "runs/swingup7_uniform/seven_link_swingup_success.mp4",
        28.02,
    ),
    8: (
        ROOT / "runs/generalized_solver/eight_link_swingup_success.mp4",
        28.02,
    ),
    9: (
        ROOT / "runs/generalized_solver/nine_link_swingup_success.mp4",
        28.02,
    ),
    10: (
        ROOT
        / "runs/generalized_solver/n10_fddp_refined_route_feedback100_park16_targetm005.mp4",
        29.98,
    ),
}


def register_fonts() -> None:
    font_dir = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("PaperSans", str(font_dir / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("PaperSans-Bold", str(font_dir / "Arial Bold.ttf")))
    pdfmetrics.registerFont(
        TTFont("PaperSans-Italic", str(font_dir / "Arial Italic.ttf"))
    )
    pdfmetrics.registerFont(
        TTFont("PaperSans-BoldItalic", str(font_dir / "Arial Bold Italic.ttf"))
    )


def ensure_evidence_frames() -> None:
    """Extract the four state-faithful evidence frames when they are absent."""
    frame_dir = ROOT / "tmp/pdfs"
    frame_dir.mkdir(parents=True, exist_ok=True)
    missing_videos = [
        str(video.relative_to(ROOT))
        for video, _ in EVIDENCE_FRAMES.values()
        if not video.is_file()
    ]
    if missing_videos:
        raise FileNotFoundError(
            "paper evidence video(s) missing: " + ", ".join(missing_videos)
        )
    missing = [
        count
        for count, (video, _) in EVIDENCE_FRAMES.items()
        if not (frame_dir / f"n{count}_upright.png").is_file()
    ]
    if not missing:
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to extract the paper's evidence frames")
    for count in missing:
        video, timestamp = EVIDENCE_FRAMES[count]
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.2f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-y",
                str(frame_dir / f"n{count}_upright.png"),
            ],
            check=True,
        )
    pdfmetrics.registerFontFamily(
        "PaperSans",
        normal="PaperSans",
        bold="PaperSans-Bold",
        italic="PaperSans-Italic",
        boldItalic="PaperSans-BoldItalic",
    )


def inline_markup(value: str) -> str:
    escaped = html.escape(value.strip())
    escaped = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#007C85"><u>\1</u></link>',
        escaped,
    )
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(
        r"`([^`]+)`", r'<font name="Courier" color="#24465A">\1</font>', escaped
    )
    return escaped


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="PaperSans-Bold",
            fontSize=25,
            leading=29,
            textColor=colors.white,
            alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Heading2"],
            fontName="PaperSans",
            fontSize=14,
            leading=19,
            textColor=colors.HexColor("#9EDDDD"),
            alignment=TA_LEFT,
            spaceAfter=24,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["BodyText"],
            fontName="PaperSans",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#D4DEE8"),
        ),
        "cover_note": ParagraphStyle(
            "CoverNote",
            parent=base["BodyText"],
            fontName="PaperSans",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#AEBECB"),
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading1"],
            fontName="PaperSans-Bold",
            fontSize=16,
            leading=20,
            textColor=NAVY,
            spaceBefore=16,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading2"],
            fontName="PaperSans-Bold",
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#007C85"),
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="PaperSans",
            fontSize=8.65,
            leading=12.2,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=6,
            splitLongWords=False,
            allowWidows=0,
            allowOrphans=0,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="PaperSans",
            fontSize=8.55,
            leading=12,
            textColor=INK,
            leftIndent=15,
            firstLineIndent=-8,
            bulletIndent=4,
            spaceAfter=3,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName="PaperSans-Italic",
            fontSize=9.2,
            leading=13,
            textColor=colors.HexColor("#274A5A"),
            leftIndent=16,
            rightIndent=16,
            borderColor=TEAL,
            borderWidth=0,
            borderPadding=8,
            backColor=PALE,
            spaceBefore=5,
            spaceAfter=8,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.35,
            leading=9.8,
            textColor=colors.HexColor("#24465A"),
            backColor=PALE,
            borderPadding=8,
            leftIndent=5,
            rightIndent=5,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="PaperSans",
            fontSize=7.3,
            leading=9.4,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="PaperSans",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
        ),
    }


def cover_story(style: dict[str, ParagraphStyle]) -> list[object]:
    snapshot = [
        ["LINKS", "NOISY GATES", "HELD-OUT HOLD", "MAX CART"],
        ["7", "20/20 + 100/100", "15.48 s", "2.3708 m"],
        ["8", "20/20 + 100/100", "12.22 s", "2.9300 m"],
        ["9", "20/20 + 100/100", "12.18 s", "2.9800 m"],
        ["10", "20/20 + 100/100", "6.12 s", "1.9976 m"],
    ]
    table = Table(
        snapshot, colWidths=[0.65 * inch, 1.65 * inch, 1.25 * inch, 1.1 * inch]
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163248")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#88D9D4")),
                ("FONTNAME", (0, 0), (-1, 0), "PaperSans-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "PaperSans-Bold"),
                ("FONTNAME", (1, 1), (-1, -1), "PaperSans"),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#496274")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [NAVY, colors.HexColor("#0B1D2E")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    return [
        Spacer(1, 0.68 * inch),
        Paragraph(
            "RESET-FREE SWING-UP AND HOLD OF SEVEN THROUGH TEN SERIAL LINKS WITH ONE CART ACTUATOR",
            style["cover_title"],
        ),
        Paragraph(
            "Funnel Composition, Target-Plant Trajectory Optimization, and Evidence-Gated Scaling",
            style["cover_subtitle"],
        ),
        HRFlowable(width="100%", thickness=1.2, color=ORANGE, spaceAfter=17),
        Paragraph(
            "<b>Corben Sorenson</b><br/>Technical report, revision 2<br/>14 September 2026<br/>"
            '<link href="https://github.com/corbensorenson/cartpole" color="#9EDDDD">'
            "github.com/corbensorenson/cartpole</link>",
            style["cover_meta"],
        ),
        Spacer(1, 0.36 * inch),
        Paragraph(
            "EVIDENCE SNAPSHOT",
            ParagraphStyle(
                "CoverLabel",
                parent=style["cover_meta"],
                fontName="PaperSans-Bold",
                fontSize=8,
                tracking=1.5,
                textColor=colors.HexColor("#88D9D4"),
                spaceAfter=7,
            ),
        ),
        table,
        Spacer(1, 0.27 * inch),
        Paragraph(
            "PROVISIONAL PUBLIC LINK-COUNT RECORD",
            ParagraphStyle(
                "Record",
                parent=style["cover_meta"],
                fontName="PaperSans-Bold",
                fontSize=9.4,
                leading=13,
                textColor=colors.white,
                borderColor=ORANGE,
                borderWidth=0.8,
                borderPadding=9,
                backColor=colors.HexColor("#10283A"),
            ),
        ),
        Spacer(1, 0.08 * inch),
        Paragraph(
            "Scoped to the frozen simulated benchmark in this report. Independent reproduction and matched prior-art adjudication remain open.",
            style["cover_note"],
        ),
        PageBreak(),
    ]


def parse_table(
    lines: list[str], style: dict[str, ParagraphStyle], width: float
) -> Table:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines
    ]
    rows = [rows[0], *rows[2:]]
    count = len(rows[0])
    data = [
        [Paragraph(inline_markup(cell), style["small"]) for cell in row] for row in rows
    ]
    col_widths = [width / count] * count
    if count == 2:
        col_widths = [0.43 * width, 0.57 * width]
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "PaperSans-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C7CF")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ALIGN", (1, 1), (-1, -1), "LEFT"),
            ]
        )
    )
    return table


def evidence_plate(style: dict[str, ParagraphStyle]) -> list[object]:
    paths = [ROOT / "tmp/pdfs" / f"n{count}_upright.png" for count in (7, 8, 9, 10)]
    if not all(path.is_file() for path in paths):
        return []
    cells: list[list[object]] = []
    for start in (0, 2):
        row: list[object] = []
        for offset in (0, 1):
            index = start + offset
            count = 7 + index
            image = Image(str(paths[index]), width=3.18 * inch, height=1.79 * inch)
            caption = Paragraph(
                f"{count} links at t=28.02 s"
                if count < 10
                else "10 links at t=29.98 s",
                style["caption"],
            )
            row.append([image, caption])
        cells.append(row)
    plate = Table(cells, colWidths=[3.26 * inch, 3.26 * inch], hAlign="CENTER")
    plate.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    figure = KeepTogether(
        [
            plate,
            Paragraph(
                "Figure 1. Held-out state-faithful video frames. All links are upright in one uninterrupted episode; the ten-link frame is taken near the 30 s boundary after the five-second gate has passed.",
                style["caption"],
            ),
        ]
    )
    return [Spacer(1, 8), figure]


def markdown_story(
    text: str, style: dict[str, ParagraphStyle], width: float
) -> list[object]:
    lines = text.splitlines()
    abstract_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "## Abstract"),
        0,
    )
    lines = lines[abstract_index:]
    story: list[object] = []
    paragraph: list[str] = []
    code: list[str] = []
    in_code = False
    index = 0
    inserted_plate = False

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline_markup(" ".join(paragraph)), style["body"]))
            paragraph.clear()

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                story.append(Preformatted("\n".join(code), style["code"]))
                code.clear()
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code.append(raw)
            index += 1
            continue
        if (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and set(
                lines[index + 1]
                .replace("|", "")
                .replace(":", "")
                .replace("-", "")
                .strip()
            )
            <= {" "}
        ):
            flush_paragraph()
            table_lines = [raw, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(parse_table(table_lines, style, width))
            story.append(Spacer(1, 7))
            header = table_lines[0]
            if "Noisy 20" in header and not inserted_plate:
                story.extend(evidence_plate(style))
                inserted_plate = True
            continue
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped.startswith("# "):
            index += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[3:]), style["h2"]))
            index += 1
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[4:]), style["h3"]))
            index += 1
            continue
        bullet = re.match(r"^[-*] (.+)$", stripped)
        numbered = re.match(r"^(\d+)\. (.+)$", stripped)
        if bullet or numbered:
            flush_paragraph()
            marker = "-" if bullet else f"{numbered.group(1)}."
            content = bullet.group(1) if bullet else numbered.group(2)
            story.append(
                Paragraph(inline_markup(content), style["bullet"], bulletText=marker)
            )
            index += 1
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[2:]), style["quote"]))
            index += 1
            continue
        paragraph.append(stripped)
        index += 1
    flush_paragraph()
    return story


def draw_page(canvas, doc) -> None:  # noqa: ANN001
    canvas.saveState()
    width, height = LETTER
    if doc.page == 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, width, height, stroke=0, fill=1)
        canvas.setFillColor(ORANGE)
        canvas.rect(0, height - 0.11 * inch, width, 0.11 * inch, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor("#6F8190"))
        canvas.setFont("PaperSans", 7)
        canvas.drawRightString(
            width - 0.7 * inch, 0.42 * inch, "REVISION 2 / 2026-09-14"
        )
    else:
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, width, height, stroke=0, fill=1)
        canvas.setStrokeColor(colors.HexColor("#D7E0E5"))
        canvas.line(
            0.68 * inch, height - 0.48 * inch, width - 0.68 * inch, height - 0.48 * inch
        )
        canvas.setFillColor(MUTED)
        canvas.setFont("PaperSans", 7)
        canvas.drawString(
            0.68 * inch, height - 0.36 * inch, "RESET-FREE 7-10 LINK CART-POLE"
        )
        canvas.drawRightString(
            width - 0.68 * inch, height - 0.36 * inch, "TECHNICAL REPORT"
        )
        canvas.setStrokeColor(colors.HexColor("#D7E0E5"))
        canvas.line(0.68 * inch, 0.48 * inch, width - 0.68 * inch, 0.48 * inch)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            0.68 * inch, 0.31 * inch, "github.com/corbensorenson/cartpole"
        )
        canvas.drawRightString(width - 0.68 * inch, 0.31 * inch, str(doc.page - 1))
    canvas.restoreState()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="docs/seven_link_swingup_paper.md", type=Path
    )
    parser.add_argument(
        "--out", default="output/pdf/seven_to_ten_link_swingup_paper.pdf", type=Path
    )
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.out if args.out.is_absolute() else ROOT / args.out
    register_fonts()
    ensure_evidence_frames()
    style = styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=LETTER,
        rightMargin=0.68 * inch,
        leftMargin=0.68 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.62 * inch,
        title="Reset-Free Swing-Up and Hold of Seven Through Ten Serial Links",
        author="Corben Sorenson",
        subject="Multi-link cart-pole control and reproducible benchmark evidence",
        creator="cartpole research repository",
    )
    story = cover_story(style)
    story.extend(
        markdown_story(source.read_text(encoding="utf-8"), style, document.width)
    )
    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    print(output)


if __name__ == "__main__":
    main()

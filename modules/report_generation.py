"""Create a downloadable PDF containing the analysis and its limitations."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _register_unicode_font() -> str:
    """Use Matplotlib's bundled DejaVu font for multilingual transcript coverage."""
    font_name = "VBCUADejaVu"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        try:
            import matplotlib

            font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
        except Exception:
            return "Helvetica"
    return font_name


def generate_pdf_report(result: dict) -> bytes:
    """Render a compact PDF report and return its bytes."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"VBCUA Report - {result.get('concept_name', 'Concept')}",
        author="Voice-Based Concept Understanding Analyser",
    )
    font = _register_unicode_font()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="VBCUATitle", parent=styles["Title"], fontName=font,
                              alignment=TA_CENTER, textColor=colors.HexColor("#16324f")))
    styles.add(ParagraphStyle(name="VBCUAHeading", parent=styles["Heading2"], fontName=font,
                              textColor=colors.HexColor("#176b87"), spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="VBCUABody", parent=styles["BodyText"], fontName=font,
                              leading=14, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="VBCUASmall", parent=styles["BodyText"], fontName=font,
                              fontSize=8.5, leading=11, textColor=colors.HexColor("#555555")))

    def paragraph(value, style="VBCUABody"):
        safe = escape(str(value)).replace("\n", "<br/>")
        return Paragraph(safe, styles[style])

    score = result["score"]
    metrics = result["audio_metrics"]
    filler = result["filler_analysis"]
    semantic = result["semantic"]
    rows = [
        [paragraph("Metric"), paragraph("Result")],
        [paragraph("Overall practice score"), paragraph(f"{score['overall_score']:.1f} / 100")],
        [paragraph("Semantic similarity"), paragraph(f"{semantic['similarity_percent']:.1f}%")],
        [paragraph("Filler-token ratio"), paragraph(f"{filler['filler_ratio_percent']:.1f}%")],
        [paragraph("Estimated pause ratio"), paragraph(f"{metrics['pause_ratio_percent']:.1f}%")],
        [paragraph("Duration"), paragraph(f"{metrics['duration_seconds']:.2f} seconds")],
        [paragraph("RMS energy"), paragraph(f"{metrics['rms_energy']:.5f}")],
        [paragraph("Words / filler occurrences"), paragraph(f"{filler['total_words']} / {filler['filler_count']}")],
    ]
    table = Table(rows, colWidths=[73 * mm, 88 * mm], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1f5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7d2d9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story = [
        paragraph("Voice-Based Concept Understanding Analyser", "VBCUATitle"),
        paragraph(f"Practice report · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", "VBCUASmall"),
        Spacer(1, 7 * mm),
        paragraph("Concept", "VBCUAHeading"),
        paragraph(result.get("concept_name", "My concept")),
        paragraph("Audio file", "VBCUAHeading"),
        paragraph(result.get("audio_filename", "Uploaded recording")),
        paragraph("Measured results", "VBCUAHeading"),
        table,
        paragraph("Score method", "VBCUAHeading"),
        paragraph(
            "The score is a configurable weighted average. Its default weights are 70% semantic "
            "similarity, 20% filler fluency, and 10% pause fluency. Filler score falls linearly "
            "to zero at 25% filler-token ratio. Pause score is 100 through 15% estimated quiet "
            "time and falls linearly to zero at 65%. Current normalized weights: "
            + ", ".join(f"{name} {weight * 100:.0f}%" for name, weight in score["weights"].items())
            + "."
        ),
        paragraph("Transcription", "VBCUAHeading"),
        paragraph(result.get("transcript", "")),
        paragraph("Reference explanation", "VBCUAHeading"),
        paragraph(result.get("reference", "")),
        paragraph("Feedback", "VBCUAHeading"),
        paragraph(result["feedback"]["headline"]),
    ]
    for item in result["feedback"].get("strengths", []) + result["feedback"].get("suggestions", []):
        story.append(paragraph("• " + item))
    story.extend([
        Spacer(1, 5 * mm),
        paragraph(
            "Limitations: this is a practice aid, not a scientifically validated assessment. "
            "Semantic similarity measures wording and related meaning, not factual correctness. "
            "Pause ratio is estimated from frame energy and can count soft speech as quiet. "
            "Whisper transcription and embedding models can make mistakes.",
            "VBCUASmall",
        ),
    ])
    document.build(story)
    return output.getvalue()

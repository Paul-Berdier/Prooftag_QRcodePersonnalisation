from pathlib import Path
import html
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output" / "pdf" / "traduction_expliquee_diffqrcoder.md"
OUTPUT = ROOT / "output" / "pdf" / "traduction_expliquee_diffqrcoder.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2563A6")
CYAN = colors.HexColor("#DDEFF8")
GREEN = colors.HexColor("#2E7D5B")
PALE_GREEN = colors.HexColor("#E5F3EB")
ORANGE = colors.HexColor("#E18335")
PALE_ORANGE = colors.HexColor("#FFF0E2")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5F6B76")
LIGHT = colors.HexColor("#EDF1F4")


pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", r"C:\Windows\Fonts\ariali.ttf"))
pdfmetrics.registerFontFamily(
    "Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic"
)


class NumberedDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [PageTemplate(id="content", frames=[frame], onPage=self._decorate_page)]
        )

    def _decorate_page(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        if doc.page > 1:
            canvas.setStrokeColor(LIGHT)
            canvas.line(18 * mm, h - 13 * mm, w - 18 * mm, h - 13 * mm)
            canvas.setFont("Arial", 8)
            canvas.setFillColor(MUTED)
            canvas.drawString(18 * mm, h - 10 * mm, "DIFFQRCODER - TRADUCTION EXPLIQUEE")
        canvas.setFont("Arial", 8)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(w / 2, 10 * mm, str(doc.page))
        canvas.restoreState()


class PipelineDiagram(Flowable):
    def __init__(self, width=170 * mm, height=42 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        y = 10 * mm
        box_w = 45 * mm
        box_h = 24 * mm
        gap = 12 * mm
        items = [
            ("1. CREER", "Image artistique\n(ControlNet)", BLUE, CYAN),
            ("2. MESURER", "Modules mal lus\n(SRL + LPIPS)", ORANGE, PALE_ORANGE),
            ("3. CORRIGER", "QR code lisible\n(SRPG + SR-MPGD)", GREEN, PALE_GREEN),
        ]
        x = 4 * mm
        for idx, (title, subtitle, edge, fill) in enumerate(items):
            c.setFillColor(fill)
            c.setStrokeColor(edge)
            c.setLineWidth(1.4)
            c.roundRect(x, y, box_w, box_h, 4 * mm, fill=1, stroke=1)
            c.setFillColor(edge)
            c.setFont("Arial-Bold", 9)
            c.drawCentredString(x + box_w / 2, y + 15 * mm, title)
            c.setFillColor(INK)
            c.setFont("Arial", 8)
            for j, line in enumerate(subtitle.split("\n")):
                c.drawCentredString(x + box_w / 2, y + (10 - 4 * j) * mm, line)
            if idx < 2:
                start = x + box_w + 2 * mm
                end = x + box_w + gap - 2 * mm
                mid = y + box_h / 2
                c.setStrokeColor(NAVY)
                c.setFillColor(NAVY)
                c.line(start, mid, end, mid)
                c.line(end, mid, end - 2.5 * mm, mid + 1.8 * mm)
                c.line(end, mid, end - 2.5 * mm, mid - 1.8 * mm)
            x += box_w + gap


def styles():
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=9.4,
            leading=13.2,
            textColor=INK,
            spaceAfter=6,
            allowWidows=0,
            allowOrphans=0,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Arial-Bold",
            fontSize=19,
            leading=23,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=9,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Arial-Bold",
            fontSize=14,
            leading=17,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="Arial-Bold",
            fontSize=11.2,
            leading=14,
            textColor=GREEN,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h4": ParagraphStyle(
            "H4",
            parent=base["Heading4"],
            fontName="Arial-Bold",
            fontSize=9.8,
            leading=12,
            textColor=ORANGE,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName="Arial-Italic",
            fontSize=9.5,
            leading=13.5,
            leftIndent=10 * mm,
            rightIndent=7 * mm,
            borderColor=BLUE,
            borderWidth=0,
            borderPadding=7,
            backColor=CYAN,
            textColor=NAVY,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Arial",
            fontSize=8.7,
            leading=12,
            leftIndent=7 * mm,
            rightIndent=7 * mm,
            backColor=colors.HexColor("#F5F7F9"),
            borderColor=LIGHT,
            borderWidth=0.5,
            borderPadding=5,
            textColor=NAVY,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=9.3,
            leading=13,
            textColor=INK,
            leftIndent=2 * mm,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=7.6,
            leading=9.5,
            textColor=INK,
        ),
    }


def inline_markup(text):
    text = html.escape(text.strip())
    text = re.sub(r"`([^`]+)`", r'<font name="Arial" color="#17324D">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def table_flowable(rows, avail_width, st):
    parsed = [[inline_markup(cell.strip()) for cell in row] for row in rows]
    ncols = max(len(r) for r in parsed)
    parsed = [r + [""] * (ncols - len(r)) for r in parsed]
    weights = []
    for col in range(ncols):
        longest = max(len(re.sub(r"<[^>]+>", "", row[col])) for row in parsed)
        weights.append(max(8, min(longest, 32)))
    total = sum(weights)
    col_widths = [avail_width * w / total for w in weights]
    cells = [[Paragraph(cell, st["small"]) for cell in row] for row in parsed]
    table = Table(cells, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BCC7D1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def parse_markdown(md, st, avail_width):
    lines = md.splitlines()
    story = []
    paragraph = []

    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(inline_markup(" ".join(paragraph)), st["body"]))
            paragraph.clear()

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            flush_paragraph()
            i += 1
            continue
        if line == "---":
            flush_paragraph()
            story.append(Spacer(1, 4 * mm))
            i += 1
            continue
        if line.startswith("#"):
            flush_paragraph()
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            style_name = "h1" if level == 1 else "h2" if level == 2 else "h3" if level == 3 else "h4"
            story.append(Paragraph(inline_markup(text), st[style_name]))
            if text == "3.0 Vue d'ensemble":
                story.append(PipelineDiagram(width=avail_width))
                story.append(Spacer(1, 2 * mm))
            i += 1
            continue
        if line.startswith(">"):
            flush_paragraph()
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip()[1:].strip())
                i += 1
            story.append(Paragraph(inline_markup(" ".join(quote_lines)), st["quote"]))
            continue
        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c or "") for c in row):
                    table_rows.append(row)
                i += 1
            story.append(Spacer(1, 2 * mm))
            story.append(table_flowable(table_rows, avail_width, st))
            story.append(Spacer(1, 3 * mm))
            continue
        if re.match(r"^[-*] ", line) or re.match(r"^\d+\. ", line):
            flush_paragraph()
            ordered = bool(re.match(r"^\d+\. ", line))
            item_rows = []
            item_number = 1
            while i < len(lines):
                current = lines[i].strip()
                match = re.match(r"^([-*]|\d+\.)\s+(.*)$", current)
                if not match:
                    break
                label = f"{item_number}." if ordered else "•"
                item_rows.append(
                    [
                        Paragraph(f"<b>{label}</b>", st["bullet"]),
                        Paragraph(inline_markup(match.group(2)), st["bullet"]),
                    ]
                )
                item_number += 1
                i += 1
            list_table = Table(
                item_rows,
                colWidths=[7 * mm, avail_width - 7 * mm],
                hAlign="LEFT",
                splitByRow=1,
            )
            list_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                    ]
                )
            )
            story.append(list_table)
            story.append(Spacer(1, 2 * mm))
            continue
        if line.startswith("`") and line.endswith("`"):
            flush_paragraph()
            story.append(Paragraph(html.escape(line.strip("`")), st["code"]))
            i += 1
            continue
        paragraph.append(line)
        i += 1
    flush_paragraph()
    return story


def build():
    st = styles()
    doc = NumberedDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=17 * mm,
        title="DiffQRCoder - traduction expliquée en français",
        author="Traduction pédagogique produite pour le lecteur",
        subject="Explication accessible de l'article arXiv 2409.06355v3",
    )

    md = SOURCE.read_text(encoding="utf-8")
    marker = "## L'essentiel en deux minutes"
    body = marker + md.split(marker, 1)[1]

    cover_title = ParagraphStyle(
        "CoverTitle",
        fontName="Arial-Bold",
        fontSize=28,
        leading=32,
        textColor=NAVY,
        alignment=TA_LEFT,
    )
    cover_sub = ParagraphStyle(
        "CoverSub",
        fontName="Arial",
        fontSize=15,
        leading=20,
        textColor=BLUE,
        alignment=TA_LEFT,
    )
    cover_meta = ParagraphStyle(
        "CoverMeta",
        fontName="Arial",
        fontSize=9,
        leading=13,
        textColor=MUTED,
        alignment=TA_LEFT,
    )

    story = [
        Spacer(1, 22 * mm),
        Paragraph("DiffQRCoder", cover_title),
        Spacer(1, 3 * mm),
        Paragraph("Comment fabriquer un QR code artistique qui reste facile à scanner ?", cover_sub),
        Spacer(1, 16 * mm),
        PipelineDiagram(width=doc.width),
        Spacer(1, 16 * mm),
        Table(
            [[Paragraph("TRADUCTION EXPLIQUEE", st["h3"])],
             [Paragraph("Une lecture fidèle et accessible au niveau lycée", st["body"])]],
            colWidths=[doc.width],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F7FA")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#C9D8E4")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]),
        ),
        Spacer(1, 17 * mm),
        Paragraph(
            "Article original : Jia-Wei Liao et coll., arXiv:2409.06355v3, 15 février 2025. "
            "Cette version reformule les équations en langage courant et conserve les résultats, limites et précautions.",
            cover_meta,
        ),
        PageBreak(),
    ]
    story.extend(parse_markdown(body, st, doc.width))
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()

from __future__ import annotations

import io
from typing import List

try:
    from fpdf import FPDF
    PDF_EXPORT_AVAILABLE = True
except ImportError:
    PDF_EXPORT_AVAILABLE = False


def export_results_pdf(query: str, results: list) -> Optional[bytes]:
    if not PDF_EXPORT_AVAILABLE:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Document Section Finder", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "Semantic Search Results Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Query: {query}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Results found: {len(results)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    for i, r in enumerate(results, 1):
        pdf.set_fill_color(240, 240, 248)
        pdf.set_font("Helvetica", "B", 11)
        page_info = f" | Page {r.page_number}" if r.page_number else ""
        title = f"Result #{r.number}{page_info}"
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", fill=True)

        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, f"Source: {r.source} | Similarity: {r.similarity:.1%}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        pdf.set_font("Helvetica", "", 9)
        content = r.snippet_html if r.snippet_html else r.content[:500]
        content = content.replace("<mark>", "").replace("</mark>", "").replace("<br>", "\n")
        content = content[:600]

        pdf.multi_cell(0, 4.5, content)
        pdf.ln(3)

        if i < len(results):
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.getvalue()

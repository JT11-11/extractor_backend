"""Server-side PDF rendering using PyMuPDF (fitz).

Much simpler than the browser-based approach in src/lib/pdf.ts:
no Canvas Factory or native bindings required — PyMuPDF ships
pre-compiled wheels that work on Linux/macOS/Windows.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class RenderedPage:
    page_number: int
    image_base64: str


@dataclass
class RenderResult:
    page_count: int
    rendered_pages: list[RenderedPage]
    truncated: bool


def render_pdf_bytes_to_images(
    pdf_bytes: bytes,
    max_pages: int = 21,
    dpi: int = 150,
    jpeg_quality: int = 86,
) -> RenderResult:
    """Convert raw PDF bytes into a list of base64-encoded JPEG images.

    dpi=150 at A4 size gives roughly 1240x1754 px — comparable to the
    scale=1.8 used in the browser renderer.
    """
    doc: fitz.Document = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count
    page_count = min(total_pages, max_pages)
    rendered: list[RenderedPage] = []

    matrix = fitz.Matrix(dpi / 72, dpi / 72)  # 72 dpi is PDF's internal unit

    for page_num in range(page_count):
        page: fitz.Page = doc.load_page(page_num)
        pix: fitz.Pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        # Encode as JPEG directly to bytes
        jpeg_bytes: bytes = pix.tobytes("jpg")
        b64: str = base64.b64encode(jpeg_bytes).decode("utf-8")

        rendered.append(
            RenderedPage(page_number=page_num + 1, image_base64=b64)
        )

    doc.close()

    return RenderResult(
        page_count=total_pages,
        rendered_pages=rendered,
        truncated=total_pages > max_pages,
    )

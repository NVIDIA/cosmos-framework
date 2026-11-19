# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render PDF input for document benchmarks."""

from contextlib import closing

from vlmeval.smp import encode_image_to_base64


def encode_pdf_pages(pdf_data: bytes, max_pages: int) -> list[str]:
    """Render the first pages at 144 DPI and encode them for the image loader."""
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ImportError("PDF rendering requires pypdfium2. Install requirements-pdf.txt from VLMEvalKit.") from exc

    encoded_pages = []
    with closing(pdfium.PdfDocument(pdf_data)) as document:
        for index in range(min(len(document), max_pages)):
            with closing(document[index]) as page, closing(page.render(scale=2, draw_annots=True)) as bitmap:
                with bitmap.to_pil() as rendered, rendered.convert("RGB") as image:
                    encoded_pages.append(encode_image_to_base64(image))
    return encoded_pages

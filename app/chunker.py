from base64 import encode

import pymupdf
import tiktoken
from pydantic import BaseModel

ENCODING_NAME = "cl100k_base"


class ChunkData(BaseModel):
    """Represents a single tokenized text chunk extracted from a PDF page."""

    chunk_index: int
    page_number: int
    content: str
    token_count: int


def extract_and_chunk_pdf(
    pdf_bytes: bytes,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[ChunkData]:
    """Extract text page-by-page from raw PDF bytes and split into overlapping token chunks.

    Args:
        pdf_bytes: Raw binary content of a PDF file.
        max_tokens: Maximum number of `cl100k_base` tokens per chunk (must be > 0).
        overlap_tokens: Number of tokens shared between consecutive chunks on the same
            page (must satisfy 0 <= overlap_tokens < max_tokens).

    Returns:
        Ordered list of `ChunkData` with contiguous 0-based `chunk_index` across the
        document and 1-based `page_number` indicating the source PDF page.

    Raises:
        ValueError: If `max_tokens <= 0`, `overlap_tokens < 0`, or
            `overlap_tokens >= max_tokens`.
    """
    chunks: list[ChunkData] = []
    chunk_index = 0

    enc = tiktoken.get_encoding(ENCODING_NAME)
    step = max_tokens - overlap_tokens

    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Invalid window input arguments")
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for pg_no, pg in enumerate(doc.pages(), start=1):
            text = pg.get_text().strip()

            if not text:
                continue

            tokens = enc.encode(text)
            if not tokens:
                continue

            for start in range(0, len(tokens), step):
                window = tokens[start : start + max_tokens]

                content = enc.decode(window)

                chunks.append(
                    ChunkData(
                        chunk_index=chunk_index,
                        page_number=pg_no,
                        content=content,
                        token_count=len(window),
                    )
                )

                chunk_index += 1

                if start + max_tokens >= len(tokens):
                    break

    return chunks

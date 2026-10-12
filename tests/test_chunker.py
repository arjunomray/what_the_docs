import fitz
import pytest
import tiktoken

from app.chunker import ENCODING_NAME, ChunkData, extract_and_chunk_pdf


def _make_pdf(pages: list[str]) -> bytes:
    """Create an in-memory PDF with one text block per page."""
    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page()
        if page_text:
            rect = fitz.Rect(50, 50, 550, 750)
            page.insert_textbox(rect, page_text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_single_page_short_pdf() -> None:
    pdf_bytes = _make_pdf(
        ["PostgreSQL with pgvector enables fast hybrid semantic and lexical search."]
    )
    chunks = extract_and_chunk_pdf(pdf_bytes, max_tokens=400, overlap_tokens=50)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, ChunkData)
    assert chunk.chunk_index == 0
    assert chunk.page_number == 1
    assert "pgvector" in chunk.content
    assert 5 <= chunk.token_count <= 400


def test_sliding_window_token_overlap() -> None:
    enc = tiktoken.get_encoding(ENCODING_NAME)
    # Build a deterministic ~120-token page
    sentences = [
        f"Clause {i} defines strict distributed transaction invariants for shard {i}."
        for i in range(1, 13)
    ]
    pdf_bytes = _make_pdf([" ".join(sentences)])

    max_tokens = 40
    overlap_tokens = 10
    chunks = extract_and_chunk_pdf(
        pdf_bytes,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    assert len(chunks) >= 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.page_number == 1 for c in chunks)
    assert all(0 < c.token_count <= max_tokens for c in chunks)
    assert all(c.token_count == max_tokens for c in chunks[:-1])

    # Verify token overlap between chunk 0 and chunk 1
    tokens_0 = enc.encode(chunks[0].content)
    tokens_1 = enc.encode(chunks[1].content)
    assert tokens_0[-overlap_tokens:] == tokens_1[:overlap_tokens]


def test_multipage_pdf_skips_blank_pages_and_keeps_contiguous_indices() -> None:
    pdf_bytes = _make_pdf(
        [
            "Page one covers vector indexing with HNSW graphs.",
            "   \n\t  ",  # Blank page 2 should be skipped
            "Page three covers Reciprocal Rank Fusion in SQL CTE queries.",
        ]
    )

    chunks = extract_and_chunk_pdf(pdf_bytes, max_tokens=400, overlap_tokens=50)

    assert len(chunks) == 2
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert [c.page_number for c in chunks] == [1, 3]
    assert "Page one" in chunks[0].content
    assert "Page three" in chunks[1].content


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens"),
    [
        (0, 0),
        (-10, 5),
        (100, -1),
        (100, 100),
        (50, 75),
    ],
)
def test_invalid_window_parameters_raise_value_error(
    max_tokens: int,
    overlap_tokens: int,
) -> None:
    pdf_bytes = _make_pdf(["Sample text for parameter validation."])
    with pytest.raises(ValueError):
        extract_and_chunk_pdf(
            pdf_bytes,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

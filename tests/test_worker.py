import os
from uuid import uuid4

import fitz
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.cache import QueryCache
from app.llm import DeterministicFakeProvider
from app.models import Document, DocumentChunk, DocumentStatus
from app.worker import process_document_job


def _create_sample_pdf(filepath: str, pages: list[str]) -> None:
    """Helper to create a real PDF file on disk for worker testing."""
    doc = fitz.open()
    for text_content in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), text_content, fontsize=11)
    doc.save(filepath)
    doc.close()


async def test_worker_processes_pdf_and_creates_chunks(
    db_session: AsyncSession,
    fake_llm: DeterministicFakeProvider,
    query_cache: QueryCache,
    tmp_path,
) -> None:
    """Worker should parse PDF, chunk it, embed, save chunks, and mark document READY."""
    # 1. Prepare PDF file and PENDING document
    pdf_path = str(tmp_path / "sample.pdf")
    _create_sample_pdf(
        pdf_path,
        [
            "Chapter 1: PostgreSQL pgvector enables high-performance vector search.",
            "Chapter 2: Redis provides fast in-memory caching and job queuing.",
        ],
    )

    doc = Document(
        filename="sample.pdf",
        file_hash="hash_sample_123",
        status=DocumentStatus.PENDING,
    )
    db_session.add(doc)
    await db_session.commit()

    # Pre-populate cache to test that worker invalidates it upon completion
    await query_cache.set(query="sample query", answer="cached answer", citations=[])
    assert await query_cache.get(query="sample query") is not None

    # 2. Build worker ctx and execute job
    session_factory = async_sessionmaker(
        db_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    ctx = {
        "session_factory": session_factory,
        "llm": fake_llm,
        "cache": query_cache,
    }

    success = await process_document_job(ctx, doc.id, pdf_path)
    assert success is True

    # 3. Verify Document status transitioned to READY
    await db_session.refresh(doc)
    assert doc.status == DocumentStatus.READY
    assert doc.error_message is None

    # 4. Verify DocumentChunks were created in Postgres
    chunks_stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index)
    res = await db_session.execute(chunks_stmt)
    chunks = list(res.scalars().all())

    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_number == 1
    assert "pgvector" in chunks[0].content
    assert len(chunks[0].embedding) == 1536

    assert chunks[1].chunk_index == 1
    assert chunks[1].page_number == 2
    assert "Redis" in chunks[1].content

    # 5. Verify query cache was invalidated
    assert await query_cache.get(query="sample query") is None

    # 6. Verify temp PDF was removed after successful processing
    assert not os.path.exists(pdf_path)


async def test_worker_handles_missing_file_gracefully(
    db_session: AsyncSession,
    fake_llm: DeterministicFakeProvider,
    query_cache: QueryCache,
) -> None:
    """If file does not exist, worker should mark document FAILED with error message."""
    doc = Document(
        filename="missing.pdf",
        file_hash="hash_missing_123",
        status=DocumentStatus.PENDING,
    )
    db_session.add(doc)
    await db_session.commit()

    session_factory = async_sessionmaker(
        db_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    ctx = {
        "session_factory": session_factory,
        "llm": fake_llm,
        "cache": query_cache,
    }

    success = await process_document_job(ctx, doc.id, "/nonexistent/path/missing.pdf")
    assert success is False

    await db_session.refresh(doc)
    assert doc.status == DocumentStatus.FAILED
    assert doc.error_message is not None
    assert "No such file" in doc.error_message or "not found" in doc.error_message.lower()

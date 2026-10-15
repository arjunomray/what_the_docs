import hashlib
import io
from uuid import uuid4

import fitz
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import QueryCache
from app.llm import DeterministicFakeProvider
from app.main import app, get_cache, get_llm
from app.models import Document, DocumentChunk, DocumentStatus, get_session


def _make_test_pdf_bytes(text_lines: list[str]) -> bytes:
    doc = fitz.open()
    for line in text_lines:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), line, fontsize=11)
    b = doc.tobytes()
    doc.close()
    return b


@pytest.fixture
async def client(
    db_session: AsyncSession,
    fake_llm: DeterministicFakeProvider,
    query_cache: QueryCache,
) -> AsyncClient:
    """Async test client with dependencies overridden to test fixtures."""
    app.dependency_overrides[get_session] = lambda: db_session
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_cache] = lambda: query_cache

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


async def test_upload_pdf_creates_pending_document(client: AsyncClient, db_session: AsyncSession) -> None:
    pdf_bytes = _make_test_pdf_bytes(["Test document for upload."])
    files = {"file": ("architecture.pdf", io.BytesIO(pdf_bytes), "application/pdf")}

    response = await client.post("/api/documents", files=files)
    assert response.status_code == 202
    data = response.json()
    assert "id" in data
    assert data["filename"] == "architecture.pdf"
    assert data["status"] == "PENDING"


async def test_upload_duplicate_pdf_returns_existing(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    pdf_bytes = _make_test_pdf_bytes(["Duplicate check content."])
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    existing_doc = Document(
        filename="original.pdf",
        file_hash=file_hash,
        status=DocumentStatus.READY,
    )
    db_session.add(existing_doc)
    await db_session.commit()

    files = {"file": ("reupload.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    response = await client.post("/api/documents", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(existing_doc.id)
    assert data["deduplicated"] is True


async def test_get_document_status_and_not_found(client: AsyncClient, db_session: AsyncSession) -> None:
    doc = Document(
        filename="manual.pdf",
        file_hash="hash_abc_123",
        status=DocumentStatus.READY,
    )
    db_session.add(doc)
    await db_session.commit()

    # Hit existing
    resp = await client.get(f"/api/documents/{doc.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == str(doc.id)
    assert resp.json()["status"] == "READY"

    # Hit non-existent
    resp404 = await client.get(f"/api/documents/{uuid4()}")
    assert resp404.status_code == 404


async def test_list_and_delete_documents(
    client: AsyncClient,
    db_session: AsyncSession,
    query_cache: QueryCache,
) -> None:
    doc = Document(
        filename="to_delete.pdf",
        file_hash="hash_delete_123",
        status=DocumentStatus.READY,
    )
    db_session.add(doc)
    await db_session.commit()

    # Pre-populate cache to ensure delete invalidates it
    await query_cache.set(query="q_del", answer="a_del", citations=[])
    assert await query_cache.get(query="q_del") is not None

    # List documents
    resp = await client.get("/api/documents")
    assert resp.status_code == 200
    ids = [d["id"] for d in resp.json()]
    assert str(doc.id) in ids

    # Delete document
    del_resp = await client.delete(f"/api/documents/{doc.id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # Verify cache was invalidated
    assert await query_cache.get(query="q_del") is None

    # Verify document is gone
    get_resp = await client.get(f"/api/documents/{doc.id}")
    assert get_resp.status_code == 404


async def test_query_endpoint_cache_miss_then_hit(
    client: AsyncClient,
    db_session: AsyncSession,
    fake_llm: DeterministicFakeProvider,
) -> None:
    # Seed document and chunks
    doc = Document(
        filename="searchable.pdf",
        file_hash="searchable_hash",
        status=DocumentStatus.READY,
    )
    db_session.add(doc)
    await db_session.flush()

    chunk_text = "PostgreSQL pgvector provides HNSW indexes for cosine distance search."
    emb = (await fake_llm.embed_texts([chunk_text]))[0]
    db_session.add(
        DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content=chunk_text,
            token_count=12,
            embedding=emb,
        )
    )
    await db_session.commit()

    query_payload = {"query": "vector indexing in postgres", "top_k": 3}

    # 1. First query: Cache Miss
    resp1 = await client.post("/api/query", json=query_payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["cached"] is False
    assert len(data1["citations"]) > 0
    assert "pgvector" in data1["citations"][0]["content"]

    # 2. Second identical query: Cache Hit!
    resp2 = await client.post("/api/query", json=query_payload)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["cached"] is True
    assert data2["answer"] == data1["answer"]
    assert data2["citations"] == data1["citations"]

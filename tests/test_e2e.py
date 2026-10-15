import io
from uuid import UUID

import fitz
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.cache import QueryCache
from app.llm import DeterministicFakeProvider
from app.main import app, get_cache, get_llm
from app.models import get_session
from app.worker import process_document_job


def _make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=11)
    b = doc.tobytes()
    doc.close()
    return b


@pytest.fixture
async def e2e_client(
    db_session: AsyncSession,
    fake_llm: DeterministicFakeProvider,
    query_cache: QueryCache,
) -> AsyncClient:
    app.dependency_overrides[get_session] = lambda: db_session
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_cache] = lambda: query_cache

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


async def test_end_to_end_rag_lifecycle(
    e2e_client: AsyncClient,
    db_session: AsyncSession,
    fake_llm: DeterministicFakeProvider,
    query_cache: QueryCache,
    tmp_path,
) -> None:
    """Full user lifecycle:

    1. Upload multi-page PDF -> 202 Accepted (PENDING)
    2. Background worker consumes job -> transitions status to READY with chunks
    3. Document status endpoint reflects READY and chunk count
    4. First query: cache miss -> returns RRF retrieved citations + synthesized answer
    5. Second query: cache hit -> returned in <1ms from Redis
    6. Re-upload identical PDF -> returns existing document with deduplicated=True
    7. Delete document -> cascades chunks and purges query cache
    """
    # 1. Upload multi-page PDF
    pdf_bytes = _make_pdf(
        [
            "PostgreSQL pgvector extension implements HNSW graph indexes for cosine distance search.",
            "Reciprocal Rank Fusion combines vector rankings with GIN full text search.",
        ]
    )
    upload_res = await e2e_client.post(
        "/api/documents",
        files={"file": ("distributed_rag.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert upload_res.status_code == 202
    upload_data = upload_res.json()
    doc_id = UUID(upload_data["id"])
    assert upload_data["status"] == "PENDING"

    # 2. Simulate worker processing job
    temp_pdf = tmp_path / f"{doc_id}.pdf"
    temp_pdf.write_bytes(pdf_bytes)

    session_factory = async_sessionmaker(
        db_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    worker_ctx = {
        "session_factory": session_factory,
        "llm": fake_llm,
        "cache": query_cache,
    }
    worker_ok = await process_document_job(worker_ctx, doc_id, str(temp_pdf))
    assert worker_ok is True

    # 3. Check status is now READY with 2 chunks
    status_res = await e2e_client.get(f"/api/documents/{doc_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["status"] == "READY"
    assert status_data["chunks_count"] == 2

    # 4. First Query: Cache Miss
    q_payload = {"query": "how does pgvector index vectors?", "document_id": str(doc_id)}
    res1 = await e2e_client.post("/api/query", json=q_payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["cached"] is False
    assert len(data1["citations"]) > 0
    assert "pgvector" in data1["citations"][0]["content"]

    # 5. Second Query: Cache Hit
    res2 = await e2e_client.post("/api/query", json=q_payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cached"] is True
    assert data2["answer"] == data1["answer"]

    # 6. Deduplication on re-upload
    reup_res = await e2e_client.post(
        "/api/documents",
        files={"file": ("duplicate.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert reup_res.status_code == 200
    reup_data = reup_res.json()
    assert reup_data["id"] == str(doc_id)
    assert reup_data["deduplicated"] is True

    # 7. Delete document and verify cache invalidation
    del_res = await e2e_client.delete(f"/api/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # Cache should be invalidated
    cached_val = await query_cache.get(q_payload["query"], doc_id)
    assert cached_val is None

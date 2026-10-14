from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import DeterministicFakeProvider
from app.models import Document, DocumentChunk, DocumentStatus
from app.retrieval import SearchResult, hybrid_search


@pytest.fixture
async def seeded_documents(
    db_session: AsyncSession,
    fake_llm: DeterministicFakeProvider,
) -> tuple[Document, Document]:
    """Seed two documents with known chunks for retrieval testing."""
    doc_a = Document(
        filename="distributed_systems.pdf",
        file_hash="1" * 64,
        status=DocumentStatus.READY,
    )
    doc_b = Document(
        filename="web_frameworks.pdf",
        file_hash="2" * 64,
        status=DocumentStatus.READY,
    )
    db_session.add_all([doc_a, doc_b])
    await db_session.flush()

    chunks_a = [
        "PostgreSQL pgvector supports HNSW approximate nearest neighbor search over high dimensional embeddings.",
        "Redis cluster returns error code ERR_REDIS_CLUSTER_DOWN when quorum fails during failover.",
        "Apache Kafka partitions distributed streaming event logs across broker topics.",
    ]
    chunks_b = [
        "FastAPI leverages Python type hints and Pydantic models for automatic OpenAPI schema generation.",
        "PostgreSQL pgvector tuning guidelines recommend configuring m and ef_construction parameters.",
    ]

    embs_a = await fake_llm.embed_texts(chunks_a)
    for idx, (content, emb) in enumerate(zip(chunks_a, embs_a, strict=True)):
        db_session.add(
            DocumentChunk(
                document_id=doc_a.id,
                chunk_index=idx,
                page_number=1,
                content=content,
                token_count=15,
                embedding=emb,
            )
        )

    embs_b = await fake_llm.embed_texts(chunks_b)
    for idx, (content, emb) in enumerate(zip(chunks_b, embs_b, strict=True)):
        db_session.add(
            DocumentChunk(
                document_id=doc_b.id,
                chunk_index=idx,
                page_number=1,
                content=content,
                token_count=15,
                embedding=emb,
            )
        )

    await db_session.commit()
    return doc_a, doc_b


async def test_hybrid_search_semantic_query(
    db_session: AsyncSession,
    fake_llm: DeterministicFakeProvider,
    seeded_documents: tuple[Document, Document],
) -> None:
    """A semantic query should rank the closest vector chunk at the top."""
    query = "approximate nearest neighbor vector search"
    query_emb = (await fake_llm.embed_texts([query]))[0]

    results = await hybrid_search(
        session=db_session,
        query_text=query,
        query_embedding=query_emb,
        top_k=3,
    )

    assert len(results) > 0
    assert isinstance(results[0], SearchResult)
    # The top result should be the pgvector HNSW chunk
    assert "pgvector" in results[0].content
    assert results[0].rrf_score > 0.0


async def test_hybrid_search_exact_keyword_query(
    db_session: AsyncSession,
    fake_llm: DeterministicFakeProvider,
    seeded_documents: tuple[Document, Document],
) -> None:
    """An exact keyword query with a rare error code should rank that chunk first."""
    query = "ERR_REDIS_CLUSTER_DOWN"
    query_emb = (await fake_llm.embed_texts([query]))[0]

    results = await hybrid_search(
        session=db_session,
        query_text=query,
        query_embedding=query_emb,
        top_k=3,
    )

    assert len(results) > 0
    assert "ERR_REDIS_CLUSTER_DOWN" in results[0].content


async def test_hybrid_search_document_id_filter(
    db_session: AsyncSession,
    fake_llm: DeterministicFakeProvider,
    seeded_documents: tuple[Document, Document],
) -> None:
    """Filtering by document_id should only return chunks belonging to that document."""
    doc_a, doc_b = seeded_documents
    query = "pgvector"
    query_emb = (await fake_llm.embed_texts([query]))[0]

    # Search only within doc_b
    results = await hybrid_search(
        session=db_session,
        query_text=query,
        query_embedding=query_emb,
        top_k=5,
        document_id=doc_b.id,
    )

    assert len(results) > 0
    assert all(r.document_id == doc_b.id for r in results)
    # Even though doc_a has a pgvector chunk, it should not appear
    assert not any(r.document_id == doc_a.id for r in results)


async def test_hybrid_search_empty_when_no_match_for_filter(
    db_session: AsyncSession,
    fake_llm: DeterministicFakeProvider,
    seeded_documents: tuple[Document, Document],
) -> None:
    """Searching with a non-existent document_id should return an empty list."""
    query = "pgvector"
    query_emb = (await fake_llm.embed_texts([query]))[0]

    results = await hybrid_search(
        session=db_session,
        query_text=query,
        query_embedding=query_emb,
        top_k=5,
        document_id=uuid4(),
    )

    assert results == []

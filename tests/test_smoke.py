from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import EMBEDDING_DIM
from app.llm import DeterministicFakeProvider, LLMProvider
from app.models import Document, DocumentChunk, DocumentStatus


async def test_postgres_pgvector_and_indexes(
    db_session: AsyncSession,
    fake_llm: DeterministicFakeProvider,
) -> None:
    """Verify pgvector extension, HNSW + GIN indexes, and Document/DocumentChunk persistence."""
    assert isinstance(fake_llm, LLMProvider)

    # Check HNSW and GIN indexes exist in pg_indexes
    idx_result = await db_session.execute(
        text(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE tablename = 'document_chunks'"
        )
    )
    indexes = {row[0]: row[1] for row in idx_result.fetchall()}
    assert "ix_document_chunks_embedding_hnsw" in indexes
    assert "USING hnsw" in indexes["ix_document_chunks_embedding_hnsw"]
    assert "ix_document_chunks_search_vector_gin" in indexes
    assert "USING gin" in indexes["ix_document_chunks_search_vector_gin"]

    # Insert a Document and two DocumentChunks
    doc = Document(
        filename="architecture.pdf",
        file_hash="a" * 64,
        status=DocumentStatus.READY,
    )
    db_session.add(doc)
    await db_session.flush()

    texts = [
        "PostgreSQL pgvector supports HNSW approximate nearest neighbor search.",
        "Redis provides sub-millisecond caching and async job queues.",
    ]
    embeddings = await fake_llm.embed_texts(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == EMBEDDING_DIM

    for idx, (chunk_text, emb) in enumerate(zip(texts, embeddings, strict=True)):
        db_session.add(
            DocumentChunk(
                document_id=doc.id,
                chunk_index=idx,
                page_number=1,
                content=chunk_text,
                token_count=10,
                embedding=emb,
            )
        )
    await db_session.commit()

    # Query back using pgvector cosine distance (<=>) and generated TSVECTOR (@@)
    query_emb = (await fake_llm.embed_texts(["pgvector HNSW nearest neighbor"]))[0]
    vec_row = (
        await db_session.execute(
            text(
                "SELECT content, embedding <=> CAST(:qvec AS vector) AS dist "
                "FROM document_chunks ORDER BY dist ASC LIMIT 1"
            ),
            {"qvec": str(query_emb)},
        )
    ).first()
    assert vec_row is not None
    assert "pgvector" in vec_row[0]

    fts_row = (
        await db_session.execute(
            text(
                "SELECT content FROM document_chunks "
                "WHERE search_vector @@ websearch_to_tsquery('english', :q)"
            ),
            {"q": "Redis caching"},
        )
    ).first()
    assert fts_row is not None
    assert "Redis" in fts_row[0]


async def test_redis_connect_and_clean_reset(redis_client: Redis) -> None:
    """Verify Redis connection and clean database state."""
    assert await redis_client.ping() is True
    assert await redis_client.dbsize() == 0

    await redis_client.set("rag:smoke_test", "ok")
    assert await redis_client.get("rag:smoke_test") == "ok"

import asyncio
import time
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from app.cache import CachedQueryResponse, QueryCache



async def test_cache_miss_returns_none(query_cache: QueryCache) -> None:
    """Querying an empty cache should return None."""
    result = await query_cache.get(query="What is HNSW?")
    assert result is None


async def test_cache_set_and_get_hit_with_normalization(query_cache: QueryCache) -> None:
    """Should return cached response, and normalize case and whitespace in queries."""
    query = "  What is PostgreSQL pgvector?  "
    answer = "pgvector provides vector similarity search."
    citations = [{"chunk_id": str(uuid4()), "page_number": 1}]

    await query_cache.set(query=query, answer=answer, citations=citations)

    # 1. Exact hit
    hit1 = await query_cache.get(query=query)
    assert hit1 is not None
    assert isinstance(hit1, CachedQueryResponse)
    assert hit1.answer == answer
    assert hit1.citations == citations

    # 2. Normalized hit (different casing and trailing spaces)
    hit2 = await query_cache.get(query="what is postgresql pgvector?")
    assert hit2 is not None
    assert hit2.answer == answer


async def test_cache_document_id_scoping(query_cache: QueryCache) -> None:
    """The same question scoped to different document IDs should not collide."""
    query = "What is the timeout?"
    doc_1 = uuid4()
    doc_2 = uuid4()

    await query_cache.set(
        query=query,
        answer="Timeout is 30s in doc 1.",
        citations=[],
        document_id=doc_1,
    )
    await query_cache.set(
        query=query,
        answer="Timeout is 60s in doc 2.",
        citations=[],
        document_id=doc_2,
    )

    hit_doc1 = await query_cache.get(query=query, document_id=doc_1)
    hit_doc2 = await query_cache.get(query=query, document_id=doc_2)
    hit_global = await query_cache.get(query=query, document_id=None)

    assert hit_doc1 is not None and hit_doc1.answer == "Timeout is 30s in doc 1."
    assert hit_doc2 is not None and hit_doc2.answer == "Timeout is 60s in doc 2."
    assert hit_global is None


async def test_cache_ttl_expiration(query_cache: QueryCache) -> None:
    """Keys should expire after their TTL seconds."""
    query = "Temporary query"
    await query_cache.set(
        query=query,
        answer="Expiring soon",
        citations=[],
        ttl_seconds=1,
    )

    assert await query_cache.get(query=query) is not None

    # Wait for TTL to expire
    await asyncio.sleep(1.1)

    assert await query_cache.get(query=query) is None


async def test_cache_invalidation_does_not_flush_other_keys(
    query_cache: QueryCache,
    redis_client: Redis,
) -> None:
    """Invalidate all should clear RAG cache entries without wiping queue/other keys."""
    # Add RAG cache entries
    await query_cache.set(query="q1", answer="a1", citations=[])
    await query_cache.set(query="q2", answer="a2", citations=[])

    # Add a non-cache key (simulating an arq background job queue key)
    await redis_client.set("arq:queue:default", "active_job_id")

    # Invalidate query cache
    deleted_count = await query_cache.invalidate_all()
    assert deleted_count == 2

    # Query cache entries are gone
    assert await query_cache.get(query="q1") is None
    assert await query_cache.get(query="q2") is None

    # The queue key is still safe and intact!
    assert await redis_client.get("arq:queue:default") == "active_job_id"

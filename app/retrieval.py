from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

RRF_QUERY = """
WITH vector_search AS (
    SELECT
        id,
        ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:query_vec AS vector)) as rank
    FROM document_chunks
    WHERE (CAST(:doc_id AS uuid) IS NULL OR document_id = CAST(:doc_id AS uuid))
    ORDER BY embedding <=> CAST(:query_vec AS vector)
    LIMIT 20
),
fts_search AS (
    SELECT
        id,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', :query_text)) DESC) as rank
    FROM document_chunks
    WHERE search_vector @@ websearch_to_tsquery('english', :query_text)
      AND (CAST(:doc_id AS uuid) IS NULL OR document_id = CAST(:doc_id AS uuid))
    ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', :query_text)) DESC
    LIMIT 20
)
SELECT
    c.id,
    c.document_id,
    c.chunk_index,
    c.page_number,
    c.content,
    COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + f.rank), 0.0) AS rrf_score
FROM vector_search v
FULL OUTER JOIN fts_search f ON v.id = f.id
JOIN document_chunks c ON c.id = COALESCE(v.id, f.id)
ORDER BY rrf_score DESC
LIMIT :top_k;
"""


class SearchResult(BaseModel):
    id: UUID
    document_id: UUID
    chunk_index: int
    page_number: int
    content: str
    rrf_score: float


async def hybrid_search(
    session: AsyncSession,
    query_text: str,
    query_embedding: list[float],
    top_k: int = 5,
    document_id: UUID | None = None,
) -> list[SearchResult]:
    result = await session.execute(
        text(RRF_QUERY),
        {
            "query_vec": str(query_embedding),
            "query_text": query_text,
            "doc_id": document_id,
            "top_k": top_k,
        },
    )
    rows = result.fetchall()
    return [
        SearchResult(
            id=row[0],
            document_id=row[1],
            chunk_index=row[2],
            page_number=row[3],
            content=row[4],
            rrf_score=row[5],
        )
        for row in rows
    ]

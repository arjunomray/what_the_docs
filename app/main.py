import hashlib
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID, uuid4

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col, select

from app.cache import QueryCache
from app.config import settings
from app.llm import DeterministicFakeProvider, LLMProvider, OpenAIProvider
from app.models import (
    Document,
    DocumentChunk,
    DocumentStatus,
    get_session,
    init_db,
)
from app.retrieval import hybrid_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Ensure DB extension & tables exist
    await init_db()

    # 2. Setup Redis and Query Cache
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    app.state.redis = redis
    app.state.cache = QueryCache(redis=redis)
    if settings.openai_api_key:
        app.state.llm = OpenAIProvider()
    else:
        app.state.llm = DeterministicFakeProvider()

    # 3. Setup arq Redis pool for enqueueing background jobs
    try:
        app.state.arq_pool = await create_pool(
            RedisSettings.from_dsn(settings.redis_url)
        )
    except Exception:
        app.state.arq_pool = None

    yield

    # Cleanup on shutdown (guards against None for static type checkers)
    arq_pool = getattr(app.state, "arq_pool", None)
    if arq_pool is not None:
        await arq_pool.aclose()

    redis_conn = getattr(app.state, "redis", None)
    if redis_conn is not None:
        await redis_conn.aclose()


app = FastAPI(title="What The Docs", lifespan=lifespan)
def get_llm(request: Request) -> LLMProvider:
    llm = getattr(request.app.state, "llm", None)
    if llm is not None:
        return llm
    if settings.openai_api_key:
        return OpenAIProvider()
    return DeterministicFakeProvider()

def get_cache(request: Request) -> QueryCache | None:
    return getattr(request.app.state, "cache", None)


async def get_arq(request: Request) -> Any:
    return getattr(request.app.state, "arq_pool", None)


# --- Request / Response Schemas ---

class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    status: DocumentStatus
    chunks_count: int = 0
    error_message: str | None = None
    created_at: datetime
    deduplicated: bool = False


class QueryRequest(BaseModel):
    query: str
    document_id: UUID | None = None
    top_k: int = 5


class Citation(BaseModel):
    chunk_id: str
    page_number: int
    content: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    cached: bool = False


# --- Endpoints ---

@app.post(
    "/api/documents",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentResponse,
)
async def upload_document(
    response: Response,
    file: Annotated[UploadFile, File(...)],
    session: Annotated[AsyncSession, Depends(get_session)],
    arq_pool: Annotated[Any, Depends(get_arq)],
):
    pdf_bytes = await file.read()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Check for deduplication (same content already processed)
    stmt = select(Document).where(col(Document.file_hash) == file_hash)
    res = await session.execute(stmt)
    existing = res.scalar_one_or_none()

    if existing and existing.status == DocumentStatus.READY:
        count_stmt = select(func.count(col(DocumentChunk.id))).where(
            col(DocumentChunk.document_id) == existing.id
        )
        chunks_count = (await session.execute(count_stmt)).scalar() or 0
        response.status_code = status.HTTP_200_OK
        return DocumentResponse(
            id=existing.id,
            filename=existing.filename,
            status=existing.status,
            chunks_count=int(chunks_count),
            created_at=existing.created_at,
            deduplicated=True,
        )

    # Save to disk for background worker
    os.makedirs(settings.upload_dir, exist_ok=True)
    doc_id = uuid4()
    saved_path = os.path.join(settings.upload_dir, f"{doc_id}_{file.filename}")
    with open(saved_path, "wb") as f:
        f.write(pdf_bytes)

    # Insert PENDING document
    doc = Document(
        id=doc_id,
        filename=file.filename or "upload.pdf",
        file_hash=file_hash,
        status=DocumentStatus.PENDING,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)

    # Enqueue background processing job
    if arq_pool:
        await arq_pool.enqueue_job("process_document_job", doc.id, saved_path)

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,
        chunks_count=0,
        created_at=doc.created_at,
        deduplicated=False,
    )


@app.get("/api/documents/{id}", response_model=DocumentResponse)
async def get_document(
    id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
):
    stmt = select(Document).where(col(Document.id) == id)
    res = await session.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    count_stmt = select(func.count(col(DocumentChunk.id))).where(
        col(DocumentChunk.document_id) == doc.id
    )
    chunks_count = (await session.execute(count_stmt)).scalar() or 0

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        status=doc.status,
        chunks_count=int(chunks_count),
        error_message=doc.error_message,
        created_at=doc.created_at,
    )


@app.get("/api/documents", response_model=list[DocumentResponse])
async def list_documents(
    session: Annotated[AsyncSession, Depends(get_session)],
):
    stmt = (
        select(Document, func.count(col(DocumentChunk.id)).label("chunks_count"))
        .outerjoin(DocumentChunk, col(Document.id) == col(DocumentChunk.document_id))
        .group_by(col(Document.id))
        .order_by(col(Document.created_at).desc())
    )
    res = await session.execute(stmt)
    rows = res.all()
    return [
        DocumentResponse(
            id=row[0].id,
            filename=row[0].filename,
            status=row[0].status,
            chunks_count=int(row[1]),
            error_message=row[0].error_message,
            created_at=row[0].created_at,
        )
        for row in rows
    ]


@app.delete("/api/documents/{id}")
async def delete_document(
    id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    cache: Annotated[QueryCache | None, Depends(get_cache)],
):
    stmt = select(Document).where(col(Document.id) == id)
    res = await session.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    await session.delete(doc)
    await session.commit()

    if cache:
        await cache.invalidate_all()

    return {"deleted": True, "id": str(id)}


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    llm: Annotated[LLMProvider, Depends(get_llm)],
    cache: Annotated[QueryCache | None, Depends(get_cache)],
):
    # 1. Check Redis Cache
    if cache:
        cached_hit = await cache.get(payload.query, payload.document_id)
        if cached_hit:
            return QueryResponse(
                answer=cached_hit.answer,
                citations=[Citation(**c) for c in cached_hit.citations],
                cached=True,
            )

    # 2. Embed query
    query_embeddings = await llm.embed_texts([payload.query])
    query_vec = query_embeddings[0]

    # 3. Hybrid RRF Retrieval (HNSW + GIN)
    search_hits = await hybrid_search(
        session=session,
        query_text=payload.query,
        query_embedding=query_vec,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )

    # 4. Synthesize Answer
    context_chunks = [h.content for h in search_hits]
    answer = await llm.generate_answer(payload.query, context_chunks)

    # 5. Format citations
    citations = [
        Citation(
            chunk_id=str(h.id),
            page_number=h.page_number,
            content=h.content,
            score=h.rrf_score,
        )
        for h in search_hits
    ]

    # 6. Save to Redis Cache
    if cache:
        await cache.set(
            query=payload.query,
            answer=answer,
            citations=[c.model_dump() for c in citations],
            document_id=payload.document_id,
        )

    return QueryResponse(answer=answer, citations=citations, cached=False)

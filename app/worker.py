from datetime import UTC, datetime
import os
from typing import Any
from uuid import UUID

from arq.connections import RedisSettings
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import select

from app.cache import QueryCache
from app.chunker import extract_and_chunk_pdf
from app.config import DATABASE_URL, settings
from app.llm import DeterministicFakeProvider, LLMProvider, OpenAIProvider
from app.models import (
    Document,
    DocumentChunk,
    DocumentStatus,
    async_session_factory,
    create_db_engine,
)


async def process_document_job(
    ctx: dict[str, Any], document_id: UUID, file_path: str
) -> bool:
    session_factory: async_sessionmaker[AsyncSession] = ctx.get("session_factory", async_session_factory)
    llm: LLMProvider = ctx.get("llm") or DeterministicFakeProvider()
    cache: QueryCache | None = ctx.get("cache")
    async with session_factory() as session:
        stmt = select(Document).where(Document.id == document_id)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()
        if not doc:
            return False
        doc.status = DocumentStatus.PROCESSING
        doc.updated_at = datetime.now(UTC)
        await session.commit()

        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"No such file exists: {file_path}")
            with open(file_path, "rb") as f:
               pdf_bytes = f.read()
            chunks = extract_and_chunk_pdf(pdf_bytes)
            if not chunks:
                raise ValueError("No extractable data in the pdf")
            texts = [c.content for c in chunks]
            embeddings = await llm.embed_texts(texts)

            for chunk_data, embed in zip(chunks, embeddings, strict=True):
                chunk_row = DocumentChunk(
                    document_id=doc.id,
                    page_number=chunk_data.page_number,
                    chunk_index=chunk_data.chunk_index,
                    content=chunk_data.content,
                    token_count=chunk_data.token_count,
                    embedding=embed
                )
                session.add(chunk_row)
            doc.status = DocumentStatus.READY
            doc.error_message = None
            doc.updated_at = datetime.now(UTC)
            await session.commit()

            if cache:
                await cache.invalidate_all()
            if os.path.exists(file_path):
                os.remove(file_path)

            return True

        except Exception as exc:
            await session.rollback()
            doc.status = DocumentStatus.FAILED
            doc.error_message = str(exc)
            doc.updated_at = datetime.now(UTC)
            await session.commit()

            if os.path.exists(file_path):
                os.remove(file_path)

            return False


# --- arq Worker Settings & Hooks ---


async def startup(ctx: dict[str, Any]) -> None:
    engine = create_db_engine(DATABASE_URL)
    ctx["session_factory"] = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    if settings.openai_api_key:
        ctx["llm"] = OpenAIProvider()
    else:
        ctx["llm"] = DeterministicFakeProvider()

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    ctx["redis"] = redis
    ctx["cache"] = QueryCache(redis=redis)


async def shutdown(ctx: dict[str, Any]) -> None:
    redis: Redis | None = ctx.get("redis")
    if redis is not None:
        await redis.aclose()


class WorkerSettings:
    functions = [process_document_job]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    on_startup = startup
    on_shutdown = shutdown

# Alias so both `WorkerSettings` (PascalCase) and `worker_settings` (snake_case) work with arq CLI
worker_settings = WorkerSettings

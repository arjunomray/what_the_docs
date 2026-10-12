from collections.abc import AsyncGenerator

import pytest_asyncio
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import DATABASE_URL, REDIS_URL
from app.llm import DeterministicFakeProvider
from app.models import init_db


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a clean PostgreSQL AsyncSession with pgvector initialized and tables truncated."""
    test_engine = create_async_engine(DATABASE_URL, echo=False, future=True)
    await init_db(test_engine)

    async with test_engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE TABLE document_chunks, documents RESTART IDENTITY CASCADE")
        )

    session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_maker() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE TABLE document_chunks, documents RESTART IDENTITY CASCADE")
        )
    await test_engine.dispose()


@pytest_asyncio.fixture
async def session(db_session: AsyncSession) -> AsyncSession:
    """Alias for db_session so tests can inject either `session` or `db_session`."""
    return db_session


@pytest_asyncio.fixture
async def redis_client() -> AsyncGenerator[Redis, None]:
    """Provide a clean async Redis client flushed before and after each test."""
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


@pytest_asyncio.fixture
def fake_llm() -> DeterministicFakeProvider:
    """Provide an offline DeterministicFakeProvider instance."""
    return DeterministicFakeProvider()

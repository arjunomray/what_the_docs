from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from enum import Enum
from typing import ClassVar
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Computed, DateTime, Index, text
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import Field, Relationship, SQLModel

from app.config import DATABASE_URL, EMBEDDING_DIM


class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class Document(SQLModel, table=True):
    __tablename__: ClassVar[str] = "documents"  # type: ignore[assignment]

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    filename: str = Field(nullable=False)
    file_hash: str = Field(index=True, unique=True, nullable=False, max_length=64)
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, nullable=False)
    error_message: str | None = Field(default=None, nullable=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    chunks: list["DocumentChunk"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class DocumentChunk(SQLModel, table=True):
    __tablename__: ClassVar[str] = "document_chunks"  # type: ignore[assignment]
    __table_args__ = (
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_document_chunks_search_vector_gin",
            "search_vector",
            postgresql_using="gin",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    document_id: UUID = Field(
        foreign_key="documents.id",
        ondelete="CASCADE",
        index=True,
        nullable=False,
    )
    chunk_index: int = Field(nullable=False)
    page_number: int = Field(default=1, nullable=False)
    content: str = Field(nullable=False)
    token_count: int = Field(default=0, nullable=False)
    embedding: list[float] = Field(
        sa_column=Column(Vector(EMBEDDING_DIM), nullable=False)
    )
    search_vector: str | None = Field(
        default=None,
        sa_column=Column(
            TSVECTOR,
            Computed("to_tsvector('english', content)", persisted=True),
        ),
    )

    document: Document | None = Relationship(back_populates="chunks")


def create_db_engine(database_url: str = DATABASE_URL) -> AsyncEngine:
    return create_async_engine(database_url, echo=False, future=True)


engine: AsyncEngine = create_db_engine()
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db(db_engine: AsyncEngine = engine) -> None:
    """Ensure pgvector extension exists and create all SQLModel tables."""
    async with db_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session

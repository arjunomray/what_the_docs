import hashlib
import time
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from redis.asyncio import Redis


class CachedQueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    cached_at: float = Field(default_factory=time.time)


class QueryCache:

    def __init__(
        self,
        redis: Redis,
        default_ttl_seconds: int = 3600,
        key_prefix: str = "rag:query:",
    ) -> None:
        self.redis = redis
        self.default_ttl_seconds = default_ttl_seconds
        self.key_prefix = key_prefix

    def _make_key(self, query: str, document_id: UUID | None = None):
        raw = f"{query.lower().strip()}:{str(document_id) if document_id else "all"}"
        hashed = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}{hashed}"

    async def set(
        self,
        query: str,
        answer: str,
        citations,
        ttl_seconds: int | None = None,
        document_id: UUID | None = None,
    ):
        key = self._make_key(query, document_id)
        payload = CachedQueryResponse(answer=answer, citations=citations)
        ttl = ttl_seconds if ttl_seconds else self.default_ttl_seconds
        await self.redis.set(key, payload.model_dump_json(), ex=ttl)

    async def get(
        self, query: str, document_id: UUID | None = None
    ) -> CachedQueryResponse | None:
        key = self._make_key(query, document_id)
        raw = await self.redis.get(key)
        if raw:
            return CachedQueryResponse.model_validate_json(raw)
        return None

    async def invalidate_all(self):
        keys = [k async for k in self.redis.scan_iter(match=f"{self.key_prefix}*")]
        if keys:
            return await self.redis.delete(*keys)
        return 0

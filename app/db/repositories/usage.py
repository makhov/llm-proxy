from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Integer, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import UsageRecord


async def record_usage(
    db: AsyncSession,
    *,
    user_id: str,
    team_id: str | None,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    request_id: str,
    cost_usd: float = 0.0,
    cache_hit: bool = False,
    was_rag_used: bool = False,
    pii_entities_scrubbed: int = 0,
    status: str = "success",
    error_code: str | None = None,
) -> UsageRecord:
    record = UsageRecord(
        id=str(uuid.uuid4()),
        user_id=user_id,
        team_id=team_id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
        request_id=request_id,
        cost_usd=cost_usd,
        cache_hit=cache_hit,
        was_rag_used=was_rag_used,
        pii_entities_scrubbed=pii_entities_scrubbed,
        status=status,
        error_code=error_code,
    )
    db.add(record)
    await db.commit()
    return record


async def get_usage_summary(
    db: AsyncSession,
    user_id: str | None = None,
    team_id: str | None = None,
    since: datetime | None = None,
) -> dict:
    q = select(
        UsageRecord.model,
        func.sum(UsageRecord.prompt_tokens).label("prompt_tokens"),
        func.sum(UsageRecord.completion_tokens).label("completion_tokens"),
        func.sum(UsageRecord.total_tokens).label("total_tokens"),
        func.sum(UsageRecord.cost_usd).label("cost_usd"),
        func.count(UsageRecord.id).label("requests"),
        func.sum(UsageRecord.cache_hit.cast(Integer)).label("cache_hits"),
    ).group_by(UsageRecord.model)

    if user_id:
        q = q.where(UsageRecord.user_id == user_id)
    if team_id:
        q = q.where(UsageRecord.team_id == team_id)
    if since:
        q = q.where(UsageRecord.created_at >= since)

    result = await db.execute(q)
    rows = result.all()
    return {
        "rows": [
            {
                "model": r.model,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "cost_usd": round(r.cost_usd or 0.0, 6),
                "requests": r.requests,
                "cache_hits": r.cache_hits or 0,
            }
            for r in rows
        ]
    }

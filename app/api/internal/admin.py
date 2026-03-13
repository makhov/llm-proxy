"""Admin endpoints for usage reporting and user/key management."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import require_admin
from app.db.engine import get_db
from app.db.repositories.usage import get_usage_summary
from app.db.repositories.users import create_api_key, create_team, create_user

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/usage")
async def usage_report(
    user_id: str | None = None,
    team_id: str | None = None,
    since: datetime | None = None,
    db: AsyncSession = Depends(get_db),
):
    return await get_usage_summary(db, user_id=user_id, team_id=team_id, since=since)


@router.post("/teams")
async def create_team_endpoint(
    name: str,
    tpm_limit: int = 500_000,
    daily_token_limit: int = 5_000_000,
    db: AsyncSession = Depends(get_db),
):
    team = await create_team(
        db, name=name, tpm_limit=tpm_limit, daily_token_limit=daily_token_limit
    )
    return {"id": team.id, "name": team.name}


@router.post("/users")
async def create_user_endpoint(
    external_id: str,
    team_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    user = await create_user(db, external_id=external_id, team_id=team_id)
    return {"id": user.id, "external_id": user.external_id}


@router.post("/api-keys")
async def create_api_key_endpoint(
    user_id: str,
    name: str = "default",
    db: AsyncSession = Depends(get_db),
):
    raw_key, api_key = await create_api_key(db, user_id=user_id, name=name)
    return {
        "key": raw_key,  # shown once
        "key_prefix": api_key.key_prefix,
        "id": api_key.id,
    }

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models(settings: Settings = Depends(get_settings)):
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "proxy"}
            for m in settings.allowed_models
        ],
    }

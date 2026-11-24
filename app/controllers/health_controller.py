from datetime import datetime, timezone

from fastapi import APIRouter
from app.schemas.api import HealthResponse

router = APIRouter(
    prefix="/health",
    tags=["health"]
)


@router.get("", response_model=HealthResponse, summary="Health check end-point")
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok/healthy",
        timestamp=datetime.now(timezone.utc),
    )

"""FastAPI service: top-k movie recommendations from the best trained model.

Run locally:
    uvicorn recsys.api:app --host 0.0.0.0 --port 8000

Configuration (environment variables):
    RECSYS_ARTIFACTS_DIR  directory with ``models/`` and ``results.json`` (default ``artifacts``)
    RECSYS_DATA_DIR       processed MovieLens tables (default ``data/processed``)
    RECSYS_MODEL          serve this model instead of the best one by validation NDCG@10
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

from recsys import __version__
from recsys.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR
from recsys.serving import RecommendationService

MAX_K = 100


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model: str = Field(description="Model serving known users", examples=["sasrec"])
    n_users: int
    n_items: int
    version: str


class MovieRecommendation(BaseModel):
    rank: int = Field(ge=1)
    movie_id: int = Field(description="MovieLens movie id")
    title: str
    score: float = Field(description="Model score (higher is better; not a probability)")


class RecommendationResponse(BaseModel):
    user_id: int = Field(description="MovieLens user id")
    model: str = Field(description="Model that produced the list")
    fallback: bool = Field(description="True if the user is unknown and popular items are served")
    recommendations: list[MovieRecommendation]


def create_app(service: RecommendationService | None = None) -> FastAPI:
    """Build the app. If ``service`` is ``None`` it is loaded from disk at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or RecommendationService.from_artifacts(
            artifacts_dir=Path(os.getenv("RECSYS_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)),
            data_dir=Path(os.getenv("RECSYS_DATA_DIR", DEFAULT_DATA_DIR)),
            model_name=os.getenv("RECSYS_MODEL") or None,
        )
        yield

    app = FastAPI(
        title="Movie Recommender",
        description="Top-k MovieLens recommendations from PyTorch recommender models.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        svc: RecommendationService = request.app.state.service
        return HealthResponse(
            status="ok",
            model=svc.model_name,
            n_users=svc.n_users,
            n_items=svc.n_items,
            version=__version__,
        )

    @app.get("/recommend/{user_id}", response_model=RecommendationResponse)
    def recommend(
        request: Request,
        user_id: Annotated[int, PathParam(description="MovieLens user id")],
        k: Annotated[int, Query(ge=1, le=MAX_K, description="Number of movies")] = 10,
    ) -> RecommendationResponse:
        svc: RecommendationService = request.app.state.service
        result = svc.recommend(user_id, k=k)
        return RecommendationResponse(
            user_id=result.user_id,
            model=result.model,
            fallback=result.fallback,
            recommendations=[
                MovieRecommendation(rank=r.rank, movie_id=r.movie_id, title=r.title, score=r.score)
                for r in result.items
            ],
        )

    return app


app = create_app()

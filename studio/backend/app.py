"""FastAPI application entry-point for Legoizer."""
from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from starlette.exceptions import HTTPException as StarletteHTTPException
from studio.backend.i18n import error_reference
from fastapi.staticfiles import StaticFiles

from studio.backend.api.routes import router
from studio.backend.benchmark import router as benchmark_router
from studio.backend.config import STUDIO_ROOT
from studio.backend.registry.registry import get_catalog

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the shared model dependencies before the browser requests the
    # catalog, templates, and environment concurrently.
    get_catalog()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Legoizer API",
    version="1.0.0",
    description="Visual programming for modular recommendation models.",
)

@app.exception_handler(StarletteHTTPException)
async def localized_http_error(_request, exc):
    return JSONResponse(status_code=exc.status_code, headers=exc.headers,
                        content={"detail": jsonable_encoder(exc.detail), "translation": error_reference(exc.detail)})


app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.environ.get("STUDIO_CORS_ORIGINS", "").split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(benchmark_router)

DIST = STUDIO_ROOT / "frontend" / "dist"
if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str):
        candidate = (DIST / full_path).resolve()
        if candidate.is_relative_to(DIST.resolve()) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def index_fallback():
        return {
            "service": "Legoizer API",
            "frontend": "not built — run `npm install && npm run build` in studio/frontend, or use the Vite dev server",
            "docs": "/docs",
        }

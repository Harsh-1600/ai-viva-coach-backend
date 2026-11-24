import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Depends, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import get_settings
from app.database import init_db, get_db
from app.schemas import HealthCheckResponse

from app.routers import auth, interviews, reports, assistant, teacher, tasks

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize database and resources on startup."""
    init_db()
    if settings.JWT_SECRET_KEY == "dev-secret-key-change-in-production-viva-coach":
        logger.warning(
            "SECURITY WARNING: Running with default dev JWT_SECRET_KEY. "
            "Set a cryptographically strong secret in the JWT_SECRET_KEY environment variable for production."
        )
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for AI-powered mock viva and technical interview preparation.",
    lifespan=lifespan
)

# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Configure Cross-Origin Resource Sharing (CORS)
cors_origins = list(settings.CORS_ORIGINS)
if settings.FRONTEND_URL:
    clean_url = settings.FRONTEND_URL.strip().rstrip("/")
    if clean_url not in cors_origins:
        cors_origins.append(clean_url)
    if not clean_url.startswith("http://") and not clean_url.startswith("https://"):
        cors_origins.append(f"https://{clean_url}")
        cors_origins.append(f"http://{clean_url}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"^(https?|capacitor|ionic)://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
)

# Include Routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(interviews.router, prefix=settings.API_V1_STR)
app.include_router(reports.router, prefix=settings.API_V1_STR)
app.include_router(assistant.router, prefix=settings.API_V1_STR)
app.include_router(teacher.router, prefix=settings.API_V1_STR)
app.include_router(tasks.router, prefix=settings.API_V1_STR)

# Uploaded user static files directory (/uploads)
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
AVATARS_DIR = UPLOAD_DIR / "avatars"
SUBMISSIONS_DIR = UPLOAD_DIR / "submissions"
AVATARS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Frontend static distribution path
FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

# Mount compiled frontend static assets (/assets)
if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


@app.get("/", tags=["Root"])
def read_root(request: Request):
    """Root endpoint welcoming API clients with JSON or serving the web app in browsers."""
    accept = request.headers.get("accept", "")
    dist_index = FRONTEND_DIST / "index.html"
    if "text/html" in accept and dist_index.exists():
        return FileResponse(dist_index)
    return {
        "message": "Welcome to AI Viva & Interview Preparation Coach API",
        "docs_url": "/docs",
        "health_check": f"{settings.API_V1_STR}/health"
    }


@app.get(
    f"{settings.API_V1_STR}/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    tags=["System"]
)
def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint to verify backend service status and database connectivity.
    Returns:
        HealthCheckResponse: Status, service name, version, database status, and timestamp.
    """
    db_status = "connected"
    try:
        # Perform a lightweight ping query on SQLite
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"

    return HealthCheckResponse(
        status="healthy" if db_status == "connected" else "degraded",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        database=db_status,
        timestamp=datetime.now(timezone.utc)
    )


@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa_fallback(full_path: str):
    """Fallback route serving static files or index.html for React SPA client-side routing."""
    # Prevent intercepting docs, OpenAPI, or API endpoints
    if full_path.startswith("api") or full_path.startswith("uploads") or full_path in ("docs", "redoc", "openapi.json"):
        raise HTTPException(status_code=404, detail="Not Found")

    target_file = FRONTEND_DIST / full_path
    if target_file.is_file():
        return FileResponse(target_file)

    dist_index = FRONTEND_DIST / "index.html"
    if dist_index.exists():
        return FileResponse(dist_index)

    raise HTTPException(status_code=404, detail="Frontend build index.html not found.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    import os
    import uvicorn
    from app.config import get_settings

    settings = get_settings()
    host = os.getenv("HOST") or os.getenv("BACKEND_HOST") or settings.BACKEND_HOST
    port = int(os.getenv("PORT") or os.getenv("BACKEND_PORT") or settings.effective_port)
    reload = os.getenv("ENVIRONMENT", "development").lower() == "development" and os.getenv("RELOAD", "false").lower() == "true"

    print("\n=======================================================")
    print(" Starting AI Viva & Interview Preparation Coach Backend ")
    print(f" URL: http://{host}:{port}")
    print(f" Swagger Docs: http://{host}:{port}/docs")
    print(f" Health Check: http://{host}:{port}/api/health")
    print("=======================================================\n")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)

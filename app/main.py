import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import router
from app.core.config import RuntimeConfigManager, build_defaults_from_env

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("reranker-service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Reranker Service...")

    runtime_path = os.getenv("CONFIG_RUNTIME_PATH", "config_runtime.json")
    defaults = build_defaults_from_env()
    config_manager = RuntimeConfigManager(runtime_path=runtime_path, defaults=defaults)

    # Apply logging level from config (best-effort)
    try:
        logging.getLogger().setLevel(config_manager.current_dict()["logging"]["level"])
    except Exception:
        pass

    app.state.runtime_config_manager = config_manager
    
    logger.info(f"Runtime config initialized (path={runtime_path})")
    yield
    
    logger.info("Shutting down Reranker Service...")

app = FastAPI(
    title="Reranker Service",
    description="Reranks retrieved items using LLM-based intent analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

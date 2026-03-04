import logging
import os
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Request, Depends

from app.api.schemas import (
    RerankRequest, RerankResponse, HealthResponse, ConfigGetResponse, MetricsResponse
)
from app.llm.client import llm_client
from app.core.dependencies import get_config_manager
from app.observability.metrics import metrics

router = APIRouter()
logger = logging.getLogger("reranker-api")

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint."""
    mgr = get_config_manager(request.app)
    cfg = mgr.current_dict()
    return HealthResponse(
        status="healthy",
        service="reranker-service",
        llm_configured=True,
        llm_base_url=cfg.get("llm", {}).get("base_url"),
        llm_default_model=cfg.get("llm", {}).get("model"),
    )

@router.get("/v1/config", response_model=ConfigGetResponse)
async def get_config(request: Request):
    """Get current runtime configuration and defaults."""
    mgr = get_config_manager(request.app)
    mgr.reload()
    return ConfigGetResponse(
        current=mgr.current_dict(),
        defaults=mgr.defaults_dict(),
        runtime_path=mgr.runtime_path,
        has_overrides_file=os.path.exists(mgr.runtime_path),
    )

@router.put("/v1/config")
async def update_config(request: Request, patch: Dict[str, Any]):
    """Apply a partial configuration update."""
    mgr = get_config_manager(request.app)
    try:
        updated = mgr.update(patch)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Apply logging level immediately (best-effort).
    try:
        logging.getLogger().setLevel(updated.logging.level)
    except Exception:
        pass

    return {"current": updated.model_dump()}

@router.post("/v1/config/reset")
async def reset_config(request: Request):
    """Reset configuration to defaults (remove runtime overrides file)."""
    mgr = get_config_manager(request.app)
    mgr.reset()
    return {"ok": True, "current": mgr.current_dict()}

@router.get("/v1/config/schema")
async def get_config_schema(request: Request):
    """Return JSON schema for the runtime config (for UI generation)."""
    mgr = get_config_manager(request.app)
    return {"schema": mgr.schema()}


@router.get("/v1/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Lightweight reranker observability counters."""
    return MetricsResponse(metrics=metrics.snapshot())

@router.post("/v1/rerank", response_model=RerankResponse)
async def rerank_items(request: Request, req: RerankRequest):
    """
    Rerank a list of candidates based on query and intent.
    """
    try:
        # If no candidates, return empty
        if not req.candidates:
            return RerankResponse(results=[], usage={"total_tokens": 0})

        mgr = get_config_manager(request.app)
        cfg = mgr.current_dict()
        llm_defaults = cfg.get("llm", {})
        reranker_defaults = cfg.get("reranker", {})

        # Determine parameters from request or runtime config
        model = req.model or str(llm_defaults.get("model"))
        top_n = req.top_n or int(reranker_defaults.get("default_top_n"))
        caller = req.caller or "unknown"
        logger.info(
            "Rerank request: caller=%s candidates=%d top_n=%d model=%s",
            caller,
            len(req.candidates),
            top_n,
            model,
        )
        
        # Override client settings for this request (conceptually)
        # Since LLMClient is a singleton currently, we pass these params to it.
        # But we also need to respect base_url/api_key if they changed in runtime config.
        # Ideally LLMClient should read from config or accept config.
        
        # Let's pass the config values to the rerank method
        reasoning_cfg = llm_defaults.get("reasoning")
        results = await llm_client.rerank(
            query=req.query,
            candidates=req.candidates,
            top_n=top_n,
            intent=req.intent,
            model=model,
            caller=caller,
            # Pass runtime config overrides
            base_url=llm_defaults.get("base_url"),
            api_key=llm_defaults.get("api_key"),
            timeout=llm_defaults.get("timeout"),
            max_tokens=llm_defaults.get("max_tokens"),
            temperature=llm_defaults.get("temperature"),
            reasoning=reasoning_cfg if isinstance(reasoning_cfg, dict) else None
        )
        
        # Usage tracking placeholder
        usage = {"total_tokens": 0} 
        
        return RerankResponse(results=results, usage=usage)
        
    except Exception as e:
        logger.error(f"Reranking failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

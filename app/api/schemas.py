from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Candidate(BaseModel):
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class RerankRequest(BaseModel):
    query: str
    candidates: List[Candidate]
    top_n: Optional[int] = None # Optional, defaults to config
    intent: Optional[str] = None
    model: Optional[str] = None
    caller: Optional[str] = Field(default="unknown", description="Originating service/component name")

class RerankResult(BaseModel):
    id: str
    score: float
    index: int
    content: Optional[str] = None

class RerankResponse(BaseModel):
    results: List[RerankResult]
    usage: Optional[Dict[str, int]] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    llm_configured: bool
    llm_base_url: Optional[str]
    llm_default_model: Optional[str]

class ConfigGetResponse(BaseModel):
    current: Dict[str, Any]
    defaults: Dict[str, Any]
    runtime_path: str
    has_overrides_file: bool


class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]

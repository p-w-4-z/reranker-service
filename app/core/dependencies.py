from fastapi import FastAPI
from app.core.config import RuntimeConfigManager

def get_config_manager(app: FastAPI) -> RuntimeConfigManager:
    """Get the RuntimeConfigManager from app state."""
    return app.state.runtime_config_manager

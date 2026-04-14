"""Configuration via environment variables with sensible defaults."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # NousResearch Portal
    nous_portal_url: str = "https://portal.nousresearch.com"
    nous_inference_url: str = "https://inference-api.nousresearch.com/v1"
    nous_client_id: str = "hermes-cli"
    nous_scope: str = "inference:mint_agent_key"

    # Proxy server
    proxy_host: str = "0.0.0.0"
    proxy_port: int = 8090

    # Comma-separated API keys for inbound auth
    proxy_api_keys: str = ""

    # Data directory
    data_dir: str = "./data"

    # Logging
    log_level: str = "info"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def tokens_file(self) -> Path:
        return Path(self.data_dir) / "tokens.json"

    @property
    def api_keys_file(self) -> Path:
        return Path(self.data_dir) / "api_keys.json"

    @property
    def parsed_api_keys(self) -> set[str]:
        if not self.proxy_api_keys:
            return set()
        return {k.strip() for k in self.proxy_api_keys.split(",") if k.strip()}


settings = Settings()

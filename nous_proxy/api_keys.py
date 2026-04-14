"""API key validation for inbound proxy requests.

Supports two modes:
1. Static keys from PROXY_API_KEYS env var
2. Generated keys persisted in data/api_keys.json
"""

from __future__ import annotations

import json
import logging
import secrets

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from nous_proxy.config import settings

logger = logging.getLogger(__name__)

_loaded_keys: set[str] = set()


def load_api_keys() -> set[str]:
    """Load API keys from env + file. Generate default if none exist."""
    global _loaded_keys

    # From environment
    env_keys = settings.parsed_api_keys

    # From file
    file_keys: set[str] = set()
    if settings.api_keys_file.exists():
        try:
            data = json.loads(settings.api_keys_file.read_text())
            if isinstance(data, list):
                file_keys = {str(k) for k in data}
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load API keys file: %s", e)

    _loaded_keys = env_keys | file_keys

    # Generate default if empty
    if not _loaded_keys:
        key = generate_api_key()
        _loaded_keys.add(key)
        save_api_keys(_loaded_keys)
        logger.info("Generated default API key: %s", key)
        logger.info("Save this key! It's your proxy API key.")

    return _loaded_keys


def save_api_keys(keys: set[str]) -> None:
    """Persist API keys to disk."""
    settings.api_keys_file.parent.mkdir(parents=True, exist_ok=True)
    settings.api_keys_file.write_text(json.dumps(sorted(keys), indent=2))


def generate_api_key() -> str:
    """Generate a new API key with nous-proxy prefix."""
    return f"np-{secrets.token_urlsafe(32)}"


def verify_api_key(request: Request) -> str:
    """FastAPI dependency: extract and validate API key from Authorization header."""
    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Use: Bearer <api_key>",
        )

    key = auth_header[7:].strip()
    if key not in _loaded_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return key

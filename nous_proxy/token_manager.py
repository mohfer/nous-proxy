"""Token lifecycle manager: persist, refresh access tokens, cache agent keys.

State stored in data/tokens.json:
{
    "access_token": "...",
    "refresh_token": "...",
    "token_obtained_at": 1700000000.0,
    "expires_in": 3600,
    "agent_key": "...",
    "agent_key_id": "...",
    "agent_key_obtained_at": 1700000000.0,
    "agent_key_expires_in": 1800,
    "inference_base_url": "https://..."
}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import httpx

from nous_proxy.auth import refresh_token, mint_agent_key, OAuthError, create_portal_client
from nous_proxy.config import settings

logger = logging.getLogger(__name__)

# Refresh 2 min before access token expires
ACCESS_TOKEN_REFRESH_SKEW = 120
# Refresh 1 min before agent key expires
AGENT_KEY_REFRESH_SKEW = 60


@dataclass
class TokenState:
    access_token: str = ""
    refresh_token: str = ""
    token_obtained_at: float = 0.0
    expires_in: int = 3600
    agent_key: str = ""
    agent_key_id: str = ""
    agent_key_obtained_at: float = 0.0
    agent_key_expires_in: int = 1800
    inference_base_url: str = ""

    @property
    def token_expires_at(self) -> float:
        return self.token_obtained_at + self.expires_in

    @property
    def agent_key_expires_at(self) -> float:
        return self.agent_key_obtained_at + self.agent_key_expires_in

    @property
    def is_token_valid(self) -> bool:
        return bool(self.access_token) and time.time() < self.token_expires_at - ACCESS_TOKEN_REFRESH_SKEW

    @property
    def is_agent_key_valid(self) -> bool:
        return bool(self.agent_key) and time.time() < self.agent_key_expires_at - AGENT_KEY_REFRESH_SKEW

    @property
    def has_credentials(self) -> bool:
        return bool(self.access_token and self.refresh_token)

    @property
    def effective_inference_url(self) -> str:
        return self.inference_base_url or settings.nous_inference_url


class TokenManager:
    """Manages OAuth tokens and agent keys with persistence and auto-refresh."""

    def __init__(self):
        self._state = TokenState()
        self._lock = asyncio.Lock()
        self._refresh_task: asyncio.Task | None = None

    @property
    def state(self) -> TokenState:
        return self._state

    def load(self) -> None:
        """Load persisted token state from disk."""
        path = settings.tokens_file
        if not path.exists():
            logger.info("No persisted tokens found at %s", path)
            return
        try:
            data = json.loads(path.read_text())
            self._state = TokenState(
                access_token=data.get("access_token", ""),
                refresh_token=data.get("refresh_token", ""),
                token_obtained_at=data.get("token_obtained_at", 0.0),
                expires_in=data.get("expires_in", 3600),
                agent_key=data.get("agent_key", ""),
                agent_key_id=data.get("agent_key_id", ""),
                agent_key_obtained_at=data.get("agent_key_obtained_at", 0.0),
                agent_key_expires_in=data.get("agent_key_expires_in", 1800),
                inference_base_url=data.get("inference_base_url", ""),
            )
            logger.info("Loaded tokens from %s", path)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load tokens: %s", e)

    def save(self) -> None:
        """Persist current token state to disk."""
        path = settings.tokens_file
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "access_token": self._state.access_token,
            "refresh_token": self._state.refresh_token,
            "token_obtained_at": self._state.token_obtained_at,
            "expires_in": self._state.expires_in,
            "agent_key": self._state.agent_key,
            "agent_key_id": self._state.agent_key_id,
            "agent_key_obtained_at": self._state.agent_key_obtained_at,
            "agent_key_expires_in": self._state.agent_key_expires_in,
            "inference_base_url": self._state.inference_base_url,
        }
        path.write_text(json.dumps(data, indent=2))

    def update_from_token_response(self, data: dict) -> None:
        """Update state from a token response (poll or refresh)."""
        self._state.access_token = data["access_token"]
        self._state.refresh_token = data.get("refresh_token", self._state.refresh_token)
        self._state.token_obtained_at = time.time()
        self._state.expires_in = data.get("expires_in", 3600)
        self._state.inference_base_url = data.get("inference_base_url", self._state.inference_base_url)
        self.save()

    def update_from_agent_key_response(self, data: dict) -> None:
        """Update state from an agent key response."""
        self._state.agent_key = data["api_key"]
        self._state.agent_key_id = data.get("key_id", "")
        self._state.agent_key_obtained_at = time.time()
        self._state.agent_key_expires_in = data.get("expires_in", 1800)
        if data.get("inference_base_url"):
            self._state.inference_base_url = data["inference_base_url"]
        self.save()

    async def ensure_valid_token(self, client: httpx.AsyncClient) -> None:
        """Ensure we have a valid access token, refreshing if needed."""
        if self._state.is_token_valid:
            return

        async with self._lock:
            if self._state.is_token_valid:
                return
            if not self._state.has_credentials:
                raise OAuthError("no_credentials", "No OAuth credentials. Run /auth/device-code first.")
            logger.info("Refreshing access token...")
            data = await refresh_token(client, self._state.refresh_token)
            self.update_from_token_response(data)

    async def ensure_valid_agent_key(self, client: httpx.AsyncClient) -> str:
        """Ensure we have a valid agent key, minting if needed.

        Returns the agent key string.
        """
        if self._state.is_agent_key_valid:
            return self._state.agent_key

        async with self._lock:
            if self._state.is_agent_key_valid:
                return self._state.agent_key

            # Refresh token directly (don't call ensure_valid_token to avoid nested lock)
            if not self._state.is_token_valid:
                if not self._state.has_credentials:
                    raise OAuthError("no_credentials", "No OAuth credentials. Run /auth/device-code first.")
                logger.info("Refreshing access token for agent key...")
                data = await refresh_token(client, self._state.refresh_token)
                self.update_from_token_response(data)

            logger.info("Minting new agent key...")
            data = await mint_agent_key(client, self._state.access_token)
            self.update_from_agent_key_response(data)
            return self._state.agent_key

    def start_background_refresh(self) -> asyncio.Task:
        """Start background task that keeps tokens and agent keys fresh."""
        if self._refresh_task and not self._refresh_task.done():
            return self._refresh_task
        self._refresh_task = asyncio.create_task(self._background_refresh_loop())
        return self._refresh_task

    async def _background_refresh_loop(self) -> None:
        """Periodically refresh tokens and agent keys."""
        async with create_portal_client() as client:
            while True:
                try:
                    # Refresh access token if expiring soon
                    if (
                        self._state.has_credentials
                        and not self._state.is_token_valid
                    ):
                        logger.info("Background: refreshing access token")
                        try:
                            data = await refresh_token(client, self._state.refresh_token)
                            self.update_from_token_response(data)
                        except OAuthError as e:
                            if e.error in ("invalid_grant", "invalid_token"):
                                logger.error("Token revoked — re-authentication needed! POST /auth/device-code")
                                # Clear invalid credentials so proxy returns clear error
                                self._state.refresh_token = ""
                            else:
                                raise

                    # Refresh agent key if expiring soon
                    if (
                        self._state.agent_key
                        and not self._state.is_agent_key_valid
                    ):
                        logger.info("Background: refreshing agent key")
                        if self._state.access_token:
                            try:
                                data = await mint_agent_key(client, self._state.access_token)
                                self.update_from_agent_key_response(data)
                            except OAuthError as e:
                                logger.error("Agent key refresh failed: %s", e)

                except OAuthError as e:
                    logger.warning("Background refresh error: %s", e)
                except httpx.HTTPError as e:
                    logger.warning("Background refresh HTTP error: %s", e)
                except Exception:
                    logger.exception("Unexpected error in background refresh")

                await asyncio.sleep(30)


# Singleton instance
token_manager = TokenManager()

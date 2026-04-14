"""OAuth 2.0 Device Authorization Grant handler for NousResearch Portal.

Flow:
1. POST /api/oauth/device/code  -> device_code, user_code, verification_uri
2. User opens verification_uri, enters user_code
3. Poll POST /api/oauth/token   -> access_token, refresh_token
"""

from __future__ import annotations

import logging

import httpx

from nous_proxy.config import settings

logger = logging.getLogger(__name__)


class OAuthError(Exception):
    """OAuth flow error."""

    def __init__(self, error: str, description: str = ""):
        self.error = error
        self.description = description
        super().__init__(f"{error}: {description}")


async def request_device_code(client: httpx.AsyncClient) -> dict:
    """Step 1: Request a device code from the portal.

    Returns dict with:
        device_code, user_code, verification_uri,
        verification_uri_complete, expires_in, interval
    """
    resp = await client.post(
        f"{settings.nous_portal_url}/api/oauth/device/code",
        data={
            "client_id": settings.nous_client_id,
            "scope": settings.nous_scope,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    logger.info("Device code requested: user_code=%s", data.get("user_code"))
    return data


async def poll_for_token(
    client: httpx.AsyncClient,
    device_code: str,
    interval: int = 5,
    timeout: int = 300,
) -> dict:
    """Step 2: Poll for access token after user authorizes.

    Args:
        device_code: From request_device_code()
        interval: Seconds between polls (from device code response)
        timeout: Max seconds to wait

    Returns dict with:
        access_token, refresh_token, token_type, expires_in, scope,
        inference_base_url (optional)
    """
    import asyncio

    elapsed = 0
    poll_interval = max(interval, 1)

    while elapsed < timeout:
        resp = await client.post(
            f"{settings.nous_portal_url}/api/oauth/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": settings.nous_client_id,
                "device_code": device_code,
            },
        )

        if resp.status_code == 200:
            data = resp.json()
            if "access_token" in data:
                logger.info("Token obtained successfully")
                return data

        # Handle errors
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        error = data.get("error", "")

        if error == "authorization_pending":
            # User hasn't authorized yet, keep polling
            pass
        elif error == "slow_down":
            # Server asking us to slow down
            poll_interval = min(poll_interval + 5, 30)
            logger.debug("Server requested slow_down, interval=%d", poll_interval)
        elif error:
            raise OAuthError(error, data.get("error_description", ""))

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    raise OAuthError("timeout", f"User did not authorize within {timeout}s")


async def refresh_token(
    client: httpx.AsyncClient,
    refresh_token_value: str,
) -> dict:
    """Refresh an expired access token.

    Returns same format as poll_for_token().
    """
    resp = await client.post(
        f"{settings.nous_portal_url}/api/oauth/token",
        data={
            "grant_type": "refresh_token",
            "client_id": settings.nous_client_id,
            "refresh_token": refresh_token_value,
        },
    )

    if resp.status_code != 200:
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        raise OAuthError(
            data.get("error", "refresh_failed"),
            data.get("error_description", f"HTTP {resp.status_code}"),
        )

    data = resp.json()
    if "access_token" not in data:
        raise OAuthError("refresh_failed", "No access_token in response")

    logger.info("Token refreshed successfully")
    return data


async def mint_agent_key(
    client: httpx.AsyncClient,
    access_token: str,
    min_ttl_seconds: int = 1800,
) -> dict:
    """Mint a short-lived agent key for inference API access.

    Args:
        access_token: Valid OAuth access token
        min_ttl_seconds: Minimum TTL (default 30 minutes, min 60)

    Returns dict with:
        api_key, key_id, expires_at, expires_in, reused,
        inference_base_url (optional)
    """
    resp = await client.post(
        f"{settings.nous_portal_url}/api/oauth/agent-key",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"min_ttl_seconds": max(60, min_ttl_seconds)},
    )

    if resp.status_code != 200:
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        raise OAuthError(
            data.get("error", "agent_key_failed"),
            data.get("error_description", f"HTTP {resp.status_code}"),
        )

    data = resp.json()
    logger.info(
        "Agent key minted: key_id=%s expires_in=%s reused=%s",
        data.get("key_id"),
        data.get("expires_in"),
        data.get("reused"),
    )
    return data

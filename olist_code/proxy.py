"""HTTP forwarding layer for proxying requests to the upstream API."""

from __future__ import annotations

from typing import Any

import httpx

from .auth import get_bearer_token
from .models import AdapterConfig


async def _build_headers(config: AdapterConfig, **extra: str) -> dict[str, str]:
    token = await get_bearer_token(config)
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        **extra,
    }


async def fetch_gateway_models(config: AdapterConfig) -> list[dict[str, Any]]:
    base_url = config.base_url.rstrip("/")
    url = f"{base_url}/v1/models"
    headers = await _build_headers(config)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])


async def forward_request(
    config: AdapterConfig,
    request_data: dict[str, Any],
) -> httpx.Response:
    base_url = config.base_url.rstrip("/")
    url = f"{base_url}/v1/chat/completions"

    headers = await _build_headers(config)

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=request_data, headers=headers)
        return response


async def open_upstream_stream(
    config: AdapterConfig,
    request_data: dict[str, Any],
) -> tuple[httpx.AsyncClient, httpx.Response]:
    """Open a streaming connection and return (client, response) before consuming the body.

    The response status code is available immediately. The caller must close
    both client and response when finished (typically in a generator finally block).
    """
    base_url = config.base_url.rstrip("/")
    url = f"{base_url}/v1/chat/completions"
    headers = await _build_headers(config, Accept="text/event-stream")
    request_data = {**request_data, "stream": True}

    client = httpx.AsyncClient(timeout=300.0)
    try:
        response = await client.send(
            client.build_request("POST", url, json=request_data, headers=headers),
            stream=True,
        )
        return client, response
    except Exception:
        await client.aclose()
        raise

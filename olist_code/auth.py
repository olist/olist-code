"""Keycloak SSO authentication (Authorization Code + PKCE) for the backoffice realm.

The interactive `login()` opens the browser, captures the callback on a local
HTTP server and stores the token set on disk. `get_bearer_token()` is used by
the proxy on every upstream request: it returns the static api_key when one is
configured, otherwise a valid SSO access token (refreshing it when expired).
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import secrets
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

import httpx

from .config import CONFIG_DIR, ensure_config_dir
from .models import AdapterConfig, SSOConfig, TokenSet

TOKENS_FILE = CONFIG_DIR / "tokens.json"

# Refresh the access token when it expires within this window (seconds).
EXPIRY_SLACK = 30.0
LOGIN_TIMEOUT = 300.0


class AuthError(Exception):
    pass


# ── Token storage ─────────────────────────────────────────────────────────────


def load_tokens(path: Path = TOKENS_FILE) -> TokenSet | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return TokenSet(**json.load(f))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def save_tokens(tokens: TokenSet, path: Path = TOKENS_FILE) -> None:
    ensure_config_dir()
    path.write_text(json.dumps(tokens.model_dump(), indent=2))
    path.chmod(0o600)


def clear_tokens(path: Path = TOKENS_FILE) -> bool:
    if path.exists():
        path.unlink()
        return True
    return False


def _token_set_from_response(data: dict[str, Any]) -> TokenSet:
    now = time.time()
    return TokenSet(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", ""),
        expires_at=now + float(data.get("expires_in", 60)),
        refresh_expires_at=now + float(data["refresh_expires_in"])
        if data.get("refresh_expires_in")
        else None,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


def _auth_endpoint(sso: SSOConfig) -> str:
    return f"{sso.issuer.rstrip('/')}/protocol/openid-connect/auth"


def _token_endpoint(sso: SSOConfig) -> str:
    return f"{sso.issuer.rstrip('/')}/protocol/openid-connect/token"


# ── Interactive login (Authorization Code + PKCE) ─────────────────────────────


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    result: dict[str, str] = {}
    event: threading.Event

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        type(self).result = {k: v[0] for k, v in params.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            "<html><body style='font-family:sans-serif'>"
            "<h2>Login conclu&iacute;do</h2>"
            "<p>Pode fechar esta janela e voltar ao terminal.</p>"
            "</body></html>".encode()
        )
        type(self).event.set()

    def log_message(self, *args: object) -> None:  # silence request logging
        pass


def login(sso: SSOConfig) -> TokenSet:
    """Run the browser SSO flow and persist the resulting token set."""
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    state = secrets.token_urlsafe(16)
    redirect_uri = f"http://localhost:{sso.callback_port}/callback"

    auth_url = f"{_auth_endpoint(sso)}?" + urllib.parse.urlencode(
        {
            "client_id": sso.client_id,
            "response_type": "code",
            "scope": "openid email profile",
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )

    _CallbackHandler.result = {}
    _CallbackHandler.event = threading.Event()
    server = http.server.HTTPServer(("localhost", sso.callback_port), _CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        webbrowser.open(auth_url)
        if not _CallbackHandler.event.wait(timeout=LOGIN_TIMEOUT):
            raise AuthError("Tempo esgotado aguardando o login no navegador.")
    finally:
        server.shutdown()
        server.server_close()

    result = _CallbackHandler.result
    if result.get("state") != state:
        raise AuthError("State invalido no callback do SSO.")
    if "error" in result:
        raise AuthError(f"SSO retornou erro: {result.get('error_description') or result['error']}")
    code = result.get("code")
    if not code:
        raise AuthError("Callback do SSO sem authorization code.")

    response = httpx.post(
        _token_endpoint(sso),
        data={
            "grant_type": "authorization_code",
            "client_id": sso.client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        raise AuthError(f"Troca do code falhou (HTTP {response.status_code}): {response.text[:300]}")

    tokens = _token_set_from_response(response.json())
    save_tokens(tokens)
    return tokens


# ── Token resolution for the proxy ────────────────────────────────────────────

_refresh_lock = threading.Lock()


async def _refresh(sso: SSOConfig, tokens: TokenSet) -> TokenSet:
    if not tokens.refresh_token:
        raise AuthError("Sessao SSO expirada. Rode `olist-code login` novamente.")
    if tokens.refresh_expires_at and time.time() >= tokens.refresh_expires_at:
        raise AuthError("Sessao SSO expirada. Rode `olist-code login` novamente.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            _token_endpoint(sso),
            data={
                "grant_type": "refresh_token",
                "client_id": sso.client_id,
                "refresh_token": tokens.refresh_token,
            },
        )
    if response.status_code != 200:
        raise AuthError(
            "Nao foi possivel renovar a sessao SSO. Rode `olist-code login` novamente."
        )

    new_tokens = _token_set_from_response(response.json())
    with _refresh_lock:
        save_tokens(new_tokens)
    return new_tokens


async def get_bearer_token(config: AdapterConfig) -> str:
    """Credential sent upstream: static api_key, or a valid SSO access token."""
    if config.api_key:
        return config.api_key

    sso = config.sso or SSOConfig()
    tokens = load_tokens()
    if tokens is None:
        raise AuthError("Nenhuma credencial configurada. Rode `olist-code login` ou informe --api-key.")

    if time.time() >= tokens.expires_at - EXPIRY_SLACK:
        tokens = await _refresh(sso, tokens)

    return tokens.access_token

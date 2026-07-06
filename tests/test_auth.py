"""Tests for SSO token storage and bearer-token resolution."""

from __future__ import annotations

import time

import pytest

from olist_code import auth
from olist_code.auth import AuthError, get_bearer_token, load_tokens, save_tokens
from olist_code.models import AdapterConfig, ModelConfig, SSOConfig, TokenSet


@pytest.fixture(autouse=True)
def tokens_file(tmp_path, monkeypatch):
    path = tmp_path / "tokens.json"
    monkeypatch.setattr(auth, "TOKENS_FILE", path)
    monkeypatch.setattr(auth.load_tokens, "__defaults__", (path,))
    monkeypatch.setattr(auth.save_tokens, "__defaults__", (path,))
    monkeypatch.setattr(auth.clear_tokens, "__defaults__", (path,))
    return path


def make_config(api_key: str = "") -> AdapterConfig:
    return AdapterConfig(
        base_url="http://localhost:8000",
        api_key=api_key,
        sso=SSOConfig(),
        models=ModelConfig(opus="gpt-test"),
    )


class TestTokenStorage:
    def test_roundtrip(self):
        tokens = TokenSet(access_token="abc", refresh_token="def", expires_at=time.time() + 60)
        save_tokens(tokens)
        loaded = load_tokens()
        assert loaded is not None
        assert loaded.access_token == "abc"
        assert loaded.refresh_token == "def"

    def test_missing_file_returns_none(self):
        assert load_tokens() is None


class TestGetBearerToken:
    async def test_static_api_key_wins(self):
        assert await get_bearer_token(make_config(api_key="sk-olist-x")) == "sk-olist-x"

    async def test_no_credentials_raises(self):
        with pytest.raises(AuthError, match="login"):
            await get_bearer_token(make_config())

    async def test_valid_token_is_returned(self):
        save_tokens(TokenSet(access_token="tok", expires_at=time.time() + 300))
        assert await get_bearer_token(make_config()) == "tok"

    async def test_expired_without_refresh_raises(self):
        save_tokens(TokenSet(access_token="tok", expires_at=time.time() - 10))
        with pytest.raises(AuthError, match="login"):
            await get_bearer_token(make_config())

    async def test_expired_refresh_window_raises(self):
        save_tokens(
            TokenSet(
                access_token="tok",
                refresh_token="ref",
                expires_at=time.time() - 10,
                refresh_expires_at=time.time() - 5,
            )
        )
        with pytest.raises(AuthError, match="login"):
            await get_bearer_token(make_config())

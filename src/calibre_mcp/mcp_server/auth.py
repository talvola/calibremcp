"""OAuth 2.1 provider for the Calibre MCP server.

Mirrors the bar-assistant-mcp pattern: a public MCP client (e.g. Claude.ai)
goes through OAuth dance, our /auth/login page accepts the user's Calibre-Web
credentials, we validate them by attempting a real Calibre-Web login, and on
success we issue an MCP access token whose stored value carries the captured
Calibre-Web session cookie. Subsequent tool calls reach Calibre-Web by
constructing a :class:`CalibreWebClient` seeded with that cookie.

Storage is in-memory (single-instance deployment is the target). Tokens
expire in an hour; refresh tokens rotate; everything cleans itself up.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from .api import CalibreWebClient, CalibreWebError


class StoredAccessToken(AccessToken):
    """OAuth access token with the Calibre-Web session cookie attached.

    Extends the SDK's :class:`AccessToken` so it works with
    :class:`mcp.server.auth.provider.ProviderTokenVerifier`. The extra fields
    are everything :func:`get_calibre_web_client` needs to talk to
    Calibre-Web on behalf of this user.
    """

    cw_url: str
    cw_session_cookie: str
    cw_username: str  # for diagnostics / logging only


@dataclass
class PendingAuth:
    """An authorization request waiting for the user to log in."""

    params: AuthorizationParams
    client: OAuthClientInformationFull
    code: str
    created_at: float = field(default_factory=time.time)


class CalibreWebOAuthProvider:
    """OAuth provider that validates against a Calibre-Web instance.

    In-memory only — fine for a single-tenant deployment. If we ever need
    persistence (e.g. survive container restarts without forcing re-login),
    swap the dicts for a SQLite-backed store.
    """

    def __init__(self, cw_url: str, issuer_url: str = "") -> None:
        self.cw_url = cw_url
        self.issuer_url = issuer_url

        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._pending_auths: dict[str, PendingAuth] = {}
        self._auth_codes: dict[str, AuthorizationCode] = {}
        self._auth_code_sessions: dict[str, tuple[str, str]] = {}  # code -> (cookie, username)
        self._access_tokens: dict[str, StoredAccessToken] = {}
        self._refresh_tokens: dict[str, RefreshToken] = {}
        self._refresh_token_sessions: dict[str, tuple[str, str]] = {}

    # ===== Client registration =====

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if client_info.client_id:
            self._clients[client_info.client_id] = client_info

    # ===== Authorization flow =====

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        """Start the auth flow — return the URL of our login page."""
        code = secrets.token_urlsafe(32)
        self._pending_auths[code] = PendingAuth(
            params=params,
            client=client,
            code=code,
        )
        return f"{self.issuer_url}/auth/login?code_id={code}"

    async def complete_authorization(
        self, code_id: str, username: str, password: str
    ) -> str:
        """Called by the login form. Validate credentials against Calibre-Web."""
        pending = self._pending_auths.pop(code_id, None)
        if not pending:
            raise ValueError("Invalid or expired authorization request")
        if time.time() - pending.created_at > 600:
            raise ValueError("Authorization request expired")

        # Validate credentials against Calibre-Web. login() raises on failure.
        client = CalibreWebClient(self.cw_url)
        try:
            try:
                client.login(username, password)
            except CalibreWebError as e:
                raise ValueError(str(e)) from None
            cookie = client.get_session_cookie()
            if not cookie:
                raise ValueError("Calibre-Web did not set a session cookie")
        finally:
            client.close()

        auth_code = AuthorizationCode(
            code=pending.code,
            scopes=pending.params.scopes or [],
            expires_at=time.time() + 300,
            client_id=pending.client.client_id or "",
            code_challenge=pending.params.code_challenge,
            redirect_uri=pending.params.redirect_uri,
            redirect_uri_provided_explicitly=pending.params.redirect_uri_provided_explicitly,
            resource=pending.params.resource,
        )
        self._auth_codes[pending.code] = auth_code
        self._auth_code_sessions[pending.code] = (cookie, username)

        redirect = str(pending.params.redirect_uri)
        sep = "&" if "?" in redirect else "?"
        redirect += f"{sep}code={pending.code}"
        if pending.params.state:
            redirect += f"&state={pending.params.state}"
        return redirect

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        code = self._auth_codes.get(authorization_code)
        if code and code.expires_at > time.time():
            return code
        return None

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        self._auth_codes.pop(authorization_code.code, None)
        cookie, username = self._auth_code_sessions.pop(
            authorization_code.code, ("", "")
        )

        access_token_str = secrets.token_urlsafe(48)
        refresh_token_str = secrets.token_urlsafe(48)
        expires_in = 3600

        self._access_tokens[access_token_str] = StoredAccessToken(
            token=access_token_str,
            client_id=client.client_id or "",
            scopes=authorization_code.scopes,
            expires_at=int(time.time()) + expires_in,
            cw_url=self.cw_url,
            cw_session_cookie=cookie,
            cw_username=username,
            resource=authorization_code.resource,
        )
        self._refresh_tokens[refresh_token_str] = RefreshToken(
            token=refresh_token_str,
            client_id=client.client_id or "",
            scopes=authorization_code.scopes,
        )
        self._refresh_token_sessions[refresh_token_str] = (cookie, username)

        return OAuthToken(
            access_token=access_token_str,
            token_type="Bearer",
            expires_in=expires_in,
            scope=(
                " ".join(authorization_code.scopes)
                if authorization_code.scopes
                else None
            ),
            refresh_token=refresh_token_str,
        )

    # ===== Refresh =====

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        return self._refresh_tokens.get(refresh_token)

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        old = refresh_token.token
        cookie, username = self._refresh_token_sessions.pop(old, ("", ""))
        self._refresh_tokens.pop(old, None)

        new_access = secrets.token_urlsafe(48)
        new_refresh = secrets.token_urlsafe(48)
        expires_in = 3600
        effective_scopes = scopes if scopes else refresh_token.scopes

        self._access_tokens[new_access] = StoredAccessToken(
            token=new_access,
            client_id=client.client_id or "",
            scopes=effective_scopes,
            expires_at=int(time.time()) + expires_in,
            cw_url=self.cw_url,
            cw_session_cookie=cookie,
            cw_username=username,
        )
        self._refresh_tokens[new_refresh] = RefreshToken(
            token=new_refresh,
            client_id=client.client_id or "",
            scopes=effective_scopes,
        )
        self._refresh_token_sessions[new_refresh] = (cookie, username)

        return OAuthToken(
            access_token=new_access,
            token_type="Bearer",
            expires_in=expires_in,
            scope=" ".join(effective_scopes) if effective_scopes else None,
            refresh_token=new_refresh,
        )

    # ===== Verification + revocation =====

    async def load_access_token(self, token: str) -> StoredAccessToken | None:
        stored = self._access_tokens.get(token)
        if stored and stored.expires_at > time.time():
            return stored
        if stored:
            self._access_tokens.pop(token, None)
        return None

    async def revoke_token(
        self, token: StoredAccessToken | RefreshToken
    ) -> None:
        if isinstance(token, StoredAccessToken):
            self._access_tokens.pop(token.token, None)
        elif isinstance(token, RefreshToken):
            self._refresh_tokens.pop(token.token, None)
            self._refresh_token_sessions.pop(token.token, None)


LOGIN_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Calibre - Sign In</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #1a1a2e;
            color: #eee;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .card {
            background: #16213e;
            border-radius: 12px;
            padding: 2rem;
            width: 100%;
            max-width: 380px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        h1 { font-size: 1.4rem; margin-bottom: 0.5rem; text-align: center; }
        .subtitle { color: #888; font-size: 0.85rem; text-align: center; margin-bottom: 1.5rem; }
        label { display: block; font-size: 0.85rem; color: #aaa; margin-bottom: 0.3rem; }
        input {
            width: 100%; padding: 0.7rem; border: 1px solid #333;
            border-radius: 6px; background: #0f3460; color: #eee;
            font-size: 1rem; margin-bottom: 1rem;
        }
        input:focus { outline: none; border-color: #4a8fe7; }
        button {
            width: 100%; padding: 0.8rem; border: none; border-radius: 6px;
            background: #4a8fe7; color: #fff; font-size: 1rem;
            cursor: pointer; font-weight: 600;
        }
        button:hover { background: #3b75c4; }
        .error {
            background: #3d1a1a; border: 1px solid #e94560; border-radius: 6px;
            padding: 0.7rem; margin-bottom: 1rem; font-size: 0.85rem; color: #ff8a8a;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Calibre Library</h1>
        <p class="subtitle">Sign in with your Calibre-Web credentials to connect Claude</p>
        $error
        <form method="POST" action="/auth/login">
            <input type="hidden" name="code_id" value="$code_id">
            <label for="username">Username</label>
            <input type="text" id="username" name="username" required autofocus autocapitalize="off">
            <label for="password">Password</label>
            <input type="password" id="password" name="password" required>
            <button type="submit">Sign In</button>
        </form>
    </div>
</body>
</html>"""

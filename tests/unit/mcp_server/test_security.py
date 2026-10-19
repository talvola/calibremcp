"""Security regressions for the HTTP-deployed MCP server (2026-10-19 review).

- Cover fetches must not reach the LAN: every redirect hop is checked before
  we connect, connections are pinned to the checked IP, and Calibre-Web is
  handed bytes, never a URL to re-fetch.
- Open OAuth client registration must not allow phishing a Calibre-Web
  login: redirect URIs are allowlisted, and the login page names the client
  and the redirect host.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

import httpx
import pytest
from mcp.server.auth.provider import AuthorizationParams, RegistrationError
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from calibre_mcp.mcp_server import server
from calibre_mcp.mcp_server.auth import CalibreWebOAuthProvider, redirect_uri_problem
from calibre_mcp.mcp_server.server import _fetch_public_image

_PUBLIC = {"covers.example.org": "104.16.0.1", "archive.example.org": "104.16.0.2"}
_PRIVATE = {"evil.example.org": "192.168.1.64", "localhost": "127.0.0.1"}
_JPEG = b"\xff\xd8\xff" + b"x" * 5000


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    table = {**_PUBLIC, **_PRIVATE}

    def getaddrinfo(host: str, port: Any, *_a: Any, **_k: Any) -> list[tuple[Any, ...]]:
        try:
            ip = str(ipaddress.ip_address(host))  # literal IPs resolve to themselves
        except ValueError:
            if host not in table:
                raise socket.gaierror(f"unknown host {host}") from None
            ip = table[host]
        return [(None, None, None, None, (ip, port))]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


class Recorder:
    """httpx MockTransport handler that records every request it serves."""

    def __init__(self, routes: dict[str, httpx.Response]) -> None:
        self.routes = routes  # keyed by "Host header + path"
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        return self.routes[request.headers["host"] + request.url.path]


def _fetch(url: str, routes: dict[str, httpx.Response]) -> tuple[tuple[bytes, str, str] | None, Recorder, str]:
    rec = Recorder(routes)
    try:
        return _fetch_public_image(url, transport=httpx.MockTransport(rec)), rec, ""
    except ValueError as e:
        return None, rec, str(e)


# ===== Cover fetch =====


def test_fetch_connects_to_pinned_ip_with_real_host_and_sni() -> None:
    routes = {"covers.example.org/b/1.jpg": httpx.Response(200, content=_JPEG, headers={"content-type": "image/jpeg"})}
    result, rec, _ = _fetch("https://covers.example.org/b/1.jpg", routes)
    assert result == (_JPEG, "image/jpeg", "https://covers.example.org/b/1.jpg")
    (req,) = rec.seen
    assert req.url.host == "104.16.0.1"
    assert req.headers["host"] == "covers.example.org"
    assert req.extensions["sni_hostname"] == "covers.example.org"


def test_fetch_follows_public_redirects_hop_by_hop() -> None:
    routes = {
        "covers.example.org/b/1.jpg": httpx.Response(302, headers={"location": "https://archive.example.org/v/1.jpg"}),
        "archive.example.org/v/1.jpg": httpx.Response(200, content=_JPEG, headers={"content-type": "image/jpeg"}),
    }
    result, rec, _ = _fetch("https://covers.example.org/b/1.jpg", routes)
    assert result is not None
    assert result[2] == "https://archive.example.org/v/1.jpg"
    assert [r.url.host for r in rec.seen] == ["104.16.0.1", "104.16.0.2"]


@pytest.mark.parametrize(
    "location",
    ["http://evil.example.org:19943/api/stacks", "http://localhost/x.jpg", "http://192.168.1.64:38083/admin"],
)
def test_redirect_to_lan_is_refused_before_connecting(location: str) -> None:
    routes = {"covers.example.org/b/1.jpg": httpx.Response(302, headers={"location": location})}
    result, rec, err = _fetch("https://covers.example.org/b/1.jpg", routes)
    assert result is None
    assert "non-public address" in err
    assert len(rec.seen) == 1  # only the public first hop was ever contacted


def test_redirect_loop_is_capped() -> None:
    routes = {"covers.example.org/loop": httpx.Response(302, headers={"location": "/loop"})}
    result, rec, err = _fetch("https://covers.example.org/loop", routes)
    assert result is None
    assert "redirects" in err
    assert len(rec.seen) == server._COVER_MAX_REDIRECTS + 1


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"}), "not a JPEG/PNG/WebP"),
        (httpx.Response(200, content=b"GIF89a", headers={"content-type": "image/gif"}), "not a JPEG/PNG/WebP"),
        (httpx.Response(404), "HTTP 404"),
        (httpx.Response(200, content=b"", headers={"content-type": "image/png"}), "empty"),
    ],
)
def test_fetch_rejects_non_images(response: httpx.Response, expected: str) -> None:
    result, _, err = _fetch("https://covers.example.org/b/1.jpg", {"covers.example.org/b/1.jpg": response})
    assert result is None
    assert expected in err


def test_fetch_caps_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_COVER_MAX_BYTES", 1000)
    routes = {"covers.example.org/b/1.jpg": httpx.Response(200, content=_JPEG, headers={"content-type": "image/jpeg"})}
    result, _, err = _fetch("https://covers.example.org/b/1.jpg", routes)
    assert result is None
    assert "larger than" in err


def test_set_cover_from_url_uploads_bytes_instead_of_url(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[dict[str, Any], Any]] = []

    class FakeCW:
        def edit_book(self, book_id: int, updates: dict[str, Any], *, cover_file: Any = None) -> None:
            calls.append((updates, cover_file))

        def get_cover_bytes(self, book_id: int) -> tuple[bytes, str]:
            return _JPEG, "image/jpeg"

    monkeypatch.setattr(server, "_calibre_web", FakeCW())
    monkeypatch.setattr(
        server, "_fetch_public_image", lambda _url: (_JPEG, "image/jpeg", "https://archive.example.org/v/1.jpg")
    )
    out = server.calibre_set_cover_from_url(42, "https://covers.example.org/b/1.jpg")
    assert calls == [({}, ("cover.jpg", _JPEG, "image/jpeg"))]
    assert "cover_url" not in calls[0][0]
    assert "5,003 bytes" in out


def test_set_cover_from_url_refuses_private_url_without_touching_calibre_web(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_calibre_web", None)  # any Calibre-Web call would raise
    out = server.calibre_set_cover_from_url(42, "http://evil.example.org/x.jpg")
    assert out.startswith("Refusing cover URL")


# ===== OAuth registration + login page =====

_ALLOWED = ("claude.ai", "claude.com")


@pytest.mark.parametrize(
    "uri",
    [
        "https://claude.ai/api/mcp/auth_callback",
        "https://claude.com/api/mcp/auth_callback",
        "http://localhost:33418/callback",
        "http://127.0.0.1:5000/cb",
        "http://[::1]:5000/cb",
    ],
)
def test_redirect_uri_allowed(uri: str) -> None:
    assert redirect_uri_problem(uri, _ALLOWED) is None


@pytest.mark.parametrize(
    "uri",
    [
        "https://attacker.example/cb",
        "http://claude.ai/api/mcp/auth_callback",  # not https
        "https://claude.ai.attacker.example/cb",
        "https://evil.claude.ai/cb",  # exact hosts only
        "https://user:pw@claude.ai/cb",
        "javascript:alert(1)",
    ],
)
def test_redirect_uri_refused(uri: str) -> None:
    assert redirect_uri_problem(uri, _ALLOWED) is not None


def _client(*redirects: str, name: str | None = "Claude") -> OAuthClientInformationFull:
    return OAuthClientInformationFull(client_id="c1", client_name=name, redirect_uris=[AnyUrl(u) for u in redirects])


async def test_register_client_rejects_attacker_redirect() -> None:
    provider = CalibreWebOAuthProvider("http://cw", "https://books.example")
    with pytest.raises(RegistrationError) as exc:
        await provider.register_client(_client("https://claude.ai/cb", "https://attacker.example/cb"))
    assert exc.value.error == "invalid_redirect_uri"
    assert await provider.get_client("c1") is None


async def test_register_client_accepts_claude_and_honors_configured_hosts() -> None:
    provider = CalibreWebOAuthProvider("http://cw", "https://books.example")
    await provider.register_client(_client("https://claude.ai/api/mcp/auth_callback"))
    assert await provider.get_client("c1") is not None
    custom = CalibreWebOAuthProvider("http://cw", allowed_redirect_hosts=("app.example",))
    await custom.register_client(_client("https://app.example/cb"))
    with pytest.raises(RegistrationError):
        await custom.register_client(_client("https://claude.ai/cb"))


async def test_describe_pending_names_client_and_redirect_host() -> None:
    provider = CalibreWebOAuthProvider("http://cw", "https://books.example")
    client = _client("https://claude.ai/api/mcp/auth_callback")
    params = AuthorizationParams(
        state="s",
        scopes=[],
        code_challenge="x" * 43,
        redirect_uri=AnyUrl("https://claude.ai/api/mcp/auth_callback"),
        redirect_uri_provided_explicitly=True,
    )
    login_url = await provider.authorize(client, params)
    code_id = login_url.split("code_id=", 1)[1]
    assert provider.describe_pending(code_id) == ("Claude", "claude.ai")
    assert provider.describe_pending("bogus") is None


async def test_failed_login_keeps_pending_request_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from calibre_mcp.mcp_server import auth

    class FailingClient:
        def __init__(self, *_a: Any, **_k: Any) -> None: ...
        def login(self, *_a: Any) -> None:
            raise auth.CalibreWebError("bad password")

        def close(self) -> None: ...

    monkeypatch.setattr(auth, "CalibreWebClient", FailingClient)
    provider = CalibreWebOAuthProvider("http://cw", "https://books.example")
    params = AuthorizationParams(
        state=None,
        scopes=[],
        code_challenge="x" * 43,
        redirect_uri=AnyUrl("https://claude.ai/cb"),
        redirect_uri_provided_explicitly=True,
    )
    code_id = (await provider.authorize(_client("https://claude.ai/cb"), params)).split("code_id=", 1)[1]
    with pytest.raises(ValueError, match="bad password"):
        await provider.complete_authorization(code_id, "erik", "wrong")
    assert provider.describe_pending(code_id) is not None

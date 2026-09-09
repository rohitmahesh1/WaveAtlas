from __future__ import annotations

import secrets
from http.cookies import SimpleCookie
from urllib.parse import parse_qs

from starlette.responses import JSONResponse, RedirectResponse


class DesktopAccess:
    """Protect *all* local content, including static files and WebSockets.

    The shell receives a single-use launch URL over a private pipe. It exchanges
    that nonce for an HttpOnly cookie; neither identity nor credentials come from
    the ambient hosted environment or the frontend's localStorage.
    """

    def __init__(self, app, runtime):
        self.app = app
        self.runtime = runtime

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        runtime = self.runtime
        headers = {k.decode("latin1"): v.decode("latin1") for k, v in scope["headers"]}
        host = headers.get("host", "")
        origin = headers.get("origin")
        valid_origin = origin is None or origin == runtime.origin
        valid_host = f"http://{host}" == runtime.origin

        async def deny(status=403):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4403})
            else:
                await JSONResponse(
                    {"detail": "Open WaveAtlas from its application shortcut."},
                    status_code=status,
                )(scope, receive, send)

        if (
            not valid_host
            or not valid_origin
            or headers.get("sec-fetch-site") == "cross-site"
        ):
            return await deny()
        if (
            scope["type"] == "http"
            and scope["path"] == "/desktop/launch"
            and scope["method"] == "GET"
        ):
            token = parse_qs(scope.get("query_string", b"").decode()).get(
                "token", [""]
            )[0]
            if not runtime.launch_token or not secrets.compare_digest(
                token, runtime.launch_token
            ):
                return await deny()
            runtime.launch_token = ""
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(
                runtime.cookie_name,
                runtime.token,
                httponly=True,
                samesite="strict",
                path="/",
            )
            response.headers["Cache-Control"] = "no-store"
            response.headers["Referrer-Policy"] = "no-referrer"
            return await response(scope, receive, send)
        cookie = SimpleCookie()
        try:
            cookie.load(headers.get("cookie", ""))
            supplied = (
                cookie[runtime.cookie_name].value
                if runtime.cookie_name in cookie
                else ""
            )
        except Exception:
            supplied = ""
        bearer = headers.get("authorization", "").removeprefix("Bearer ")
        if not (
            secrets.compare_digest(supplied, runtime.token)
            or secrets.compare_digest(bearer, runtime.token)
        ):
            return await deny(401)

        async def secured_send(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).extend(
                    [
                        (b"cache-control", b"no-store"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"x-content-type-options", b"nosniff"),
                        (
                            b"content-security-policy",
                            (
                                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self' "
                                + runtime.origin.replace("http://", "ws://")
                                + "; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
                            ).encode(),
                        ),
                    ]
                )
            await send(message)

        await self.app(scope, receive, secured_send)

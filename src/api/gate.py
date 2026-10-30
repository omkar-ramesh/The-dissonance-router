"""Access code for a hosted demo.

Only active when DEMO_ACCESS_CODE is set. Every request needs the code, so only people who were
given it (the judges) can reach the app and spend credits. The code is sent once, on the
unlock page or as /?code=..., and is then remembered in a cookie.
"""
import hashlib
import hmac
import time
from urllib.parse import parse_qs

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from src import config

COOKIE = "dr_access"
MAX_TRIES_PER_MIN = 8
_tries: dict[str, list[float]] = {}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Dissonance Router</title>
<style>:root{color-scheme:dark}body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0b0e14;color:#d7deea;font:15px/1.5 ui-monospace,Consolas,monospace}
form{width:min(360px,90vw);background:#131822;border:1px solid #232b3a;border-radius:14px;padding:24px}
h1{font-size:16px;margin:0 0 6px;letter-spacing:.08em}p{margin:0 0 16px;color:#97a3b8;font-size:13px}
input{width:100%;box-sizing:border-box;background:#0b0e14;color:#d7deea;border:1px solid #232b3a;border-radius:8px;padding:10px;font:inherit}
button{margin-top:12px;width:100%;padding:10px;border-radius:8px;border:1px solid #3ddc84;background:#3ddc8420;color:#3ddc84;font:inherit;cursor:pointer}
.err{color:#ff5d6c;margin-top:10px;font-size:13px}</style></head><body>
<form method="post" action="/unlock"><h1>THE DISSONANCE ROUTER</h1>
<p>This demo is open to the hackathon judges. Enter the access code from the submission.</p>
<input name="code" type="password" autocomplete="off" aria-label="Access code" autofocus required>
<button type="submit">Open the demo</button>__ERR__</form></body></html>"""


def _token(code: str) -> str:
    return hashlib.sha256(("dr-demo|" + code).encode()).hexdigest()


def _allowed(request: Request) -> bool:
    return hmac.compare_digest(request.cookies.get(COOKIE, ""), _token(config.DEMO_ACCESS_CODE))


def _too_many(ip: str) -> bool:
    now = time.time()
    recent = [t for t in _tries.get(ip, []) if now - t < 60]
    _tries[ip] = recent
    return len(recent) >= MAX_TRIES_PER_MIN


def _unlocked(request: Request) -> RedirectResponse:
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(COOKIE, _token(config.DEMO_ACCESS_CODE), max_age=60 * 60 * 24 * 14, httponly=True, samesite="lax",
                    secure=request.headers.get("x-forwarded-proto", request.url.scheme) == "https")
    return resp


def _wrong(request: Request, ip: str, shown: bool) -> HTMLResponse:
    _tries.setdefault(ip, []).append(time.time())
    return HTMLResponse(PAGE.replace("__ERR__", '<div class="err">That code is not right.</div>' if shown else ""),
                        status_code=401 if shown else 200)


async def gate(request: Request, call_next):
    if not config.DEMO_ACCESS_CODE:
        return await call_next(request)
    ip = request.client.host if request.client else "?"
    if request.url.path == "/unlock" and request.method == "POST":
        if _too_many(ip):
            return JSONResponse(status_code=429, content={"detail": "Too many tries. Wait a minute."})
        body = parse_qs((await request.body()).decode("utf-8", "ignore"))
        code = (body.get("code") or [""])[0]
        if hmac.compare_digest(code.encode(), config.DEMO_ACCESS_CODE.encode()):
            return _unlocked(request)
        return _wrong(request, ip, shown=True)
    if _allowed(request):
        return await call_next(request)
    code = request.query_params.get("code")
    if request.method == "GET" and request.url.path == "/" and code is not None:
        if _too_many(ip):
            return JSONResponse(status_code=429, content={"detail": "Too many tries. Wait a minute."})
        if hmac.compare_digest(code.encode(), config.DEMO_ACCESS_CODE.encode()):
            return _unlocked(request)
        return _wrong(request, ip, shown=True)
    if request.method == "GET" and request.url.path == "/":
        return HTMLResponse(PAGE.replace("__ERR__", ""))
    return JSONResponse(status_code=401, content={"detail": "Access code required."})

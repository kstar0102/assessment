"""Minimal HS256 JWT signing and verification using only the stdlib."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

ALLOWED_ROLES = {"admin", "viewer"}


class AuthError(Exception):
    pass


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(sub: str, role: str, secret: str, ttl: int = 3600) -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    now = int(time.time())
    payload = _b64url(json.dumps({"sub": sub, "role": role, "iat": now, "exp": now + ttl}, separators=(",", ":")).encode())
    sig = hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url(sig)}"


def verify_token(token: str, secret: str, leeway: int = 30) -> dict:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
        sig = _b64url_decode(sig_b64)
    except Exception as e:
        raise AuthError("malformed token") from e

    if header.get("alg") != "HS256":  # blocks alg=none and alg confusion
        raise AuthError("unsupported alg")
    expected = hmac.new(secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise AuthError("bad signature")
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or exp + leeway < time.time():
        raise AuthError("expired")
    if payload.get("role") not in ALLOWED_ROLES:
        raise AuthError("unknown role")
    return payload


def parse_bearer(header_value: str | None) -> str:
    if not header_value:
        raise AuthError("missing Authorization header")
    scheme, _, token = header_value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("expected Bearer token")
    return token.strip()


if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(description="Mint a test token")
    p.add_argument("role", choices=sorted(ALLOWED_ROLES))
    p.add_argument("--sub", default="demo-user")
    p.add_argument("--ttl", type=int, default=3600)
    a = p.parse_args()
    print(make_token(a.sub, a.role, os.environ.get("GATEWAY_JWT_SECRET", "dev-secret-change-me"), a.ttl))

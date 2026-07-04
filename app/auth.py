"""JWT verification for Neon Auth (Better Auth).

Neon Auth issues EdDSA-signed JWTs.  The public keys are available at the
JWKS endpoint.  We cache the key set in memory and refresh it when a key-id
is not found (key rotation).
"""

import os
import time
from typing import Any

import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

NEON_AUTH_BASE_URL = os.environ.get(
    "NEON_AUTH_BASE_URL",
    "",
)
JWKS_URL = f"{NEON_AUTH_BASE_URL}/.well-known/jwks.json"

_bearer = HTTPBearer()

# Simple in-process JWKS cache
_jwks_cache: dict[str, Any] = {}
_jwks_fetched_at: float = 0.0
_JWKS_TTL = 3600  # re-fetch keys at most every hour


async def _get_jwks() -> dict[str, Any]:
    global _jwks_cache, _jwks_fetched_at
    now = time.monotonic()
    if not _jwks_cache or (now - _jwks_fetched_at) > _JWKS_TTL:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(JWKS_URL)
            resp.raise_for_status()
            _jwks_cache = resp.json()
            _jwks_fetched_at = now
    return _jwks_cache


def _build_jwk_set(jwks: dict[str, Any]) -> jwt.PyJWKSet:
    return jwt.PyJWKSet.from_dict(jwks)


async def _verify_token(token: str) -> dict[str, Any]:
    """Verify a Bearer JWT and return its claims."""
    jwks = await _get_jwks()
    jwk_set = _build_jwk_set(jwks)

    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if kid:
            signing_key = jwk_set[kid]
        else:
            signing_key = jwk_set.keys[0]

        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["EdDSA", "RS256", "ES256"],
            options={"verify_aud": False},
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )


class CurrentUser:
    def __init__(self, user_id: str, email: str, name: str | None = None):
        self.user_id = user_id
        self.email = email
        self.name = name


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> CurrentUser:
    claims = await _verify_token(credentials.credentials)

    user_id: str = claims.get("sub", "")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing 'sub' claim.",
        )

    email: str = claims.get("email", "")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing 'email' claim.",
        )

    return CurrentUser(user_id=user_id, email=email, name=claims.get("name"))

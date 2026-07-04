"""Async Postgres connection pool using psycopg (v3)."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

_pool: AsyncConnectionPool | None = None


async def init_pool() -> None:
    global _pool
    url = os.environ["DATABASE_URL"]
    _pool = AsyncConnectionPool(
        conninfo=url,
        min_size=1,
        max_size=10,
        kwargs={"row_factory": dict_row},
        open=False,
    )
    await _pool.open(wait=True)


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_conn() -> AsyncIterator[psycopg.AsyncConnection]:
    """Yield a connection from the pool."""
    if _pool is None:
        raise RuntimeError("Database pool not initialised.")
    async with _pool.connection() as conn:
        yield conn

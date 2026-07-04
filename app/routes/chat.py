"""Q&A chat over extracted statement data.

Ports the logic from src/routes/api.chat.ts and buildStatementContext
from src/lib/statement.ts, calling the same Ollama model.
"""

from __future__ import annotations

import json
import os

import ollama
from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import CurrentUser, get_current_user
from app.db import get_conn
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/chat", tags=["chat"])

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5-vl")


def _ollama_client() -> ollama.Client:
    headers: dict = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return ollama.Client(host=OLLAMA_HOST, headers=headers)


def _build_context(result: dict) -> str:
    """Mirror buildStatementContext from src/lib/statement.ts."""
    context = {
        "metadata": {
            "institution": result.get("institution"),
            "accountName": result.get("accountName"),
            "accountNumber": result.get("accountNumber"),
            "cardName": result.get("cardName"),
            "nameOnCard": result.get("nameOnCard"),
            "statementPeriod": result.get("statementPeriod"),
            "currency": result.get("currency"),
        },
        "fields": result.get("fields", []),
        "transactions": result.get("transactions", []),
        "notes": result.get("notes", []),
    }
    return json.dumps(context, indent=2)


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
) -> ChatResponse:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT result FROM extraction_jobs
                WHERE id = %s AND user_id = %s AND status = 'complete'
                """,
                (body.job_id, user.user_id),
            )
            row = await cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job not found or extraction not yet complete.",
        )

    result_data = row["result"]
    if isinstance(result_data, str):
        result_data = json.loads(result_data)

    system_prompt = (
        "You answer questions using only the extracted bank statement data below. "
        "If the answer is not supported by the extracted transactions, say so. "
        "Be concise and show calculations when useful.\n\n"
        + _build_context(result_data)
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in body.messages:
        messages.append({"role": msg.role, "content": msg.content})

    client = _ollama_client()
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": 0.1},
    )

    reply: str = response["message"]["content"]
    return ChatResponse(reply=reply)

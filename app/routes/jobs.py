"""Job management routes: upload PDF, poll status, list history."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.auth import CurrentUser, get_current_user
from app.db import get_conn
from app.schemas import JobStatus

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("", status_code=202)
async def create_job(
    file: Annotated[UploadFile, File(description="PDF bank statement")],
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO extraction_jobs (user_id, user_email, file_name, pdf_bytes, status)
                VALUES (%s, %s, %s, %s, 'queued')
                RETURNING id::text
                """,
                (
                    user.user_id,
                    user.email,
                    file.filename or "statement.pdf",
                    pdf_bytes,
                ),
            )
            row = await cur.fetchone()

    return {"jobId": row["id"]}


@router.get("", response_model=list[JobStatus])
async def list_jobs(
    user: CurrentUser = Depends(get_current_user),
) -> list[dict]:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id::text, status, file_name,
                       result, error,
                       created_at::text, updated_at::text
                FROM extraction_jobs
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 50
                """,
                (user.user_id,),
            )
            rows = await cur.fetchall()

    return [_format_row(r) for r in rows]


@router.get("/{job_id}", response_model=JobStatus)
async def get_job(
    job_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> dict:
    async with get_conn() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id::text, status, file_name,
                       result, error,
                       created_at::text, updated_at::text
                FROM extraction_jobs
                WHERE id = %s AND user_id = %s
                """,
                (job_id, user.user_id),
            )
            row = await cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")

    return _format_row(row)


def _format_row(row: dict) -> dict:
    result = row.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            pass
    return {
        "id": row["id"],
        "status": row["status"],
        "file_name": row["file_name"],
        "result": result,
        "error": row.get("error"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }

"""Background worker: poll Neon Postgres for queued jobs and process them.

Uses SELECT ... FOR UPDATE SKIP LOCKED so multiple worker instances can
run in parallel without claiming the same job.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

# Allow imports from the backend root when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from worker.mailer import send_completion_email, send_failure_email
from worker.extraction import extract_statement_from_pages
from worker.pdf_render import render_pdf_bytes_to_images

DATABASE_URL = os.environ["DATABASE_URL"]
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SECONDS", "2"))
MAX_PAGES = int(os.environ.get("EXTRACTION_MAX_PAGES", "21"))
RENDER_DPI = int(os.environ.get("EXTRACTION_RENDER_DPI", "120"))


def _get_conn() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def _update_status(cur: psycopg.Cursor, job_id: str, status: str, **extras) -> None:
    set_clauses = ["status = %s", "updated_at = NOW()"]
    values: list = [status]
    for col, val in extras.items():
        set_clauses.append(f"{col} = %s")
        values.append(val)
    values.append(job_id)
    cur.execute(
        f"UPDATE extraction_jobs SET {', '.join(set_clauses)} WHERE id = %s",
        values,
    )


def process_job(job: dict) -> None:
    job_id: str = str(job["id"])
    file_name: str = job["file_name"]
    pdf_bytes: bytes = bytes(job["pdf_bytes"])
    user_email: str = job["user_email"]

    print(f"[Worker] Processing job {job_id} ({file_name}) for {user_email}", flush=True)

    with _get_conn() as conn:
        with conn.cursor() as cur:
            try:
                # --- Render ---
                print(f"[Worker] Rendering PDF pages for job {job_id}", flush=True)
                render_result = render_pdf_bytes_to_images(
                    pdf_bytes,
                    max_pages=MAX_PAGES,
                    dpi=RENDER_DPI,
                )
                _update_status(cur, job_id, "extracting")
                conn.commit()

                # --- Extract ---
                print(
                    f"[Worker] Calling vision model for job {job_id} "
                    f"({len(render_result.rendered_pages)} pages)",
                    flush=True,
                )
                extraction = extract_statement_from_pages(
                    file_name, render_result.rendered_pages
                )

                # --- Save result ---
                _update_status(
                    cur,
                    job_id,
                    "complete",
                    result=json.dumps(extraction),
                )
                conn.commit()
                print(f"[Worker] Job {job_id} complete", flush=True)

                # --- Email ---
                send_completion_email(user_email, file_name, job_id)
                cur.execute(
                    "UPDATE extraction_jobs SET notified_at = NOW() WHERE id = %s",
                    (job_id,),
                )
                conn.commit()

            except Exception as exc:
                print(f"[Worker] Job {job_id} failed: {exc}", flush=True)
                try:
                    conn.rollback()
                    with conn.cursor() as err_cur:
                        _update_status(err_cur, job_id, "failed", error=str(exc))
                        conn.commit()
                    send_failure_email(user_email, file_name, str(exc))
                    with conn.cursor() as notify_cur:
                        notify_cur.execute(
                            "UPDATE extraction_jobs SET notified_at = NOW() WHERE id = %s",
                            (job_id,),
                        )
                        conn.commit()
                except Exception as inner:
                    print(f"[Worker] Could not mark job {job_id} as failed: {inner}", flush=True)


def poll_loop() -> None:
    print("[Worker] Python background worker started. Polling for jobs...", flush=True)

    while True:
        try:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    # Atomically claim one queued job
                    cur.execute("""
                        UPDATE extraction_jobs
                        SET status = 'rendering', updated_at = NOW()
                        WHERE id = (
                            SELECT id FROM extraction_jobs
                            WHERE status = 'queued'
                            ORDER BY created_at ASC
                            LIMIT 1
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING id, file_name, pdf_bytes, user_email
                    """)
                    job = cur.fetchone()
                    conn.commit()

                if not job:
                    time.sleep(POLL_INTERVAL)
                    continue

        except Exception as db_err:
            print(f"[Worker] DB error in poll loop: {db_err}", flush=True)
            time.sleep(5)
            continue

        process_job(job)


if __name__ == "__main__":
    poll_loop()

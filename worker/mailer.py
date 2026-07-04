"""Send job completion/failure emails via Resend."""

import os

import resend

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")
FROM_ADDRESS = os.environ.get(
    "RESEND_FROM",
    "Bank Statement Converter <notifications@resend.dev>",
)


def _client() -> None:
    resend.api_key = RESEND_API_KEY


def send_completion_email(
    to_email: str,
    file_name: str,
    job_id: str,
) -> None:
    """Send a 'your extraction is ready' email."""
    if not RESEND_API_KEY:
        print(f"[Email] RESEND_API_KEY not set — skipping email to {to_email}")
        return

    _client()
    # There's no dedicated /jobs/:id page — the dashboard resumes the active
    # job via localStorage, and job history is listed on the root page.
    job_url = FRONTEND_URL

    params: resend.Emails.SendParams = {
        "from": FROM_ADDRESS,
        "to": [to_email],
        "subject": f"Your statement is ready: {file_name}",
        "html": f"""
        <div style="font-family:sans-serif;max-width:520px;margin:auto">
          <h2 style="color:#0f172a">Extraction complete</h2>
          <p>Your bank statement <strong>{file_name}</strong> has been successfully
          processed and is ready to view.</p>
          <a href="{job_url}"
             style="display:inline-block;background:#0f172a;color:#fff;padding:12px 24px;
                    border-radius:6px;text-decoration:none;font-weight:600;">
            View results
          </a>
          <p style="color:#6b7280;font-size:13px;margin-top:24px;">
            You are receiving this because you uploaded a statement to Bank Statement Converter.
          </p>
        </div>
        """,
    }

    email = resend.Emails.send(params)
    print(f"[Email] Sent completion email to {to_email}, id={email.get('id')}")


def send_failure_email(
    to_email: str,
    file_name: str,
    error_message: str,
) -> None:
    """Send a 'extraction failed' email."""
    if not RESEND_API_KEY:
        print(f"[Email] RESEND_API_KEY not set — skipping failure email to {to_email}")
        return

    _client()

    params: resend.Emails.SendParams = {
        "from": FROM_ADDRESS,
        "to": [to_email],
        "subject": f"Extraction failed: {file_name}",
        "html": f"""
        <div style="font-family:sans-serif;max-width:520px;margin:auto">
          <h2 style="color:#dc2626">Extraction failed</h2>
          <p>Unfortunately, we could not extract transactions from
          <strong>{file_name}</strong>.</p>
          <p><strong>Reason:</strong> {error_message}</p>
          <p>Please try uploading the file again. If the problem persists,
          make sure the PDF is a valid bank statement and not password-protected.</p>
        </div>
        """,
    }

    email = resend.Emails.send(params)
    print(f"[Email] Sent failure email to {to_email}, id={email.get('id')}")

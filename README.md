# extractor_backend

## Docker

Build the image:

```bash
docker build -t extractor-backend .
```

Run the API locally:

```bash
docker run --env-file .env -p 8000:8000 -e PORT=8000 extractor-backend
```

Run the worker locally:

```bash
docker build -f Dockerfile.worker -t extractor-worker .
docker run --env-file .env extractor-worker
```

## Render

This repo includes a `render.yaml` Blueprint with two Docker services:

- `statement-api`: FastAPI web service, health check at `/health`
- `statement-worker`: background worker for queued extraction jobs, using `Dockerfile.worker`

Set these environment variables in Render:

- `DATABASE_URL`
- `NEON_AUTH_BASE_URL`
- `OLLAMA_HOST`
- `OLLAMA_API_KEY`
- `OLLAMA_MODEL`
- `OLLAMA_TIMEOUT_SECONDS`
- `EXTRACTION_PAGES_PER_REQUEST`
- `EXTRACTION_MAX_PAGES`
- `EXTRACTION_RENDER_DPI`
- `STALE_JOB_MINUTES`
- `FRONTEND_ORIGIN`
- `RESEND_API_KEY`
- `RESEND_FROM`
- `FRONTEND_URL`

## Manual Render worker setup

If Render does not show a start command field for Docker, use the worker
Dockerfile instead:

- Service type: Background Worker
- Runtime: Docker
- Root directory: `backend`
- Dockerfile path: `./Dockerfile.worker`
- Docker context: `.`

The worker Dockerfile already runs:

```bash
python worker/worker.py
```

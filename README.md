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
docker run --env-file .env extractor-backend python worker/worker.py
```

## Render

This repo includes a `render.yaml` Blueprint with two Docker services:

- `statement-api`: FastAPI web service, health check at `/health`
- `statement-worker`: background worker for queued extraction jobs

Set these environment variables in Render:

- `DATABASE_URL`
- `NEON_AUTH_BASE_URL`
- `OLLAMA_HOST`
- `OLLAMA_API_KEY`
- `OLLAMA_MODEL`
- `FRONTEND_ORIGIN`
- `RESEND_API_KEY`
- `RESEND_FROM`
- `FRONTEND_URL`

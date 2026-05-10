# Smartour

Smartour is a conversational travel planning application. It collects trip requirements through a stateful chat flow, confirms the structured trip brief, generates an itinerary with Google Maps Platform data, and displays the result in a Next.js workspace.

The repository contains a Python FastAPI backend and a separate Next.js frontend:

- `src/smartour`: backend API, domain models, services, SQLite persistence, Google Maps integrations, and supervised requirement extraction.
- `app`: browser workspace, typed backend API client, itinerary display, photo gallery, route maps, and theme support.

## Architecture

The backend uses a layered `src/` architecture:

```text
src/smartour/
|-- api/              HTTP routes and dependency wiring
|-- application/      conversation, extraction, planning, and job services
|-- core/             configuration and shared errors
|-- domain/           Pydantic business models
|-- infrastructure/   SQLite repositories, cache, metrics, and rate limiting
|-- integrations/     Google Maps and requirement model clients
`-- main.py           FastAPI application entrypoint
```

The frontend lives in `app/` and calls the backend through `app/src/lib/smartourApi.tsx`.

See [docs/architecture.md](docs/architecture.md) for the detailed architecture guide.

## Requirements

- Python 3.12
- `uv`
- Node.js 20 or newer
- `pnpm`
- A Google Maps Platform API key with the required server-side APIs enabled
- A trained local requirement model artifact, or the development fallback enabled

## Environment

Create a repository-level `.env` file:

```text
GOOGLE_MAPS_API_KEY=your-server-side-google-maps-key
SMARTOUR_SQLITE_PATH=data/smartour.sqlite3

# Supervised requirement extraction
REQUIREMENT_MODEL_PATH=models/requirement_model/latest
REQUIREMENT_MODEL_CONFIDENCE_THRESHOLD=0.35
REQUIREMENT_MODEL_DEVELOPMENT_FALLBACK_ENABLED=true

# Optional frontend and local development settings
NEXT_PUBLIC_SMARTOUR_API_BASE_URL=http://127.0.0.1:8000/api
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your-browser-restricted-google-maps-key
SMARTOUR_CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

Use a separate browser-restricted Google Maps key for `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY` in real deployments.

## Backend Setup

Install Python dependencies:

```bash
uv sync
```

Run the API server:

```bash
uv run smartour-api
```

The backend listens on `http://127.0.0.1:8000`. The health endpoint is available at:

```text
GET http://127.0.0.1:8000/api/health
```

## Frontend Setup

Install frontend dependencies:

```bash
cd app
pnpm install
```

Run the Next.js development server:

```bash
pnpm dev
```

The frontend starts on `http://localhost:3000` by default and calls `http://127.0.0.1:8000/api` unless `NEXT_PUBLIC_SMARTOUR_API_BASE_URL` overrides it.

## Common Commands

Backend checks from the repository root:

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src tests
```

Frontend checks from `app/`:

```bash
pnpm exec tsc --noEmit
pnpm exec eslint .
```

Integration probes:

```bash
uv run smartour-google-maps-probe
uv run python scripts/requirement_model/generate_data.py --count 3000 --language en --llm-augment --validate
uv run python scripts/requirement_model/audit_data.py --data-dir data/requirement_model --reviewed-test --strict
uv run python scripts/requirement_model/train.py --quick
uv run python scripts/requirement_model/evaluate.py --split reviewed_test
```

The Google Maps API also exposes a safe backend probe:

```text
GET /api/google-maps/probe
GET /api/google-maps/probe?live=true
```

## Requirement Model Data

The active requirement model data workflow generates English-only training data.
It writes 3000 validated JSONL records under `data/requirement_model`: 2400
training records, 300 validation records, 300 test records, and a manually
reviewed `reviewed_test.jsonl` split for evaluation.

LLM augmentation uses the official OpenAI Python SDK with OpenAI-compatible chat
completion fields. Configure the primary endpoint with `OPENAI_API_BASEURL`,
`OPENAI_API_KEY`, and `OPENAI_API_MODEL`. Optional backup variables
`OPENAI_API_BASEURL_BACKUP`, `OPENAI_API_KEY_BACKUP`, and
`OPENAI_API_MODEL_BACKUP` are used only when the primary endpoint is unavailable
or retryable failures exhaust the primary attempts.

Generate and validate the full dataset:

```bash
uv run python scripts/requirement_model/generate_data.py --count 3000 --language en --llm-augment --validate
```

Audit the generated and reviewed splits:

```bash
uv run python scripts/requirement_model/audit_data.py --data-dir data/requirement_model --reviewed-test --strict
```

Run a quick training smoke test and evaluate the reviewed split:

```bash
uv run python scripts/requirement_model/train.py --quick
uv run python scripts/requirement_model/evaluate.py --split reviewed_test
```

## Main API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check backend health |
| `POST` | `/api/conversations` | Create a planning conversation |
| `GET` | `/api/conversations/{conversationId}` | Fetch conversation state |
| `POST` | `/api/conversations/{conversationId}/messages` | Send a user requirement message |
| `POST` | `/api/conversations/{conversationId}/confirm` | Confirm completed requirements |
| `POST` | `/api/conversations/{conversationId}/itinerary-jobs` | Queue itinerary generation |
| `GET` | `/api/itinerary-jobs/{jobId}` | Fetch itinerary job state |
| `GET` | `/api/itinerary-jobs/{jobId}/events` | Stream itinerary job updates |
| `GET` | `/api/itineraries/{itineraryId}` | Fetch a generated itinerary |

## Documentation

- [Architecture](docs/architecture.md)
- [Backend design](docs/backend-design.md)
- [Frontend design](docs/frontend-design.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

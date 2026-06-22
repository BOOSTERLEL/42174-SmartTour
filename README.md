# Smartour

Smartour is a conversational travel planning application. It collects trip requirements through a stateful chat flow, confirms the structured trip brief, generates an itinerary with Google Maps Platform data, and displays the result in a Next.js workspace.

The project includes a Python FastAPI backend, a Next.js frontend, Google Maps integrations, itinerary generation workflows, an operations dashboard, and downloadable trip reports.

## Screenshots

### Main planning workspace

The main workspace supports conversational requirement collection, trip brief confirmation, itinerary generation, route overview, restaurant suggestions, and map-based travel planning.

![Smartour main planning workspace](assets\smartour-main-ui.png)

### Generated itinerary and report

Smartour generates a structured multi-day travel guide and a report view that can be shared or downloaded.

![Smartour generated itinerary and report](assets\smartour-report-view.png)

### Operations dashboard

The admin dashboard tracks users, conversations, itineraries, shared trips, jobs, estimated API costs, Google Maps service usage, and job status distribution.

![Smartour operations dashboard](assets\smartour-admin-dashboard.png)

## What the system does

- Collects natural-language travel requirements through a chat interface.
- Converts user requirements into a structured trip brief.
- Confirms missing or uncertain trip details before itinerary generation.
- Generates multi-day itineraries using Google Maps data.
- Displays attractions, restaurants, route steps, travel time, and daily themes.
- Provides an admin dashboard for monitoring jobs, usage, and estimated cost.
- Supports report-style itinerary output for sharing or review.

## Tech stack

- Backend: Python, FastAPI, Pydantic, SQLite
- Frontend: Next.js, TypeScript
- Maps and itinerary data: Google Maps Platform
- Workflow support: GitHub Actions, ClearML-compatible model workflow scripts
- Testing and quality checks: pytest, Ruff, mypy, ESLint, TypeScript checks

## Repository structure

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
NEXT_PUBLIC_GOOGLE_MAPS_MAP_ID=your-browser-map-id
SMARTOUR_CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

Use a separate browser-restricted Google Maps key for `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY` in real deployments.
If `NEXT_PUBLIC_GOOGLE_MAPS_MAP_ID` is omitted, the frontend uses Google's demo map ID for Advanced Marker compatibility.

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

## CI/CD

The repository has two GitHub Actions workflows:

- `CI`: runs on pull requests, pushes to `main`, and manual dispatch. It runs
  backend tests, Ruff, mypy, frontend Prettier check, ESLint, TypeScript, and
  `pnpm build`. It uses dummy Google Maps values and does not require ClearML or
  production secrets.
- `Model Ops`: manual-only workflow for requirement-model operations. It can run
  data audit, quick train/eval, HPO, model comparison, or all operations.
  `promote_winner` defaults to `false`, so comparison runs do not update
  `models/requirement_model/latest` unless promotion is explicitly enabled.

Configure these GitHub repository secrets only when the `Model Ops` workflow is
run with `clearml=true`:

```text
CLEARML_API_ACCESS_KEY
CLEARML_API_SECRET_KEY
CLEARML_API_HOST
CLEARML_WEB_HOST
CLEARML_FILES_HOST
```

The `Model Ops` workflow uploads only summary artifacts:
`hpo_summary.json`, `hpo_trials.csv`, `comparison_summary.json`, and
`comparison_metrics.csv`. It does not commit model binaries, generated datasets,
or logs.

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

## ClearML Requirement Model Tracking

The requirement model scripts can publish workflow state to ClearML when the
`--clearml` flag is supplied. ClearML tracking is disabled by default, so local
data, training, and evaluation commands still work without ClearML credentials.

Configure the repository `.env` file with ClearML credentials:

```text
CLEARML_API_ACCESS_KEY=your-clearml-access-key
CLEARML_API_SECRET_KEY=your-clearml-secret-key
CLEARML_API_HOST=https://api.clear.ml
CLEARML_WEB_HOST=https://app.clear.ml
CLEARML_FILES_HOST=https://files.clear.ml
```

The default ClearML project is `Smartour`. Publish the audited requirement model
dataset, run tracked quick training, and report reviewed-test metrics:

```bash
uv run python scripts/requirement_model/audit_data.py --data-dir data/requirement_model --reviewed-test --strict --clearml
uv run python scripts/requirement_model/train.py --quick --clearml
uv run python scripts/requirement_model/evaluate.py --model-dir models/requirement_model/latest --split reviewed_test --clearml
```

Use the detailed reporting flags when the ClearML task should include richer
data, model, and evaluation views:

```bash
uv run python scripts/requirement_model/audit_data.py --data-dir data/requirement_model --reviewed-test --strict --clearml --clearml-report-data-profile
uv run python scripts/requirement_model/train.py --quick --clearml --clearml-model-report
uv run python scripts/requirement_model/evaluate.py --model-dir models/requirement_model/latest --split reviewed_test --clearml --clearml-detailed-report
```

The detailed data audit reports split counts, token and text length histograms,
BIO label distribution, and slot coverage. The detailed evaluation reports a BIO
confusion matrix, per-label precision/recall/F1, per-slot accuracy, and failed
examples. The model report uploads model configuration, label maps, and the file
manifest; add `--clearml-register-model` to register the trained output model in
ClearML.

Run a convergence-based model comparison when selecting the default requirement
model:

```bash
uv run python scripts/requirement_model/compare_models.py --clearml --clearml-project Smartour --device cuda --batch-size 4 --max-epochs 20 --min-epochs 3 --patience 3 --min-delta 0.001 --register-winner
```

Run a deterministic HPO search for the current baseline model:

```bash
uv run python scripts/requirement_model/hpo.py \
  --model-name distilbert-base-multilingual-cased \
  --learning-rate-values 2e-5,3e-5,5e-5 \
  --batch-size-values 4,8 \
  --max-length-values 128,192 \
  --trial-limit 6 \
  --max-epochs 8 \
  --min-epochs 3 \
  --patience 2 \
  --objective-split validation \
  --objective-metric macro_f1 \
  --clearml
```

HPO writes per-trial model artifacts and training reports under
`models/requirement_model/hpo/<run-id>/<trial-id>`, plus
`hpo_summary.json` and `hpo_trials.csv` under the run directory. The summary
records every trial config, status, objective score, metrics, output directory,
and the selected best trial.

Use an HPO summary to tune matching model parameters during comparison:

```bash
uv run python scripts/requirement_model/compare_models.py \
  --hpo-summary models/requirement_model/hpo/<run-id>/hpo_summary.json \
  --model-name distilbert-base-multilingual-cased \
  --model-name distilbert-base-uncased \
  --model-name bert-base-cased \
  --no-promote \
  --clearml
```

The comparison command trains the current baseline
`distilbert-base-multilingual-cased` plus the English-focused candidates
`distilbert-base-uncased` and `bert-base-cased`. Each model trains on
`train.jsonl` and monitors `validation.jsonl` with early stopping on validation
`macro_f1`: training runs for at least 3 epochs, stops after 3 epochs without a
minimum 0.001 improvement, and caps at 20 epochs. The saved candidate artifact is
the best validation checkpoint, not necessarily the final epoch.

Each candidate is evaluated on `validation`, `test`, and `reviewed_test`. The
comparison reports slot accuracy, exact-match accuracy, micro F1, macro F1,
per-slot accuracy, per-label precision/recall/F1, BIO confusion matrices, and
failed examples. Ranking uses reviewed-test slot accuracy first, then
reviewed-test exact-match accuracy, reviewed-test macro F1, and validation macro
F1 as tie breakers. The winning model is copied to
`models/requirement_model/latest` and registered in ClearML when
`--register-winner` is supplied. Add `--no-promote` for workflow or review runs
that should compare models without changing the default runtime artifact. Local
comparison summaries are written under `models/requirement_model/experiments/<run-id>`,
while model and data directories remain ignored by Git.

Run the local workflow DAG when the ClearML UI should show the audit, quick
training, and evaluation steps as a pipeline without using a ClearML Agent queue:

```bash
uv run python scripts/requirement_model/clearml_pipeline.py --local --quick
```

For a full local training pipeline, omit `--quick` and pass the desired training
settings:

```bash
uv run python scripts/requirement_model/clearml_pipeline.py --local --device cuda --batch-size 4 --epochs 3
```

For full training, remove `--quick` and pass the desired training device and
batch size. The training task reports per-epoch loss and uploads the saved model
directory as a ClearML artifact. ClearML tasks created by these scripts clear
the captured script diff so local git changes are not stored in the task record.

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

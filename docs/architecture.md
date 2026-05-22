# Smartour Architecture

## 1. Purpose

Smartour is a conversational travel planning application. It collects trip requirements through a chat-like workflow, confirms the structured requirement snapshot, generates an itinerary with Google Maps Platform data, and renders the result in a browser workspace.

The implementation is split into two applications:

- `src/smartour`: a Python 3.12 FastAPI backend that owns state, validation, planning, persistence, and external API calls.
- `app`: a Next.js frontend that owns the user workspace, typed API client, route display, photo gallery, and interaction state.

The core architectural rule is that deterministic backend code owns canonical state. The supervised requirement model can propose requirement updates, and Google Maps can supply place and route data, but Smartour validates, stores, merges, and exposes the final application models.

## 2. System Context

```text
Browser
  |
  | HTTP JSON
  v
Next.js frontend
  |
  | HTTP JSON / optional SSE
  v
FastAPI backend
  |
  +-- SQLite persistence
  +-- Google Maps Platform APIs
  +-- Supervised requirement extraction model
```

The frontend communicates with the backend through `/api` endpoints. The backend calls Google Maps services from the server so unrestricted server-side keys are never sent to the browser. The frontend may use a browser-safe Google Maps key for map rendering, but it can still operate with fallback route previews when that key is unavailable.

## 3. Repository Layout

```text
.
|-- app/
|   |-- src/app/
|   |   |-- page.tsx
|   |   |-- page.module.css
|   |   `-- layout.tsx
|   |-- src/components/
|   |   |-- DailyPhotoGallery.tsx
|   |   `-- RouteMap.tsx
|   |-- src/lib/
|   |   |-- googleMapsLoader.tsx
|   |   |-- placePhotos.tsx
|   |   |-- routeGeometry.tsx
|   |   `-- smartourApi.tsx
|   |-- package.json
|   `-- next.config.ts
|-- docs/
|   |-- architecture.md
|   |-- backend-design.md
|   |-- frontend-design.md
|   `-- frontend-page-mockup.png
|-- src/smartour/
|   |-- api/
|   |-- application/
|   |-- core/
|   |-- domain/
|   |-- infrastructure/
|   |-- integrations/
|   `-- main.py
|-- tests/
|-- pyproject.toml
`-- README.md
```

The backend follows a layered `src/` layout. The frontend is isolated under `app/` so browser dependencies do not leak into the Python package.

## 4. Backend Architecture

### 4.1 API Layer

The API layer is implemented under `src/smartour/api`. It contains route handlers and dependency wiring.

Main responsibilities:

- Create the FastAPI application in `smartour.main`.
- Configure CORS for local Next.js origins.
- Expose health, conversation, itinerary, job, and Google Maps diagnostic routes.
- Translate application errors into HTTP responses.
- Keep HTTP request and response models separate from service orchestration.

Implemented route groups:

| File | Responsibility |
|---|---|
| `api/routes/health.py` | `GET /api/health` service health check |
| `api/routes/conversations.py` | Conversation creation, retrieval, messages, and requirement confirmation |
| `api/routes/itineraries.py` | Immediate itinerary generation, queued jobs, job polling, SSE job events, and itinerary fetch |
| `api/routes/google_maps.py` | Sanitized Google Maps probe endpoint |
| `api/dependencies.py` | Settings, repositories, services, rate limiters, and Google Maps client construction |

The dependency module uses process-local cached factories for settings, SQLite database handles, repositories, services, and rate limiters. Google Maps clients are request-scoped because they wrap an `httpx.AsyncClient`.

### 4.2 Domain Layer

The domain layer is implemented under `src/smartour/domain`. It contains Pydantic models and small domain behaviors.

Key models:

- `TravelRequirement`: canonical structured requirements collected from conversation.
- `TravelRequirementUpdate`: partial requirement updates produced by extractors.
- `Conversation`: conversation state, messages, timestamps, and requirement snapshot.
- `ItineraryJob`: queued, running, succeeded, or failed itinerary generation job.
- `Itinerary`: generated plan with hotels, days, items, routes, photos, and guide markdown.

Domain models are not database table classes. They are serialized into repository payloads, which keeps application logic decoupled from SQLite schema details.

### 4.3 Application Layer

The application layer is implemented under `src/smartour/application`. It coordinates use cases and owns business flow.

| Service | Responsibility |
|---|---|
| `ConversationService` | Creates conversations, processes user turns, merges requirement updates, asks for missing slots, and confirms complete requirements |
| `RequirementExtractor` | Defines the extraction interface and development fallback extractor |
| `ItineraryJobService` | Creates queued jobs, enforces rate limits, runs jobs, and updates conversation state when jobs finish |
| `PlanningService` | Resolves destinations, discovers hotels and places, scores candidates, builds daily routes, supplements photos, renders guide markdown, and persists itineraries |

The conversation flow is intentionally stateful:

```text
collecting_requirements
  -> confirming_requirements
  -> planning
  -> ready_for_review
```

Failures move the conversation to `failed`. A conversation can only create an itinerary job after all required requirement slots are present.

### 4.4 Infrastructure Layer

The infrastructure layer is implemented under `src/smartour/infrastructure`. It provides local persistence, Google API cache and metrics storage, and rate limiting.

SQLite is the current persistence engine. The schema is created lazily on first connection and includes:

- `conversations`
- `itineraries`
- `itinerary_jobs`
- `google_api_cache_entries`
- `google_api_request_metrics`
- `rate_limit_events`

The repository classes store full domain payloads as JSON. This keeps the MVP schema compact and makes domain model iteration cheap. The trade-off is lower queryability for analytics or reporting. If the product needs rich search, reporting, or multi-user dashboards later, the JSON payloads should be complemented by normalized projections rather than replacing the domain models with database records.

### 4.5 Integration Layer

The integration layer is implemented under `src/smartour/integrations`.

Google Maps integration is split by service:

- `google_maps/places.py`
- `google_maps/routes.py`
- `google_maps/geocoding.py`
- `google_maps/timezone.py`
- `google_maps/client.py`
- `google_maps/field_masks.py`

`GoogleMapsHttpClient` centralizes API keys, field masks, request hashing, caching, metrics, status error handling, invalid JSON handling, and sanitized exceptions. The service-specific clients expose higher-level operations so planning code does not construct raw HTTP requests directly.

Requirement extraction uses a supervised token-classification model when `REQUIREMENT_MODEL_PATH` points to a local model artifact. When no model path is configured, the backend uses an explicit development fallback extractor so local conversation tests and demos can still run without a trained artifact.

## 5. Frontend Architecture

The frontend is a Next.js application under `app/`.

Main responsibilities:

- Render the travel planning workspace.
- Maintain local UI state for messages, selected day, expanded route rows, active job, and generated itinerary.
- Call the FastAPI backend through a typed API client.
- Display requirement collection, planning status, itinerary summaries, route stops, photos, and route maps.
- Load Google Maps JavaScript only when a browser key is available.

Important files:

| File | Responsibility |
|---|---|
| `app/src/app/page.tsx` | Main Smartour workspace and interaction flow |
| `app/src/lib/smartourApi.tsx` | Typed backend API client and frontend response contracts |
| `app/src/components/DailyPhotoGallery.tsx` | Itinerary photo display and fallback states |
| `app/src/components/RouteMap.tsx` | Route leg map rendering and fallback route preview |
| `app/src/lib/googleMapsLoader.tsx` | Browser Google Maps loader |
| `app/src/lib/placePhotos.tsx` | Google Places photo URL helpers |
| `app/src/lib/routeGeometry.tsx` | Route geometry helpers for fallback display |

The frontend currently polls itinerary jobs through `GET /api/itinerary-jobs/{jobId}`. The backend also exposes an SSE endpoint at `GET /api/itinerary-jobs/{jobId}/events`, so the frontend can switch from polling to `EventSource` when streaming UX becomes a priority.

## 6. Main Runtime Flow

### 6.1 Requirement Collection

```text
User submits a message
  -> frontend POSTs /api/conversations or /api/conversations/{id}/messages
  -> ConversationService stores the user message
  -> RequirementExtractor returns a partial TravelRequirementUpdate
  -> TravelRequirement.merge updates canonical state
  -> ConversationService checks missing required slots
  -> backend returns ConversationResponse
  -> frontend updates the chat panel and requirement summary
```

Required slots are:

- destination
- trip dates or trip length
- travelers
- budget level
- travel pace
- interests
- hotel area
- transportation mode

When all required slots are present, the conversation moves to `confirming_requirements` and the frontend enables the confirmation action.

### 6.2 Itinerary Job Generation

```text
User confirms requirements
  -> frontend POSTs /api/conversations/{id}/confirm
  -> frontend POSTs /api/conversations/{id}/itinerary-jobs
  -> ItineraryJobService checks conversation and IP rate limits
  -> backend persists a queued job
  -> FastAPI background task runs the job
  -> PlanningService generates and stores the itinerary
  -> ItineraryJobService marks the job succeeded or failed
  -> frontend polls job status and fetches the itinerary when ready
```

This job model avoids tying slow Google Maps and itinerary work to one blocking browser request.

### 6.3 Planning Pipeline

The planning service uses a pragmatic pipeline:

1. Resolve the destination with geocoding.
2. Discover hotels, attractions, and restaurants through Places API queries.
3. Normalize Google payloads into `PlaceRecommendation` models.
4. Score places by preference match, rating quality, review count, budget fit, distance, and route friendliness.
5. Cluster attractions by geography and theme.
6. Select daily items according to trip length and travel pace.
7. Select lunch and dinner candidates near daily clusters.
8. Build route legs with Routes API.
9. Supplement itinerary items with photo metadata.
10. Render a guide markdown string.
11. Persist the final `Itinerary`.

The current implementation favors clear product behavior and maintainable heuristics over mathematical route optimization. More advanced optimization can be introduced later behind the planning service without changing the public API contract.

## 7. API Surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Return basic service health |
| `POST` | `/api/conversations` | Create a conversation, optionally with an initial message |
| `GET` | `/api/conversations/{conversationId}` | Fetch a conversation |
| `POST` | `/api/conversations/{conversationId}/messages` | Send a user message |
| `POST` | `/api/conversations/{conversationId}/confirm` | Confirm complete requirements |
| `POST` | `/api/conversations/{conversationId}/itinerary` | Generate an itinerary immediately |
| `POST` | `/api/conversations/{conversationId}/itinerary-jobs` | Queue itinerary generation |
| `GET` | `/api/itinerary-jobs/{jobId}` | Fetch job state |
| `GET` | `/api/itinerary-jobs/{jobId}/events` | Stream job updates as server-sent events |
| `GET` | `/api/itineraries/{itineraryId}` | Fetch a generated itinerary |
| `GET` | `/api/google-maps/probe` | Run a sanitized Google Maps diagnostic probe |

Errors are normalized at the route boundary. Incomplete planning input returns `409`, rate limits return `429`, missing resources return `404`, and external service failures return `502`.

## 8. Configuration

Backend settings are loaded from environment variables and `.env` through `pydantic-settings`.

| Variable | Required | Purpose |
|---|---:|---|
| `GOOGLE_MAPS_API_KEY` | Yes | Server-side Google Maps Platform key |
| `SMARTOUR_SQLITE_PATH` | No | SQLite path, default `data/smartour.sqlite3` |
| `SMARTOUR_CORS_ALLOWED_ORIGINS` | No | Comma-separated CORS allowlist |
| `GOOGLE_MAPS_TIMEOUT_SECONDS` | No | Google Maps HTTP timeout |
| `GOOGLE_MAPS_CACHE_TTL_SECONDS` | No | Default cache TTL for Google API responses |
| `GOOGLE_MAPS_ROUTES_CACHE_TTL_SECONDS` | No | Routes-specific cache TTL |
| `REQUIREMENT_MODEL_PATH` | No | Local supervised requirement model artifact path |
| `REQUIREMENT_MODEL_CONFIDENCE_THRESHOLD` | No | Minimum average slot-span confidence, default `0.35` |
| `REQUIREMENT_MODEL_DEVELOPMENT_FALLBACK_ENABLED` | No | Enables the deterministic development fallback when no model path is configured |
| `ITINERARY_JOB_CONVERSATION_RATE_LIMIT_COUNT` | No | Per-conversation job limit |
| `ITINERARY_JOB_IP_RATE_LIMIT_COUNT` | No | Per-IP job limit |
| `ITINERARY_JOB_RATE_LIMIT_WINDOW_SECONDS` | No | Rate limit window |

Frontend configuration:

| Variable | Required | Purpose |
|---|---:|---|
| `NEXT_PUBLIC_SMARTOUR_API_BASE_URL` | No | Backend API base URL, default `http://127.0.0.1:8000/api` |
| `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY` | No | Browser-safe key for Maps JavaScript rendering |
| `NEXT_PUBLIC_GOOGLE_MAPS_MAP_ID` | No | Browser map ID for Maps JavaScript Advanced Marker rendering; defaults to Google's demo map ID when omitted |

`app/next.config.ts` can read a repository-level `.env` file to populate `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY` from either `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY` or `GOOGLE_MAPS_API_KEY`, and `NEXT_PUBLIC_GOOGLE_MAPS_MAP_ID` from either `NEXT_PUBLIC_GOOGLE_MAPS_MAP_ID` or `GOOGLE_MAPS_MAP_ID`. In production, use a separate browser-restricted key instead of reusing a server key.

## 9. Data and State Boundaries

The backend state boundary is SQLite. The frontend does not own canonical trip state; it mirrors the latest backend response and keeps local UI-only state.

Canonical backend state:

- Conversation message history.
- Current requirement snapshot.
- Conversation lifecycle state.
- Itinerary job lifecycle state.
- Generated itinerary payloads.
- Google API cache entries and request metrics.
- Rate limit events.

Local frontend-only state:

- Current text input.
- Visible chat transcript derived from responses.
- Active day tab.
- Expanded route row.
- Loading and error display state.

## 10. Quality and Verification

Backend checks:

```text
uv run pytest
uv run ruff check src tests
uv run mypy src tests
```

Frontend checks from `app/`:

```text
pnpm exec tsc --noEmit
pnpm exec eslint .
```

The current test suite focuses on persistence, conversation handling, requirement extraction, itinerary jobs, Google Maps clients, and planning behavior. Future high-value additions are frontend tests for API states and backend route tests for job failure, rate limiting, and Google probe behavior.

## 11. Extension Guidelines

Use these rules when extending the architecture:

- Put HTTP request handling in `api`, not in services.
- Put use-case orchestration in `application`, not in route handlers.
- Keep canonical business models in `domain`.
- Put storage details in `infrastructure`.
- Keep provider-specific request and response handling in `integrations`.
- Add normalized database projections only when query needs justify them.
- Keep the frontend API types aligned with backend response models.
- Confirm requirements before launching expensive planning work.
- Cache idempotent Google API calls and keep field masks minimal.
- Prefer small planning heuristics with tests before adding broad abstractions.

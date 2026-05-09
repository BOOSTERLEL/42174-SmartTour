# Backend Design for Conversational Travel Planning

## 1. Scope

This document designs the backend for a product that uses multi-turn conversation to collect travel requirements, calls Google Maps Platform APIs to discover real places and routes, and returns a complete travel guide with hotels, attractions, restaurants, and daily travel routes.

The current repository is a Python 3.12 project with no backend application code yet. The design assumes a Python backend and proposes a clean directory structure for the first implementation.

## 2. Assumptions

- The first version targets a single traveler or small group planning a leisure trip.
- The itinerary is generated for one destination city or a small destination area.
- The system recommends hotels, attractions, restaurants, and daily routes, but does not book hotels, tickets, tables, flights, trains, or cars.
- Google Maps Platform is the source of truth for place identity, place metadata, coordinates, map links, route distances, and travel times.
- A large language model is used for requirement extraction, clarification, ranking explanation, and final guide writing, but deterministic backend code owns state, validation, scoring, and API calls.
- The backend calls Google Maps APIs from the server. The frontend must not receive unrestricted server-side API keys.
- The MVP can use a synchronous API for short plans, but full itinerary generation should be modeled as a background job because Maps calls and LLM generation can exceed normal request latency.

## 3. Simpler MVP Path

The smallest useful version should avoid complex trip optimization. It should:

1. Collect required travel slots through conversation.
2. Search candidate hotels, attractions, and restaurants with Places API (New).
3. Enrich selected candidates with Place Details (New).
4. Group attractions and restaurants into days by area and preference.
5. Use Routes API for daily leg distances and ETAs.
6. Generate one structured itinerary and one human-readable guide.

Route Optimization API should be deferred unless the product needs strict time windows, many stops, multi-vehicle logic, or advanced optimization constraints.

## 4. Google Maps Platform API Strategy

### 4.1 Places API (New)

Use Places API (New) for place discovery and enrichment.

Recommended use cases:

- `Text Search (New)` for broad discovery, such as "family friendly attractions in Kyoto", "moderate hotels near Shinjuku Station", or "vegetarian restaurants near Sydney CBD".
- `Nearby Search (New)` for area-based recommendations around a hotel, station, attraction cluster, or route stop.
- `Place Details (New)` after a candidate has a `placeId`.
- `Place Photos (New)` only when the frontend needs images.
- `Search along route` for road trips or meals and stops near a computed route.

Important implementation rules:

- Always send `X-Goog-FieldMask`; Places API (New) has no default returned fields and omitting the field mask returns an error.
- Do not use `*` field masks in production.
- Request only fields needed for the current stage. Search can use small fields first, then Place Details can enrich shortlisted places.
- Store and reuse Google `placeId` values as stable place references.

Use staged field masks. Broad discovery should return enough data to identify and locate candidates. Ranking and final rendering should enrich only shortlisted places.

```text
Discovery search fields:
places.id,places.displayName,places.formattedAddress,places.location,places.primaryType,places.types,nextPageToken

Ranking enrichment fields:
id,displayName,formattedAddress,location,googleMapsUri,rating,userRatingCount,priceLevel,regularOpeningHours,currentOpeningHours,businessStatus,types

Final detail fields:
id,displayName,formattedAddress,location,googleMapsUri,websiteUri,rating,userRatingCount,priceLevel,regularOpeningHours,currentOpeningHours,businessStatus,types,photos,utcOffsetMinutes
```

### 4.2 Routes API

Use Routes API for real travel time, distance, daily route legs, and route polylines.

Recommended use cases:

- `Compute Routes` for a day's ordered stops.
- `Compute Routes` with waypoint optimization for a moderate number of daily stops when the order is flexible.
- `Compute Route Matrix` for candidate scoring and grouping, because it returns many origin-to-destination durations and distances in one server-side request.
- Keep route matrix requests within the documented route element limit. At the time of research, Compute Route Matrix supports up to 625 route elements per server-side request.

Important implementation rules:

- Always send a Routes response field mask; there is no default returned field list.
- Keep route field masks minimal to reduce latency and cost.
- Support `DRIVE`, `WALK`, `BICYCLE`, and `TRANSIT` where available and appropriate.
- Use route polylines only when needed by the frontend or for Places search along a route.

Initial field masks:

```text
Compute Routes:
routes.distanceMeters,routes.duration,routes.staticDuration,routes.polyline.encodedPolyline,routes.legs.distanceMeters,routes.legs.duration,routes.legs.startLocation,routes.legs.endLocation

Compute Route Matrix:
originIndex,destinationIndex,duration,distanceMeters,status,condition
```

### 4.3 Geocoding API

Use Geocoding API to normalize free-form destination input, convert addresses to coordinates, and resolve coordinates or Place IDs into readable addresses.

Recommended use cases:

- Convert user-provided destination names into canonical location data.
- Validate hotel addresses or custom start/end points.
- Convert manually entered meeting points into coordinates before routing.

### 4.4 Time Zone API

Use Time Zone API when itinerary dates and local opening hours matter across time zones.

Recommended use cases:

- Convert user travel dates into destination-local dates and times.
- Avoid scheduling items outside destination-local opening hours.
- Show local time labels in generated guides.

### 4.5 Route Optimization API

Do not use Route Optimization API in the MVP. It is designed for optimized route plans with objectives and constraints such as time windows, vehicle capacity, driver hours, and multiple vehicles. A tourist itinerary generally needs recommendation quality, opening-hour checks, and reasonable pacing more than fleet-grade optimization.

Consider Route Optimization API later if the product adds:

- Strict visit time windows.
- Large stop counts per day.
- Multi-city road trips with many required stops.
- Group transportation or fleet-like constraints.

### 4.6 Frontend Map Display

The backend should return structured coordinates, Google Maps URLs, and encoded polylines. The frontend can render them with Maps JavaScript API or link out through Google Maps URLs. The server-side backend should not expose unrestricted web service keys to the browser.

## 5. Backend Architecture

### 5.1 Recommended Stack

- Runtime: Python 3.12.
- Web framework: FastAPI or another ASGI framework.
- Validation: Pydantic models.
- HTTP client: `httpx`.
- Database: PostgreSQL for production; SQLite is acceptable for coursework/demo.
- Cache: Redis for API response cache, rate limiting, and job status; in-memory cache is acceptable only for local demo.
- Background jobs: RQ, Celery, Dramatiq, or a simple async worker. Keep the first implementation simple.
- LLM integration: provider-agnostic adapter behind an interface.

### 5.2 High-Level Components

```text
Client
  |
  v
API layer
  |
  +-- Conversation service
  |     +-- Requirement extractor
  |     +-- Clarification policy
  |
  +-- Planning service
  |     +-- Candidate discovery
  |     +-- Candidate scoring
  |     +-- Itinerary scheduler
  |     +-- Route planner
  |     +-- Guide renderer
  |
  +-- Google Maps integration
  |     +-- Places API client
  |     +-- Routes API client
  |     +-- Geocoding API client
  |     +-- Time Zone API client
  |
  +-- Persistence
        +-- Conversations
        +-- Messages
        +-- Requirement snapshots
        +-- Place cache
        +-- Itinerary artifacts
```

### 5.3 State Ownership

The LLM should not own canonical state. The backend should store a structured `TravelRequirement` snapshot after every user turn. The LLM can propose updates, but the backend validates and merges them.

Conversation states:

```text
collecting_requirements
confirming_requirements
planning
ready_for_review
revising
completed
failed
```

## 6. Conversation Design

### 6.1 Required Slots

The backend should not call the expensive planning pipeline until the required slots are present.

Required:

- Destination.
- Trip dates or trip length.
- Number and type of travelers.
- Approximate budget level.
- Travel pace.
- Main interests.
- Hotel preference or acceptable area.
- Primary transportation mode.

Strongly recommended:

- Food preferences and dietary restrictions.
- Mobility or accessibility constraints.
- Must-see places.
- Places to avoid.
- Daily start and end time.
- Preferred language for the final guide.

### 6.2 Turn Handling

For each user message:

1. Save the raw message.
2. Extract possible slot updates with the LLM.
3. Validate extracted values with deterministic code.
4. Merge updates into the current requirement snapshot.
5. Detect conflicts, such as "budget hotel" plus "five-star only".
6. Decide whether to ask a clarification, confirm the summary, or start planning.
7. Return a concise assistant response.

Clarification policy:

- Ask at most two questions per turn.
- Ask about blockers before preferences.
- Do not ask for optional details if a reasonable default exists.
- Confirm all important requirements before calling Maps APIs for a full plan.

## 7. Planning Pipeline

### 7.1 Pipeline Steps

1. Normalize destination.
   - Use Geocoding API to resolve the destination.
   - Store canonical name, coordinates, viewport, country/region, and time zone if available.

2. Build search queries.
   - Convert user interests into place categories and text queries.
   - Separate hotel, attraction, restaurant, and optional transit/station queries.

3. Discover candidates.
   - Use Places Text Search and Nearby Search.
   - Limit each category to a bounded number of candidates.
   - Cache raw API responses by normalized request hash.

4. Enrich candidates.
   - Use Place Details only for shortlisted candidates.
   - Include opening hours, rating, review count, price level, Maps URL, and photos when needed.

5. Score candidates.
   - Score by preference match, rating, review count confidence, price fit, distance to hotel area, opening-hour compatibility, and diversity.

6. Select hotel candidates.
   - Recommend a small set of hotel options, not just one.
   - Prefer areas that reduce total travel time and match budget/pacing.

7. Cluster itinerary days.
   - Group attractions and meals by geographic proximity and theme.
   - Avoid overloading a day. Keep realistic buffers.

8. Sequence daily stops.
   - Use Compute Route Matrix for travel-time-aware ordering.
   - Use Compute Routes for final daily leg details and route polyline.
   - Re-run sequencing if opening hours conflict with the first order.

9. Render the guide.
   - Return structured JSON for the frontend.
   - Generate a human-readable guide from the structured itinerary.
   - Include reasoning for choices, but keep operational data structured.

10. Persist artifacts.
   - Save the requirement snapshot, candidates used, route legs, generated guide, and revision history.

### 7.2 Candidate Scoring

Use deterministic scoring before LLM narration.

Example scoring inputs:

```text
score =
  preference_match * 0.30
  + rating_quality * 0.20
  + review_confidence * 0.10
  + budget_fit * 0.15
  + distance_fit * 0.15
  + opening_hours_fit * 0.10
```

The exact weights should be configuration, not hard-coded constants buried in business logic.

### 7.3 Scheduling Rules

Initial defaults:

- Breakfast: optional, only if user asks or hotel breakfast is not assumed.
- Lunch window: 11:30-13:30 local time.
- Dinner window: 18:00-20:30 local time.
- Attraction visit duration: 60-150 minutes depending on category.
- Restaurant visit duration: 60-90 minutes.
- Hotel check-in buffer: 60 minutes.
- Travel buffer: route duration plus 10-20 minutes.
- Maximum major attractions per day: 3-4 for normal pace, 2-3 for relaxed pace.

These defaults should be visible in configuration and overrideable by user preference.

## 8. API Design

### 8.1 REST Endpoints

```text
POST /api/conversations
POST /api/conversations/{conversationId}/messages
GET  /api/conversations/{conversationId}
POST /api/conversations/{conversationId}/confirm
POST /api/conversations/{conversationId}/itinerary-jobs
GET  /api/itinerary-jobs/{jobId}
GET  /api/itinerary-jobs/{jobId}/events
GET  /api/itineraries/{itineraryId}
POST /api/itineraries/{itineraryId}/revisions
GET  /api/health
```

### 8.2 Message Response Shape

```json
{
  "conversationId": "conv_123",
  "state": "collecting_requirements",
  "assistantMessage": "Which dates are you considering, and what budget level should I use?",
  "requirementSnapshot": {
    "destination": "Tokyo",
    "tripLengthDays": null,
    "travelers": {
      "adults": 2,
      "children": 0
    },
    "budgetLevel": null,
    "interests": ["food", "museums"]
  },
  "missingRequiredSlots": ["tripDatesOrLength", "budgetLevel"]
}
```

### 8.3 Itinerary Response Shape

```json
{
  "itineraryId": "itin_123",
  "title": "4-Day Tokyo Food and Culture Trip",
  "destination": {
    "name": "Tokyo, Japan",
    "placeId": "example",
    "location": {
      "latitude": 35.6764,
      "longitude": 139.6500
    }
  },
  "hotels": [
    {
      "placeId": "example",
      "name": "Example Hotel",
      "address": "Example address",
      "priceLevel": "PRICE_LEVEL_MODERATE",
      "rating": 4.4,
      "googleMapsUri": "https://maps.google.com/?cid=example"
    }
  ],
  "days": [
    {
      "date": "2026-06-10",
      "summary": "Museums and food around Ueno and Asakusa.",
      "items": [
        {
          "time": "09:30",
          "type": "attraction",
          "placeId": "example",
          "name": "Example Museum",
          "durationMinutes": 120
        }
      ],
      "route": {
        "travelMode": "TRANSIT",
        "distanceMeters": 8200,
        "durationSeconds": 3100,
        "encodedPolyline": "example"
      }
    }
  ],
  "guideMarkdown": "..."
}
```

## 9. Data Model

Core tables:

```text
users
conversations
messages
travel_requirement_snapshots
itinerary_jobs
itineraries
itinerary_days
itinerary_items
places
place_candidates
route_legs
google_api_cache_entries
```

Important fields:

- `conversations.state`: current conversation state.
- `messages.role`: `user`, `assistant`, or `system`.
- `travel_requirement_snapshots.payload`: validated JSON requirement model.
- `places.place_id`: Google place ID.
- `places.source_payload`: normalized subset of Google place data, not the full raw response unless needed.
- `route_legs.origin_place_id` and `route_legs.destination_place_id`: route endpoints.
- `google_api_cache_entries.expires_at`: TTL-based cache expiry.

## 10. Code Directory Design

Use a `src` layout for application code. The current `pyproject.toml` package discovery targets `scripts*`; implementation should update package discovery when code is added.

```text
src/
  smartour/
    __init__.py
    main.py
    api/
      __init__.py
      dependencies.py
      routes/
        __init__.py
        conversations.py
        health.py
        itineraries.py
    application/
      __init__.py
      conversation_service.py
      itinerary_renderer.py
      planning_service.py
      requirement_extractor.py
      revision_service.py
    core/
      __init__.py
      config.py
      errors.py
      logging.py
      security.py
    domain/
      __init__.py
      conversation.py
      itinerary.py
      place.py
      requirement.py
      route.py
    infrastructure/
      __init__.py
      cache.py
      database.py
      jobs.py
      repositories/
        __init__.py
        conversations.py
        itineraries.py
        places.py
    integrations/
      __init__.py
      google_maps/
        __init__.py
        cache_keys.py
        client.py
        field_masks.py
        geocoding.py
        places.py
        routes.py
        timezone.py
      llm/
        __init__.py
        client.py
        prompts.py
        schemas.py
tests/
  unit/
    test_requirement_merging.py
    test_candidate_scoring.py
    test_itinerary_scheduling.py
  integration/
    test_google_maps_client_contracts.py
  fixtures/
    google_maps/
      places_text_search.json
      place_details.json
      route_matrix.json
```

Directory responsibilities:

- `api`: HTTP route definitions and request/response models.
- `application`: orchestration services and use cases.
- `domain`: pure business models and rules.
- `integrations`: external API clients and provider-specific schemas.
- `infrastructure`: database, cache, job queue, repositories.
- `tests`: unit tests first, integration tests gated by environment variables.

## 11. Google Maps Client Design

The Google Maps integration should expose high-level methods instead of leaking HTTP details into services.

Suggested methods:

```text
GooglePlacesClient.search_text(...)
GooglePlacesClient.search_nearby(...)
GooglePlacesClient.get_place_details(...)
GooglePlacesClient.get_photo_metadata(...)
GoogleRoutesClient.compute_routes(...)
GoogleRoutesClient.compute_route_matrix(...)
GoogleGeocodingClient.geocode(...)
GoogleTimeZoneClient.get_time_zone(...)
```

Client requirements:

- Centralize base URLs, authentication headers, timeouts, retries, and field masks.
- Use short request timeouts and bounded retries for `429` and `5xx`.
- Never let client request bodies directly choose arbitrary Google endpoints.
- Normalize API errors into internal exception types.
- Add request logging without logging API keys.
- Cache idempotent GET/POST lookups by normalized request body and field mask.

## 12. Security and Cost Controls

Security:

- Store Google Maps API keys outside source code, preferably in environment variables or secret manager.
- Restrict server-side Maps keys by API and, where possible, server public IP address.
- Use separate keys/projects for frontend map rendering and backend web service calls.
- Disable unused Google Maps services in the Google Cloud project.
- Do not expose server-side API keys to browsers or mobile clients.
- Rate-limit itinerary generation by user and IP.
- Sanitize LLM outputs before using them as API parameters.

Cost controls:

- Use field masks for every Places and Routes request.
- Do not use wildcard field masks in production.
- Avoid requesting rating, price, photos, reviews, or website fields during broad discovery if a later shortlist enrichment step can fetch them.
- Confirm requirements before calling the full planning pipeline.
- Cap search pages and candidate counts per category.
- Cache Place Details and route matrix responses with a TTL.
- Fetch photos lazily.
- Track per-job Maps request counts for observability.

## 13. Failure Handling

Expected failure modes:

- Ambiguous destination.
- No hotel or restaurant results after filters.
- Candidate place is permanently closed.
- Opening hours conflict with planned time.
- Route unavailable for selected travel mode.
- Google API quota or billing error.
- Requirement model extraction returns invalid or contradictory updates.

Fallback behavior:

- Ask a clarification for ambiguous destinations.
- Relax filters gradually and explain the relaxation.
- Replace closed places before rendering the final guide.
- Switch travel mode only with explicit user confirmation when it materially changes the trip.
- Return partial planning errors with a retryable job state rather than losing conversation state.

## 14. Testing Strategy

Unit tests:

- Requirement extraction merge logic.
- Missing-slot detection.
- Conflict detection.
- Candidate scoring.
- Day clustering and scheduling.
- Opening-hour compatibility.

Contract tests:

- Google Maps request builders create the expected URL, method, headers, field masks, and body.
- Google Maps response parsers handle real fixture shapes.

Integration tests:

- Run only when `GOOGLE_MAPS_API_KEY` is present.
- Use low-cost queries and strict candidate limits.
- Do not run in default CI unless explicitly enabled.

End-to-end tests:

- Simulate a multi-turn conversation.
- Confirm requirement summary.
- Generate a mocked itinerary from fixtures.
- Verify the response contains hotels, attractions, restaurants, daily route data, and guide Markdown.

## 15. Implementation Phases

### Phase 1: Conversation Core

- Add FastAPI app skeleton.
- Add conversation endpoints.
- Add requirement schema and merge logic.
- Add mocked LLM requirement extraction.
- Persist conversations and messages.

Verification:

- Unit tests for required slots and merging.
- API tests for starting a conversation and sending messages.

### Phase 2: Google Maps Integration

- Add Google Maps client wrappers.
- Add Places Text Search, Place Details, Geocoding, and Routes clients.
- Add field mask constants.
- Add fixture-based parser tests.

Verification:

- Contract tests with fixtures.
- Optional live integration test behind `GOOGLE_MAPS_API_KEY`.

### Phase 3: Planning Pipeline

- Add candidate discovery and scoring.
- Add day clustering and routing.
- Add itinerary persistence.
- Add background job execution.

Verification:

- Golden tests for itinerary output.
- Route matrix fixture tests.

### Phase 4: Revision and Production Hardening

- Add itinerary revision endpoint.
- Add cache TTLs, rate limits, and request metrics.
- Add API key restriction checklist to deployment docs.
- Add better error reporting and retries.

Verification:

- End-to-end mocked conversation-to-itinerary test.
- Failure-mode tests for empty Places results and route errors.

## 16. Open Decisions

- LLM provider and model choice.
- Database choice for the first runnable version: SQLite for demo or PostgreSQL from day one.
- Whether itinerary generation should stream progress by SSE or simple polling.
- Whether the frontend will render Maps JavaScript API directly or mostly use Google Maps URLs.
- Whether hotel recommendations should include booking links from a separate provider. Google Maps can identify hotels, but booking and price availability usually require another integration.

## 17. References

- Google Maps Platform Documentation: https://developers.google.com/maps/documentation/
- Places API (New) overview: https://developers.google.com/maps/documentation/places/web-service/overview
- Places Text Search (New): https://developers.google.com/maps/documentation/places/web-service/text-search
- Place Details (New): https://developers.google.com/maps/documentation/places/web-service/place-details
- Search along route: https://developers.google.com/maps/documentation/places/web-service/search-along-route
- Routes API Compute Routes overview: https://developers.google.com/maps/documentation/routes/overview
- Routes API field masks: https://developers.google.com/maps/documentation/routes/choose_fields
- Routes API Compute Route Matrix overview: https://developers.google.com/maps/documentation/routes/compute-route-matrix-over
- Geocoding API overview: https://developers.google.com/maps/documentation/geocoding/overview
- Time Zone API overview: https://developers.google.com/maps/documentation/timezone/overview
- Route Optimization API overview: https://developers.google.com/maps/documentation/route-optimization/overview
- Google Maps Platform security guidance: https://developers.google.com/maps/api-security-best-practices

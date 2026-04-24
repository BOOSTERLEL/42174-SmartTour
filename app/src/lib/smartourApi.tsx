/**
 * Typed client for the Smartour backend API.
 */

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000/api";

/**
 * Supported conversation states returned by the backend.
 */
export type ConversationState =
  | "collecting_requirements"
  | "confirming_requirements"
  | "planning"
  | "ready_for_review"
  | "completed"
  | "failed";

/**
 * Supported itinerary generation job states returned by the backend.
 */
export type ItineraryJobStatus = "queued" | "running" | "succeeded" | "failed";

/**
 * Traveler counts collected during requirement intake.
 */
export type Travelers = {
  adults: number | null;
  children: number;
};

/**
 * Canonical travel requirement snapshot returned by the backend.
 */
export type TravelRequirement = {
  destination: string | null;
  trip_dates: string | null;
  trip_length_days: number | null;
  travelers: Travelers;
  budget_level: string | null;
  travel_pace: string | null;
  interests: string[];
  hotel_area: string | null;
  transportation_mode: string | null;
  food_preferences: string[];
  language: string;
};

/**
 * Conversation state response returned by the backend.
 */
export type ConversationResponse = {
  conversation_id: string;
  state: ConversationState;
  assistant_message: string | null;
  requirement_snapshot: TravelRequirement;
  missing_required_slots: string[];
};

/**
 * Geographic coordinates returned for places and destinations.
 */
export type Coordinates = {
  latitude: number;
  longitude: number;
};

/**
 * Normalized place recommendation returned by the itinerary planner.
 */
export type PlaceRecommendation = {
  place_id: string;
  name: string;
  category: string;
  address: string | null;
  location: Coordinates | null;
  google_maps_uri: string | null;
  rating: number | null;
  user_rating_count: number | null;
  price_level: string | null;
  types: string[];
  regular_opening_hours: Record<string, unknown> | null;
  current_opening_hours: Record<string, unknown> | null;
  score: number;
};

/**
 * Route leg returned between two itinerary places.
 */
export type RouteLeg = {
  origin_place_id: string;
  destination_place_id: string;
  travel_mode: string;
  distance_meters: number | null;
  duration_seconds: number | null;
  encoded_polyline: string | null;
};

/**
 * Aggregated route summary for one itinerary day.
 */
export type RouteSummary = {
  travel_mode: string;
  distance_meters: number;
  duration_seconds: number;
  legs: RouteLeg[];
};

/**
 * Scheduled itinerary item returned by the backend.
 */
export type ItineraryItem = {
  time: string;
  type: "attraction" | "lunch" | "dinner";
  place: PlaceRecommendation;
  duration_minutes: number;
};

/**
 * One day of a generated itinerary.
 */
export type ItineraryDay = {
  day_number: number;
  date: string | null;
  theme: string;
  summary: string;
  items: ItineraryItem[];
  route: RouteSummary | null;
};

/**
 * Generated itinerary returned by the backend.
 */
export type Itinerary = {
  id: string;
  conversation_id: string;
  title: string;
  destination_name: string;
  destination_location: Coordinates | null;
  hotels: PlaceRecommendation[];
  days: ItineraryDay[];
  guide_markdown: string;
  created_at: string;
};

/**
 * Itinerary generation job returned by the backend.
 */
export type ItineraryJob = {
  id: string;
  conversation_id: string;
  status: ItineraryJobStatus;
  itinerary_id: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
};

/**
 * Return the configured backend API base URL.
 *
 * @returns The normalized API base URL without a trailing slash.
 */
export function getApiBaseUrl(): string {
  return (
    process.env.NEXT_PUBLIC_SMARTOUR_API_BASE_URL ?? DEFAULT_API_BASE_URL
  ).replace(/\/$/, "");
}

/**
 * Create a new backend conversation.
 *
 * @param initialMessage - Optional initial user message.
 * @returns The created conversation response.
 */
export async function createConversation(
  initialMessage: string | null,
): Promise<ConversationResponse> {
  return requestJson<ConversationResponse>("/conversations", {
    body: { initial_message: initialMessage },
    method: "POST",
  });
}

/**
 * Send a user message to an existing conversation.
 *
 * @param conversationId - The conversation identifier.
 * @param message - The user message content.
 * @returns The updated conversation response.
 */
export async function sendConversationMessage(
  conversationId: string,
  message: string,
): Promise<ConversationResponse> {
  return requestJson<ConversationResponse>(
    `/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      body: { message },
      method: "POST",
    },
  );
}

/**
 * Confirm the collected requirements for a conversation.
 *
 * @param conversationId - The conversation identifier.
 * @returns The confirmed conversation response.
 */
export async function confirmConversation(
  conversationId: string,
): Promise<ConversationResponse> {
  return requestJson<ConversationResponse>(
    `/conversations/${encodeURIComponent(conversationId)}/confirm`,
    {
      method: "POST",
    },
  );
}

/**
 * Create an asynchronous itinerary generation job.
 *
 * @param conversationId - The conversation identifier.
 * @returns The queued itinerary job.
 */
export async function createItineraryJob(
  conversationId: string,
): Promise<ItineraryJob> {
  return requestJson<ItineraryJob>(
    `/conversations/${encodeURIComponent(conversationId)}/itinerary-jobs`,
    {
      method: "POST",
    },
  );
}

/**
 * Fetch an itinerary generation job by ID.
 *
 * @param jobId - The itinerary job identifier.
 * @returns The current itinerary job.
 */
export async function getItineraryJob(jobId: string): Promise<ItineraryJob> {
  return requestJson<ItineraryJob>(
    `/itinerary-jobs/${encodeURIComponent(jobId)}`,
  );
}

/**
 * Fetch a generated itinerary by ID.
 *
 * @param itineraryId - The itinerary identifier.
 * @returns The generated itinerary.
 */
export async function getItinerary(itineraryId: string): Promise<Itinerary> {
  return requestJson<Itinerary>(
    `/itineraries/${encodeURIComponent(itineraryId)}`,
  );
}

type RequestOptions = {
  body?: unknown;
  method?: "GET" | "POST";
};

/**
 * Execute a JSON request against the configured backend API.
 *
 * @param path - The backend API path.
 * @param options - Optional request options.
 * @returns The parsed JSON response.
 */
async function requestJson<ResponseBody>(
  path: string,
  options: RequestOptions = {},
): Promise<ResponseBody> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
    },
    method: options.method ?? "GET",
  });
  const payload = await parseJson(response);
  if (!response.ok) {
    throw new Error(readErrorMessage(payload, response.statusText));
  }
  return payload as ResponseBody;
}

/**
 * Parse a response body as JSON when present.
 *
 * @param response - The fetch response.
 * @returns The parsed JSON payload or null.
 */
async function parseJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }
  return JSON.parse(text) as unknown;
}

/**
 * Return a readable error message from a backend error payload.
 *
 * @param payload - The parsed error payload.
 * @param fallbackMessage - The fallback status message.
 * @returns A readable error message.
 */
function readErrorMessage(payload: unknown, fallbackMessage: string): string {
  if (isErrorPayload(payload)) {
    return payload.detail;
  }
  return fallbackMessage || "Request failed";
}

/**
 * Return whether a payload matches the FastAPI error shape.
 *
 * @param payload - The payload to inspect.
 * @returns True when the payload contains a string detail field.
 */
function isErrorPayload(payload: unknown): payload is { detail: string } {
  return (
    typeof payload === "object" &&
    payload !== null &&
    "detail" in payload &&
    typeof (payload as { detail: unknown }).detail === "string"
  );
}

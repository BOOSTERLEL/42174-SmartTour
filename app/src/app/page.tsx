"use client";

import type { FormEvent, ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import { useTheme } from "next-themes";
import {
  BarChart2,
  Bus,
  Calendar,
  Check,
  ChevronDown,
  Clock,
  Footprints,
  Hotel,
  LoaderCircle,
  Moon,
  MoreHorizontal,
  Send,
  Ship,
  Sparkles,
  Sun,
  User,
  Utensils,
} from "lucide-react";
import {
  confirmConversation,
  createConversation,
  createItineraryJob,
  getApiBaseUrl,
  getItinerary,
  getItineraryJob,
  type ConversationResponse,
  type Itinerary,
  type ItineraryDay,
  type ItineraryItem,
  type ItineraryJob,
  type RouteLeg,
  type TravelRequirement,
  sendConversationMessage,
} from "@/lib/smartourApi";
import { RouteMap } from "@/components/RouteMap";
import styles from "./page.module.css";

const DEFAULT_PROMPT =
  "We are 2 adults visiting Sydney for 3 days. We want a moderate budget, relaxed pace, museums, harbour views, food, a CBD hotel area, and transit.";
const JOB_POLL_INTERVAL_MS = 1500;
const JOB_MAX_POLLS = 180;

type UiMessage = {
  content: string;
  createdAt: string;
  id: string;
  role: "assistant" | "user";
  tone?: "error" | "normal";
};

type RequirementRow = {
  label: string;
  value: string;
};

/**
 * Render the Smartour frontend application.
 *
 * @returns The Smartour app page.
 */
export default function Home() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [messages, setMessages] = useState<UiMessage[]>(createInitialMessages);
  const [inputValue, setInputValue] = useState(DEFAULT_PROMPT);
  const [conversation, setConversation] = useState<ConversationResponse | null>(
    null,
  );
  const [job, setJob] = useState<ItineraryJob | null>(null);
  const [itinerary, setItinerary] = useState<Itinerary | null>(null);
  const [activeDayIndex, setActiveDayIndex] = useState(0);
  const [isSending, setIsSending] = useState(false);
  const [isPlanning, setIsPlanning] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    const frameId = window.requestAnimationFrame(() => {
      setMounted(true);
    });
    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, []);

  const activeDay = itinerary?.days[activeDayIndex] ?? null;
  const requirement = conversation?.requirement_snapshot ?? null;
  const requirementRows = useMemo(
    () => buildRequirementRows(requirement),
    [requirement],
  );
  const routeTotals = useMemo(() => summarizeRoutes(itinerary), [itinerary]);
  const canConfirm =
    conversation?.state === "confirming_requirements" &&
    conversation.missing_required_slots.length === 0 &&
    !isPlanning;
  const pageTitle = itinerary?.title ?? buildDraftTitle(requirement);

  /**
   * Submit the current user message to the backend conversation API.
   */
  async function handleSubmitMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedInput = inputValue.trim();
    if (!trimmedInput || isSending || isPlanning) {
      return;
    }
    setInputValue("");
    setIsSending(true);
    setErrorMessage(null);
    appendMessage("user", trimmedInput);
    try {
      const nextConversation =
        conversation === null
          ? await createConversation(trimmedInput)
          : await sendConversationMessage(
              conversation.conversation_id,
              trimmedInput,
            );
      setConversation(nextConversation);
      appendAssistantMessage(nextConversation);
    } catch (error) {
      handleRequestError(error);
    } finally {
      setIsSending(false);
    }
  }

  /**
   * Confirm requirements and start the backend itinerary generation job.
   */
  async function handleConfirmRequirements() {
    if (conversation === null || !canConfirm) {
      return;
    }
    setIsPlanning(true);
    setErrorMessage(null);
    try {
      const confirmedConversation = await confirmConversation(
        conversation.conversation_id,
      );
      setConversation(confirmedConversation);
      appendAssistantMessage(confirmedConversation);
      const queuedJob = await createItineraryJob(
        confirmedConversation.conversation_id,
      );
      setJob(queuedJob);
      appendMessage(
        "assistant",
        "Planning job queued. I am checking progress.",
      );
      const generatedItinerary = await pollJobUntilComplete(queuedJob.id);
      setItinerary(generatedItinerary);
      setActiveDayIndex(0);
      appendMessage("assistant", `${generatedItinerary.title} is ready.`);
    } catch (error) {
      handleRequestError(error);
    } finally {
      setIsPlanning(false);
    }
  }

  /**
   * Poll an itinerary job until it reaches a terminal state.
   *
   * @param jobId - The itinerary job identifier.
   * @returns The generated itinerary.
   */
  async function pollJobUntilComplete(jobId: string): Promise<Itinerary> {
    for (let pollIndex = 0; pollIndex < JOB_MAX_POLLS; pollIndex += 1) {
      const latestJob = await getItineraryJob(jobId);
      setJob(latestJob);
      if (latestJob.status === "succeeded" && latestJob.itinerary_id !== null) {
        return getItinerary(latestJob.itinerary_id);
      }
      if (latestJob.status === "failed") {
        throw new Error(
          latestJob.error_message ?? "Itinerary generation failed",
        );
      }
      await sleep(JOB_POLL_INTERVAL_MS);
    }
    throw new Error("Itinerary generation timed out");
  }

  /**
   * Append a local chat message.
   *
   * @param role - The message role.
   * @param content - The message content.
   * @param tone - Optional visual tone.
   */
  function appendMessage(
    role: UiMessage["role"],
    content: string,
    tone: UiMessage["tone"] = "normal",
  ) {
    setMessages((currentMessages) => [
      ...currentMessages,
      createUiMessage(role, content, tone),
    ]);
  }

  /**
   * Append the latest backend assistant message when present.
   *
   * @param nextConversation - The backend conversation response.
   */
  function appendAssistantMessage(nextConversation: ConversationResponse) {
    if (nextConversation.assistant_message !== null) {
      appendMessage("assistant", nextConversation.assistant_message);
    }
  }

  /**
   * Convert a failed request into local error state.
   *
   * @param error - The thrown error.
   */
  function handleRequestError(error: unknown) {
    const message =
      error instanceof Error ? error.message : "The backend request failed";
    setErrorMessage(message);
    appendMessage("assistant", message, "error");
  }

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.logo}>Smartour</div>
        <nav className={styles.navLinks} aria-label="Primary navigation">
          <a href="#" className={styles.navLink}>
            Trips
          </a>
          <a href="#" className={styles.navLink}>
            Explore
          </a>
          <a href="#" className={styles.navLink}>
            API
          </a>
        </nav>
        <div className={styles.headerActions}>
          {mounted ? (
            <button
              className="btn btn-ghost"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label="Toggle dark mode"
              type="button"
            >
              {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          ) : null}
          <button className="btn btn-secondary" type="button">
            Sign in
          </button>
          <button
            className="btn btn-primary"
            onClick={() => {
              setMessages(createInitialMessages());
              setConversation(null);
              setJob(null);
              setItinerary(null);
              setInputValue(DEFAULT_PROMPT);
              setErrorMessage(null);
            }}
            type="button"
          >
            New trip
          </button>
        </div>
      </header>

      <div className={styles.subHeader}>
        <WorkflowStep
          accentClass={styles.collect}
          isActive={
            conversation === null ||
            conversation.state === "collecting_requirements"
          }
          label="Collect"
          number="1"
        />
        <div className={styles.stepLine} />
        <WorkflowStep
          accentClass={styles.plan}
          isActive={isPlanning || conversation?.state === "planning"}
          label="Plan"
          number="2"
        />
        <div className={styles.stepLine} />
        <WorkflowStep
          accentClass={styles.review}
          isActive={
            itinerary !== null || conversation?.state === "ready_for_review"
          }
          label="Review"
          number="3"
        />
      </div>

      <main className={styles.mainContent}>
        <aside className={styles.leftSidebar}>
          <div className={`card-elevated ${styles.chatCard}`}>
            <div className={styles.chatHeader}>
              <div className={styles.chatHeaderTitle}>
                <Sparkles size={16} />
                <span className="text-body-semibold">Trip brief</span>
              </div>
              <button className="btn btn-ghost" type="button">
                <MoreHorizontal size={16} />
              </button>
            </div>

            <div className={styles.chatMessages}>
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
            </div>

            {canConfirm ? (
              <div className={styles.confirmBar}>
                <button
                  className="btn btn-primary"
                  disabled={isPlanning}
                  onClick={handleConfirmRequirements}
                  type="button"
                >
                  {isPlanning ? (
                    <LoaderCircle className={styles.spin} size={16} />
                  ) : (
                    <Check size={16} />
                  )}
                  Confirm and plan
                </button>
              </div>
            ) : null}

            <form
              className={styles.chatInputContainer}
              onSubmit={handleSubmitMessage}
            >
              <input
                aria-label="Travel requirement message"
                className={styles.chatInput}
                disabled={isSending || isPlanning}
                onChange={(event) => setInputValue(event.target.value)}
                placeholder="Type travel requirements..."
                type="text"
                value={inputValue}
              />
              <button
                aria-label="Send message"
                className={styles.sendBtn}
                disabled={isSending || isPlanning || !inputValue.trim()}
                type="submit"
              >
                {isSending ? (
                  <LoaderCircle className={styles.spin} size={18} />
                ) : (
                  <Send size={18} />
                )}
              </button>
            </form>
          </div>
        </aside>

        <section className={styles.middleColumn}>
          <div className={styles.titleArea}>
            <div>
              <h1 className="text-section">{pageTitle}</h1>
              <div className={styles.titleMeta}>
                {conversation?.state ?? "not started"}
              </div>
            </div>
            <StatusBadge
              icon={
                itinerary !== null ? (
                  <Check size={14} />
                ) : (
                  <Sparkles size={14} />
                )
              }
              label={buildStatusLabel(conversation, job, itinerary)}
            />
          </div>

          <div className={styles.tabs} role="tablist">
            {(itinerary?.days ?? createDraftDays(requirement)).map(
              (day, index) => (
                <button
                  className={`${styles.tab} ${
                    index === activeDayIndex ? styles.active : ""
                  }`}
                  key={`${day.day_number}-${day.theme}`}
                  onClick={() => setActiveDayIndex(index)}
                  role="tab"
                  type="button"
                >
                  Day {day.day_number} - {day.theme}
                </button>
              ),
            )}
          </div>

          <RouteMap activeDay={activeDay} isPlanning={isPlanning} />

          <div>
            <h2 className={styles.routeStopsTitle}>Route stops</h2>
            <div className={styles.routeList}>
              {activeDay === null ? (
                <EmptyRouteState isPlanning={isPlanning} />
              ) : (
                activeDay.items.map((item, index) => (
                  <RouteItemRow
                    index={index}
                    item={item}
                    key={`${item.place.place_id}-${item.time}`}
                    leg={activeDay.route?.legs[index] ?? null}
                  />
                ))
              )}
            </div>
          </div>
        </section>

        <aside className={styles.rightSidebar}>
          <div className="card">
            <div className={styles.summarySection}>
              <h2 className="text-body-semibold">Itinerary summary</h2>
              {errorMessage !== null ? (
                <div className={styles.errorText}>{errorMessage}</div>
              ) : null}
            </div>

            <SummaryBlock icon={<Hotel size={16} />} title="Stay">
              {itinerary?.hotels[0] ? (
                <div className={styles.stayInfo}>
                  <span className={styles.summaryValue}>
                    {itinerary.hotels[0].name}
                  </span>
                  <span className={styles.stayAddress}>
                    {itinerary.hotels[0].address ?? "Address unavailable"}
                  </span>
                  <span className="text-secondary">
                    {formatTravelers(requirement)}
                  </span>
                </div>
              ) : (
                <RequirementRows rows={requirementRows} />
              )}
            </SummaryBlock>

            <SummaryBlock icon={<Calendar size={16} />} title="Daily themes">
              <div className={styles.summaryStack}>
                {(itinerary?.days ?? createDraftDays(requirement)).map(
                  (day) => (
                    <div className={styles.summaryRow} key={day.day_number}>
                      <span className="text-link">Day {day.day_number}</span>
                      <span>{day.theme}</span>
                    </div>
                  ),
                )}
              </div>
            </SummaryBlock>

            <SummaryBlock icon={<Utensils size={16} />} title="Restaurants">
              <div className={styles.summaryStack}>
                {selectRestaurants(itinerary).map((restaurant) => (
                  <div className={styles.summaryRow} key={restaurant.label}>
                    <span className="text-link">{restaurant.label}</span>
                    <span>{restaurant.name}</span>
                  </div>
                ))}
              </div>
            </SummaryBlock>

            <SummaryBlock icon={<Clock size={16} />} title="Route overview">
              <div className={styles.summaryStack}>
                <SummaryRow
                  label="Total duration"
                  value={formatDuration(routeTotals.durationSeconds)}
                />
                <SummaryRow
                  label="Total distance"
                  value={formatDistance(routeTotals.distanceMeters)}
                />
                <SummaryRow
                  label="Route legs"
                  value={`${routeTotals.legCount}`}
                />
              </div>
            </SummaryBlock>

            <SummaryBlock icon={<BarChart2 size={16} />} title="Backend">
              <div className={styles.summaryStack}>
                <SummaryRow label="API base" value={getApiBaseUrl()} />
                <SummaryRow
                  label="Conversation"
                  value={conversation?.conversation_id ?? "-"}
                />
                <SummaryRow label="Job status" value={job?.status ?? "-"} />
              </div>
            </SummaryBlock>

            <div className={styles.summarySection}>
              <div className="text-caption text-secondary">
                Route distances and durations come from the generated backend
                itinerary.
              </div>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

type WorkflowStepProps = {
  accentClass: string;
  isActive: boolean;
  label: string;
  number: string;
};

/**
 * Render one workflow step in the app header.
 *
 * @param props - The workflow step props.
 * @returns A workflow step element.
 */
function WorkflowStep({
  accentClass,
  isActive,
  label,
  number,
}: WorkflowStepProps) {
  return (
    <div
      className={`${styles.step} ${accentClass} ${
        isActive ? styles.stepActive : ""
      }`}
    >
      <div className={styles.stepNumber}>{number}</div>
      {label}
    </div>
  );
}

type MessageBubbleProps = {
  message: UiMessage;
};

/**
 * Render a local chat message.
 *
 * @param props - The message bubble props.
 * @returns A chat message element.
 */
function MessageBubble({ message }: MessageBubbleProps) {
  const isAssistant = message.role === "assistant";
  return (
    <div className={styles.message}>
      <div className={styles.messageHeader}>
        <div
          className={`${styles.avatar} ${isAssistant ? styles.assistantAvatar : ""}`}
        >
          {isAssistant ? <Sparkles size={14} /> : <User size={14} />}
        </div>
        <span className="text-body-medium">
          {isAssistant ? "Smartour" : "You"}
        </span>
        <span>{message.createdAt}</span>
      </div>
      <div
        className={`${styles.messageBubble} ${
          isAssistant ? styles.assistantBubble : ""
        } ${message.tone === "error" ? styles.errorBubble : ""}`}
      >
        {message.content}
      </div>
    </div>
  );
}

type StatusBadgeProps = {
  icon: ReactNode;
  label: string;
};

/**
 * Render a compact status badge.
 *
 * @param props - The status badge props.
 * @returns A status badge element.
 */
function StatusBadge({ icon, label }: StatusBadgeProps) {
  return (
    <div className="badge">
      {icon}
      <span className={styles.badgeText}>{label}</span>
    </div>
  );
}

type EmptyRouteStateProps = {
  isPlanning: boolean;
};

/**
 * Render the route list empty state.
 *
 * @param props - The empty state props.
 * @returns A route list empty state element.
 */
function EmptyRouteState({ isPlanning }: EmptyRouteStateProps) {
  return (
    <div className={styles.emptyRouteState}>
      {isPlanning ? <LoaderCircle className={styles.spin} size={18} /> : null}
      <span>
        {isPlanning
          ? "Waiting for the backend itinerary job."
          : "Send requirements to start a planning conversation."}
      </span>
    </div>
  );
}

type RouteItemRowProps = {
  index: number;
  item: ItineraryItem;
  leg: RouteLeg | null;
};

/**
 * Render one itinerary item row with route leg details.
 *
 * @param props - The route item row props.
 * @returns A route item row element.
 */
function RouteItemRow({ index, item, leg }: RouteItemRowProps) {
  return (
    <div className={styles.routeItem}>
      <div className={styles.routeNumber}>{index + 1}</div>
      <div>
        <div className={styles.routePlaceName}>{item.place.name}</div>
        <div className={styles.routeAddress}>
          {item.place.address ?? item.place.category}
        </div>
      </div>
      <div className={styles.routeTime}>
        <span>{item.time}</span>
        <span>{item.duration_minutes} min</span>
      </div>
      <div className={styles.routeTransit}>
        <div className={styles.routeTransitIcon}>
          {renderTravelIcon(leg?.travel_mode ?? "walk")}
          {formatDuration(leg?.duration_seconds ?? 0)}
        </div>
        <div className={styles.badgeSmall}>
          {formatTravelMode(leg?.travel_mode)}
        </div>
        <ChevronDown size={16} />
      </div>
    </div>
  );
}

type SummaryBlockProps = {
  children: ReactNode;
  icon: ReactNode;
  title: string;
};

/**
 * Render a summary section.
 *
 * @param props - The summary block props.
 * @returns A summary block element.
 */
function SummaryBlock({ children, icon, title }: SummaryBlockProps) {
  return (
    <div className={styles.summarySection}>
      <div className={styles.summaryTitle}>
        <span className="text-secondary">{icon}</span>
        {title}
      </div>
      {children}
    </div>
  );
}

type RequirementRowsProps = {
  rows: RequirementRow[];
};

/**
 * Render collected requirement rows.
 *
 * @param props - The requirement rows props.
 * @returns Requirement rows.
 */
function RequirementRows({ rows }: RequirementRowsProps) {
  if (rows.length === 0) {
    return (
      <span className="text-secondary">No requirements collected yet.</span>
    );
  }
  return (
    <div className={styles.summaryStack}>
      {rows.map((row) => (
        <SummaryRow key={row.label} label={row.label} value={row.value} />
      ))}
    </div>
  );
}

type SummaryRowProps = {
  label: string;
  value: string;
};

/**
 * Render one summary row.
 *
 * @param props - The summary row props.
 * @returns A summary row element.
 */
function SummaryRow({ label, value }: SummaryRowProps) {
  return (
    <div className={styles.summaryRow}>
      <span className={styles.summaryLabel}>{label}</span>
      <span className={styles.summaryValue}>{value}</span>
    </div>
  );
}

/**
 * Build the initial local chat transcript.
 *
 * @returns Initial chat messages.
 */
function createInitialMessages(): UiMessage[] {
  return [
    {
      content:
        "Tell me your destination, length, travelers, budget, pace, interests, hotel area, and transport mode.",
      createdAt: "Ready",
      id: "initial-assistant-message",
      role: "assistant",
    },
  ];
}

/**
 * Create a local UI message.
 *
 * @param role - The message role.
 * @param content - The message content.
 * @param tone - Optional visual tone.
 * @returns The local UI message.
 */
function createUiMessage(
  role: UiMessage["role"],
  content: string,
  tone: UiMessage["tone"],
): UiMessage {
  return {
    content,
    createdAt: formatMessageTime(new Date()),
    id: createId(),
    role,
    tone,
  };
}

/**
 * Build a stable local identifier.
 *
 * @returns A local identifier string.
 */
function createId(): string {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

/**
 * Format a local chat timestamp.
 *
 * @param date - The date to format.
 * @returns A compact timestamp.
 */
function formatMessageTime(date: Date): string {
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * Pause for a fixed number of milliseconds.
 *
 * @param milliseconds - The sleep duration in milliseconds.
 * @returns A promise that resolves after the duration.
 */
function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });
}

/**
 * Build a draft itinerary title from collected requirements.
 *
 * @param requirement - The requirement snapshot.
 * @returns A display title.
 */
function buildDraftTitle(requirement: TravelRequirement | null): string {
  const destination = requirement?.destination ?? "New trip";
  const days = requirement?.trip_length_days;
  if (days === null || days === undefined) {
    return destination;
  }
  return `${destination}, ${days} days`;
}

/**
 * Build requirement summary rows for the sidebar.
 *
 * @param requirement - The requirement snapshot.
 * @returns Requirement summary rows.
 */
function buildRequirementRows(
  requirement: TravelRequirement | null,
): RequirementRow[] {
  if (requirement === null) {
    return [];
  }
  return [
    { label: "Destination", value: requirement.destination ?? "-" },
    {
      label: "Length",
      value: requirement.trip_length_days
        ? `${requirement.trip_length_days} days`
        : (requirement.trip_dates ?? "-"),
    },
    { label: "Travelers", value: formatTravelers(requirement) },
    { label: "Budget", value: requirement.budget_level ?? "-" },
    { label: "Pace", value: requirement.travel_pace ?? "-" },
    {
      label: "Interests",
      value:
        requirement.interests.length > 0
          ? requirement.interests.join(", ")
          : "-",
    },
    { label: "Hotel area", value: requirement.hotel_area ?? "-" },
    {
      label: "Transport",
      value: requirement.transportation_mode ?? "-",
    },
  ];
}

/**
 * Format traveler counts from a requirement.
 *
 * @param requirement - The requirement snapshot.
 * @returns A readable traveler count.
 */
function formatTravelers(requirement: TravelRequirement | null): string {
  if (requirement === null || requirement.travelers.adults === null) {
    return "-";
  }
  const childLabel =
    requirement.travelers.children > 0
      ? `, ${requirement.travelers.children} children`
      : "";
  return `${requirement.travelers.adults} adults${childLabel}`;
}

/**
 * Create placeholder day tabs before itinerary generation.
 *
 * @param requirement - The requirement snapshot.
 * @returns Draft itinerary days for tab labels.
 */
function createDraftDays(
  requirement: TravelRequirement | null,
): ItineraryDay[] {
  const dayCount = Math.min(requirement?.trip_length_days ?? 3, 7);
  const interests =
    requirement?.interests && requirement.interests.length > 0
      ? requirement.interests
      : ["highlights"];
  return Array.from({ length: dayCount }, (_, index) => ({
    date: null,
    day_number: index + 1,
    items: [],
    route: null,
    summary: "Draft day pending backend generation.",
    theme: interests[index % interests.length],
  }));
}

/**
 * Build a high-level status label for the current plan.
 *
 * @param conversation - The current conversation response.
 * @param job - The current itinerary job.
 * @param itinerary - The generated itinerary.
 * @returns A compact status label.
 */
function buildStatusLabel(
  conversation: ConversationResponse | null,
  job: ItineraryJob | null,
  itinerary: Itinerary | null,
): string {
  if (itinerary !== null) {
    return "Itinerary ready";
  }
  if (job !== null) {
    return `Job ${job.status}`;
  }
  if (conversation !== null) {
    return conversation.state.replaceAll("_", " ");
  }
  return "Backend idle";
}

/**
 * Summarize route metrics across an itinerary.
 *
 * @param itinerary - The generated itinerary.
 * @returns Route totals.
 */
function summarizeRoutes(itinerary: Itinerary | null): {
  distanceMeters: number;
  durationSeconds: number;
  legCount: number;
} {
  if (itinerary === null) {
    return { distanceMeters: 0, durationSeconds: 0, legCount: 0 };
  }
  return itinerary.days.reduce(
    (totals, day) => ({
      distanceMeters: totals.distanceMeters + (day.route?.distance_meters ?? 0),
      durationSeconds:
        totals.durationSeconds + (day.route?.duration_seconds ?? 0),
      legCount: totals.legCount + (day.route?.legs.length ?? 0),
    }),
    { distanceMeters: 0, durationSeconds: 0, legCount: 0 },
  );
}

/**
 * Select restaurant rows from a generated itinerary.
 *
 * @param itinerary - The generated itinerary.
 * @returns Restaurant labels and names.
 */
function selectRestaurants(
  itinerary: Itinerary | null,
): Array<{ label: string; name: string }> {
  if (itinerary === null) {
    return [{ label: "-", name: "Pending itinerary" }];
  }
  return itinerary.days.flatMap((day) =>
    day.items
      .filter((item) => item.type === "lunch" || item.type === "dinner")
      .map((item) => ({
        label: `Day ${day.day_number} ${item.type}`,
        name: item.place.name,
      })),
  );
}

/**
 * Format a route distance.
 *
 * @param meters - The route distance in meters.
 * @returns A readable distance.
 */
function formatDistance(meters: number): string {
  if (meters <= 0) {
    return "-";
  }
  if (meters < 1000) {
    return `${meters} m`;
  }
  return `${(meters / 1000).toFixed(1)} km`;
}

/**
 * Format a route duration.
 *
 * @param seconds - The route duration in seconds.
 * @returns A readable duration.
 */
function formatDuration(seconds: number): string {
  if (seconds <= 0) {
    return "-";
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes} min`;
  }
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return remainingMinutes === 0
    ? `${hours} hr`
    : `${hours} hr ${remainingMinutes} min`;
}

/**
 * Format a backend travel mode.
 *
 * @param travelMode - The backend travel mode.
 * @returns A readable travel mode.
 */
function formatTravelMode(travelMode: string | null | undefined): string {
  if (!travelMode) {
    return "Move";
  }
  return travelMode.replaceAll("_", " ").toLowerCase();
}

/**
 * Render a travel mode icon.
 *
 * @param travelMode - The backend travel mode.
 * @returns A travel mode icon.
 */
function renderTravelIcon(travelMode: string): ReactNode {
  const normalizedMode = travelMode.toLowerCase();
  if (normalizedMode.includes("transit")) {
    return <Bus size={14} />;
  }
  if (normalizedMode.includes("ferry")) {
    return <Ship size={14} />;
  }
  return <Footprints size={14} />;
}

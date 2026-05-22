import Link from "next/link";
import {
  getSharedItinerary,
  type ItineraryDay,
  type ItineraryItem,
  type SharedItineraryResponse,
} from "@/lib/smartourApi";
import styles from "./page.module.css";

type SharedItineraryPageProps = {
  params: Promise<{ token: string }>;
};

/**
 * Render a public read-only shared itinerary page.
 *
 * @param props - The dynamic route props.
 * @returns The shared itinerary page.
 */
export default async function SharedItineraryPage({
  params,
}: SharedItineraryPageProps) {
  const { token } = await params;
  const sharedItinerary = await loadSharedItinerary(token);
  if (sharedItinerary === null) {
    return <NotFoundState />;
  }
  const { itinerary, report } = sharedItinerary;
  return (
    <main className={styles.container}>
      <header className={styles.header}>
        <Link className={styles.logo} href="/">
          Smartour
        </Link>
        <span className={styles.badge}>Shared trip</span>
      </header>

      <section className={styles.hero}>
        <div>
          <p className={styles.kicker}>{itinerary.destination_name}</p>
          <h1>{itinerary.title}</h1>
        </div>
        <div className={styles.metaGrid}>
          <Metric label="Days" value={`${itinerary.days.length}`} />
          <Metric label="Hotels" value={`${itinerary.hotels.length}`} />
          <Metric label="Generated" value={formatDate(itinerary.created_at)} />
        </div>
      </section>

      <section className={styles.contentGrid}>
        <div className={styles.daysColumn}>
          {itinerary.days.map((day) => (
            <DaySection day={day} key={day.day_number} />
          ))}
        </div>
        <aside className={styles.reportColumn}>
          <h2>Report</h2>
          <pre className={styles.reportPreview}>{report.markdown}</pre>
        </aside>
      </section>
    </main>
  );
}

/**
 * Load a shared itinerary and normalize lookup failures.
 *
 * @param token - The share token.
 * @returns The shared itinerary when found.
 */
async function loadSharedItinerary(
  token: string,
): Promise<SharedItineraryResponse | null> {
  try {
    return await getSharedItinerary(token);
  } catch {
    return null;
  }
}

type MetricProps = {
  label: string;
  value: string;
};

/**
 * Render a compact metadata metric.
 *
 * @param props - The metric props.
 * @returns A metric element.
 */
function Metric({ label, value }: MetricProps) {
  return (
    <div className={styles.metric}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

type DaySectionProps = {
  day: ItineraryDay;
};

/**
 * Render one shared itinerary day.
 *
 * @param props - The day section props.
 * @returns A day section.
 */
function DaySection({ day }: DaySectionProps) {
  return (
    <section className={styles.daySection}>
      <div className={styles.dayHeader}>
        <div>
          <p className={styles.kicker}>Day {day.day_number}</p>
          <h2>{day.theme}</h2>
        </div>
        {day.date !== null ? <span>{day.date}</span> : null}
      </div>
      <p className={styles.summary}>{day.summary}</p>
      <div className={styles.itemList}>
        {day.items.map((item) => (
          <ItineraryItemRow
            item={item}
            key={`${item.time}-${item.place.name}`}
          />
        ))}
      </div>
      {day.route !== null ? (
        <div className={styles.routeSummary}>
          <span>{formatDistance(day.route.distance_meters)}</span>
          <span>{formatDuration(day.route.duration_seconds)}</span>
          <span>
            {day.route.travel_mode.toLowerCase().replaceAll("_", " ")}
          </span>
        </div>
      ) : null}
    </section>
  );
}

type ItineraryItemRowProps = {
  item: ItineraryItem;
};

/**
 * Render one itinerary stop.
 *
 * @param props - The itinerary item row props.
 * @returns An itinerary item row.
 */
function ItineraryItemRow({ item }: ItineraryItemRowProps) {
  const content = (
    <>
      <span>{item.time}</span>
      <strong>{item.place.name}</strong>
      <span>{`${item.type}, ${item.duration_minutes} min`}</span>
    </>
  );
  if (item.place.google_maps_uri === null) {
    return <div className={styles.itemRow}>{content}</div>;
  }
  return (
    <a className={styles.itemRow} href={item.place.google_maps_uri}>
      {content}
    </a>
  );
}

/**
 * Render the not-found state for invalid share tokens.
 *
 * @returns A not-found state.
 */
function NotFoundState() {
  return (
    <main className={styles.container}>
      <header className={styles.header}>
        <Link className={styles.logo} href="/">
          Smartour
        </Link>
      </header>
      <section className={styles.emptyState}>
        <span className={styles.badge}>Not found</span>
        <h1>Shared itinerary unavailable</h1>
      </section>
    </main>
  );
}

/**
 * Format an ISO date string.
 *
 * @param value - The ISO date value.
 * @returns A readable date.
 */
function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
  }).format(new Date(value));
}

/**
 * Format a route distance.
 *
 * @param meters - The route distance in meters.
 * @returns A readable distance.
 */
function formatDistance(meters: number): string {
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

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  BarChart2,
  Database,
  DollarSign,
  LoaderCircle,
  RefreshCw,
  Shield,
} from "lucide-react";
import {
  ApiRequestError,
  getAdminStats,
  getAuthState,
  type AdminStatsResponse,
  type AuthStateResponse,
  type GoogleMapsCostBreakdown,
} from "@/lib/smartourApi";
import styles from "./page.module.css";

const WINDOW_OPTIONS = [24, 168, 720];

/**
 * Render the admin operations dashboard.
 *
 * @returns The admin dashboard page.
 */
export default function AdminPage() {
  const [authState, setAuthState] = useState<AuthStateResponse | null>(null);
  const [stats, setStats] = useState<AdminStatsResponse | null>(null);
  const [windowHours, setWindowHours] = useState(24);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let isActive = true;
    void getAuthState()
      .then(async (nextAuthState) => {
        if (!isActive) {
          return;
        }
        setAuthState(nextAuthState);
        if (!nextAuthState.authenticated || !nextAuthState.user?.is_admin) {
          setStats(null);
          return;
        }
        const nextStats = await getAdminStats(windowHours);
        if (isActive) {
          setStats(nextStats);
        }
      })
      .catch((error: unknown) => {
        if (isActive) {
          setErrorMessage(readErrorMessage(error));
          if (isAuthenticationError(error)) {
            setAuthState({ authenticated: false, user: null });
          }
          setStats(null);
        }
      })
      .finally(() => {
        if (isActive) {
          setIsLoading(false);
        }
      });
    return () => {
      isActive = false;
    };
  }, [windowHours]);

  /**
   * Refresh admin statistics for the current window.
   */
  async function handleRefresh() {
    setIsRefreshing(true);
    setErrorMessage(null);
    try {
      setStats(await getAdminStats(windowHours));
    } catch (error) {
      setErrorMessage(readErrorMessage(error));
      if (isAuthenticationError(error)) {
        setAuthState({ authenticated: false, user: null });
        setStats(null);
      }
    } finally {
      setIsRefreshing(false);
    }
  }

  const currentUser = authState?.user ?? null;
  return (
    <main className={styles.container}>
      <header className={styles.header}>
        <Link className={styles.logo} href="/">
          Smartour
        </Link>
        <div className={styles.headerActions}>
          <Link className="btn btn-secondary" href="/account">
            Account
          </Link>
          <Link className="btn btn-primary" href="/">
            New trip
          </Link>
        </div>
      </header>

      <section className={styles.hero}>
        <div>
          <p className={styles.kicker}>Admin</p>
          <h1>Operations dashboard</h1>
        </div>
        <div className={styles.toolbar}>
          <label>
            <span>Window</span>
            <select
              onChange={(event) => setWindowHours(Number(event.target.value))}
              value={windowHours}
            >
              {WINDOW_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}h
                </option>
              ))}
            </select>
          </label>
          <button
            className="btn btn-secondary"
            disabled={isRefreshing || currentUser?.is_admin !== true}
            onClick={() => {
              void handleRefresh();
            }}
            type="button"
          >
            {isRefreshing ? (
              <LoaderCircle className={styles.spin} size={16} />
            ) : (
              <RefreshCw size={16} />
            )}
            Refresh
          </button>
        </div>
      </section>

      {errorMessage !== null ? (
        <div className={styles.errorText}>{errorMessage}</div>
      ) : null}

      {isLoading ? (
        <LoadingState />
      ) : authState?.authenticated !== true ? (
        <AccessState
          message="Sign in with an admin account to view backend statistics."
          title="Authentication required"
        />
      ) : currentUser?.is_admin !== true ? (
        <AccessState
          message="This account does not have admin access."
          title="Admin access required"
        />
      ) : stats === null ? (
        <AccessState
          message="Refresh the page or sign in again if the session expired."
          title="Admin data unavailable"
        />
      ) : (
        <StatsDashboard stats={stats} />
      )}
    </main>
  );
}

type StatsDashboardProps = {
  stats: AdminStatsResponse;
};

/**
 * Render backend statistics and cost summaries.
 *
 * @param props - The stats dashboard props.
 * @returns The admin stats dashboard.
 */
function StatsDashboard({ stats }: StatsDashboardProps) {
  const costSummary = stats.google_maps_cost_summary;
  return (
    <section className={styles.dashboardStack}>
      <div className={styles.metricGrid}>
        <MetricCard
          icon={<Database size={18} />}
          label="Users"
          value={formatNumber(stats.record_counts.users)}
        />
        <MetricCard
          icon={<Database size={18} />}
          label="Conversations"
          value={formatNumber(stats.record_counts.conversations)}
        />
        <MetricCard
          icon={<Database size={18} />}
          label="Itineraries"
          value={formatNumber(stats.record_counts.itineraries)}
        />
        <MetricCard
          icon={<Database size={18} />}
          label="Share links"
          value={formatNumber(stats.record_counts.share_links)}
        />
        <MetricCard
          icon={<BarChart2 size={18} />}
          label="Jobs"
          value={formatNumber(stats.record_counts.itinerary_jobs)}
        />
        <MetricCard
          icon={<DollarSign size={18} />}
          label="Est. cost"
          value={formatCost(costSummary.estimated_cost_usd)}
        />
      </div>

      <section className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <p className={styles.kicker}>Google Maps</p>
            <h2>Cost management</h2>
          </div>
          <span>{formatDate(stats.generated_at)}</span>
        </div>
        <div className={styles.costGrid}>
          <SummaryRow
            label="Requests"
            value={formatNumber(costSummary.total_requests)}
          />
          <SummaryRow
            label="Billable est."
            value={formatNumber(costSummary.estimated_billable_requests)}
          />
          <SummaryRow
            label="Cache hits"
            value={formatNumber(costSummary.cache_hits)}
          />
          <SummaryRow
            label="Errors"
            value={formatNumber(costSummary.error_requests)}
          />
          <SummaryRow
            label="Avg duration"
            value={`${costSummary.average_duration_ms.toFixed(1)} ms`}
          />
          <SummaryRow label="Currency" value={costSummary.currency} />
        </div>
        <CostBreakdownTable services={costSummary.services} />
      </section>

      <section className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <p className={styles.kicker}>Jobs</p>
            <h2>Status distribution</h2>
          </div>
        </div>
        <div className={styles.statusList}>
          {stats.job_status_counts.length > 0 ? (
            stats.job_status_counts.map((item) => (
              <SummaryRow
                key={item.status}
                label={item.status}
                value={formatNumber(item.count)}
              />
            ))
          ) : (
            <div className={styles.emptyText}>No jobs recorded yet.</div>
          )}
        </div>
      </section>
    </section>
  );
}

type MetricCardProps = {
  icon: React.ReactNode;
  label: string;
  value: string;
};

/**
 * Render one admin metric card.
 *
 * @param props - The metric card props.
 * @returns A metric card.
 */
function MetricCard({ icon, label, value }: MetricCardProps) {
  return (
    <div className={styles.metricCard}>
      <div className={styles.metricIcon}>{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

type CostBreakdownTableProps = {
  services: GoogleMapsCostBreakdown[];
};

/**
 * Render Google Maps cost breakdown rows.
 *
 * @param props - The cost table props.
 * @returns The cost breakdown table.
 */
function CostBreakdownTable({ services }: CostBreakdownTableProps) {
  if (services.length === 0) {
    return (
      <div className={styles.emptyText}>No Google Maps usage recorded.</div>
    );
  }
  return (
    <div className={styles.table}>
      <div className={styles.tableHeader}>
        <span>Service</span>
        <span>Endpoint</span>
        <span>Billable</span>
        <span>Cost</span>
      </div>
      {services.map((service) => (
        <div
          className={styles.tableRow}
          key={`${service.service}-${service.endpoint}`}
        >
          <span>{formatServiceName(service.service)}</span>
          <span>{service.endpoint}</span>
          <span>
            {formatNumber(service.estimated_billable_requests)}/
            {formatNumber(service.total_requests)}
          </span>
          <span>{formatCost(service.estimated_cost_usd)}</span>
        </div>
      ))}
    </div>
  );
}

type SummaryRowProps = {
  label: string;
  value: string;
};

/**
 * Render one label/value summary row.
 *
 * @param props - The summary row props.
 * @returns A summary row.
 */
function SummaryRow({ label, value }: SummaryRowProps) {
  return (
    <div className={styles.summaryRow}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

type AccessStateProps = {
  message: string;
  title: string;
};

/**
 * Render an admin access state.
 *
 * @param props - The access state props.
 * @returns An access state.
 */
function AccessState({ message, title }: AccessStateProps) {
  return (
    <section className={styles.accessState}>
      <Shield size={28} />
      <h2>{title}</h2>
      <p>{message}</p>
      <Link className="btn btn-primary" href="/account">
        Go to account
      </Link>
    </section>
  );
}

/**
 * Render an admin loading state.
 *
 * @returns The loading state.
 */
function LoadingState() {
  return (
    <section className={styles.loadingState}>
      <LoaderCircle className={styles.spin} size={24} />
      <span>Loading admin data</span>
    </section>
  );
}

/**
 * Return a readable message from an unknown error.
 *
 * @param error - The thrown error.
 * @returns A readable message.
 */
function readErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Request failed";
}

/**
 * Return whether an error came from an unauthenticated backend response.
 *
 * @param error - The thrown error.
 * @returns True when the backend returned 401.
 */
function isAuthenticationError(error: unknown): boolean {
  return error instanceof ApiRequestError && error.status === 401;
}

/**
 * Format a number for admin display.
 *
 * @param value - The number to format.
 * @returns A readable number.
 */
function formatNumber(value: number): string {
  return value.toLocaleString();
}

/**
 * Format an estimated cost in USD.
 *
 * @param value - The cost value.
 * @returns A readable cost.
 */
function formatCost(value: number): string {
  return `$${value.toFixed(6)}`;
}

/**
 * Format a backend service key.
 *
 * @param service - The service key.
 * @returns A readable service name.
 */
function formatServiceName(service: string): string {
  return service.replaceAll("_", " ");
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
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
    year: "numeric",
  }).format(new Date(value));
}

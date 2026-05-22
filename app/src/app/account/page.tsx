"use client";

import type { FormEvent } from "react";
import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ExternalLink,
  Link2,
  LoaderCircle,
  LogIn,
  LogOut,
  Map,
  UserPlus,
} from "lucide-react";
import {
  ApiRequestError,
  getAuthState,
  getUserDashboard,
  loginAuthUser,
  logoutAuthUser,
  registerAuthUser,
  type AuthStateResponse,
  type UserDashboardResponse,
} from "@/lib/smartourApi";
import styles from "./page.module.css";

type AuthMode = "login" | "register";

/**
 * Render the current user's account dashboard.
 *
 * @returns The account dashboard page.
 */
export default function AccountPage() {
  const [authState, setAuthState] = useState<AuthStateResponse | null>(null);
  const [dashboard, setDashboard] = useState<UserDashboardResponse | null>(
    null,
  );
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let isActive = true;
    void getAuthState()
      .then(async (nextAuthState) => {
        if (!isActive) {
          return;
        }
        const nextDashboard = nextAuthState.authenticated
          ? await getUserDashboard()
          : null;
        if (!isActive) {
          return;
        }
        if (nextAuthState.authenticated) {
          setDashboard(nextDashboard);
        } else {
          setDashboard(null);
        }
        setAuthState(nextAuthState);
      })
      .catch((error: unknown) => {
        if (isActive) {
          setErrorMessage(readErrorMessage(error));
          setAuthState({ authenticated: false, user: null });
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
  }, []);

  const currentUser = authState?.user ?? null;

  /**
   * Submit the login or registration form.
   *
   * @param event - The form submit event.
   */
  async function handleAuthSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSubmitting) {
      return;
    }
    setIsSubmitting(true);
    setErrorMessage(null);
    try {
      const nextAuthState =
        authMode === "register"
          ? await registerAuthUser(username, password)
          : await loginAuthUser(username, password);
      const nextDashboard = nextAuthState.authenticated
        ? await getUserDashboard()
        : null;
      setAuthState(nextAuthState);
      setDashboard(nextDashboard);
      setPassword("");
    } catch (error) {
      setErrorMessage(readErrorMessage(error));
      if (isAuthenticationError(error)) {
        setAuthState({ authenticated: false, user: null });
        setDashboard(null);
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  /**
   * Logout the current user and clear dashboard state.
   */
  async function handleLogout() {
    setErrorMessage(null);
    try {
      setAuthState(await logoutAuthUser());
      setDashboard(null);
    } catch (error) {
      setErrorMessage(readErrorMessage(error));
    }
  }

  return (
    <main className={styles.container}>
      <header className={styles.header}>
        <Link className={styles.logo} href="/">
          Smartour
        </Link>
        <div className={styles.headerActions}>
          {currentUser?.is_admin ? (
            <Link className="btn btn-secondary" href="/admin">
              Admin
            </Link>
          ) : null}
          <Link className="btn btn-primary" href="/">
            New trip
          </Link>
        </div>
      </header>

      <section className={styles.hero}>
        <div>
          <p className={styles.kicker}>Account</p>
          <h1>
            {currentUser === null
              ? "Sign in to Smartour"
              : currentUser.username}
          </h1>
        </div>
        {currentUser !== null ? (
          <button
            className="btn btn-secondary"
            onClick={() => {
              void handleLogout();
            }}
            type="button"
          >
            <LogOut size={16} />
            Sign out
          </button>
        ) : null}
      </section>

      {errorMessage !== null ? (
        <div className={styles.errorText}>{errorMessage}</div>
      ) : null}

      {isLoading ? (
        <LoadingState />
      ) : currentUser === null ? (
        <AuthForm
          authMode={authMode}
          isSubmitting={isSubmitting}
          onAuthModeChange={setAuthMode}
          onPasswordChange={setPassword}
          onSubmit={handleAuthSubmit}
          onUsernameChange={setUsername}
          password={password}
          username={username}
        />
      ) : dashboard === null ? (
        <DashboardUnavailableState />
      ) : (
        <DashboardView dashboard={dashboard} />
      )}
    </main>
  );
}

type AuthFormProps = {
  authMode: AuthMode;
  isSubmitting: boolean;
  onAuthModeChange: (authMode: AuthMode) => void;
  onPasswordChange: (password: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onUsernameChange: (username: string) => void;
  password: string;
  username: string;
};

/**
 * Render the local account login/register form.
 *
 * @param props - The auth form props.
 * @returns The auth form.
 */
function AuthForm({
  authMode,
  isSubmitting,
  onAuthModeChange,
  onPasswordChange,
  onSubmit,
  onUsernameChange,
  password,
  username,
}: AuthFormProps) {
  return (
    <section className={styles.authPanel}>
      <div className={styles.modeTabs} role="tablist">
        <button
          className={authMode === "login" ? styles.activeMode : ""}
          onClick={() => onAuthModeChange("login")}
          type="button"
        >
          <LogIn size={16} />
          Login
        </button>
        <button
          className={authMode === "register" ? styles.activeMode : ""}
          onClick={() => onAuthModeChange("register")}
          type="button"
        >
          <UserPlus size={16} />
          Register
        </button>
      </div>
      <form className={styles.authForm} onSubmit={onSubmit}>
        <label>
          <span>Username</span>
          <input
            autoComplete="username"
            onChange={(event) => onUsernameChange(event.target.value)}
            value={username}
          />
        </label>
        <label>
          <span>Password</span>
          <input
            autoComplete={
              authMode === "register" ? "new-password" : "current-password"
            }
            onChange={(event) => onPasswordChange(event.target.value)}
            type="password"
            value={password}
          />
        </label>
        <button
          className="btn btn-primary"
          disabled={isSubmitting || !username.trim() || !password}
          type="submit"
        >
          {isSubmitting ? (
            <LoaderCircle className={styles.spin} size={16} />
          ) : authMode === "register" ? (
            <UserPlus size={16} />
          ) : (
            <LogIn size={16} />
          )}
          {authMode === "register" ? "Register" : "Login"}
        </button>
      </form>
    </section>
  );
}

type DashboardViewProps = {
  dashboard: UserDashboardResponse;
};

/**
 * Render saved plans and share links for the current user.
 *
 * @param props - The dashboard view props.
 * @returns The dashboard content.
 */
function DashboardView({ dashboard }: DashboardViewProps) {
  return (
    <section className={styles.dashboardGrid}>
      <DashboardSection
        emptyText="No created plans yet."
        icon={<Map size={18} />}
        title="Created plans"
      >
        {dashboard.created_itineraries.map((itinerary) => (
          <Link
            className={styles.listItem}
            href={itinerary.open_path}
            key={itinerary.itinerary_id}
          >
            <div>
              <strong>{itinerary.title}</strong>
              <span>{itinerary.destination_name}</span>
            </div>
            <span>{formatDate(itinerary.created_at)}</span>
            <ExternalLink size={16} />
          </Link>
        ))}
      </DashboardSection>

      <DashboardSection
        emptyText="No share links created yet."
        icon={<Link2 size={18} />}
        title="Shared plans"
      >
        {dashboard.share_links.map((shareLink) => (
          <Link
            className={styles.listItem}
            href={shareLink.share_path}
            key={shareLink.token}
          >
            <div>
              <strong>{shareLink.itinerary_title}</strong>
              <span>{shareLink.token}</span>
            </div>
            <span>{formatDate(shareLink.created_at)}</span>
            <ExternalLink size={16} />
          </Link>
        ))}
      </DashboardSection>
    </section>
  );
}

/**
 * Render a recoverable dashboard unavailable state.
 *
 * @returns The dashboard unavailable state.
 */
function DashboardUnavailableState() {
  return (
    <section className={styles.loadingState}>
      <span>Account dashboard unavailable.</span>
    </section>
  );
}

type DashboardSectionProps = {
  children: React.ReactNode[];
  emptyText: string;
  icon: React.ReactNode;
  title: string;
};

/**
 * Render one dashboard list section.
 *
 * @param props - The dashboard section props.
 * @returns A dashboard section.
 */
function DashboardSection({
  children,
  emptyText,
  icon,
  title,
}: DashboardSectionProps) {
  return (
    <section className={styles.dashboardSection}>
      <div className={styles.sectionTitle}>
        {icon}
        <h2>{title}</h2>
      </div>
      <div className={styles.itemList}>
        {children.length > 0 ? (
          children
        ) : (
          <div className={styles.emptyText}>{emptyText}</div>
        )}
      </div>
    </section>
  );
}

/**
 * Render a loading state.
 *
 * @returns The loading state.
 */
function LoadingState() {
  return (
    <section className={styles.loadingState}>
      <LoaderCircle className={styles.spin} size={24} />
      <span>Loading account</span>
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

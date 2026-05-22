"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

type Theme = "light" | "dark";

type ThemeContextValue = {
  setTheme: (theme: Theme) => void;
  theme: Theme;
};

type ThemeProviderProps = {
  children: ReactNode;
};

const THEME_STORAGE_KEY = "smartour-theme";
const DARK_THEME_QUERY = "(prefers-color-scheme: dark)";
const ThemeContext = createContext<ThemeContextValue | null>(null);

/**
 * Provide theme state to the application tree without injecting scripts.
 *
 * @param props - The theme provider props.
 * @returns The configured theme provider.
 */
export function ThemeProvider({ children }: ThemeProviderProps) {
  const [theme, setThemeState] = useState<Theme>(initialTheme);

  useEffect(() => {
    const storedTheme = readStoredTheme();
    applyTheme(theme);

    if (storedTheme !== null) {
      return;
    }
    const mediaQueryList = window.matchMedia(DARK_THEME_QUERY);

    /**
     * Sync theme state when the operating system preference changes.
     */
    function handleSystemThemeChange() {
      const nextTheme = systemTheme();
      setThemeState(nextTheme);
      applyTheme(nextTheme);
    }

    mediaQueryList.addEventListener("change", handleSystemThemeChange);
    return () => {
      mediaQueryList.removeEventListener("change", handleSystemThemeChange);
    };
  }, [theme]);

  const setTheme = useCallback((nextTheme: Theme) => {
    window.localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    setThemeState(nextTheme);
    applyTheme(nextTheme);
  }, []);

  const value = useMemo<ThemeContextValue>(
    () => ({
      setTheme,
      theme,
    }),
    [setTheme, theme],
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

/**
 * Return the current theme context.
 *
 * @returns The current theme state and setter.
 */
export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === null) {
    throw new Error("useTheme must be used within ThemeProvider");
  }
  return context;
}

/**
 * Return the initial theme for the current render environment.
 *
 * @returns The initial theme.
 */
function initialTheme(): Theme {
  if (typeof window === "undefined") {
    return "light";
  }
  return readStoredTheme() ?? systemTheme();
}

/**
 * Read a persisted theme value from local storage.
 *
 * @returns The stored theme when valid.
 */
function readStoredTheme(): Theme | null {
  if (typeof window === "undefined") {
    return null;
  }
  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (storedTheme === "light" || storedTheme === "dark") {
    return storedTheme;
  }
  return null;
}

/**
 * Return the current operating-system theme preference.
 *
 * @returns The resolved system theme.
 */
function systemTheme(): Theme {
  return window.matchMedia(DARK_THEME_QUERY).matches ? "dark" : "light";
}

/**
 * Apply a theme to the document root.
 *
 * @param theme - The theme to apply.
 */
function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle("dark", theme === "dark");
}

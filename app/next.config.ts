/**
 * Next.js config for the Smartour frontend.
 */

import type { NextConfig } from "next";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const REPOSITORY_ENV_FILE = resolve(process.cwd(), "..", ".env");

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_GOOGLE_MAPS_API_KEY:
      process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY ??
      readRepositoryEnvValue("NEXT_PUBLIC_GOOGLE_MAPS_API_KEY") ??
      process.env.GOOGLE_MAPS_API_KEY ??
      readRepositoryEnvValue("GOOGLE_MAPS_API_KEY") ??
      "",
  },
};

/**
 * Read one value from the repository-level `.env` file.
 *
 * @param variableName - The environment variable name to read.
 * @returns The configured value when present.
 */
function readRepositoryEnvValue(variableName: string): string | undefined {
  try {
    const envFile = readFileSync(REPOSITORY_ENV_FILE, "utf-8");
    for (const line of envFile.split(/\r?\n/)) {
      const value = parseEnvLine(line, variableName);
      if (value !== undefined) {
        return value;
      }
    }
  } catch {
    return undefined;
  }
  return undefined;
}

/**
 * Parse one `.env` line for a specific variable.
 *
 * @param line - The raw `.env` line.
 * @param variableName - The variable name to match.
 * @returns The parsed value when the line matches.
 */
function parseEnvLine(line: string, variableName: string): string | undefined {
  const trimmedLine = line.trim();
  if (!trimmedLine || trimmedLine.startsWith("#")) {
    return undefined;
  }
  const normalizedLine = trimmedLine.startsWith("export ")
    ? trimmedLine.slice("export ".length).trim()
    : trimmedLine;
  const prefix = `${variableName}=`;
  if (!normalizedLine.startsWith(prefix)) {
    return undefined;
  }
  return unwrapEnvValue(normalizedLine.slice(prefix.length).trim());
}

/**
 * Remove simple quote wrappers from an `.env` value.
 *
 * @param value - The raw value.
 * @returns The unwrapped value.
 */
function unwrapEnvValue(value: string): string {
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }
  return value;
}

export default nextConfig;

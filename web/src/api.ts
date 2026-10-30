import { useQuery } from "@tanstack/react-query";
import type { Session } from "./types";
let csrf = "";
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}
export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...options,
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrf,
        ...options.headers,
      },
    });
  } catch {
    throw new ApiError(
      "The research API is unavailable. Check that the Python server is running.",
      0,
    );
  }
  if (!response.ok) {
    const body = await response
      .json()
      .catch(() => ({ detail: "The request could not be completed." }));
    const detail = body.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail
              .map(
                (x: { loc?: string[]; msg: string }) =>
                  `${x.loc?.slice(1).join(".")}: ${x.msg}`,
              )
              .join(" · ")
          : Object.values(detail ?? {}).join(" · ");
    throw new ApiError(message, response.status);
  }
  return response.json();
}
export async function session() {
  try {
    const s = await api<Session>("/auth/session");
    csrf = s.csrf ?? "";
    return s;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      throw new ApiError(
        "The sign-in service is unavailable. Check the TA-EFL server connection and try again.",
        error.status,
      );
    }
    throw error;
  }
}
export function setSession(s: Session) {
  csrf = s.csrf ?? "";
}
export function useData<T>(
  path: string,
  interval: number | false | ((data: T | undefined) => number | false) = false,
  enabled = true,
) {
  return useQuery({
    queryKey: [path],
    queryFn: () => api<T>(path),
    refetchInterval:
      typeof interval === "function"
        ? (query) => interval(query.state.data)
        : interval,
    enabled,
    retry: 1,
  });
}
export const post = <T>(path: string, body: unknown = {}) =>
  api<T>(path, { method: "POST", body: JSON.stringify(body) });
export const fmt = (n: number | null | undefined, digits = 3) =>
  n == null ? "—" : n.toFixed(digits);
export const percent = (n: number | null | undefined) =>
  n == null ? "—" : `${(n * 100).toFixed(1)}%`;
export const strategyName = (v: string) =>
  ({ trust: "Trust-aware", fedavg: "FedAvg", median: "Median" })[v] ?? v;
export const label = (v: string) =>
  v.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());
export const date = (v: string) =>
  new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(v));

import { useCallback, useEffect, useState } from "react";
import { ApiError, listJobs } from "../api";
import type { JobRead } from "../api";

function jobHistoryErrorMessage(error: unknown) {
  if (error instanceof ApiError) {
    const detail = error.body ? `: ${error.body.slice(0, 160)}` : "";
    return `Failed to load jobs (${error.status} ${error.statusText}${detail})`;
  }
  if (error instanceof Error && error.message) {
    return `Failed to load jobs (${error.message})`;
  }
  return "Failed to load jobs";
}

export function useJobHistory() {
  const [jobs, setJobs] = useState<JobRead[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let data: JobRead[] | null = null;
      for (let attempt = 0; attempt < 2; attempt += 1) {
        try {
          data = await listJobs(50, 0);
          break;
        } catch {
          if (attempt === 0) {
            await new Promise((resolve) => window.setTimeout(resolve, 750));
          }
        }
      }
      if (data == null) {
        throw new Error("list jobs failed");
      }
      setJobs(data);
    } catch (error) {
      setError(jobHistoryErrorMessage(error));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { jobs, loading, error, refresh, setJobs };
}

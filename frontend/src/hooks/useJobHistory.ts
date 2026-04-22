import { useCallback, useEffect, useState } from "react";
import { listJobs } from "../api";
import type { JobRead } from "../api";

export function useJobHistory() {
  const [jobs, setJobs] = useState<JobRead[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listJobs(50, 0);
      setJobs(data);
    } catch {
      setError("Failed to load jobs");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { jobs, loading, error, refresh, setJobs };
}

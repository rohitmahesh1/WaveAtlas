import type { ReactNode } from "react";
import { useJobSession } from "../hooks/useJobSession";
import { JobSessionContext } from "./jobSessionContext";

export function JobSessionProvider({ children }: { children: ReactNode }) {
  const session = useJobSession({ resumeOnMount: true });

  return <JobSessionContext.Provider value={session}>{children}</JobSessionContext.Provider>;
}

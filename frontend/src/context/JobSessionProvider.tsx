import type { ReactNode } from "react";
import { useJobSession } from "../hooks/useJobSession";
import { JobSessionContext } from "./jobSessionContext";

export function JobSessionProvider({ children }: { children: ReactNode }) {
  const session = useJobSession({ resumeOnMount: false });

  return <JobSessionContext.Provider value={session}>{children}</JobSessionContext.Provider>;
}

import { useContext } from "react";
import { JobSessionContext } from "../context/jobSessionContext";

export function useSharedJobSession() {
  const session = useContext(JobSessionContext);
  if (!session) {
    throw new Error("useSharedJobSession must be used within JobSessionProvider");
  }
  return session;
}

import { createContext } from "react";
import type { JobSessionState } from "../hooks/useJobSession";

export const JobSessionContext = createContext<JobSessionState | null>(null);

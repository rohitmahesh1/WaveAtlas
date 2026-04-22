import { useMemo, useState } from "react";
import type { OverlayTrackEvent } from "../OverlayCanvas";
import type { FieldDef, FilterField, FilterRule, SummaryStats } from "../types";

export function useFilters(tracks: OverlayTrackEvent[], fields: FieldDef[]) {
  const [filters, setFilters] = useState<FilterRule[]>([]);

  const fieldMap = useMemo(() => {
    const m = new Map<FilterField, FieldDef>();
    for (const f of fields) m.set(f.key, f);
    return m;
  }, [fields]);

  const addFilterRule = () => {
    const id = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    setFilters((prev) => [
      ...prev,
      { id, field: "mean_amplitude", op: ">", value: "20" },
    ]);
  };

  const updateFilterRule = (id: string, patch: Partial<FilterRule>) => {
    setFilters((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  };

  const removeFilterRule = (id: string) => {
    setFilters((prev) => prev.filter((r) => r.id !== id));
  };

  const clearFilters = () => setFilters([]);

  const filteredTracks = useMemo(() => {
    const matchRule = (t: OverlayTrackEvent, rule: FilterRule) => {
      const def = fieldMap.get(rule.field);
      if (!def) return true;
      const raw = def.get(t);
      if (def.type === "number") {
        const v = Number(raw);
        if (!Number.isFinite(v)) return false;
        const hasVal = rule.value !== undefined && rule.value !== "";
        const hasVal2 = rule.value2 !== undefined && rule.value2 !== "";
        const n1 = Number(rule.value);
        const n2 = Number(rule.value2);
        if (!hasVal && rule.op !== "between") return true;
        switch (rule.op) {
          case ">":
            return Number.isFinite(n1) ? v > n1 : true;
          case "<":
            return Number.isFinite(n1) ? v < n1 : true;
          case ">=":
            return Number.isFinite(n1) ? v >= n1 : true;
          case "<=":
            return Number.isFinite(n1) ? v <= n1 : true;
          case "==":
            return Number.isFinite(n1) ? v === n1 : true;
          case "!=":
            return Number.isFinite(n1) ? v !== n1 : true;
          case "between":
            if (!hasVal || !hasVal2 || !Number.isFinite(n1) || !Number.isFinite(n2)) return true;
            return v >= Math.min(n1, n2) && v <= Math.max(n1, n2);
          default:
            return true;
        }
      }

      const s = String(raw ?? "");
      const q = String(rule.value ?? "").toLowerCase();
      if (q === "") return true;
      switch (rule.op) {
        case "contains":
          return s.toLowerCase().includes(q);
        case "==":
          return s.toLowerCase() === q;
        case "!=":
          return s.toLowerCase() !== q;
        default:
          return true;
      }
    };

    if (filters.length === 0) return tracks;
    return tracks.filter((t) => filters.every((r) => matchRule(t, r)));
  }, [tracks, filters, fieldMap]);

  const filteredStats = useMemo<SummaryStats>(() => {
    if (!filteredTracks.length) {
      return { count: 0, points: 0, avgAmplitude: null as number | null, avgFrequency: null as number | null };
    }
    let sumAmp = 0;
    let cntAmp = 0;
    let sumFreq = 0;
    let cntFreq = 0;
    let pts = 0;
    for (const t of filteredTracks) {
      const a = Number(t.metrics?.mean_amplitude);
      const f = Number(t.metrics?.dominant_frequency);
      if (Number.isFinite(a)) {
        sumAmp += a;
        cntAmp += 1;
      }
      if (Number.isFinite(f)) {
        sumFreq += f;
        cntFreq += 1;
      }
      pts += t.poly?.length ?? 0;
    }
    return {
      count: filteredTracks.length,
      points: pts,
      avgAmplitude: cntAmp ? sumAmp / cntAmp : null,
      avgFrequency: cntFreq ? sumFreq / cntFreq : null,
    };
  }, [filteredTracks]);

  return {
    filters,
    setFilters,
    addFilterRule,
    updateFilterRule,
    removeFilterRule,
    clearFilters,
    fieldMap,
    filteredTracks,
    filteredStats,
  };
}

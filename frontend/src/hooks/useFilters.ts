import { useCallback, useMemo, useState, type Dispatch, type SetStateAction } from "react";
import type { OverlayTrackEvent } from "../OverlayCanvas";
import type { FieldDef, FilterField, FilterRule, SummaryStats } from "../types";

type FilterState = {
  scopeKey: string;
  filters: FilterRule[];
};

const EMPTY_FILTERS: FilterRule[] = [];

export function useFilters(tracks: OverlayTrackEvent[], fields: FieldDef[], scopeKey = "default") {
  const [filterState, setFilterState] = useState<FilterState>({ scopeKey, filters: [] });
  const filters = filterState.scopeKey === scopeKey ? filterState.filters : EMPTY_FILTERS;
  const setFilters = useCallback<Dispatch<SetStateAction<FilterRule[]>>>((next) => {
    setFilterState((current) => {
      const currentFilters = current.scopeKey === scopeKey ? current.filters : [];
      const filters = typeof next === "function" ? next(currentFilters) : next;
      return { scopeKey, filters };
    });
  }, [scopeKey]);

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
      return {
        count: 0,
        points: 0,
        avgAmplitude: null,
        avgFrequency: null,
        avgSlope: null,
        avgSpeed: null,
        avgAngle: null,
        familyCount: 0,
      };
    }
    let sumAmp = 0;
    let cntAmp = 0;
    let sumFreq = 0;
    let cntFreq = 0;
    let sumSlope = 0;
    let cntSlope = 0;
    let sumSpeed = 0;
    let cntSpeed = 0;
    let sumAngle = 0;
    let cntAngle = 0;
    let pts = 0;
    const familyIds = new Set<string>();
    for (const t of filteredTracks) {
      const a = Number(t.metrics?.mean_amplitude);
      const f = Number(t.metrics?.dominant_frequency);
      const slope = Number(t.metrics?.slope_px_per_frame);
      const speedValue = t.metrics?.speed_px_per_s ?? (
        t.metrics?.velocity_px_per_s != null ? Math.abs(t.metrics.velocity_px_per_s) : null
      );
      const speed = Number(speedValue);
      const angle = Number(t.metrics?.angle_from_time_axis_deg ?? t.metrics?.angle_deg);
      if (Number.isFinite(a)) {
        sumAmp += a;
        cntAmp += 1;
      }
      if (Number.isFinite(f)) {
        sumFreq += f;
        cntFreq += 1;
      }
      if (Number.isFinite(slope)) {
        sumSlope += slope;
        cntSlope += 1;
      }
      if (Number.isFinite(speed)) {
        sumSpeed += speed;
        cntSpeed += 1;
      }
      if (Number.isFinite(angle)) {
        sumAngle += angle;
        cntAngle += 1;
      }
      if (t.metrics?.family_id) familyIds.add(t.metrics.family_id);
      pts += t.poly?.length ?? 0;
    }
    return {
      count: filteredTracks.length,
      points: pts,
      avgAmplitude: cntAmp ? sumAmp / cntAmp : null,
      avgFrequency: cntFreq ? sumFreq / cntFreq : null,
      avgSlope: cntSlope ? sumSlope / cntSlope : null,
      avgSpeed: cntSpeed ? sumSpeed / cntSpeed : null,
      avgAngle: cntAngle ? sumAngle / cntAngle : null,
      familyCount: familyIds.size,
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

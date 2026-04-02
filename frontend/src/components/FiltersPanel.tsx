import React from "react";
import type { FieldDef, FilterField, FilterOp, FilterRule } from "../types";

export function FiltersPanel(props: {
  filters: FilterRule[];
  fields: FieldDef[];
  fieldMap: Map<FilterField, FieldDef>;
  onAdd: () => void;
  onClear: () => void;
  onUpdate: (id: string, patch: Partial<FilterRule>) => void;
  onRemove: (id: string) => void;
}) {
  const { filters, fields, fieldMap, onAdd, onClear, onUpdate, onRemove } = props;

  return (
    <section className="panel">
      <div className="panel-title">Filters</div>
      <div className="panel-body">
        <div className="filter-actions">
          <button className="ghost-btn" onClick={onAdd}>
            Add rule
          </button>
          <button className="ghost-btn" onClick={onClear}>
            Clear
          </button>
        </div>
        {filters.length === 0 ? (
          <div className="empty-text">No filters yet. Add a rule to narrow tracks.</div>
        ) : (
          <div className="filter-list">
            {filters.map((rule) => {
              const def = fieldMap.get(rule.field) ?? fields[0];
              return (
                <div key={rule.id} className="filter-row">
                  <select
                    value={rule.field}
                    onChange={(e) => {
                      const nextField = e.target.value as FilterField;
                      const nextDef = fieldMap.get(nextField) ?? fields[0];
                      onUpdate(rule.id, {
                        field: nextField,
                        op: nextDef.ops[0],
                        value: "",
                        value2: "",
                      });
                    }}
                  >
                    {fields.map((f) => (
                      <option key={f.key} value={f.key}>
                        {f.label}
                      </option>
                    ))}
                  </select>
                  <select value={rule.op} onChange={(e) => onUpdate(rule.id, { op: e.target.value as FilterOp })}>
                    {def.ops.map((op) => (
                      <option key={op} value={op}>
                        {op}
                      </option>
                    ))}
                  </select>
                  <input
                    type={def.type === "number" ? "number" : "text"}
                    value={rule.value ?? ""}
                    onChange={(e) => onUpdate(rule.id, { value: e.target.value })}
                    placeholder="value"
                  />
                  {rule.op === "between" ? (
                    <input
                      type="number"
                      value={rule.value2 ?? ""}
                      onChange={(e) => onUpdate(rule.id, { value2: e.target.value })}
                      placeholder="and"
                    />
                  ) : null}
                  <button className="icon-btn" onClick={() => onRemove(rule.id)} title="Remove rule">
                    ×
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}

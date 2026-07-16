import { useEffect, useMemo, useRef, useState } from "react";
import {
  AggregateMetrics,
  AggregateMetricSpec,
  getAggregateMetrics,
} from "../api/client";
import "./AggregateHeatmap.css";

interface AggregateHeatmapProps {
  predictionSet: string | null;
}

/**
 * Sequential blue ramp (dataviz reference palette, steps 700 -> 100) ordered
 * dark -> light for this app's dark chart surface: a near-zero cell recedes
 * toward the surface, a high cell reads brightest.
 */
const BLUE_RAMP = [
  "#0d366b",
  "#104281",
  "#184f95",
  "#1c5cab",
  "#256abf",
  "#2a78d6",
  "#3987e5",
  "#5598e7",
  "#6da7ec",
  "#86b6ef",
  "#9ec5f4",
  "#b7d3f6",
  "#cde2fb",
] as const;

/** Ramp index past which the fill is light enough to need dark ink. */
const DARK_INK_FROM = 8;

interface CellStats {
  min: number;
  max: number;
}

/** Per-metric min/max across samples — each column normalizes independently. */
function computeStats(data: AggregateMetrics): Record<string, CellStats | null> {
  const stats: Record<string, CellStats | null> = {};
  for (const metric of data.metrics) {
    let min = Infinity;
    let max = -Infinity;
    for (const sample of data.samples) {
      const v = data.values[sample]?.[metric.key];
      if (typeof v === "number" && Number.isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    stats[metric.key] = Number.isFinite(min) ? { min, max } : null;
  }
  return stats;
}

/** Normalized position of `value` in its column, 0..1 (0.5 for a flat column). */
function normalize(value: number, stats: CellStats): number {
  if (stats.max === stats.min) return 0.5;
  return (value - stats.min) / (stats.max - stats.min);
}

function rampIndex(t: number): number {
  const i = Math.round(t * (BLUE_RAMP.length - 1));
  return Math.min(BLUE_RAMP.length - 1, Math.max(0, i));
}

/** Compact display value for an in-cell direct label. */
function fmtCell(value: number): string {
  if (Number.isInteger(value)) return String(value);
  if (value === 0) return "0";
  if (Math.abs(value) >= 1e-3) return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  return value.toExponential(1);
}

interface TooltipState {
  x: number;
  y: number;
  sample: string;
  metric: AggregateMetricSpec;
  value: number | null;
  norm: number | null;
}

export function AggregateHeatmap({ predictionSet }: AggregateHeatmapProps) {
  const [data, setData] = useState<AggregateMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState<string>("0.5");
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getAggregateMetrics(predictionSet, threshold)
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setError(null);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setData(null);
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [predictionSet, threshold]);

  const stats = useMemo(() => (data ? computeStats(data) : {}), [data]);

  function showTooltip(
    event: React.MouseEvent,
    sample: string,
    metric: AggregateMetricSpec,
    value: number | null,
  ) {
    const bounds = wrapRef.current?.getBoundingClientRect();
    const st = stats[metric.key];
    setTooltip({
      x: event.clientX - (bounds?.left ?? 0),
      y: event.clientY - (bounds?.top ?? 0),
      sample,
      metric,
      value,
      norm: value !== null && st ? normalize(value, st) : null,
    });
  }

  return (
    <div className="agg" ref={wrapRef}>
      <div className="agg__toolbar">
        <div>
          <h2 className="agg__title">Aggregate Metrics</h2>
          <p className="agg__subtitle">
            Samples × metrics. Each column is colour-scaled independently to its
            own min/max across samples.
          </p>
        </div>

        <label className="agg__threshold">
          <span className="agg__threshold-label">Threshold</span>
          <select
            value={threshold}
            onChange={(e) => setThreshold(e.target.value)}
            title="Detection threshold used by the threshold-dependent columns"
          >
            {(data?.thresholdChoices ?? ["0.3", "0.5", "0.75"]).map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
      </div>

      {loading && <p className="agg__hint">Loading aggregate metrics…</p>}
      {error && <p className="agg__error">{error}</p>}

      {!loading && !error && data && data.samples.length === 0 && (
        <p className="agg__hint">
          No scoring metrics for this prediction set. Metrics are currently
          BiaPy-only.
        </p>
      )}

      {!loading && !error && data && data.samples.length > 0 && (
        <>
          <div className="agg__legend">
            <span className="agg__legend-label">low</span>
            <div className="agg__legend-ramp">
              {BLUE_RAMP.map((c) => (
                <span
                  key={c}
                  className="agg__legend-step"
                  style={{ background: c }}
                />
              ))}
            </div>
            <span className="agg__legend-label">high</span>
            <span className="agg__legend-note">
              per-column min → max · higher is better for every column shown
            </span>
          </div>

          <div className="agg__table-wrap">
            <table className="agg__table">
              <caption className="agg__caption">
                Scoring metrics per sample for the selected prediction set;
                threshold-dependent columns evaluated at {data.threshold}.
              </caption>
              <thead>
                <tr>
                  <th className="agg__corner" scope="col">
                    sample
                  </th>
                  {data.metrics.map((m) => (
                    <th key={m.key} scope="col" title={`${m.label} (${m.source})`}>
                      {m.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.samples.map((sample) => (
                  <tr key={sample}>
                    <th scope="row" className="agg__rowhead" title={sample}>
                      {sample}
                    </th>
                    {data.metrics.map((m) => {
                      const value = data.values[sample]?.[m.key] ?? null;
                      const st = stats[m.key];
                      const hasValue =
                        typeof value === "number" && Number.isFinite(value) && st;
                      const idx = hasValue
                        ? rampIndex(normalize(value as number, st as CellStats))
                        : -1;
                      return (
                        <td
                          key={m.key}
                          className={
                            hasValue ? "agg__cell" : "agg__cell agg__cell--empty"
                          }
                          style={
                            hasValue
                              ? {
                                  background: BLUE_RAMP[idx],
                                  color:
                                    idx >= DARK_INK_FROM ? "#0b0b0b" : "#ffffff",
                                }
                              : undefined
                          }
                          onMouseEnter={(e) =>
                            showTooltip(e, sample, m, hasValue ? (value as number) : null)
                          }
                          onMouseMove={(e) =>
                            showTooltip(e, sample, m, hasValue ? (value as number) : null)
                          }
                          onMouseLeave={() => setTooltip(null)}
                        >
                          {hasValue ? fmtCell(value as number) : "–"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
              {/* Derived column summaries, not samples: rendered as plain
                  tabular numbers so they never read as another data row. */}
              <tfoot className="agg__foot">
                {(["mean", "median"] as const).map((stat) => (
                  <tr key={stat}>
                    <th scope="row" className="agg__rowhead agg__rowhead--summary">
                      {stat}
                    </th>
                    {data.metrics.map((m) => {
                      const value = data.summary?.[stat]?.[m.key] ?? null;
                      const n = data.summary?.n?.[m.key] ?? 0;
                      return (
                        <td
                          key={m.key}
                          className="agg__cell agg__cell--summary"
                          title={
                            value === null
                              ? `No ${stat} — no samples scored for ${m.label}`
                              : `${stat} of ${m.label} over ${n} sample${n === 1 ? "" : "s"}: ${value}`
                          }
                        >
                          {value === null ? "–" : fmtCell(value)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tfoot>
            </table>
          </div>

          {tooltip && (
            <div
              className="agg__tooltip"
              style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
            >
              <div className="agg__tooltip-sample">{tooltip.sample}</div>
              <div className="agg__tooltip-metric">{tooltip.metric.label}</div>
              <div className="agg__tooltip-value">
                {tooltip.value === null ? "no value" : String(tooltip.value)}
              </div>
              {tooltip.norm !== null && (
                <div className="agg__tooltip-norm">
                  column position {(tooltip.norm * 100).toFixed(0)}%
                </div>
              )}
              <div className="agg__tooltip-source">{tooltip.metric.source}</div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

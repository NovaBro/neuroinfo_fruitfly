import { useEffect, useMemo, useRef, useState } from "react";
import {
  AggregateGroup,
  AggregateGroupKey,
  AggregateMetrics,
  AggregateMetricSpec,
  getAggregateMetrics,
  peekAggregateMetrics,
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

const DEFAULT_GROUP_ORDER: AggregateGroupKey[] = [
  "test/completely",
  "test/partly",
  "test/synthetic",
  "val/completely",
  "val/partly",
  "val/synthetic",
];

function groupTitle(key: string): string {
  const [split, dataset] = key.split("/", 2);
  const splitLabel =
    split === "test" ? "Test" : split === "val" ? "Val" : split;
  return dataset ? `${splitLabel} · ${dataset}` : splitLabel;
}

interface CellStats {
  min: number;
  max: number;
}

/** Per-metric min/max across samples in one group — column-normalized. */
function computeStats(
  metrics: AggregateMetricSpec[],
  group: AggregateGroup,
): Record<string, CellStats | null> {
  const stats: Record<string, CellStats | null> = {};
  for (const metric of metrics) {
    let min = Infinity;
    let max = -Infinity;
    for (const sample of group.samples) {
      const v = group.values[sample]?.[metric.key];
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

interface HeatmapTableProps {
  title: string;
  group: AggregateGroup;
  metrics: AggregateMetricSpec[];
  threshold: string;
  onTooltip: (
    event: React.MouseEvent | null,
    sample: string,
    metric: AggregateMetricSpec,
    value: number | null,
    stats: Record<string, CellStats | null>,
  ) => void;
}

function HeatmapTable({
  title,
  group,
  metrics,
  threshold,
  onTooltip,
}: HeatmapTableProps) {
  const stats = useMemo(
    () => computeStats(metrics, group),
    [metrics, group],
  );

  if (group.samples.length === 0) {
    return (
      <section className="agg__section">
        <h3 className="agg__section-title">{title}</h3>
        <p className="agg__hint">No samples scored.</p>
      </section>
    );
  }

  return (
    <section className="agg__section">
      <h3 className="agg__section-title">
        {title}
        <span className="agg__section-count">
          {group.samples.length} sample{group.samples.length === 1 ? "" : "s"}
        </span>
      </h3>
      <div className="agg__table-wrap">
        <table className="agg__table">
          <caption className="agg__caption">
            {title} scoring metrics; threshold-dependent columns at {threshold}.
          </caption>
          <thead>
            <tr>
              <th className="agg__corner" scope="col">
                sample
              </th>
              {metrics.map((m) => (
                <th key={m.key} scope="col" title={`${m.label} (${m.source})`}>
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {group.samples.map((sample) => (
              <tr key={sample}>
                <th scope="row" className="agg__rowhead" title={sample}>
                  {sample}
                </th>
                {metrics.map((m) => {
                  const value = group.values[sample]?.[m.key] ?? null;
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
                        onTooltip(
                          e,
                          sample,
                          m,
                          hasValue ? (value as number) : null,
                          stats,
                        )
                      }
                      onMouseMove={(e) =>
                        onTooltip(
                          e,
                          sample,
                          m,
                          hasValue ? (value as number) : null,
                          stats,
                        )
                      }
                      onMouseLeave={() =>
                        onTooltip(null, sample, m, null, stats)
                      }
                    >
                      {hasValue ? fmtCell(value as number) : "–"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
          <tfoot className="agg__foot">
            {(["mean", "median"] as const).map((stat) => (
              <tr key={stat}>
                <th scope="row" className="agg__rowhead agg__rowhead--summary">
                  {stat}
                </th>
                {metrics.map((m) => {
                  const value = group.summary?.[stat]?.[m.key] ?? null;
                  const n = group.summary?.n?.[m.key] ?? 0;
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
    </section>
  );
}

export function AggregateHeatmap({ predictionSet }: AggregateHeatmapProps) {
  const [threshold, setThreshold] = useState<string>("0.5");
  const [data, setData] = useState<AggregateMetrics | null>(
    () => peekAggregateMetrics(predictionSet, "0.5") ?? null,
  );
  const [loading, setLoading] = useState(
    () => peekAggregateMetrics(predictionSet, "0.5") == null,
  );
  const [error, setError] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const hit = peekAggregateMetrics(predictionSet, threshold);
    if (hit) {
      setData(hit);
      setLoading(false);
      setError(null);
      return;
    }

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

  const groupKeys = useMemo(() => {
    if (!data?.groups) return DEFAULT_GROUP_ORDER;
    const order = data.groupOrder?.length
      ? data.groupOrder
      : DEFAULT_GROUP_ORDER;
    // Include any extra keys the server sent, then drop empty tables from display
    // only for scoring check — still render empty? Plan: emit all keys; show
    // only non-empty tables to avoid six empty heatmaps.
    const keys = [
      ...order.filter((k) => k in data.groups),
      ...Object.keys(data.groups).filter((k) => !order.includes(k)),
    ];
    return keys.filter((k) => (data.groups[k]?.samples.length ?? 0) > 0);
  }, [data]);

  const anyScored = groupKeys.length > 0;

  function showTooltip(
    event: React.MouseEvent | null,
    sample: string,
    metric: AggregateMetricSpec,
    value: number | null,
    stats: Record<string, CellStats | null>,
  ) {
    if (event === null) {
      setTooltip(null);
      return;
    }
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
            One table per FISBe split × dataset (e.g. Test · completely).
            Each column is colour-scaled independently within its table.
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

      {!loading && !error && data && !anyScored && (
        <p className="agg__hint">
          No scoring metrics for this prediction set.
        </p>
      )}

      {!loading && !error && data && anyScored && (
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
              per-column min → max within each table · higher is better
            </span>
          </div>

          <div className="agg__sections">
            {groupKeys.map((key) => (
              <HeatmapTable
                key={key}
                title={groupTitle(key)}
                group={data.groups[key]}
                metrics={data.metrics}
                threshold={data.threshold}
                onTooltip={showTooltip}
              />
            ))}
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

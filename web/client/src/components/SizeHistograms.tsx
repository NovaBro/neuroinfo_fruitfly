import { useEffect, useMemo, useState } from "react";
import {
  AggregateGroupKey,
  getInstanceSizeHistograms,
  InstanceSizeHistograms,
  InstanceSizeSampleHist,
  peekInstanceSizeHistograms,
} from "../api/client";
import "./SizeHistograms.css";

interface SizeHistogramsProps {
  predictionSet: string | null;
}

const DEFAULT_GROUP_ORDER: AggregateGroupKey[] = [
  "test/completely",
  "test/partly",
  "test/synth_vh",
  "test/synth_vh_u8",
  "test/1k_v1",
  "test/1k_v1_u8",
  "test/himanshu",
  "val/completely",
  "val/partly",
  "val/synth_vh",
  "val/synth_vh_u8",
  "val/1k_v1",
  "val/1k_v1_u8",
  "val/himanshu",
];

const CHART_W = 240;
const CHART_H = 96;
const PAD = { top: 8, right: 6, bottom: 18, left: 28 };

/** Decade edges for x-axis labels (match server logspace 1…1e6). */
const X_DECADE_VALUES = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6];

function groupTitle(key: string): string {
  const [split, dataset] = key.split("/", 2);
  const splitLabel =
    split === "test" ? "Test" : split === "val" ? "Val" : split;
  return `${splitLabel} · ${dataset}`;
}

function formatEdge(v: number): string {
  if (v <= 1) return "1";
  const exp = Math.round(Math.log10(v));
  if (exp >= 1 && Math.abs(v - 10 ** exp) / v < 0.05) return `1e${exp}`;
  if (v >= 100) return String(Math.round(v));
  return String(Math.round(v));
}

function nearestEdgeIndex(binEdges: number[], target: number): number {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < binEdges.length; i++) {
    const d = Math.abs(binEdges[i] - target);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

function SampleChart({
  sample,
  binEdges,
}: {
  sample: InstanceSizeSampleHist;
  binEdges: number[];
}) {
  const nBins = sample.counts.length;
  const maxCount = Math.max(1, ...sample.counts);
  const midCount = Math.ceil(maxCount / 2);
  const innerW = CHART_W - PAD.left - PAD.right;
  const innerH = CHART_H - PAD.top - PAD.bottom;
  const barGap = 1;
  const barW = Math.max(1, (innerW - barGap * (nBins - 1)) / nBins);
  const baselineY = PAD.top + innerH;

  const xTicks = X_DECADE_VALUES.map((target) => {
    const edgeIndex = nearestEdgeIndex(binEdges, target);
    return {
      key: target,
      x: PAD.left + (edgeIndex / nBins) * innerW,
      label: formatEdge(target),
      edgeIndex,
    };
  });

  const yTickCandidates = [
    { value: maxCount, y: PAD.top },
    { value: midCount, y: PAD.top + innerH / 2 },
    { value: 0, y: baselineY },
  ];
  // Drop duplicate count labels when maxCount is small (e.g. 1).
  const yTicks = yTickCandidates.filter(
    (t, i, arr) => arr.findIndex((o) => o.value === t.value) === i,
  );

  return (
    <div className="sizes__card">
      <div className="sizes__card-head">
        <span className="sizes__sample" title={sample.sample}>
          {sample.sample}
        </span>
        <span className="sizes__n">{sample.n_instances} inst</span>
      </div>
      <svg
        className="sizes__chart"
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        role="img"
        aria-label={`Size histogram for ${sample.sample}`}
      >
        {sample.counts.map((c, i) => {
          const h = (c / maxCount) * innerH;
          const x = PAD.left + i * (barW + barGap);
          const y = baselineY - h;
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={barW}
              height={Math.max(c > 0 ? 1 : 0, h)}
              className={c > 0 ? "sizes__bar" : "sizes__bar sizes__bar--empty"}
            >
              <title>
                {formatEdge(binEdges[i])}–{formatEdge(binEdges[i + 1])} vx: {c}
              </title>
            </rect>
          );
        })}
        <line
          x1={PAD.left}
          y1={PAD.top}
          x2={PAD.left}
          y2={baselineY}
          className="sizes__axis"
        />
        <line
          x1={PAD.left}
          y1={baselineY}
          x2={PAD.left + innerW}
          y2={baselineY}
          className="sizes__axis"
        />
        {yTicks.map((t) => (
          <text
            key={`y-${t.value}-${t.y}`}
            x={PAD.left - 3}
            y={t.y + 2.5}
            className="sizes__tick sizes__tick--y"
            textAnchor="end"
          >
            {t.value}
          </text>
        ))}
        {xTicks.map((t) => (
          <text
            key={`x-${t.key}`}
            x={t.x}
            y={CHART_H - 3}
            className="sizes__tick"
            textAnchor={
              t.edgeIndex === 0
                ? "start"
                : t.edgeIndex >= nBins
                  ? "end"
                  : "middle"
            }
          >
            {t.label}
          </text>
        ))}
      </svg>
    </div>
  );
}

/** Per-sample predicted-instance size histograms, grouped by split × dataset. */
export function SizeHistograms({ predictionSet }: SizeHistogramsProps) {
  const [data, setData] = useState<InstanceSizeHistograms | null>(
    () => peekInstanceSizeHistograms(predictionSet) ?? null,
  );
  const [loading, setLoading] = useState(
    () => peekInstanceSizeHistograms(predictionSet) == null,
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const cached = peekInstanceSizeHistograms(predictionSet);
    if (cached) {
      setData(cached);
      setLoading(false);
      setError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    getInstanceSizeHistograms(predictionSet)
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
  }, [predictionSet]);

  const groupKeys = useMemo(() => {
    if (!data?.groups) return DEFAULT_GROUP_ORDER;
    const order = data.groupOrder?.length
      ? data.groupOrder
      : DEFAULT_GROUP_ORDER;
    const keys = [
      ...order.filter((k) => k in data.groups),
      ...Object.keys(data.groups).filter((k) => !order.includes(k)),
    ];
    return keys.filter((k) => (data.groups[k]?.samples.length ?? 0) > 0);
  }, [data]);

  const anySamples = groupKeys.length > 0;

  return (
    <div className="sizes">
      <div className="sizes__toolbar">
        <div>
          <h2 className="sizes__title">Instance sizes</h2>
          <p className="sizes__subtitle">
            Predicted instance voxel-count histograms (log bins 1…1e6). One
            chart per sample, grouped by FISBe split × dataset.
          </p>
        </div>
      </div>

      {loading && (
        <p className="sizes__hint">
          Scanning predicted label volumes… (first load per set can take a
          while)
        </p>
      )}
      {error && <p className="sizes__error">{error}</p>}

      {!loading && !error && data && !data.applicable && (
        <p className="sizes__hint">
          {data.reason ?? "Not applicable for this prediction set."}
        </p>
      )}

      {!loading && !error && data && data.applicable && !anySamples && (
        <p className="sizes__hint">
          {data.reason ??
            "No predicted instances with body sizes for this set."}
        </p>
      )}

      {!loading && !error && data && data.applicable && anySamples && (
        <div className="sizes__sections">
          {groupKeys.map((key) => (
            <section key={key} className="sizes__section">
              <h3 className="sizes__section-title">
                {groupTitle(key)}
                <span className="sizes__section-count">
                  {data.groups[key].samples.length} samples
                </span>
              </h3>
              <div className="sizes__grid">
                {data.groups[key].samples.map((s) => (
                  <SampleChart
                    key={s.sample}
                    sample={s}
                    binEdges={data.bin_edges}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

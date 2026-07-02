import { CsvMetrics, MetricThresholds, TomlMetrics } from "../api/client";
import "./MetricsPanel.css";

interface MetricsPanelProps {
  sampleName: string | null;
  toml: TomlMetrics | null;
  csv: CsvMetrics | null;
  loading: boolean;
  error: string | null;
}

/** Format a metric value compactly for a narrow column. */
function fmt(value: unknown): string {
  if (value === null || value === undefined) return "–";
  if (Array.isArray(value)) return value.map(fmt).join(", ");
  if (typeof value !== "number") return String(value);
  if (Number.isInteger(value)) return String(value);
  if (value === 0) return "0";
  return Math.abs(value) >= 1e-3
    ? value.toFixed(4).replace(/\.?0+$/, "")
    : value.toExponential(2);
}

function KeyValues({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data);
  if (entries.length === 0) return null;
  return (
    <dl className="metrics-kv">
      {entries.map(([k, v]) => (
        <div className="metrics-kv__row" key={k}>
          <dt className="metrics-kv__key">{k}</dt>
          <dd className="metrics-kv__val">{fmt(v)}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Metric-rows × threshold-columns table (horizontally scrollable). */
function ThresholdTable({ thresholds }: { thresholds: MetricThresholds }) {
  const cols = Object.keys(thresholds);
  if (cols.length === 0) return null;

  const metricNames: string[] = [];
  const seen = new Set<string>();
  for (const th of cols) {
    for (const m of Object.keys(thresholds[th])) {
      if (!seen.has(m)) {
        seen.add(m);
        metricNames.push(m);
      }
    }
  }

  return (
    <div className="metrics-table-wrap">
      <table className="metrics-table">
        <thead>
          <tr>
            <th className="metrics-table__corner">metric</th>
            {cols.map((th) => (
              <th key={th}>{th}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metricNames.map((m) => (
            <tr key={m}>
              <th scope="row">{m}</th>
              {cols.map((th) => (
                <td key={th}>{fmt(thresholds[th][m])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function MetricsPanel({
  sampleName,
  toml,
  csv,
  loading,
  error,
}: MetricsPanelProps) {
  return (
    <div className="metrics-panel">
      <h2 className="metrics-panel__title">Scoring Metrics</h2>
      {sampleName && <p className="metrics-panel__sample">{sampleName}</p>}

      {!sampleName && (
        <p className="metrics-panel__hint">Select a sample to see its scores.</p>
      )}
      {sampleName && loading && (
        <p className="metrics-panel__hint">Loading metrics…</p>
      )}
      {sampleName && error && (
        <p className="metrics-panel__error">{error}</p>
      )}

      {sampleName && !loading && !error && (
        <>
          <section className="metrics-section">
            <h3 className="metrics-section__title">tests/metrics</h3>
            {toml ? (
              <>
                <p className="metrics-section__source">{toml.source}</p>
                <h4 className="metrics-section__sub">General</h4>
                <KeyValues data={toml.general} />
                <h4 className="metrics-section__sub">Confusion matrix (avg)</h4>
                <KeyValues data={toml.summary} />
                <h4 className="metrics-section__sub">Per threshold</h4>
                <ThresholdTable thresholds={toml.thresholds} />
              </>
            ) : (
              <p className="metrics-panel__hint">
                No <code>tests/metrics</code> file for this sample/set.
              </p>
            )}
          </section>

          <section className="metrics-section">
            <h3 className="metrics-section__title">CSV score results</h3>
            {csv ? (
              <>
                <p className="metrics-section__source">{csv.source}</p>
                <h4 className="metrics-section__sub">Scores</h4>
                <KeyValues data={csv.scalars} />
                <h4 className="metrics-section__sub">Per threshold</h4>
                <ThresholdTable thresholds={csv.thresholds} />
              </>
            ) : (
              <p className="metrics-panel__hint">
                No CSV row for this sample in this set.
              </p>
            )}
          </section>
        </>
      )}
    </div>
  );
}

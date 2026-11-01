import { useEffect, useState } from "react";
import {
  checkHealth,
  listPredictionSets,
  listSamples,
  PredictionSet,
  SampleInfo,
  SampleRef,
} from "./api/client";
import { AggregateHeatmap } from "./components/AggregateHeatmap";
import { Layout } from "./components/Layout";
import { MetricsPanel } from "./components/MetricsPanel";
import { OrthoSliceViewer } from "./components/OrthoSliceViewer";
import { SampleBrowser } from "./components/SampleBrowser";
import { SizeHistograms } from "./components/SizeHistograms";
import { VolumeViewer3D } from "./components/VolumeViewer3D";
import { useSampleMeta, useSampleMetrics } from "./hooks/useSample";
import "./App.css";

type Tab = "slices" | "3d" | "aggregate" | "sizes";

function sameSample(a: SampleRef | null, b: SampleInfo): boolean {
  return a != null && a.name === b.name && a.dataset === b.dataset;
}

function App() {
  const [samples, setSamples] = useState<SampleInfo[]>([]);
  const [samplesLoading, setSamplesLoading] = useState(true);
  const [selected, setSelected] = useState<SampleRef | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("3d");
  // Cross-sample tabs mount on first visit only: their requests scan every
  // sample in the set and would otherwise compete with the 3D view on load.
  const [visitedTabs, setVisitedTabs] = useState<ReadonlySet<Tab>>(
    () => new Set([activeTab]),
  );
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [fisbeRootExists, setFisbeRootExists] = useState(false);
  const [predictionSets, setPredictionSets] = useState<PredictionSet[]>([]);
  const [predictionSet, setPredictionSet] = useState<string | null>(null);

  const { meta, loading: metaLoading, error: metaError } = useSampleMeta(
    selected,
    predictionSet,
  );

  const {
    metrics,
    loading: metricsLoading,
    error: metricsError,
  } = useSampleMetrics(selected, predictionSet);

  useEffect(() => {
    checkHealth()
      .then((h) => {
        setApiOk(true);
        setFisbeRootExists(h.fisbe_root_exists);
      })
      .catch(() => setApiOk(false));
  }, []);

  useEffect(() => {
    listPredictionSets()
      .then((sets) => {
        setPredictionSets(sets);
        const def = sets.find((s) => s.default) ?? sets[0];
        if (def) setPredictionSet(def.id);
      })
      .catch(() => setPredictionSets([]));
  }, []);

  // Refetch sample flags scoped to the selected prediction set; jump to a
  // sample that actually has an overlay when the current one does not.
  useEffect(() => {
    let cancelled = false;
    setSamplesLoading(true);
    listSamples(predictionSet)
      .then((data) => {
        if (cancelled) return;
        setSamples(data);
        const stillOk =
          selected != null &&
          data.some((s) => sameSample(selected, s) && s.path_exists);
        const stillHasPred =
          stillOk &&
          data.some((s) => sameSample(selected, s) && s.has_predicted);
        if (stillHasPred) return;
        const firstWithPredicted = data.find(
          (s) => s.path_exists && s.has_predicted,
        );
        const stillEntry =
          stillOk && selected
            ? data.find((s) => sameSample(selected, s))
            : null;
        const firstAvailable = data.find((s) => s.path_exists);
        const nextEntry = firstWithPredicted ?? stillEntry ?? firstAvailable;
        setSelected(
          nextEntry
            ? { dataset: nextEntry.dataset, name: nextEntry.name }
            : null,
        );
      })
      .catch(() => {
        if (!cancelled) setSamples([]);
      })
      .finally(() => {
        if (!cancelled) setSamplesLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // Intentionally omit `selected` so changing the sample does not refetch;
    // only prediction-set changes (and the initial null) drive this effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictionSet]);

  useEffect(() => {
    setVisitedTabs((prev) =>
      prev.has(activeTab) ? prev : new Set(prev).add(activeTab),
    );
  }, [activeTab]);

  const metaMatches =
    selected != null &&
    meta != null &&
    meta.name === selected.name &&
    meta.dataset === selected.dataset;

  const header = (
    <div className="app-header">
      <div>
        <h1 className="app-header__title">FISBe 3D Volume Viewer</h1>
        <p className="app-header__subtitle">
          Browse FlyLight instance-segmentation volumes (Zarr, CZYX)
        </p>
      </div>
      <div className="app-header__status">
        <span
          className={
            apiOk === true
              ? "app-header__dot app-header__dot--ok"
              : apiOk === false
                ? "app-header__dot app-header__dot--err"
                : "app-header__dot"
          }
        />
        API {apiOk === null ? "…" : apiOk ? "connected" : "offline"}
        {apiOk && !fisbeRootExists && (
          <span className="app-header__warn"> · FISBE_ROOT not found</span>
        )}
      </div>
    </div>
  );

  const main = (
    <div className="app-main">
      <div className="app-main__toolbar">
        <div className="app-main__tabs">
          <button
            type="button"
            className={
              activeTab === "slices"
                ? "app-main__tab app-main__tab--active"
                : "app-main__tab"
            }
            onClick={() => setActiveTab("slices")}
          >
            Slice / MIP Viewer
          </button>
          <button
            type="button"
            className={
              activeTab === "3d"
                ? "app-main__tab app-main__tab--active"
                : "app-main__tab"
            }
            onClick={() => setActiveTab("3d")}
          >
            3D Viewer
          </button>
          <button
            type="button"
            className={
              activeTab === "aggregate"
                ? "app-main__tab app-main__tab--active"
                : "app-main__tab"
            }
            onClick={() => setActiveTab("aggregate")}
          >
            Aggregate
          </button>
          <button
            type="button"
            className={
              activeTab === "sizes"
                ? "app-main__tab app-main__tab--active"
                : "app-main__tab"
            }
            onClick={() => setActiveTab("sizes")}
          >
            Instance sizes
          </button>
        </div>

        {predictionSets.length > 0 && (
          <label className="app-main__pred-set">
            <span className="app-main__pred-set-label">Predictions</span>
            <select
              value={predictionSet ?? ""}
              onChange={(e) => setPredictionSet(e.target.value)}
              title={
                predictionSets.find((s) => s.id === predictionSet)?.path ?? ""
              }
            >
              {(() => {
                const groups: { label: string; source: string }[] = [
                  { label: "BiaPy", source: "biapy" },
                  { label: "PatchPerPix", source: "ppp" },
                ];
                return groups
                  .map(({ label, source }) => {
                    const sets = predictionSets.filter(
                      (s) => (s.source ?? "biapy") === source
                    );
                    if (sets.length === 0) return null;
                    return (
                      <optgroup key={source} label={label}>
                        {sets.map((s) => (
                          <option key={s.id} value={s.id}>
                            {s.name}
                            {s.default ? " (default)" : ""}
                          </option>
                        ))}
                      </optgroup>
                    );
                  })
                  .filter(Boolean);
              })()}
            </select>
          </label>
        )}
      </div>

      {/* Aggregate / Instance sizes span all samples — mounted on first visit,
          then kept mounted (hidden on other tabs) so per-prediction caches
          survive tab switches. */}
      {visitedTabs.has("aggregate") && (
        <div hidden={activeTab !== "aggregate"}>
          <AggregateHeatmap predictionSet={predictionSet} />
        </div>
      )}
      {visitedTabs.has("sizes") && (
        <div hidden={activeTab !== "sizes"}>
          <SizeHistograms predictionSet={predictionSet} />
        </div>
      )}
      {activeTab !== "aggregate" && activeTab !== "sizes" && (
        <>
          {!selected && (
            <p className="app-main__placeholder">
              Select a sample from the sidebar.
            </p>
          )}

          {selected && metaLoading && (
            <p className="app-main__placeholder">Loading volume metadata…</p>
          )}

          {selected && metaError && (
            <p className="app-main__error">{metaError}</p>
          )}

          {selected && metaMatches && activeTab === "slices" && meta && (
            <OrthoSliceViewer
              sampleName={selected.name}
              dataset={selected.dataset}
              meta={meta}
            />
          )}

          {selected && metaMatches && activeTab === "3d" && meta && (
            <VolumeViewer3D
              sampleName={selected.name}
              dataset={selected.dataset}
              meta={meta}
              predictionSet={predictionSet}
              predictedLabel={
                (() => {
                  const set = predictionSets.find((s) => s.id === predictionSet);
                  if (!set) return "Predicted instances";
                  if (set.kind === "numinst") return "Predicted numinst";
                  if (set.source === "ppp") return "Predicted instances (PatchPerPix)";
                  return "Predicted instances";
                })()
              }
              bodyFilterEnabled={
                !!predictionSet &&
                !(
                  predictionSets.find((s) => s.id === predictionSet)?.kind ===
                    "numinst" ||
                  predictionSet.startsWith("ppp-numinst:")
                )
              }
            />
          )}
        </>
      )}
    </div>
  );

  return (
    <Layout
      header={header}
      sidebar={
        <SampleBrowser
          samples={samples}
          selected={selected}
          onSelect={setSelected}
          loading={samplesLoading}
        />
      }
      main={main}
      rightPanel={
        // Aggregate / Instance sizes already cover all samples; hide the
        // per-sample metrics column beside those tabs.
        activeTab === "aggregate" || activeTab === "sizes" ? undefined : (
          <MetricsPanel
            sampleName={
              selected ? `${selected.dataset}/${selected.name}` : null
            }
            toml={metrics?.toml ?? null}
            csv={metrics?.csv ?? null}
            leaderboard={metrics?.leaderboard ?? null}
            loading={metricsLoading}
            error={metricsError}
          />
        )
      }
    />
  );
}

export default App;

import { useEffect, useState } from "react";
import {
  checkHealth,
  listPredictionSets,
  listSamples,
  PredictionSet,
  SampleInfo,
} from "./api/client";
import { Layout } from "./components/Layout";
import { OrthoSliceViewer } from "./components/OrthoSliceViewer";
import { SampleBrowser } from "./components/SampleBrowser";
import { VolumeViewer3D } from "./components/VolumeViewer3D";
import { useSampleMeta } from "./hooks/useSample";
import "./App.css";

type Tab = "slices" | "3d";

function App() {
  const [samples, setSamples] = useState<SampleInfo[]>([]);
  const [samplesLoading, setSamplesLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("3d");
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [fisbeRootExists, setFisbeRootExists] = useState(false);
  const [predictionSets, setPredictionSets] = useState<PredictionSet[]>([]);
  const [predictionSet, setPredictionSet] = useState<string | null>(null);

  const { meta, loading: metaLoading, error: metaError } = useSampleMeta(
    selected,
    predictionSet,
  );

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

  useEffect(() => {
    listSamples()
      .then((data) => {
        setSamples(data);
        const firstWithPredicted = data.find(
          (s) => s.path_exists && s.has_predicted,
        );
        const firstAvailable = data.find((s) => s.path_exists);
        setSelected(
          (firstWithPredicted ?? firstAvailable)?.name ?? null,
        );
      })
      .catch(() => setSamples([]))
      .finally(() => setSamplesLoading(false));
  }, []);

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
              {predictionSets.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                  {s.default ? " (default)" : ""}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      {!selected && (
        <p className="app-main__placeholder">Select a sample from the sidebar.</p>
      )}

      {selected && metaLoading && (
        <p className="app-main__placeholder">Loading volume metadata…</p>
      )}

      {selected && metaError && (
        <p className="app-main__error">{metaError}</p>
      )}

      {selected && meta && meta.name === selected && activeTab === "slices" && (
        <OrthoSliceViewer sampleName={selected} meta={meta} />
      )}

      {selected && meta && meta.name === selected && activeTab === "3d" && (
        <VolumeViewer3D
          sampleName={selected}
          meta={meta}
          predictionSet={predictionSet}
        />
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
    />
  );
}

export default App;

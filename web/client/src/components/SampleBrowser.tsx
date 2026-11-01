import { useEffect, useMemo, useState } from "react";
import { Dataset, SampleInfo, SampleRef, Split } from "../api/client";
import "./SampleBrowser.css";

const SPLITS: Split[] = ["train", "val", "test"];
const DATASET_ORDER: Dataset[] = [
  "completely",
  "partly",
  "synth_vh",
  "synth_vh_u8",
  "1k_v1",
  "1k_v1_u8",
  "himanshu",
];

interface SampleBrowserProps {
  samples: SampleInfo[];
  selected: SampleRef | null;
  onSelect: (ref: SampleRef) => void;
  loading?: boolean;
}

export function SampleBrowser({
  samples,
  selected,
  onSelect,
  loading,
}: SampleBrowserProps) {
  const [activeSplit, setActiveSplit] = useState<Split>("train");
  const [activeDataset, setActiveDataset] = useState<Dataset>("completely");
  const [query, setQuery] = useState("");

  // Only offer a dataset toggle for datasets that are actually present.
  const datasets = useMemo<Dataset[]>(() => {
    const present = new Set(samples.map((s) => s.dataset));
    return DATASET_ORDER.filter((d) => present.has(d));
  }, [samples]);

  // Keep activeDataset on a tab that exists once samples load.
  useEffect(() => {
    if (datasets.length > 0 && !datasets.includes(activeDataset)) {
      setActiveDataset(datasets[0]);
    }
  }, [datasets, activeDataset]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return samples
      .filter((s) => s.dataset === activeDataset)
      .filter((s) => s.split === activeSplit)
      .filter((s) => !q || s.name.toLowerCase().includes(q));
  }, [samples, activeDataset, activeSplit, query]);

  return (
    <div className="sample-browser">
      {datasets.length > 1 && (
        <div className="sample-browser__tabs sample-browser__tabs--dataset">
          {datasets.map((dataset) => (
            <button
              key={dataset}
              type="button"
              className={
                activeDataset === dataset
                  ? "sample-browser__tab sample-browser__tab--active"
                  : "sample-browser__tab"
              }
              onClick={() => setActiveDataset(dataset)}
            >
              {dataset}
            </button>
          ))}
        </div>
      )}

      <div className="sample-browser__tabs">
        {SPLITS.map((split) => (
          <button
            key={split}
            type="button"
            className={
              activeSplit === split
                ? "sample-browser__tab sample-browser__tab--active"
                : "sample-browser__tab"
            }
            onClick={() => setActiveSplit(split)}
          >
            {split}
          </button>
        ))}
      </div>

      <input
        className="sample-browser__search"
        type="search"
        placeholder="Search samples…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      {!loading && (
        <p className="sample-browser__count">
          {filtered.length} {activeDataset} / {activeSplit} sample
          {filtered.length === 1 ? "" : "s"}
        </p>
      )}

      {loading ? (
        <p className="sample-browser__status">Loading samples…</p>
      ) : (
        <ul className="sample-browser__list">
          {filtered.map((sample) => {
            const hasPred = Boolean(sample.has_predicted);
            const isSelected =
              selected?.name === sample.name &&
              selected?.dataset === sample.dataset;
            const titleParts = [
              sample.name,
              hasPred ? "Prediction available" : "No prediction",
              !sample.path_exists ? "Data not on disk" : null,
            ].filter(Boolean);
            return (
              <li key={`${sample.dataset}/${sample.name}`}>
                <button
                  type="button"
                  className={
                    isSelected
                      ? "sample-browser__item sample-browser__item--selected"
                      : "sample-browser__item"
                  }
                  onClick={() =>
                    onSelect({ dataset: sample.dataset, name: sample.name })
                  }
                  title={titleParts.join(" · ")}
                >
                  <span
                    className={
                      hasPred
                        ? "sample-browser__pred-dot sample-browser__pred-dot--on"
                        : "sample-browser__pred-dot sample-browser__pred-dot--off"
                    }
                    aria-hidden
                  />
                  <span className="sample-browser__name">{sample.name}</span>
                  {!sample.path_exists && (
                    <span className="sample-browser__badge">missing</span>
                  )}
                </button>
              </li>
            );
          })}
          {filtered.length === 0 && (
            <li className="sample-browser__status">No samples in this split.</li>
          )}
        </ul>
      )}
    </div>
  );
}

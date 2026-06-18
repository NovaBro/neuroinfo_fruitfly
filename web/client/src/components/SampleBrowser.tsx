import { useMemo, useState } from "react";
import { SampleInfo, Split } from "../api/client";
import "./SampleBrowser.css";

const SPLITS: Split[] = ["train", "val", "test"];

interface SampleBrowserProps {
  samples: SampleInfo[];
  selected: string | null;
  onSelect: (name: string) => void;
  loading?: boolean;
}

export function SampleBrowser({
  samples,
  selected,
  onSelect,
  loading,
}: SampleBrowserProps) {
  const [activeSplit, setActiveSplit] = useState<Split>("train");
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return samples
      .filter((s) => s.split === activeSplit)
      .filter((s) => !q || s.name.toLowerCase().includes(q));
  }, [samples, activeSplit, query]);

  return (
    <div className="sample-browser">
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

      {loading ? (
        <p className="sample-browser__status">Loading samples…</p>
      ) : (
        <ul className="sample-browser__list">
          {filtered.map((sample) => (
            <li key={sample.name}>
              <button
                type="button"
                className={
                  selected === sample.name
                    ? "sample-browser__item sample-browser__item--selected"
                    : "sample-browser__item"
                }
                onClick={() => onSelect(sample.name)}
                title={sample.path_exists ? sample.name : "Data not on disk"}
              >
                <span className="sample-browser__name">{sample.name}</span>
                {!sample.path_exists && (
                  <span className="sample-browser__badge">missing</span>
                )}
              </button>
            </li>
          ))}
          {filtered.length === 0 && (
            <li className="sample-browser__status">No samples in this split.</li>
          )}
        </ul>
      )}
    </div>
  );
}

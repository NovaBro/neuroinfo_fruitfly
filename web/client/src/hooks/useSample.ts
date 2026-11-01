import { useCallback, useEffect, useState } from "react";
import {
  Dataset,
  getMeta,
  getMetrics,
  SampleMeta,
  SampleMetrics,
  SampleRef,
} from "../api/client";

export function useSampleMeta(
  sample: SampleRef | null,
  predictionSet?: string | null,
) {
  const [meta, setMeta] = useState<SampleMeta | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!sample) {
      setMeta(null);
      return;
    }
    setLoading(true);
    setError(null);
    setMeta(null);
    try {
      const data = await getMeta(sample.name, sample.dataset, predictionSet);
      setMeta(data);
    } catch (err) {
      setMeta(null);
      setError(err instanceof Error ? err.message : "Failed to load metadata");
    } finally {
      setLoading(false);
    }
  }, [sample?.name, sample?.dataset, predictionSet]);

  useEffect(() => {
    reload();
  }, [reload]);

  return { meta, loading, error, reload };
}

export function useSampleMetrics(
  sample: SampleRef | null,
  predictionSet?: string | null,
) {
  const [metrics, setMetrics] = useState<SampleMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!sample) {
      setMetrics(null);
      return;
    }
    setLoading(true);
    setError(null);
    setMetrics(null);
    try {
      const data = await getMetrics(sample.name, sample.dataset, predictionSet);
      setMetrics(data);
    } catch (err) {
      setMetrics(null);
      setError(err instanceof Error ? err.message : "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  }, [sample?.name, sample?.dataset, predictionSet]);

  useEffect(() => {
    reload();
  }, [reload]);

  return { metrics, loading, error, reload };
}

/** Convenience: build a SampleRef from dataset + name. */
export function sampleRef(dataset: Dataset, name: string): SampleRef {
  return { dataset, name };
}

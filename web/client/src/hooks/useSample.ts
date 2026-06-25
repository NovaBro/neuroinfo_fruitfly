import { useCallback, useEffect, useState } from "react";
import { getMeta, SampleMeta } from "../api/client";

export function useSampleMeta(
  sampleName: string | null,
  predictionSet?: string | null,
) {
  const [meta, setMeta] = useState<SampleMeta | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!sampleName) {
      setMeta(null);
      return;
    }
    setLoading(true);
    setError(null);
    setMeta(null);
    try {
      const data = await getMeta(sampleName, predictionSet);
      setMeta(data);
    } catch (err) {
      setMeta(null);
      setError(err instanceof Error ? err.message : "Failed to load metadata");
    } finally {
      setLoading(false);
    }
  }, [sampleName, predictionSet]);

  useEffect(() => {
    reload();
  }, [reload]);

  return { meta, loading, error, reload };
}

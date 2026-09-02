export type Split = "train" | "val" | "test";

export interface SampleInfo {
  split: Split;
  name: string;
  dataset: "completely" | "partly";
  path_exists: boolean;
  has_predicted?: boolean;
}

export interface VolumeMeta {
  shape: number[];
  dtype: string;
}

export interface SampleMeta {
  name: string;
  split: Split;
  dataset: "completely" | "partly";
  zarr_path: string;
  prediction_set?: string | null;
  raw: VolumeMeta;
  gt_instances: VolumeMeta;
  predicted_instances?: VolumeMeta | null;
}

/** One metric source's numbers grouped by detection threshold. */
export type MetricThresholds = Record<string, Record<string, number | null>>;

export interface TomlMetrics {
  source: string;
  general: Record<string, unknown>;
  summary: Record<string, number>;
  thresholds: MetricThresholds;
}

export interface CsvMetrics {
  source: string;
  file: string;
  scalars: Record<string, number | null>;
  thresholds: MetricThresholds;
}

export interface SampleMetrics {
  name: string;
  prediction_set?: string | null;
  toml: TomlMetrics | null;
  csv: CsvMetrics | null;
}

/** One heatmap column: a single metric across all samples. */
export interface AggregateMetricSpec {
  key: string;
  label: string;
  source: string;
  higherIsBetter: boolean;
}

/** Per-column mean/median across samples, plus the n each was taken over. */
export interface AggregateSummary {
  mean: Record<string, number | null>;
  median: Record<string, number | null>;
  n: Record<string, number>;
}

export interface AggregateMetrics {
  prediction_set?: string | null;
  threshold: string;
  thresholdChoices: string[];
  metrics: AggregateMetricSpec[];
  /** Sample names in row order. Derived summary rows are NOT included here. */
  samples: string[];
  /** values[sample][metricKey] -> number | null (missing cell). */
  values: Record<string, Record<string, number | null>>;
  summary: AggregateSummary;
}

export interface PredictionSet {
  /** Opaque selection handle (BiaPy: run dir relative to biapy_work_folder/results;
   *  PatchPerPix: source-prefixed, e.g. "ppp-numinst:.../8000"). */
  id: string;
  /** Human-readable set name, e.g. "train_3d_instance_segmentation_1". */
  name: string;
  /** Absolute path on the server (for display). */
  path: string;
  /** True for the server's default prediction set. */
  default: boolean;
  /** Model that produced this set. */
  source?: "biapy" | "ppp";
  /** PatchPerPix overlay flavour ("numinst" | "instances"); BiaPy: "instances". */
  kind?: string;
}

export interface HealthResponse {
  status: string;
  fisbe_root: string;
  fisbe_root_exists: boolean;
}

export type VolumeKind = "raw" | "gt" | "predicted";
export type AxisKind = "z" | "y" | "x";
export type ChannelParam = number | "all" | "off";

export const RAW_CHANNEL_CHOICES: { id: ChannelParam; label: string }[] = [
  { id: "off", label: "Off" },
  { id: 0, label: "R" },
  { id: 1, label: "G" },
  { id: 2, label: "B" },
  { id: "all", label: "All" },
];

const API_BASE = "/api";

function channelQuery(channel: ChannelParam): string {
  if (channel === "all") return "all";
  if (channel === "off") return "0";
  return String(channel);
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>(`${API_BASE}/health`);
}

export async function listSamples(): Promise<SampleInfo[]> {
  return fetchJson<SampleInfo[]>(`${API_BASE}/samples`);
}

export async function listPredictionSets(): Promise<PredictionSet[]> {
  return fetchJson<PredictionSet[]>(`${API_BASE}/prediction-sets`);
}

export async function getMeta(
  name: string,
  predictionSet?: string | null,
): Promise<SampleMeta> {
  const qs = predictionSet
    ? `?prediction_set=${encodeURIComponent(predictionSet)}`
    : "";
  return fetchJson<SampleMeta>(
    `${API_BASE}/samples/${encodeURIComponent(name)}/meta${qs}`,
  );
}

export async function getMetrics(
  name: string,
  predictionSet?: string | null,
): Promise<SampleMetrics> {
  const qs = predictionSet
    ? `?prediction_set=${encodeURIComponent(predictionSet)}`
    : "";
  return fetchJson<SampleMetrics>(
    `${API_BASE}/samples/${encodeURIComponent(name)}/metrics${qs}`,
  );
}

export async function getAggregateMetrics(
  predictionSet?: string | null,
  threshold?: string | null,
): Promise<AggregateMetrics> {
  const params = new URLSearchParams();
  if (predictionSet) params.set("prediction_set", predictionSet);
  if (threshold) params.set("threshold", threshold);
  const qs = params.toString();
  return fetchJson<AggregateMetrics>(
    `${API_BASE}/aggregate-metrics${qs ? `?${qs}` : ""}`,
  );
}

export function sliceUrl(
  name: string,
  opts: {
    volume?: VolumeKind;
    channel?: ChannelParam;
    axis?: AxisKind;
    index?: number;
  } = {},
): string {
  const params = new URLSearchParams();
  if (opts.volume) params.set("volume", opts.volume);
  if (opts.channel !== undefined) params.set("channel", channelQuery(opts.channel));
  if (opts.axis) params.set("axis", opts.axis);
  if (opts.index !== undefined) params.set("index", String(opts.index));
  const qs = params.toString();
  return `${API_BASE}/samples/${encodeURIComponent(name)}/slice.png${qs ? `?${qs}` : ""}`;
}

export function mipUrl(
  name: string,
  opts: { volume?: VolumeKind; channel?: ChannelParam } = {},
): string {
  const params = new URLSearchParams();
  if (opts.volume) params.set("volume", opts.volume);
  if (opts.channel !== undefined) params.set("channel", channelQuery(opts.channel));
  const qs = params.toString();
  return `${API_BASE}/samples/${encodeURIComponent(name)}/mip.png${qs ? `?${qs}` : ""}`;
}

export type FisbeMipHalf = "full" | "raw" | "gt";

/**
 * URL for a sample's pre-generated FISBe MIP PNG (from fisbe/mips). Unlike
 * {@link mipUrl}, this serves the shipped image rather than projecting the zarr.
 * `half` selects the raw (original colored) half, the GT-segmentation half, or
 * the full side-by-side composite.
 */
export function fisbeMipUrl(
  name: string,
  opts: { half?: FisbeMipHalf } = {},
): string {
  const params = new URLSearchParams();
  if (opts.half) params.set("half", opts.half);
  const qs = params.toString();
  return `${API_BASE}/samples/${encodeURIComponent(name)}/fisbe_mip.png${qs ? `?${qs}` : ""}`;
}

export interface VolumeData {
  data: Uint8Array;
  shape: [number, number, number];
  originalShape: [number, number, number];
  downsampleFactor: number;
  components: number;
}

export const VOLUME_MAX_SIZE_OPTIONS = [128, 192, 256, 320, 384, 512] as const;
export const DEFAULT_VOLUME_MAX_SIZE = 128;

function parseShapeHeader(header: string | null): [number, number, number] {
  if (!header) {
    throw new Error("Missing volume shape header");
  }
  const parts = header.split(",").map((s) => Number(s.trim()));
  if (parts.length !== 3 || parts.some((n) => !Number.isFinite(n))) {
    throw new Error(`Invalid volume shape header: ${header}`);
  }
  return [parts[0], parts[1], parts[2]];
}

export function volumeDataUrl(
  name: string,
  opts: {
    volume?: VolumeKind;
    channel?: ChannelParam;
    maxSize?: number;
    predictionSet?: string | null;
  } = {},
): string {
  const params = new URLSearchParams();
  if (opts.volume) params.set("volume", opts.volume);
  if (opts.channel !== undefined) params.set("channel", channelQuery(opts.channel));
  if (opts.maxSize !== undefined) params.set("max_size", String(opts.maxSize));
  if (opts.predictionSet) params.set("prediction_set", opts.predictionSet);
  const qs = params.toString();
  return `${API_BASE}/samples/${encodeURIComponent(name)}/volume.bin${qs ? `?${qs}` : ""}`;
}

/**
 * In-memory LRU cache of decoded {@link VolumeData}, keyed by the request URL
 * (which encodes sample/volume/channel/maxSize/predictionSet). Switching among
 * raw channels, or toggling overlays that don't depend on the channel, re-uses
 * the already-downloaded bytes instead of hitting the network + server
 * downsample again. FISBe volumes and BiaPy predictions are static on disk, so
 * a cached entry never goes stale.
 *
 * Bounded by a byte budget rather than an entry count: a single max_size=512
 * RGB volume is ~400 MB, so a count cap could blow memory, whereas at the
 * default max_size=128 every raw channel (~2–6 MB) of many samples fits.
 * Insertion order in the Map doubles as the LRU order (re-inserting on hit
 * moves an entry to the most-recently-used end).
 */
const VOLUME_CACHE_MAX_BYTES = 256 * 1024 * 1024;
const volumeCache = new Map<string, VolumeData>();
let volumeCacheBytes = 0;

function volumeEntryBytes(vol: VolumeData): number {
  return vol.data.byteLength;
}

function touchCacheEntry(key: string): VolumeData | undefined {
  const vol = volumeCache.get(key);
  if (!vol) return undefined;
  // Re-insert to move to the most-recently-used end.
  volumeCache.delete(key);
  volumeCache.set(key, vol);
  return vol;
}

function storeCacheEntry(key: string, vol: VolumeData): void {
  const bytes = volumeEntryBytes(vol);
  // Don't cache a single entry larger than the whole budget — it would evict
  // everything else and still not fit meaningfully.
  if (bytes > VOLUME_CACHE_MAX_BYTES) return;
  const existing = volumeCache.get(key);
  if (existing) volumeCacheBytes -= volumeEntryBytes(existing);
  volumeCache.set(key, vol);
  volumeCacheBytes += bytes;
  // Evict least-recently-used (front of insertion order) until under budget.
  while (volumeCacheBytes > VOLUME_CACHE_MAX_BYTES && volumeCache.size > 1) {
    const oldestKey = volumeCache.keys().next().value as string | undefined;
    if (oldestKey === undefined) break;
    const oldest = volumeCache.get(oldestKey);
    volumeCache.delete(oldestKey);
    if (oldest) volumeCacheBytes -= volumeEntryBytes(oldest);
  }
}

/**
 * Synchronously return a cached volume for these params without fetching, or
 * `undefined` on a miss. Callers use this to skip the loading state when a
 * channel/overlay switch can be satisfied entirely from cache.
 */
export function peekVolumeCache(
  name: string,
  opts: {
    volume?: VolumeKind;
    channel?: ChannelParam;
    maxSize?: number;
    predictionSet?: string | null;
  } = {},
): VolumeData | undefined {
  return touchCacheEntry(volumeDataUrl(name, opts));
}

/** Drop all cached volumes (frees the retained bytes). */
export function clearVolumeCache(): void {
  volumeCache.clear();
  volumeCacheBytes = 0;
}

export async function fetchVolumeData(
  name: string,
  opts: {
    volume?: VolumeKind;
    channel?: ChannelParam;
    maxSize?: number;
    predictionSet?: string | null;
    signal?: AbortSignal;
  } = {},
): Promise<VolumeData> {
  const url = volumeDataUrl(name, opts);
  const cached = touchCacheEntry(url);
  if (cached) return cached;

  const res = await fetch(url, { signal: opts.signal });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `Volume request failed: ${res.status}`);
  }

  const shape = parseShapeHeader(res.headers.get("X-Volume-Shape"));
  const originalShape = parseShapeHeader(res.headers.get("X-Original-Shape"));
  const factorHeader = res.headers.get("X-Downsample-Factor");
  const downsampleFactor = factorHeader ? Number(factorHeader) : 1;
  if (!Number.isFinite(downsampleFactor)) {
    throw new Error(`Invalid downsample factor header: ${factorHeader}`);
  }

  const componentsHeader = res.headers.get("X-Volume-Components");
  const components = componentsHeader ? Number(componentsHeader) : 1;
  if (!Number.isFinite(components) || components < 1) {
    throw new Error(`Invalid volume components header: ${componentsHeader}`);
  }

  const buffer = await res.arrayBuffer();
  const vol: VolumeData = {
    data: new Uint8Array(buffer),
    shape,
    originalShape,
    downsampleFactor,
    components,
  };
  storeCacheEntry(url, vol);
  return vol;
}

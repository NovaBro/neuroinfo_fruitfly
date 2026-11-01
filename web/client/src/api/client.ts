export type Split = "train" | "val" | "test";

export type Dataset =
  | "completely"
  | "partly"
  | "synth_vh"
  | "synth_vh_u8"
  | "1k_v1"
  | "1k_v1_u8"
  | "himanshu";

/** Sidebar / API selection key (bare stem + dataset). */
export interface SampleRef {
  dataset: Dataset;
  name: string;
}

export interface SampleInfo {
  split: Split;
  name: string;
  dataset: Dataset;
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
  dataset: Dataset;
  zarr_path: string;
  prediction_set?: string | null;
  raw: VolumeMeta;
  /** Null when the zarr has no gt_instances (e.g. himanshu). */
  gt_instances: VolumeMeta | null;
  predicted_instances?: VolumeMeta | null;
  /** Physical voxel size [vx, vy, vz] µm (X, Y, Z). */
  voxel_size_um: [number, number, number];
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

/** One FISBe website leaderboard cell (S · avF1 · C · clDiceTP · tp · FS · FM). */
export interface LeaderboardMetric {
  key: string;
  label: string;
  value: number | null;
  higher_is_better: boolean;
}

export interface SampleMetrics {
  name: string;
  prediction_set?: string | null;
  toml: TomlMetrics | null;
  csv: CsvMetrics | null;
  /** Curated FISBe website names; null/empty when unavailable. */
  leaderboard?: LeaderboardMetric[] | null;
}

/** One heatmap column: a single metric across all samples. */
export interface AggregateMetricSpec {
  key: string;
  label: string;
  source: string;
  higherIsBetter: boolean;
  /** When false, Aggregate renders plain numbers (no blue ramp). Default true. */
  colorScale?: boolean;
}

/** Per-column mean/median across samples, plus the n each was taken over. */
export interface AggregateSummary {
  mean: Record<string, number | null>;
  median: Record<string, number | null>;
  n: Record<string, number>;
}

/** One Aggregate heatmap partition (``{split}/{dataset}``, e.g. test/completely). */
export interface AggregateGroup {
  /** Stem-only row labels (dataset is in the group key). */
  samples: string[];
  /** values[sample][metricKey] -> number | null (missing cell). */
  values: Record<string, Record<string, number | null>>;
  summary: AggregateSummary;
}

/** Group key = ``{fisbe_split}/{dataset}`` (e.g. ``test/completely``). */
export type AggregateGroupKey = string;

export interface AggregateMetrics {
  prediction_set?: string | null;
  threshold: string;
  thresholdChoices: string[];
  metrics: AggregateMetricSpec[];
  /** Scored samples partitioned for the Aggregate tab. */
  groups: Record<AggregateGroupKey, AggregateGroup>;
  /** Stable table order from the server. */
  groupOrder?: AggregateGroupKey[];
}

/** One sample's predicted-instance size histogram (shared log bins). */
export interface InstanceSizeSampleHist {
  sample: string;
  counts: number[];
  n_instances: number;
}

export interface InstanceSizeGroup {
  samples: InstanceSizeSampleHist[];
}

export interface InstanceSizeHistograms {
  prediction_set?: string | null;
  /** False for numinst / missing set; UI shows ``reason``. */
  applicable: boolean;
  reason?: string | null;
  /** Log-spaced bin edges (length = counts.length + 1). */
  bin_edges: number[];
  groups: Record<AggregateGroupKey, InstanceSizeGroup>;
  groupOrder?: AggregateGroupKey[];
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

export async function listSamples(
  predictionSet?: string | null,
): Promise<SampleInfo[]> {
  const qs = predictionSet
    ? `?prediction_set=${encodeURIComponent(predictionSet)}`
    : "";
  return fetchJson<SampleInfo[]>(`${API_BASE}/samples${qs}`);
}

export async function listPredictionSets(): Promise<PredictionSet[]> {
  return fetchJson<PredictionSet[]>(`${API_BASE}/prediction-sets`);
}

function sampleQuery(
  dataset: Dataset,
  extra?: Record<string, string | undefined | null>,
): string {
  const params = new URLSearchParams();
  params.set("dataset", dataset);
  if (extra) {
    for (const [k, v] of Object.entries(extra)) {
      if (v != null && v !== "") params.set(k, v);
    }
  }
  return `?${params.toString()}`;
}

export async function getMeta(
  name: string,
  dataset: Dataset,
  predictionSet?: string | null,
): Promise<SampleMeta> {
  return fetchJson<SampleMeta>(
    `${API_BASE}/samples/${encodeURIComponent(name)}/meta${sampleQuery(
      dataset,
      { prediction_set: predictionSet },
    )}`,
  );
}

export async function getMetrics(
  name: string,
  dataset: Dataset,
  predictionSet?: string | null,
): Promise<SampleMetrics> {
  return fetchJson<SampleMetrics>(
    `${API_BASE}/samples/${encodeURIComponent(name)}/metrics${sampleQuery(
      dataset,
      { prediction_set: predictionSet },
    )}`,
  );
}

/**
 * In-memory cache of aggregate heatmap payloads, keyed by prediction set +
 * threshold. Scoring CSVs are static for a given prediction run, so entries
 * never go stale within a session. Switching samples or remounting the
 * Aggregate tab reuses the already-fetched matrix instead of hitting the
 * server again.
 */
const aggregateMetricsCache = new Map<string, AggregateMetrics>();

function aggregateCacheKey(
  predictionSet?: string | null,
  threshold?: string | null,
): string {
  return `${predictionSet ?? ""}|${threshold ?? ""}`;
}

/** Synchronously return a cached aggregate payload, or undefined on a miss. */
export function peekAggregateMetrics(
  predictionSet?: string | null,
  threshold?: string | null,
): AggregateMetrics | undefined {
  return aggregateMetricsCache.get(aggregateCacheKey(predictionSet, threshold));
}

export async function getAggregateMetrics(
  predictionSet?: string | null,
  threshold?: string | null,
): Promise<AggregateMetrics> {
  const key = aggregateCacheKey(predictionSet, threshold);
  const cached = aggregateMetricsCache.get(key);
  if (cached) return cached;

  const params = new URLSearchParams();
  if (predictionSet) params.set("prediction_set", predictionSet);
  if (threshold) params.set("threshold", threshold);
  const qs = params.toString();
  const data = await fetchJson<AggregateMetrics>(
    `${API_BASE}/aggregate-metrics${qs ? `?${qs}` : ""}`,
  );
  aggregateMetricsCache.set(key, data);
  return data;
}

/**
 * In-memory cache of instance-size histogram payloads, keyed by prediction
 * set. Label volumes are static for a run, so entries never go stale within a
 * session. The Instance sizes tab reuses a hit instead of rescanning volumes.
 */
const instanceSizeHistCache = new Map<string, InstanceSizeHistograms>();

export function peekInstanceSizeHistograms(
  predictionSet?: string | null,
): InstanceSizeHistograms | undefined {
  return instanceSizeHistCache.get(predictionSet ?? "");
}

export async function getInstanceSizeHistograms(
  predictionSet?: string | null,
): Promise<InstanceSizeHistograms> {
  const key = predictionSet ?? "";
  const cached = instanceSizeHistCache.get(key);
  if (cached) return cached;

  const params = new URLSearchParams();
  if (predictionSet) params.set("prediction_set", predictionSet);
  const qs = params.toString();
  const data = await fetchJson<InstanceSizeHistograms>(
    `${API_BASE}/instance-size-histograms${qs ? `?${qs}` : ""}`,
  );
  instanceSizeHistCache.set(key, data);
  return data;
}

export function sliceUrl(
  name: string,
  opts: {
    dataset: Dataset;
    volume?: VolumeKind;
    channel?: ChannelParam;
    axis?: AxisKind;
    index?: number;
  },
): string {
  const params = new URLSearchParams();
  params.set("dataset", opts.dataset);
  if (opts.volume) params.set("volume", opts.volume);
  if (opts.channel !== undefined) params.set("channel", channelQuery(opts.channel));
  if (opts.axis) params.set("axis", opts.axis);
  if (opts.index !== undefined) params.set("index", String(opts.index));
  return `${API_BASE}/samples/${encodeURIComponent(name)}/slice.png?${params}`;
}

export function mipUrl(
  name: string,
  opts: { dataset: Dataset; volume?: VolumeKind; channel?: ChannelParam },
): string {
  const params = new URLSearchParams();
  params.set("dataset", opts.dataset);
  if (opts.volume) params.set("volume", opts.volume);
  if (opts.channel !== undefined) params.set("channel", channelQuery(opts.channel));
  return `${API_BASE}/samples/${encodeURIComponent(name)}/mip.png?${params}`;
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
  opts: { dataset: Dataset; half?: FisbeMipHalf },
): string {
  const params = new URLSearchParams();
  params.set("dataset", opts.dataset);
  if (opts.half) params.set("half", opts.half);
  return `${API_BASE}/samples/${encodeURIComponent(name)}/fisbe_mip.png?${params}`;
}

export interface VolumeData {
  data: Uint8Array;
  shape: [number, number, number];
  originalShape: [number, number, number];
  downsampleFactor: number;
  components: number;
  /** Physical size of one original voxel [vx, vy, vz] µm (X, Y, Z). */
  voxelSizeUm: [number, number, number];
}

const DEFAULT_VOXEL_SIZE_UM: [number, number, number] = [0.44, 0.44, 0.44];

export const VOLUME_MAX_SIZE_OPTIONS = [
  128, 192, 256, 320, 384, 512, 768, 1024, 1500,
] as const;
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

function parseVoxelSizeHeader(header: string | null): [number, number, number] {
  if (!header) {
    return DEFAULT_VOXEL_SIZE_UM;
  }
  const parts = header.split(",").map((s) => Number(s.trim()));
  if (
    parts.length !== 3 ||
    parts.some((n) => !Number.isFinite(n) || n <= 0)
  ) {
    return DEFAULT_VOXEL_SIZE_UM;
  }
  return [parts[0], parts[1], parts[2]];
}

export function volumeDataUrl(
  name: string,
  opts: {
    dataset: Dataset;
    volume?: VolumeKind;
    channel?: ChannelParam;
    maxSize?: number;
    predictionSet?: string | null;
    /** Display-only: hide predicted instances smaller than this many voxels. */
    minSize?: number;
  },
): string {
  const params = new URLSearchParams();
  params.set("dataset", opts.dataset);
  if (opts.volume) params.set("volume", opts.volume);
  if (opts.channel !== undefined) params.set("channel", channelQuery(opts.channel));
  if (opts.maxSize !== undefined) params.set("max_size", String(opts.maxSize));
  if (opts.predictionSet) params.set("prediction_set", opts.predictionSet);
  if (opts.minSize !== undefined && opts.minSize > 0) {
    params.set("min_size", String(opts.minSize));
  }
  return `${API_BASE}/samples/${encodeURIComponent(name)}/volume.bin?${params}`;
}

/**
 * In-memory LRU cache of decoded {@link VolumeData}, keyed by the request URL
 * (which encodes sample/volume/channel/maxSize/predictionSet/minSize). Switching
 * among raw channels, or toggling overlays that don't depend on the channel,
 * re-uses the already-downloaded bytes instead of hitting the network + server
 * downsample again. FISBe volumes and BiaPy predictions are static on disk, so
 * a cached entry never goes stale.
 *
 * Bounded by a byte budget rather than an entry count: a single max_size=1500
 * raw/RGB volume can be hundreds of MB, so a count cap could blow memory,
 * whereas at the default max_size=128 every raw channel (~2–6 MB) of many
 * samples fits. The 256 MiB LRU still protects default browsing.
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
    dataset: Dataset;
    volume?: VolumeKind;
    channel?: ChannelParam;
    maxSize?: number;
    predictionSet?: string | null;
    minSize?: number;
  },
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
    dataset: Dataset;
    volume?: VolumeKind;
    channel?: ChannelParam;
    maxSize?: number;
    predictionSet?: string | null;
    minSize?: number;
    signal?: AbortSignal;
  },
): Promise<VolumeData> {
  const url = volumeDataUrl(name, opts);
  const cached = touchCacheEntry(url);
  if (cached) {
    console.debug("[volume]", {
      url,
      clientCache: "hit",
      totalMs: 0,
      bytes: cached.data.byteLength,
      shape: cached.shape,
    });
    return cached;
  }

  const t0 = performance.now();
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
  const totalMs = Math.round(performance.now() - t0);
  const computeHeader = res.headers.get("X-Volume-Compute-Ms");
  const computeMs = computeHeader != null ? Number(computeHeader) : undefined;
  const cache = res.headers.get("X-Volume-Cache") ?? undefined;
  console.debug("[volume]", {
    url,
    clientCache: "miss",
    totalMs,
    computeMs: Number.isFinite(computeMs) ? computeMs : undefined,
    cache,
    bytes: buffer.byteLength,
    shape,
  });

  const voxelSizeUm = parseVoxelSizeHeader(res.headers.get("X-Voxel-Size-Um"));

  const vol: VolumeData = {
    data: new Uint8Array(buffer),
    shape,
    originalShape,
    downsampleFactor,
    components,
    voxelSizeUm,
  };
  storeCacheEntry(url, vol);
  return vol;
}

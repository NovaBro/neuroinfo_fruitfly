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
  raw: VolumeMeta;
  gt_instances: VolumeMeta;
  predicted_instances?: VolumeMeta | null;
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

export async function getMeta(name: string): Promise<SampleMeta> {
  return fetchJson<SampleMeta>(
    `${API_BASE}/samples/${encodeURIComponent(name)}/meta`,
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
  } = {},
): string {
  const params = new URLSearchParams();
  if (opts.volume) params.set("volume", opts.volume);
  if (opts.channel !== undefined) params.set("channel", channelQuery(opts.channel));
  if (opts.maxSize !== undefined) params.set("max_size", String(opts.maxSize));
  const qs = params.toString();
  return `${API_BASE}/samples/${encodeURIComponent(name)}/volume.bin${qs ? `?${qs}` : ""}`;
}

export async function fetchVolumeData(
  name: string,
  opts: {
    volume?: VolumeKind;
    channel?: ChannelParam;
    maxSize?: number;
    signal?: AbortSignal;
  } = {},
): Promise<VolumeData> {
  const url = volumeDataUrl(name, opts);
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
  return {
    data: new Uint8Array(buffer),
    shape,
    originalShape,
    downsampleFactor,
    components,
  };
}

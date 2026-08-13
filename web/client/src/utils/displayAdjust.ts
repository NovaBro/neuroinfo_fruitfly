export const DISPLAY_MIN = 50;
export const DISPLAY_MAX = 300;
export const DEFAULT_BRIGHTNESS = 100;
export const DEFAULT_CONTRAST = 100;

/** Per-channel intensity for raw RGB (`channel=all`); 100 = neutral. */
export type ChannelGains = { r: number; g: number; b: number };

export const CHANNEL_GAIN_MIN = 0;
export const CHANNEL_GAIN_MAX = 300;
export const DEFAULT_CHANNEL_GAINS: ChannelGains = { r: 100, g: 100, b: 100 };

/** Map a 0–1 intensity through brightness/contrast (100 = neutral). */
export function adjustIntensity(
  value01: number,
  brightness: number,
  contrast: number,
): number {
  const b = brightness / 100;
  const c = contrast / 100;
  const centered = (value01 - 0.5) * c + 0.5;
  return Math.max(0, Math.min(1, centered * b));
}

/** Gamma < 1 lifts mid-tones; tied to brightness so higher slider = brighter overall. */
function displayGamma(brightness: number): number {
  const b = brightness / 100;
  return Math.max(0.35, 1.25 - b * 0.45);
}

function liftScalar(
  value01: number,
  brightness: number,
  contrast: number,
): number {
  const adjusted = adjustIntensity(value01, brightness, contrast);
  return Math.min(1, Math.pow(adjusted, displayGamma(brightness)));
}

function scaleChannel(value01: number, gainPct: number): number {
  return Math.max(0, Math.min(1, value01 * (gainPct / 100)));
}

/** Remap one RGB triple (0–1) with optional channel gains, then brightness/contrast. */
function remapRgb01(
  r: number,
  g: number,
  b: number,
  brightness: number,
  contrast: number,
  channelGains?: ChannelGains,
): [number, number, number] {
  const gains = channelGains ?? DEFAULT_CHANNEL_GAINS;
  const sr = scaleChannel(r, gains.r);
  const sg = scaleChannel(g, gains.g);
  const sb = scaleChannel(b, gains.b);
  const lum = Math.max(sr, sg, sb);

  if (lum <= 0) return [0, 0, 0];

  const targetLum = liftScalar(lum, brightness, contrast);
  const gain = targetLum / lum;
  return [
    Math.min(1, sr * gain),
    Math.min(1, sg * gain),
    Math.min(1, sb * gain),
  ];
}

/** Remap uint8 volume samples for display (1- or 3-component). */
export function remapVolumeUint8(
  source: Uint8Array,
  brightness: number,
  contrast: number,
  components = 1,
  channelGains?: ChannelGains,
): Uint8Array {
  const out = new Uint8Array(source.length);

  if (components === 1) {
    for (let i = 0; i < source.length; i++) {
      out[i] = Math.round(liftScalar(source[i] / 255, brightness, contrast) * 255);
    }
    return out;
  }

  // RGB: optional per-channel gains, then uniform luminance lift to preserve hue.
  for (let i = 0; i < source.length; i += 3) {
    const [r, g, b] = remapRgb01(
      source[i] / 255,
      source[i + 1] / 255,
      source[i + 2] / 255,
      brightness,
      contrast,
      channelGains,
    );
    out[i] = Math.round(r * 255);
    out[i + 1] = Math.round(g * 255);
    out[i + 2] = Math.round(b * 255);
  }

  return out;
}

/**
 * In-place remap of canvas ImageData RGBA pixels (alpha unchanged) using the
 * same RGB gain + brightness/contrast math as {@link remapVolumeUint8}.
 */
export function remapRgbUint8Clamped(
  data: Uint8ClampedArray,
  brightness: number,
  contrast: number,
  channelGains?: ChannelGains,
): void {
  for (let i = 0; i < data.length; i += 4) {
    const [r, g, b] = remapRgb01(
      data[i] / 255,
      data[i + 1] / 255,
      data[i + 2] / 255,
      brightness,
      contrast,
      channelGains,
    );
    data[i] = Math.round(r * 255);
    data[i + 1] = Math.round(g * 255);
    data[i + 2] = Math.round(b * 255);
  }
}

export function cssBrightnessContrast(
  brightness: number,
  contrast: number,
): string {
  return `brightness(${brightness}%) contrast(${contrast}%)`;
}

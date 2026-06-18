export const DISPLAY_MIN = 50;
export const DISPLAY_MAX = 300;
export const DEFAULT_BRIGHTNESS = 100;
export const DEFAULT_CONTRAST = 100;

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

/** Remap uint8 volume samples for display (1- or 3-component). */
export function remapVolumeUint8(
  source: Uint8Array,
  brightness: number,
  contrast: number,
  components = 1,
): Uint8Array {
  const out = new Uint8Array(source.length);

  if (components === 1) {
    for (let i = 0; i < source.length; i++) {
      out[i] = Math.round(liftScalar(source[i] / 255, brightness, contrast) * 255);
    }
    return out;
  }

  // RGB: scale each voxel uniformly to preserve hue while lifting luminance.
  for (let i = 0; i < source.length; i += 3) {
    const r = source[i] / 255;
    const g = source[i + 1] / 255;
    const b = source[i + 2] / 255;
    const lum = Math.max(r, g, b);

    if (lum <= 0) {
      out[i] = 0;
      out[i + 1] = 0;
      out[i + 2] = 0;
      continue;
    }

    const targetLum = liftScalar(lum, brightness, contrast);
    const gain = targetLum / lum;
    out[i] = Math.round(Math.min(1, r * gain) * 255);
    out[i + 1] = Math.round(Math.min(1, g * gain) * 255);
    out[i + 2] = Math.round(Math.min(1, b * gain) * 255);
  }

  return out;
}

export function cssBrightnessContrast(
  brightness: number,
  contrast: number,
): string {
  return `brightness(${brightness}%) contrast(${contrast}%)`;
}

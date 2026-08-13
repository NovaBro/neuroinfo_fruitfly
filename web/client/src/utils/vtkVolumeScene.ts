import "@kitware/vtk.js/Rendering/Profiles/Volume";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkInteractorStyleTrackballCamera from "@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera";
import { VolumeData } from "../api/client";
import { ChannelGains, remapVolumeUint8 } from "./displayAdjust";

export type VolumeMode = "raw" | "rgb" | "instance_rgb";

export type GenericRenderWindow = ReturnType<
  typeof vtkGenericRenderWindow.newInstance
>;

const DEFAULT_BACKGROUND: [number, number, number] = [0.06, 0.07, 0.09];

/** Create a GenericRenderWindow mounted on `container` with a trackball camera. */
export function createRenderWindow(
  container: HTMLElement,
  background: [number, number, number] = DEFAULT_BACKGROUND,
): GenericRenderWindow {
  const grw = vtkGenericRenderWindow.newInstance({
    background,
    listenWindowResize: false,
  });
  grw.setContainer(container);
  grw.resize();
  grw
    .getInteractor()
    .setInteractorStyle(vtkInteractorStyleTrackballCamera.newInstance());
  return grw;
}

/** Resize, reset the camera to fit all volumes, and render. */
export function fitCamera(grw: GenericRenderWindow): void {
  const renderer = grw.getRenderer();
  grw.resize();
  renderer.resetCamera();
  renderer.resetCameraClippingRange();
  grw.getRenderWindow().render();
}

/**
 * Rotate the camera in response to an arrow key. Returns true if `key` was an
 * arrow key (and the caller should `preventDefault`), false otherwise.
 */
export function rotateCameraByKey(
  grw: GenericRenderWindow,
  key: string,
  step: number,
): boolean {
  const renderer = grw.getRenderer();
  const camera = renderer.getActiveCamera();
  switch (key) {
    case "ArrowLeft":
      camera.azimuth(-step);
      break;
    case "ArrowRight":
      camera.azimuth(step);
      break;
    case "ArrowUp":
      camera.elevation(step);
      break;
    case "ArrowDown":
      camera.elevation(-step);
      break;
    default:
      return false;
  }
  renderer.resetCameraClippingRange();
  grw.getRenderWindow().render();
  return true;
}

export type VolumeLayer = {
  volume: ReturnType<typeof vtkVolume.newInstance>;
  mapper: ReturnType<typeof vtkVolumeMapper.newInstance>;
  imageData: ReturnType<typeof vtkImageData.newInstance>;
};

function opacityAtLevel(level: number, floor: number) {
  const t = level / 255;
  return Math.min(1, Math.max(floor, t));
}

function configureVolumeProperty(
  layer: VolumeLayer,
  mode: VolumeMode,
  opacity = 1,
) {
  const property = layer.volume.getProperty();
  const ctfun = vtkColorTransferFunction.newInstance();
  const ofun = vtkPiecewiseFunction.newInstance();
  const dims = layer.imageData.getDimensions();
  const spacing = layer.imageData.getSpacing();
  const diagonal = Math.hypot(
    Math.max(0, dims[0] - 1) * spacing[0],
    Math.max(0, dims[1] - 1) * spacing[1],
    Math.max(0, dims[2] - 1) * spacing[2],
  );

  property.setShade(false);
  property.setUseGradientOpacity(0, false);
  property.setInterpolationTypeToLinear();
  property.setScalarOpacityUnitDistance(0, Math.max(0.5, diagonal / 80));

  if (mode === "rgb" || mode === "instance_rgb") {
    property.setIndependentComponents(false);
    // For dependent 3-component (direct RGB) rendering vtk.js ignores the color
    // curve, but it still uses the transfer function's *range* to scale the
    // stored values in-shader (colorTextureScale = sscale / rangeWidth). A null
    // function lazily defaults to range [0, 1024], so uint8 RGB (sscale 255)
    // gets scaled to ~255/1024 ≈ 25% brightness — the "dark filter" where even
    // white renders muddy. A [0, 255] range makes colorTextureScale ≈ 1 so RGB
    // renders at full brightness, matching the single-channel path.
    ctfun.addRGBPoint(0, 0, 0, 0);
    ctfun.addRGBPoint(255, 1, 1, 1);
    property.setRGBTransferFunction(0, ctfun);
    ofun.addPoint(0, 0.0);
    ofun.addPoint(1, 0.9 * opacity);
    ofun.addPoint(16, 0.95 * opacity);
    ofun.addPoint(255, 1.0 * opacity);
    property.setScalarOpacity(0, ofun);
    return;
  }

  property.setIndependentComponents(false);
  ctfun.addRGBPoint(0, 0, 0, 0);
  ctfun.addRGBPoint(255, 1, 1, 1);
  ofun.addPoint(0, 0.0);
  ofun.addPoint(32, opacityAtLevel(32, 0.1));
  ofun.addPoint(96, opacityAtLevel(96, 0.3));
  ofun.addPoint(160, opacityAtLevel(160, 0.55));
  ofun.addPoint(255, opacityAtLevel(255, 1.0));
  property.setRGBTransferFunction(0, ctfun);
  property.setScalarOpacity(0, ofun);
}

function updateMapperSampling(layer: VolumeLayer) {
  const bounds = layer.imageData.getBounds();
  const diagonal = Math.hypot(
    bounds[1] - bounds[0],
    bounds[3] - bounds[2],
    bounds[5] - bounds[4],
  );
  layer.mapper.setAutoAdjustSampleDistances(true);
  layer.mapper.setSampleDistance(Math.max(0.25, diagonal / 256));
  layer.mapper.setMaximumSamplesPerRay(2000);
}

function setLayerScalars(
  layer: VolumeLayer,
  source: Uint8Array,
  components: number,
  brightness: number,
  contrast: number,
  channelGains?: ChannelGains,
) {
  const displayData = remapVolumeUint8(
    source,
    brightness,
    contrast,
    components,
    // Per-channel gains only apply to raw RGB, never to instance label colors.
    components === 3 ? channelGains : undefined,
  );
  const scalars = vtkDataArray.newInstance({
    name: "Scalars",
    numberOfComponents: components,
    values: displayData,
  });
  layer.imageData.getPointData().setScalars(scalars);
  layer.imageData.modified();
  layer.mapper.modified();
  layer.volume.modified();
}

function applyVolumeData(
  layer: VolumeLayer,
  vol: VolumeData,
  mode: VolumeMode,
  brightness: number,
  contrast: number,
  opacity = 1,
  channelGains?: ChannelGains,
) {
  configureVolumeProperty(layer, mode, opacity);

  const [z, y, x] = vol.shape;
  const [oz, oy, ox] = vol.originalShape;

  layer.imageData.setDimensions([x, y, z]);
  layer.imageData.setSpacing([1, oy / ox, oz / ox]);
  layer.imageData.setOrigin([0, 0, 0]);

  setLayerScalars(
    layer,
    vol.data,
    vol.components,
    brightness,
    contrast,
    mode === "rgb" ? channelGains : undefined,
  );
  updateMapperSampling(layer);
}

function createVolumeLayer(mode: VolumeMode): VolumeLayer {
  const imageData = vtkImageData.newInstance();
  const mapper = vtkVolumeMapper.newInstance();
  mapper.setInputData(imageData);
  mapper.setBlendModeToMaximumIntensity();

  const volume = vtkVolume.newInstance();
  volume.setMapper(mapper);
  volume.getProperty().setInterpolationTypeToLinear();

  const layer = { volume, mapper, imageData };
  configureVolumeProperty(layer, mode);
  return layer;
}

/**
 * Bundles a single VTK volume layer with the small amount of mutable state the
 * viewer tracks alongside it: whether it is currently shown, and the last
 * fetched source volume (kept so brightness/contrast tweaks can re-remap the
 * display without re-fetching). Collapses the three identical raw/predicted/gt
 * layers in the viewer into one reusable unit.
 */
export class VolumeLayerController {
  readonly layer: VolumeLayer;
  visible = false;
  source: VolumeData | null = null;
  private mode: VolumeMode;

  constructor(mode: VolumeMode) {
    this.mode = mode;
    this.layer = createVolumeLayer(mode);
  }

  get volume() {
    return this.layer.volume;
  }

  /** Load a freshly-fetched volume, configure it, and make the layer visible. */
  applyData(
    vol: VolumeData,
    mode: VolumeMode,
    brightness: number,
    contrast: number,
    opacity = 1,
    channelGains?: ChannelGains,
  ) {
    this.mode = mode;
    applyVolumeData(
      this.layer,
      vol,
      mode,
      brightness,
      contrast,
      opacity,
      channelGains,
    );
    this.source = vol;
    this.layer.volume.setVisibility(true);
    this.visible = true;
  }

  /** Re-remap the cached source for new brightness/contrast/gains (no re-fetch). */
  refreshScalars(
    brightness: number,
    contrast: number,
    channelGains?: ChannelGains,
  ) {
    if (!this.visible || !this.source) return;
    setLayerScalars(
      this.layer,
      this.source.data,
      this.source.components,
      brightness,
      contrast,
      this.mode === "rgb" ? channelGains : undefined,
    );
  }

  /** Re-apply the transfer functions for a new opacity, if currently shown. */
  setOpacity(mode: VolumeMode, opacity: number) {
    if (!this.visible) return;
    configureVolumeProperty(this.layer, mode, opacity);
  }

  /** Hide the layer and drop its cached source. */
  hide() {
    this.layer.volume.setVisibility(false);
    this.visible = false;
    this.source = null;
  }
}

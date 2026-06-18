import { useCallback, useEffect, useRef, useState } from "react";
import "@kitware/vtk.js/Rendering/Profiles/Volume";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkInteractorStyleTrackballCamera from "@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera";
import {
  ChannelParam,
  DEFAULT_VOLUME_MAX_SIZE,
  fetchVolumeData,
  RAW_CHANNEL_CHOICES,
  SampleMeta,
  VolumeData,
  VOLUME_MAX_SIZE_OPTIONS,
} from "../api/client";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import {
  DEFAULT_BRIGHTNESS,
  DEFAULT_CONTRAST,
  DISPLAY_MAX,
  DISPLAY_MIN,
  remapVolumeUint8,
} from "../utils/displayAdjust";
import "./VolumeViewer3D.css";

const ROTATE_STEP = 5;

interface VolumeViewer3DProps {
  sampleName: string;
  meta: SampleMeta;
}

type VolumeMode = "raw" | "rgb" | "instance_rgb";

type VolumeLayer = {
  volume: ReturnType<typeof vtkVolume.newInstance>;
  mapper: ReturnType<typeof vtkVolumeMapper.newInstance>;
  imageData: ReturnType<typeof vtkImageData.newInstance>;
};

type VtkContext = {
  genericRenderWindow: ReturnType<typeof vtkGenericRenderWindow.newInstance>;
  raw: VolumeLayer;
  predicted: VolumeLayer;
};

function opacityAtLevel(level: number, floor: number) {
  const t = level / 255;
  return Math.min(1, Math.max(floor, t));
}

function configureVolumeProperty(layer: VolumeLayer, mode: VolumeMode) {
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
    property.setRGBTransferFunction(0, null);
    ofun.addPoint(0, 0.0);
    ofun.addPoint(1, 0.9);
    ofun.addPoint(16, 0.95);
    ofun.addPoint(255, 1.0);
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
) {
  const displayData = remapVolumeUint8(
    source,
    brightness,
    contrast,
    components,
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
) {
  configureVolumeProperty(layer, mode);

  const [z, y, x] = vol.shape;
  const [oz, oy, ox] = vol.originalShape;

  layer.imageData.setDimensions([x, y, z]);
  layer.imageData.setSpacing([1, oy / ox, oz / ox]);
  layer.imageData.setOrigin([0, 0, 0]);

  setLayerScalars(layer, vol.data, vol.components, brightness, contrast);
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

export function VolumeViewer3D({ sampleName, meta }: VolumeViewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const vtkRef = useRef<VtkContext | null>(null);
  const vtkGenerationRef = useRef(0);
  const rawVisibleRef = useRef(false);
  const predictedVisibleRef = useRef(false);
  const rawSourceRef = useRef<VolumeData | null>(null);
  const predictedSourceRef = useRef<VolumeData | null>(null);
  const [vtkReady, setVtkReady] = useState(false);
  const [channel, setChannel] = useState<ChannelParam>(0);
  const [brightness, setBrightness] = useState(DEFAULT_BRIGHTNESS);
  const [contrast, setContrast] = useState(DEFAULT_CONTRAST);
  const [maxSize, setMaxSize] = useState(DEFAULT_VOLUME_MAX_SIZE);
  const debouncedMaxSize = useDebouncedValue(maxSize, 400);
  const [showPredicted, setShowPredicted] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [volumeInfo, setVolumeInfo] = useState<string | null>(null);

  const hasPredicted = meta.predicted_instances != null;
  const channelCount = meta.raw.shape[0];
  const rawMode: VolumeMode = channel === "all" ? "rgb" : "raw";
  const showRaw = channel !== "off";

  const renderScene = useCallback(() => {
    const vtk = vtkRef.current;
    if (!vtk) return;
    vtk.genericRenderWindow.getRenderWindow().render();
  }, []);

  const fitCameraAndRender = useCallback(() => {
    const vtk = vtkRef.current;
    if (!vtk) return;
    const grw = vtk.genericRenderWindow;
    const renderer = grw.getRenderer();
    grw.resize();
    renderer.resetCamera();
    renderer.resetCameraClippingRange();
    grw.getRenderWindow().render();
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const generation = ++vtkGenerationRef.current;
    const genericRenderWindow = vtkGenericRenderWindow.newInstance({
      background: [0.06, 0.07, 0.09],
      listenWindowResize: false,
    });
    genericRenderWindow.setContainer(container);
    genericRenderWindow.resize();

    const interactor = genericRenderWindow.getInteractor();
    interactor.setInteractorStyle(
      vtkInteractorStyleTrackballCamera.newInstance(),
    );

    const renderer = genericRenderWindow.getRenderer();
    const raw = createVolumeLayer("raw");
    const predicted = createVolumeLayer("instance_rgb");

    renderer.addVolume(raw.volume);
    renderer.addVolume(predicted.volume);
    predicted.volume.setVisibility(false);
    raw.volume.setVisibility(false);

    vtkRef.current = { genericRenderWindow, raw, predicted };
    rawVisibleRef.current = false;
    predictedVisibleRef.current = false;
    setVtkReady(true);

    const resizeObserver = new ResizeObserver(() => {
      if (vtkGenerationRef.current === generation) {
        genericRenderWindow.resize();
        renderScene();
      }
    });
    resizeObserver.observe(container);

    requestAnimationFrame(() => {
      if (vtkGenerationRef.current === generation) {
        fitCameraAndRender();
      }
    });

    return () => {
      vtkGenerationRef.current += 1;
      setVtkReady(false);
      resizeObserver.disconnect();
      renderer.removeVolume(raw.volume);
      renderer.removeVolume(predicted.volume);
      genericRenderWindow.delete();
      vtkRef.current = null;
      rawVisibleRef.current = false;
      predictedVisibleRef.current = false;
      rawSourceRef.current = null;
      predictedSourceRef.current = null;
    };
  }, [renderScene, fitCameraAndRender]);

  useEffect(() => {
    setChannel(0);
    setBrightness(DEFAULT_BRIGHTNESS);
    setContrast(DEFAULT_CONTRAST);
    setMaxSize(DEFAULT_VOLUME_MAX_SIZE);
    setShowPredicted(hasPredicted);
    setVolumeInfo(null);
    setError(null);
    rawSourceRef.current = null;
    predictedSourceRef.current = null;
  }, [sampleName, hasPredicted]);

  useEffect(() => {
    if (!vtkReady) return;
    const vtk = vtkRef.current;
    if (!vtk) return;

    if (rawVisibleRef.current && rawSourceRef.current) {
      setLayerScalars(
        vtk.raw,
        rawSourceRef.current.data,
        rawSourceRef.current.components,
        brightness,
        contrast,
      );
    }
    if (predictedVisibleRef.current && predictedSourceRef.current) {
      setLayerScalars(
        vtk.predicted,
        predictedSourceRef.current.data,
        predictedSourceRef.current.components,
        brightness,
        contrast,
      );
    }
    renderScene();
  }, [vtkReady, brightness, contrast, renderScene]);

  useEffect(() => {
    if (!vtkReady) return;
    const generation = vtkGenerationRef.current;
    const vtk = vtkRef.current;
    if (!vtk) return;

    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    setError(null);

    async function loadVolumes() {
      try {
        const infoParts: string[] = [];

        if (showRaw) {
          const rawVol = await fetchVolumeData(sampleName, {
            volume: "raw",
            channel,
            maxSize: debouncedMaxSize,
            signal: controller.signal,
          });
          if (cancelled || vtkGenerationRef.current !== generation) return;

          rawSourceRef.current = rawVol;
          applyVolumeData(vtk!.raw, rawVol, rawMode, brightness, contrast);
          vtk!.raw.volume.setVisibility(true);
          rawVisibleRef.current = true;
          fitCameraAndRender();

          const [z, y, x] = rawVol.shape;
          const [oz, oy, ox] = rawVol.originalShape;
          const channelLabel =
            channel === "all" ? "RGB" : `ch ${channel}`;
          infoParts.push(
            `${channelLabel} ${z}×${y}×${x} (×${rawVol.downsampleFactor}, max ${debouncedMaxSize}) from ${oz}×${oy}×${ox}`,
          );
        } else {
          vtk!.raw.volume.setVisibility(false);
          rawVisibleRef.current = false;
          rawSourceRef.current = null;
          infoParts.push("Raw off");
        }

        if (hasPredicted && showPredicted) {
          try {
            const predictedVol = await fetchVolumeData(sampleName, {
              volume: "predicted",
              maxSize: debouncedMaxSize,
              signal: controller.signal,
            });
            if (cancelled || vtkGenerationRef.current !== generation) return;
            predictedSourceRef.current = predictedVol;
            applyVolumeData(
              vtk!.predicted,
              predictedVol,
              "instance_rgb",
              brightness,
              contrast,
            );
            vtk!.predicted.volume.setVisibility(true);
            predictedVisibleRef.current = true;
            infoParts.push("predicted overlay");
          } catch (predictedErr) {
            if (controller.signal.aborted || cancelled) return;
            vtk!.predicted.volume.setVisibility(false);
            predictedVisibleRef.current = false;
            predictedSourceRef.current = null;
            console.warn("Predicted overlay unavailable:", predictedErr);
          }
        } else {
          vtk!.predicted.volume.setVisibility(false);
          predictedVisibleRef.current = false;
          predictedSourceRef.current = null;
        }

        if (cancelled || vtkGenerationRef.current !== generation) return;

        setVolumeInfo(infoParts.join(" · "));
        fitCameraAndRender();
        requestAnimationFrame(() => {
          if (!cancelled && vtkGenerationRef.current === generation) {
            fitCameraAndRender();
          }
        });
      } catch (err) {
        if (controller.signal.aborted || cancelled) return;
        if (vtkGenerationRef.current !== generation) return;
        setError(err instanceof Error ? err.message : String(err));
        setVolumeInfo(null);
      } finally {
        if (!cancelled && vtkGenerationRef.current === generation) {
          setLoading(false);
        }
      }
    }

    void loadVolumes();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [
    vtkReady,
    sampleName,
    channel,
    rawMode,
    showRaw,
    hasPredicted,
    showPredicted,
    debouncedMaxSize,
    renderScene,
    fitCameraAndRender,
  ]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      const vtk = vtkRef.current;
      if (!vtk) return;

      const renderer = vtk.genericRenderWindow.getRenderer();
      const camera = renderer.getActiveCamera();
      const renderWindow = vtk.genericRenderWindow.getRenderWindow();

      let handled = true;
      switch (e.key) {
        case "ArrowLeft":
          camera.azimuth(-ROTATE_STEP);
          break;
        case "ArrowRight":
          camera.azimuth(ROTATE_STEP);
          break;
        case "ArrowUp":
          camera.elevation(ROTATE_STEP);
          break;
        case "ArrowDown":
          camera.elevation(-ROTATE_STEP);
          break;
        default:
          handled = false;
      }

      if (handled) {
        e.preventDefault();
        renderer.resetCameraClippingRange();
        renderWindow.render();
      }
    },
    [],
  );

  const maxSizeIndex = VOLUME_MAX_SIZE_OPTIONS.indexOf(
    maxSize as (typeof VOLUME_MAX_SIZE_OPTIONS)[number],
  );
  const sliderIndex = maxSizeIndex >= 0 ? maxSizeIndex : 2;

  return (
    <div className="volume-viewer-3d">
      <div className="volume-viewer-3d__controls">
        <div className="volume-viewer-3d__group">
          <span className="volume-viewer-3d__label">Channel</span>
          <div className="volume-viewer-3d__btn-row">
            {RAW_CHANNEL_CHOICES.filter(
              (c) =>
                c.id === "off" ||
                c.id === "all" ||
                (typeof c.id === "number" && c.id < channelCount),
            ).map(({ id, label }) => (
              <button
                key={label}
                type="button"
                className={
                  channel === id
                    ? "volume-viewer-3d__btn volume-viewer-3d__btn--active"
                    : "volume-viewer-3d__btn"
                }
                onClick={() => setChannel(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="volume-viewer-3d__group volume-viewer-3d__group--slider">
          <span className="volume-viewer-3d__label">
            Brightness ({brightness}%)
          </span>
          <input
            type="range"
            min={DISPLAY_MIN}
            max={DISPLAY_MAX}
            value={brightness}
            onChange={(e) => setBrightness(Number(e.target.value))}
          />
        </div>

        <div className="volume-viewer-3d__group volume-viewer-3d__group--slider">
          <span className="volume-viewer-3d__label">Contrast ({contrast}%)</span>
          <input
            type="range"
            min={DISPLAY_MIN}
            max={DISPLAY_MAX}
            value={contrast}
            onChange={(e) => setContrast(Number(e.target.value))}
          />
        </div>

        <div className="volume-viewer-3d__group volume-viewer-3d__group--slider">
          <span className="volume-viewer-3d__label">
            Resolution (max edge {maxSize}px)
          </span>
          <input
            type="range"
            min={0}
            max={VOLUME_MAX_SIZE_OPTIONS.length - 1}
            value={sliderIndex}
            onChange={(e) =>
              setMaxSize(VOLUME_MAX_SIZE_OPTIONS[Number(e.target.value)])
            }
          />
          <span className="volume-viewer-3d__slider-hint">
            Lower = faster · Higher = sharper
          </span>
        </div>

        {hasPredicted && (
          <div className="volume-viewer-3d__group">
            <span className="volume-viewer-3d__label">Overlay</span>
            <label className="volume-viewer-3d__checkbox">
              <input
                type="checkbox"
                checked={showPredicted}
                onChange={(e) => setShowPredicted(e.target.checked)}
              />
              Predicted instances (BiaPy)
            </label>
          </div>
        )}
      </div>

      <div
        className="volume-viewer-3d__viewport"
        tabIndex={0}
        onKeyDown={handleKeyDown}
        role="application"
        aria-label="3D volume viewer"
      >
        <div ref={containerRef} className="volume-viewer-3d__canvas" />
        {loading && (
          <div className="volume-viewer-3d__overlay">Loading volume…</div>
        )}
        {error && (
          <div className="volume-viewer-3d__overlay volume-viewer-3d__overlay--error">
            {error}
          </div>
        )}
        <p className="volume-viewer-3d__hint">
          Drag to rotate · Arrow keys to rotate (click viewer to focus)
        </p>
      </div>

      {volumeInfo && (
        <p className="volume-viewer-3d__footer">
          {volumeInfo}
          {hasPredicted && meta.predicted_instances
            ? ` · Predicted (Z,Y,X): ${meta.predicted_instances.shape.join(" × ")}`
            : ""}
          {" · "}Raw shape (C,Z,Y,X): {meta.raw.shape.join(" × ")}
        </p>
      )}
    </div>
  );
}

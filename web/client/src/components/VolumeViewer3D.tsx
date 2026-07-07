import { useCallback, useEffect, useRef, useState } from "react";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkInteractorStyleTrackballCamera from "@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera";
import {
  ChannelParam,
  DEFAULT_VOLUME_MAX_SIZE,
  fetchVolumeData,
  FisbeMipHalf,
  fisbeMipUrl,
  RAW_CHANNEL_CHOICES,
  SampleMeta,
  VOLUME_MAX_SIZE_OPTIONS,
} from "../api/client";
import { SliceImage } from "./SliceImage";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import {
  DEFAULT_BRIGHTNESS,
  DEFAULT_CONTRAST,
  DISPLAY_MAX,
  DISPLAY_MIN,
} from "../utils/displayAdjust";
import {
  VolumeLayerController,
  VolumeMode,
} from "../utils/vtkVolumeScene";
import "./VolumeViewer3D.css";

const ROTATE_STEP = 5;

// Returned by loadOverlay when the request finished but the viewer moved on
// (unmounted or a newer load started) — the caller must abort the whole sequence.
const STALE = Symbol("stale");

interface VolumeViewer3DProps {
  sampleName: string;
  meta: SampleMeta;
  predictionSet?: string | null;
}

type VtkContext = {
  genericRenderWindow: ReturnType<typeof vtkGenericRenderWindow.newInstance>;
  raw: VolumeLayerController;
  predicted: VolumeLayerController;
  gt: VolumeLayerController;
};

export function VolumeViewer3D({
  sampleName,
  meta,
  predictionSet,
}: VolumeViewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const vtkRef = useRef<VtkContext | null>(null);
  const vtkGenerationRef = useRef(0);
  const [vtkReady, setVtkReady] = useState(false);
  const [channel, setChannel] = useState<ChannelParam>(0);
  const [brightness, setBrightness] = useState(DEFAULT_BRIGHTNESS);
  const [contrast, setContrast] = useState(DEFAULT_CONTRAST);
  const [maxSize, setMaxSize] = useState(DEFAULT_VOLUME_MAX_SIZE);
  const debouncedMaxSize = useDebouncedValue(maxSize, 400);
  const [showPredicted, setShowPredicted] = useState(true);
  const [showGt, setShowGt] = useState(false);
  const [predictedOpacity, setPredictedOpacity] = useState(100);
  const [gtOpacity, setGtOpacity] = useState(100);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [volumeInfo, setVolumeInfo] = useState<string | null>(null);
  // Which half of the shipped FISBe MIP composite to show beneath the 3D render:
  // the original colored raw MIP, the GT instance segmentation, or both.
  const [mipHalf, setMipHalf] = useState<FisbeMipHalf>("full");

  const hasPredicted = meta.predicted_instances != null;
  const hasGt = meta.gt_instances != null;
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
    const raw = new VolumeLayerController("raw");
    const predicted = new VolumeLayerController("instance_rgb");
    const gt = new VolumeLayerController("instance_rgb");

    renderer.addVolume(raw.volume);
    renderer.addVolume(predicted.volume);
    renderer.addVolume(gt.volume);
    predicted.volume.setVisibility(false);
    gt.volume.setVisibility(false);
    raw.volume.setVisibility(false);

    vtkRef.current = { genericRenderWindow, raw, predicted, gt };
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
      renderer.removeVolume(gt.volume);
      genericRenderWindow.delete();
      vtkRef.current = null;
    };
  }, [renderScene, fitCameraAndRender]);

  useEffect(() => {
    setChannel(0);
    setBrightness(DEFAULT_BRIGHTNESS);
    setContrast(DEFAULT_CONTRAST);
    setMaxSize(DEFAULT_VOLUME_MAX_SIZE);
    setShowPredicted(hasPredicted);
    setShowGt(false);
    setPredictedOpacity(100);
    setGtOpacity(100);
    setVolumeInfo(null);
    setError(null);
    const vtk = vtkRef.current;
    if (vtk) {
      vtk.raw.source = null;
      vtk.predicted.source = null;
      vtk.gt.source = null;
    }
  }, [sampleName, hasPredicted]);

  useEffect(() => {
    if (!vtkReady) return;
    const vtk = vtkRef.current;
    if (!vtk) return;

    for (const ctrl of [vtk.raw, vtk.predicted, vtk.gt]) {
      ctrl.refreshScalars(brightness, contrast);
    }
    renderScene();
  }, [vtkReady, brightness, contrast, renderScene]);

  useEffect(() => {
    if (!vtkReady) return;
    const vtk = vtkRef.current;
    if (!vtk) return;

    vtk.predicted.setOpacity("instance_rgb", predictedOpacity / 100);
    vtk.gt.setOpacity("instance_rgb", gtOpacity / 100);
    renderScene();
  }, [vtkReady, predictedOpacity, gtOpacity, renderScene]);

  useEffect(() => {
    if (!vtkReady) return;
    const generation = vtkGenerationRef.current;
    const vtk = vtkRef.current;
    if (!vtk) return;

    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    setError(null);

    // Fetch and apply an instance overlay (predicted or gt). Returns the info
    // label to show on success, null when disabled/unavailable, or STALE when
    // the load was superseded and the caller should bail out entirely.
    async function loadOverlay(
      ctrl: VolumeLayerController,
      opts: {
        enabled: boolean;
        volume: "predicted" | "gt";
        opacity: number;
        label: string;
        predictionSet?: string | null;
      },
    ): Promise<string | null | typeof STALE> {
      if (!opts.enabled) {
        ctrl.hide();
        return null;
      }
      try {
        const vol = await fetchVolumeData(sampleName, {
          volume: opts.volume,
          maxSize: debouncedMaxSize,
          predictionSet: opts.predictionSet,
          signal: controller.signal,
        });
        if (cancelled || vtkGenerationRef.current !== generation) return STALE;
        ctrl.applyData(vol, "instance_rgb", brightness, contrast, opts.opacity);
        return opts.label;
      } catch (err) {
        if (controller.signal.aborted || cancelled) return STALE;
        ctrl.hide();
        console.warn(`${opts.label} unavailable:`, err);
        return null;
      }
    }

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

          vtk!.raw.applyData(rawVol, rawMode, brightness, contrast);
          fitCameraAndRender();

          const [z, y, x] = rawVol.shape;
          const [oz, oy, ox] = rawVol.originalShape;
          const channelLabel =
            channel === "all" ? "RGB" : `ch ${channel}`;
          infoParts.push(
            `${channelLabel} ${z}×${y}×${x} (×${rawVol.downsampleFactor}, max ${debouncedMaxSize}) from ${oz}×${oy}×${ox}`,
          );
        } else {
          vtk!.raw.hide();
          infoParts.push("Raw off");
        }

        const predictedInfo = await loadOverlay(vtk!.predicted, {
          enabled: hasPredicted && showPredicted,
          volume: "predicted",
          predictionSet,
          opacity: predictedOpacity / 100,
          label: "predicted overlay",
        });
        if (predictedInfo === STALE) return;
        if (predictedInfo) infoParts.push(predictedInfo);

        const gtInfo = await loadOverlay(vtk!.gt, {
          enabled: hasGt && showGt,
          volume: "gt",
          opacity: gtOpacity / 100,
          label: "ground-truth overlay",
        });
        if (gtInfo === STALE) return;
        if (gtInfo) infoParts.push(gtInfo);

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
    predictionSet,
    hasGt,
    showGt,
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

        {(hasGt || hasPredicted) && (
          <div className="volume-viewer-3d__group">
            <span className="volume-viewer-3d__label">Overlay</span>
            {hasGt && (
              <label className="volume-viewer-3d__checkbox">
                <input
                  type="checkbox"
                  checked={showGt}
                  onChange={(e) => setShowGt(e.target.checked)}
                />
                Ground truth (Zarr)
              </label>
            )}
            {hasGt && showGt && (
              <div className="volume-viewer-3d__group volume-viewer-3d__group--slider">
                <span className="volume-viewer-3d__label">
                  GT opacity ({gtOpacity}%)
                </span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={gtOpacity}
                  onChange={(e) => setGtOpacity(Number(e.target.value))}
                />
              </div>
            )}
            {hasPredicted && (
              <label className="volume-viewer-3d__checkbox">
                <input
                  type="checkbox"
                  checked={showPredicted}
                  onChange={(e) => setShowPredicted(e.target.checked)}
                />
                Predicted instances (BiaPy)
              </label>
            )}
            {hasPredicted && showPredicted && (
              <div className="volume-viewer-3d__group volume-viewer-3d__group--slider">
                <span className="volume-viewer-3d__label">
                  Predicted opacity ({predictedOpacity}%)
                </span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={predictedOpacity}
                  onChange={(e) => setPredictedOpacity(Number(e.target.value))}
                />
              </div>
            )}
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

      <div className="volume-viewer-3d__mip">
        <span className="volume-viewer-3d__label">FISBe MIP</span>
        <div className="volume-viewer-3d__btn-row">
          {(
            [
              ["full", "Both"],
              ["raw", "Original"],
              ["gt", "GT segmentation"],
            ] as [FisbeMipHalf, string][]
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={
                mipHalf === value
                  ? "volume-viewer-3d__btn volume-viewer-3d__btn--active"
                  : "volume-viewer-3d__btn"
              }
              onClick={() => setMipHalf(value)}
            >
              {label}
            </button>
          ))}
        </div>
        <SliceImage
          key={`${sampleName}-${mipHalf}`}
          url={fisbeMipUrl(sampleName, { half: mipHalf })}
          alt={`FISBe maximum-intensity projection of ${sampleName} (${mipHalf})`}
          className="volume-viewer-3d__mip-img"
          brightness={brightness}
          contrast={contrast}
        />
      </div>

      {volumeInfo && (
        <p className="volume-viewer-3d__footer">
          {volumeInfo}
          {hasPredicted && meta.predicted_instances
            ? ` · Predicted (Z,Y,X): ${meta.predicted_instances.shape.join(" × ")}`
            : ""}
          {showGt && hasGt
            ? ` · GT (C,Z,Y,X): ${meta.gt_instances.shape.join(" × ")}`
            : ""}
          {" · "}Raw shape (C,Z,Y,X): {meta.raw.shape.join(" × ")}
        </p>
      )}
    </div>
  );
}

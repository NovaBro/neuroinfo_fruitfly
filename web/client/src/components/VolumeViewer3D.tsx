import { useCallback, useEffect, useRef, useState } from "react";
import {
  ChannelParam,
  DEFAULT_VOLUME_MAX_SIZE,
  fetchVolumeData,
  FisbeMipHalf,
  fisbeMipUrl,
  SampleMeta,
} from "../api/client";
import { SliceImage } from "./SliceImage";
import { VolumeControls } from "./VolumeControls";
import { Volume3DView } from "./Volume3DView";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import { CameraSync } from "../utils/cameraSync";
import { DEFAULT_BRIGHTNESS, DEFAULT_CONTRAST } from "../utils/displayAdjust";
import {
  createRenderWindow,
  fitCamera,
  GenericRenderWindow,
  rotateCameraByKey,
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
  genericRenderWindow: GenericRenderWindow;
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
  // One camera-link group shared by the main render and the two sub-views.
  const cameraSyncRef = useRef<CameraSync>();
  if (!cameraSyncRef.current) cameraSyncRef.current = new CameraSync("main");
  const cameraSync = cameraSyncRef.current;
  const [linkViews, setLinkViews] = useState(true);
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
    fitCamera(vtk.genericRenderWindow);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const generation = ++vtkGenerationRef.current;
    const genericRenderWindow = createRenderWindow(container);

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

    const unregisterSync = cameraSync.register({
      id: "main",
      camera: renderer.getActiveCamera(),
      renderer,
      renderWindow: genericRenderWindow.getRenderWindow(),
    });

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
      unregisterSync();
      renderer.removeVolume(raw.volume);
      renderer.removeVolume(predicted.volume);
      renderer.removeVolume(gt.volume);
      genericRenderWindow.delete();
      vtkRef.current = null;
    };
  }, [cameraSync, renderScene, fitCameraAndRender]);

  useEffect(() => {
    cameraSync.setEnabled(linkViews);
  }, [cameraSync, linkViews]);

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

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    const vtk = vtkRef.current;
    if (!vtk) return;
    if (rotateCameraByKey(vtk.genericRenderWindow, e.key, ROTATE_STEP)) {
      e.preventDefault();
    }
  }, []);

  return (
    <div className="volume-viewer-3d">
      <VolumeControls
        channel={channel}
        onChannelChange={setChannel}
        channelCount={channelCount}
        brightness={brightness}
        onBrightnessChange={setBrightness}
        contrast={contrast}
        onContrastChange={setContrast}
        maxSize={maxSize}
        onMaxSizeChange={setMaxSize}
        hasGt={hasGt}
        showGt={showGt}
        onShowGtChange={setShowGt}
        gtOpacity={gtOpacity}
        onGtOpacityChange={setGtOpacity}
        hasPredicted={hasPredicted}
        showPredicted={showPredicted}
        onShowPredictedChange={setShowPredicted}
        predictedOpacity={predictedOpacity}
        onPredictedOpacityChange={setPredictedOpacity}
      />

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

      {/* Standalone interactive 3D renders of the exact Zarr the main viewer
          loaded (volumes/raw as RGB, volumes/gt_instances merged/colored). They
          share the brightness/contrast/resolution controls above, and (when
          linked) a synchronized camera with the main render. */}
      <label className="volume-viewer-3d__checkbox">
        <input
          type="checkbox"
          checked={linkViews}
          onChange={(e) => setLinkViews(e.target.checked)}
        />
        Link camera across all 3D views
      </label>
      <div className="volume-viewer-3d__subviews">
        <Volume3DView
          key={`${sampleName}-raw-3d`}
          sampleName={sampleName}
          label="Raw (Zarr)"
          volume="raw"
          channel="all"
          mode="rgb"
          maxSize={debouncedMaxSize}
          brightness={brightness}
          contrast={contrast}
          cameraSync={cameraSync}
          syncId="sub-raw"
        />
        {hasGt && (
          <Volume3DView
            key={`${sampleName}-gt-3d`}
            sampleName={sampleName}
            label="GT instances (Zarr)"
            volume="gt"
            mode="instance_rgb"
            maxSize={debouncedMaxSize}
            brightness={brightness}
            contrast={contrast}
            cameraSync={cameraSync}
            syncId="sub-gt"
          />
        )}
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

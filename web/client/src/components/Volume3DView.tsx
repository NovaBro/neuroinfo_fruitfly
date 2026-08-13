import { useCallback, useEffect, useRef, useState } from "react";
import { ChannelParam, fetchVolumeData, VolumeKind } from "../api/client";
import { CameraSync } from "../utils/cameraSync";
import { ChannelGains } from "../utils/displayAdjust";
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

interface Volume3DViewProps {
  sampleName: string;
  /** Panel title shown above the render. */
  label: string;
  /** Which server `volume.bin` encoding to fetch and render. */
  volume: VolumeKind;
  /** Channel selector for `raw` (e.g. "all" for RGB); ignored for `gt`. */
  channel?: ChannelParam;
  /** Transfer-function mode: "rgb" for raw RGB, "instance_rgb" for labels. */
  mode: VolumeMode;
  /** Downsample target (longest edge); typically shared with the main viewer. */
  maxSize: number;
  brightness: number;
  contrast: number;
  /** Per-channel intensity for raw RGB mode; ignored for instance overlays. */
  channelGains?: ChannelGains;
  /** Shared camera-link group; when set, this view registers under `syncId`. */
  cameraSync?: CameraSync;
  syncId?: string;
}

type Ctx = {
  grw: GenericRenderWindow;
  layer: VolumeLayerController;
};

/**
 * A single-volume interactive 3D viewer: its own vtk.js render window rendering
 * one `volume.bin` volume with a MaximumIntensity blend (so the render reads as
 * a rotatable MIP). A distilled, single-layer sibling of {@link VolumeViewer3D};
 * both share the plumbing in `vtkVolumeScene.ts`.
 */
export function Volume3DView({
  sampleName,
  label,
  volume,
  channel,
  mode,
  maxSize,
  brightness,
  contrast,
  channelGains,
  cameraSync,
  syncId,
}: Volume3DViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<Ctx | null>(null);
  const generationRef = useRef(0);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const renderScene = useCallback(() => {
    ctxRef.current?.grw.getRenderWindow().render();
  }, []);

  const fitAndRender = useCallback(() => {
    if (ctxRef.current) fitCamera(ctxRef.current.grw);
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const generation = ++generationRef.current;
    const grw = createRenderWindow(container);
    const layer = new VolumeLayerController(mode);
    grw.getRenderer().addVolume(layer.volume);
    layer.volume.setVisibility(false);

    ctxRef.current = { grw, layer };
    setReady(true);

    const unregisterSync =
      cameraSync && syncId
        ? cameraSync.register({
            id: syncId,
            camera: grw.getRenderer().getActiveCamera(),
            renderer: grw.getRenderer(),
            renderWindow: grw.getRenderWindow(),
          })
        : undefined;

    const resizeObserver = new ResizeObserver(() => {
      if (generationRef.current === generation) {
        grw.resize();
        renderScene();
      }
    });
    resizeObserver.observe(container);

    requestAnimationFrame(() => {
      if (generationRef.current === generation) fitAndRender();
    });

    return () => {
      generationRef.current += 1;
      setReady(false);
      resizeObserver.disconnect();
      unregisterSync?.();
      grw.getRenderer().removeVolume(layer.volume);
      grw.delete();
      ctxRef.current = null;
    };
  }, [mode, cameraSync, syncId, renderScene, fitAndRender]);

  useEffect(() => {
    if (!ready) return;
    const ctx = ctxRef.current;
    if (!ctx) return;

    const generation = generationRef.current;
    const controller = new AbortController();
    let cancelled = false;
    setLoading(true);
    setError(null);

    async function load() {
      try {
        const vol = await fetchVolumeData(sampleName, {
          volume,
          channel,
          maxSize,
          signal: controller.signal,
        });
        if (cancelled || generationRef.current !== generation) return;
        ctx!.layer.applyData(
          vol,
          mode,
          brightness,
          contrast,
          1,
          mode === "rgb" ? channelGains : undefined,
        );
        fitAndRender();
        requestAnimationFrame(() => {
          if (!cancelled && generationRef.current === generation) fitAndRender();
        });
      } catch (err) {
        if (controller.signal.aborted || cancelled) return;
        if (generationRef.current !== generation) return;
        ctx!.layer.hide();
        renderScene();
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled && generationRef.current === generation) {
          setLoading(false);
        }
      }
    }

    void load();

    return () => {
      cancelled = true;
      controller.abort();
    };
    // brightness/contrast/channelGains are intentionally excluded: the effect
    // below re-maps them without a re-fetch (matches VolumeViewer3D's split).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, sampleName, volume, channel, mode, maxSize, fitAndRender, renderScene]);

  useEffect(() => {
    if (!ready) return;
    ctxRef.current?.layer.refreshScalars(
      brightness,
      contrast,
      mode === "rgb" ? channelGains : undefined,
    );
    renderScene();
  }, [ready, brightness, contrast, channelGains, mode, renderScene]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    if (rotateCameraByKey(ctx.grw, e.key, ROTATE_STEP)) e.preventDefault();
  }, []);

  return (
    <div className="volume-viewer-3d__subview">
      <span className="volume-viewer-3d__label">{label}</span>
      <div
        className="volume-viewer-3d__viewport volume-viewer-3d__viewport--sub"
        tabIndex={0}
        onKeyDown={handleKeyDown}
        role="application"
        aria-label={`3D viewer: ${label}`}
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
      </div>
    </div>
  );
}

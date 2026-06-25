import { useEffect, useMemo, useState } from "react";
import {
  AxisKind,
  ChannelParam,
  mipUrl,
  RAW_CHANNEL_CHOICES,
  SampleMeta,
  sliceUrl,
} from "../api/client";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import {
  DEFAULT_BRIGHTNESS,
  DEFAULT_CONTRAST,
  DISPLAY_MAX,
  DISPLAY_MIN,
} from "../utils/displayAdjust";
import { SliceImage } from "./SliceImage";
import "./OrthoSliceViewer.css";

const AXES: AxisKind[] = ["z", "y", "x"];

interface OrthoSliceViewerProps {
  sampleName: string;
  meta: SampleMeta;
}

function axisSize(meta: SampleMeta, axis: AxisKind): number {
  const [, z, y, x] = meta.raw.shape;
  if (axis === "z") return z;
  if (axis === "y") return y;
  return x;
}

export function OrthoSliceViewer({ sampleName, meta }: OrthoSliceViewerProps) {
  const [channel, setChannel] = useState<ChannelParam>(0);
  const [brightness, setBrightness] = useState(DEFAULT_BRIGHTNESS);
  const [contrast, setContrast] = useState(DEFAULT_CONTRAST);
  const [axis, setAxis] = useState<AxisKind>("z");
  const [index, setIndex] = useState(0);
  const [showGt, setShowGt] = useState(false);
  const [gtChannel, setGtChannel] = useState(0);
  const [gtOpacity, setGtOpacity] = useState(55);
  const [viewMode, setViewMode] = useState<"slice" | "mip">("slice");

  const maxIndex = useMemo(
    () => Math.max(0, axisSize(meta, axis) - 1),
    [meta, axis],
  );

  const debouncedIndex = useDebouncedValue(index, 200);

  const setAxisAndCenter = (nextAxis: AxisKind) => {
    const nextMax = Math.max(0, axisSize(meta, nextAxis) - 1);
    setAxis(nextAxis);
    setIndex(Math.floor(nextMax / 2));
  };

  useEffect(() => {
    setIndex(Math.floor(maxIndex / 2));
    setBrightness(DEFAULT_BRIGHTNESS);
    setContrast(DEFAULT_CONTRAST);
  }, [sampleName]);

  const gtChannelCount = meta.gt_instances.shape[0];

  const channelCount = meta.raw.shape[0];

  const rawImageUrl = useMemo(() => {
    if (channel === "off") return null;
    if (viewMode === "mip") {
      return mipUrl(sampleName, { volume: "raw", channel });
    }
    return sliceUrl(sampleName, {
      volume: "raw",
      channel,
      axis,
      index: debouncedIndex,
    });
  }, [sampleName, channel, axis, debouncedIndex, viewMode]);

  const gtImageUrl = useMemo(() => {
    if (!showGt) return null;
    if (viewMode === "mip") {
      return mipUrl(sampleName, { volume: "gt", channel: gtChannel });
    }
    return sliceUrl(sampleName, {
      volume: "gt",
      channel: gtChannel,
      axis,
      index: debouncedIndex,
    });
  }, [sampleName, showGt, gtChannel, axis, debouncedIndex, viewMode]);

  return (
    <div className="ortho-viewer">
      <div className="ortho-viewer__controls">
        <div className="ortho-viewer__group">
          <span className="ortho-viewer__label">View</span>
          <div className="ortho-viewer__btn-row">
            <button
              type="button"
              className={
                viewMode === "slice"
                  ? "ortho-viewer__btn ortho-viewer__btn--active"
                  : "ortho-viewer__btn"
              }
              onClick={() => setViewMode("slice")}
            >
              Slice
            </button>
            <button
              type="button"
              className={
                viewMode === "mip"
                  ? "ortho-viewer__btn ortho-viewer__btn--active"
                  : "ortho-viewer__btn"
              }
              onClick={() => setViewMode("mip")}
            >
              MIP
            </button>
          </div>
        </div>

        <div className="ortho-viewer__group">
          <span className="ortho-viewer__label">Channel</span>
          <div className="ortho-viewer__btn-row">
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
                    ? "ortho-viewer__btn ortho-viewer__btn--active"
                    : "ortho-viewer__btn"
                }
                onClick={() => setChannel(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="ortho-viewer__group ortho-viewer__group--slider">
          <span className="ortho-viewer__label">Brightness ({brightness}%)</span>
          <input
            type="range"
            min={DISPLAY_MIN}
            max={DISPLAY_MAX}
            value={brightness}
            onChange={(e) => setBrightness(Number(e.target.value))}
          />
        </div>

        <div className="ortho-viewer__group ortho-viewer__group--slider">
          <span className="ortho-viewer__label">Contrast ({contrast}%)</span>
          <input
            type="range"
            min={DISPLAY_MIN}
            max={DISPLAY_MAX}
            value={contrast}
            onChange={(e) => setContrast(Number(e.target.value))}
          />
        </div>

        {viewMode === "slice" && (
          <div className="ortho-viewer__group">
            <span className="ortho-viewer__label">Axis</span>
            <div className="ortho-viewer__btn-row">
              {AXES.map((a) => (
                <button
                  key={a}
                  type="button"
                  className={
                    axis === a
                      ? "ortho-viewer__btn ortho-viewer__btn--active"
                      : "ortho-viewer__btn"
                  }
                  onClick={() => setAxisAndCenter(a)}
                >
                  {a.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        )}

        {viewMode === "slice" && (
          <div className="ortho-viewer__group ortho-viewer__group--slider">
            <span className="ortho-viewer__label">
              {axis.toUpperCase()} = {index} / {maxIndex}
            </span>
            <input
              type="range"
              min={0}
              max={maxIndex}
              value={index}
              onChange={(e) => setIndex(Number(e.target.value))}
            />
          </div>
        )}

        <div className="ortho-viewer__group">
          <label className="ortho-viewer__checkbox">
            <input
              type="checkbox"
              checked={showGt}
              onChange={(e) => setShowGt(e.target.checked)}
            />
            Show GT overlay
          </label>
          {showGt && (
            <div className="ortho-viewer__btn-row">
              {Array.from({ length: gtChannelCount }, (_, i) => (
                <button
                  key={i}
                  type="button"
                  className={
                    gtChannel === i
                      ? "ortho-viewer__btn ortho-viewer__btn--active"
                      : "ortho-viewer__btn"
                  }
                  onClick={() => setGtChannel(i)}
                >
                  GT {i}
                </button>
              ))}
            </div>
          )}
          {showGt && (
            <div className="ortho-viewer__group ortho-viewer__group--slider">
              <span className="ortho-viewer__label">
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
        </div>
      </div>

      <div className="ortho-viewer__canvas">
        {rawImageUrl ? (
          <SliceImage
            key={rawImageUrl}
            className="ortho-viewer__img"
            url={rawImageUrl}
            alt={`${sampleName} raw ${viewMode}`}
            brightness={brightness}
            contrast={contrast}
          />
        ) : (
          <p className="ortho-viewer__placeholder">Raw channel off</p>
        )}
        {gtImageUrl && (
          <SliceImage
            key={gtImageUrl}
            className="ortho-viewer__img ortho-viewer__img--overlay"
            url={gtImageUrl}
            alt={`${sampleName} gt overlay`}
            brightness={brightness}
            contrast={contrast}
            opacity={gtOpacity / 100}
          />
        )}
      </div>

      <p className="ortho-viewer__meta">
        Raw shape (C,Z,Y,X): {meta.raw.shape.join(" × ")} · GT instances:{" "}
        {meta.gt_instances.shape[0]} channel
        {meta.gt_instances.shape[0] !== 1 ? "s" : ""}
      </p>
    </div>
  );
}

import {
  ChannelParam,
  RAW_CHANNEL_CHOICES,
  VOLUME_MAX_SIZE_OPTIONS,
} from "../api/client";
import {
  CHANNEL_GAIN_MAX,
  CHANNEL_GAIN_MIN,
  ChannelGains,
  DISPLAY_MAX,
  DISPLAY_MIN,
} from "../utils/displayAdjust";
import { RangeSlider } from "./RangeSlider";
import "./VolumeViewer3D.css";

interface VolumeControlsProps {
  channel: ChannelParam;
  onChannelChange: (channel: ChannelParam) => void;
  channelCount: number;
  channelGains: ChannelGains;
  onChannelGainsChange: (gains: ChannelGains) => void;
  brightness: number;
  onBrightnessChange: (value: number) => void;
  contrast: number;
  onContrastChange: (value: number) => void;
  maxSize: number;
  onMaxSizeChange: (value: number) => void;
  hasGt: boolean;
  showGt: boolean;
  onShowGtChange: (value: boolean) => void;
  gtOpacity: number;
  onGtOpacityChange: (value: number) => void;
  hasPredicted: boolean;
  showPredicted: boolean;
  onShowPredictedChange: (value: boolean) => void;
  predictedOpacity: number;
  onPredictedOpacityChange: (value: number) => void;
}

/** Display/overlay control panel for {@link VolumeViewer3D} (presentational). */
export function VolumeControls({
  channel,
  onChannelChange,
  channelCount,
  channelGains,
  onChannelGainsChange,
  brightness,
  onBrightnessChange,
  contrast,
  onContrastChange,
  maxSize,
  onMaxSizeChange,
  hasGt,
  showGt,
  onShowGtChange,
  gtOpacity,
  onGtOpacityChange,
  hasPredicted,
  showPredicted,
  onShowPredictedChange,
  predictedOpacity,
  onPredictedOpacityChange,
}: VolumeControlsProps) {
  const maxSizeIndex = VOLUME_MAX_SIZE_OPTIONS.indexOf(
    maxSize as (typeof VOLUME_MAX_SIZE_OPTIONS)[number],
  );
  const sliderIndex = maxSizeIndex >= 0 ? maxSizeIndex : 2;

  return (
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
              onClick={() => onChannelChange(id)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {channel === "all" && (
        <>
          <RangeSlider
            label={`Red intensity (${channelGains.r}%)`}
            min={CHANNEL_GAIN_MIN}
            max={CHANNEL_GAIN_MAX}
            value={channelGains.r}
            onChange={(r) => onChannelGainsChange({ ...channelGains, r })}
          />
          <RangeSlider
            label={`Green intensity (${channelGains.g}%)`}
            min={CHANNEL_GAIN_MIN}
            max={CHANNEL_GAIN_MAX}
            value={channelGains.g}
            onChange={(g) => onChannelGainsChange({ ...channelGains, g })}
          />
          <RangeSlider
            label={`Blue intensity (${channelGains.b}%)`}
            min={CHANNEL_GAIN_MIN}
            max={CHANNEL_GAIN_MAX}
            value={channelGains.b}
            onChange={(b) => onChannelGainsChange({ ...channelGains, b })}
          />
        </>
      )}

      <RangeSlider
        label={`Brightness (${brightness}%)`}
        min={DISPLAY_MIN}
        max={DISPLAY_MAX}
        value={brightness}
        onChange={onBrightnessChange}
      />

      <RangeSlider
        label={`Contrast (${contrast}%)`}
        min={DISPLAY_MIN}
        max={DISPLAY_MAX}
        value={contrast}
        onChange={onContrastChange}
      />

      <RangeSlider
        label={`Resolution (max edge ${maxSize}px)`}
        min={0}
        max={VOLUME_MAX_SIZE_OPTIONS.length - 1}
        value={sliderIndex}
        onChange={(index) => onMaxSizeChange(VOLUME_MAX_SIZE_OPTIONS[index])}
        hint="Lower = faster · Higher = sharper"
      />

      {(hasGt || hasPredicted) && (
        <div className="volume-viewer-3d__group">
          <span className="volume-viewer-3d__label">Overlay</span>
          {hasGt && (
            <label className="volume-viewer-3d__checkbox">
              <input
                type="checkbox"
                checked={showGt}
                onChange={(e) => onShowGtChange(e.target.checked)}
              />
              Ground truth (Zarr)
            </label>
          )}
          {hasGt && showGt && (
            <RangeSlider
              label={`GT opacity (${gtOpacity}%)`}
              min={0}
              max={100}
              value={gtOpacity}
              onChange={onGtOpacityChange}
            />
          )}
          {hasPredicted && (
            <label className="volume-viewer-3d__checkbox">
              <input
                type="checkbox"
                checked={showPredicted}
                onChange={(e) => onShowPredictedChange(e.target.checked)}
              />
              Predicted instances (BiaPy)
            </label>
          )}
          {hasPredicted && showPredicted && (
            <RangeSlider
              label={`Predicted opacity (${predictedOpacity}%)`}
              min={0}
              max={100}
              value={predictedOpacity}
              onChange={onPredictedOpacityChange}
            />
          )}
        </div>
      )}
    </div>
  );
}

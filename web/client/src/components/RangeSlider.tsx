import "./VolumeViewer3D.css";

interface RangeSliderProps {
  label: string;
  min: number;
  max: number;
  value: number;
  onChange: (value: number) => void;
  /** Optional helper text shown beneath the slider. */
  hint?: string;
}

/** Labelled range input used for the volume viewer's display controls. */
export function RangeSlider({
  label,
  min,
  max,
  value,
  onChange,
  hint,
}: RangeSliderProps) {
  return (
    <div className="volume-viewer-3d__group volume-viewer-3d__group--slider">
      <span className="volume-viewer-3d__label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <span className="volume-viewer-3d__slider-hint">{hint}</span>}
    </div>
  );
}

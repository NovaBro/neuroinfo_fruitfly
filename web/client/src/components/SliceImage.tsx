import { useEffect, useState } from "react";
import { cssBrightnessContrast } from "../utils/displayAdjust";

interface SliceImageProps {
  url: string;
  alt: string;
  className?: string;
  brightness?: number;
  contrast?: number;
  opacity?: number;
}

export function SliceImage({
  url,
  alt,
  className,
  brightness = 100,
  contrast = 100,
  opacity = 1,
}: SliceImageProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl: string | null = null;

    setBlobUrl(null);
    fetch(url, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Slice request failed: ${res.status}`);
        }
        return res.blob();
      })
      .then((blob) => {
        objectUrl = URL.createObjectURL(blob);
        setBlobUrl(objectUrl);
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === "AbortError") {
          return;
        }
        console.error(err);
      });

    return () => {
      controller.abort();
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [url]);

  if (!blobUrl) {
    return <div className={`${className ?? ""} slice-image--loading`} aria-busy="true" />;
  }

  return (
    <img
      className={className}
      src={blobUrl}
      alt={alt}
      style={{ filter: cssBrightnessContrast(brightness, contrast), opacity }}
    />
  );
}

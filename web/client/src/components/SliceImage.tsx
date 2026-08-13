import { useEffect, useRef, useState } from "react";
import {
  ChannelGains,
  cssBrightnessContrast,
  remapRgbUint8Clamped,
} from "../utils/displayAdjust";

interface SliceImageProps {
  url: string;
  alt: string;
  className?: string;
  brightness?: number;
  contrast?: number;
  opacity?: number;
  /** When set, remaps RGB via canvas (gains + brightness/contrast); no CSS filter. */
  channelGains?: ChannelGains;
}

function remapBlobToObjectUrl(
  blob: Blob,
  brightness: number,
  contrast: number,
  channelGains: ChannelGains,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const src = URL.createObjectURL(blob);
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          URL.revokeObjectURL(src);
          reject(new Error("Canvas 2D context unavailable"));
          return;
        }
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        remapRgbUint8Clamped(imageData.data, brightness, contrast, channelGains);
        ctx.putImageData(imageData, 0, 0);
        URL.revokeObjectURL(src);
        canvas.toBlob((out) => {
          if (!out) {
            reject(new Error("Failed to encode remapped slice"));
            return;
          }
          resolve(URL.createObjectURL(out));
        }, "image/png");
      } catch (err) {
        URL.revokeObjectURL(src);
        reject(err);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(src);
      reject(new Error("Failed to decode slice image"));
    };
    img.src = src;
  });
}

export function SliceImage({
  url,
  alt,
  className,
  brightness = 100,
  contrast = 100,
  opacity = 1,
  channelGains,
}: SliceImageProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [displayUrl, setDisplayUrl] = useState<string | null>(null);
  const sourceBlobRef = useRef<Blob | null>(null);
  const remappedUrlRef = useRef<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl: string | null = null;

    setBlobUrl(null);
    setDisplayUrl(null);
    sourceBlobRef.current = null;
    if (remappedUrlRef.current) {
      URL.revokeObjectURL(remappedUrlRef.current);
      remappedUrlRef.current = null;
    }

    fetch(url, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Slice request failed: ${res.status}`);
        }
        return res.blob();
      })
      .then((blob) => {
        sourceBlobRef.current = blob;
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

  useEffect(() => {
    if (!channelGains || !blobUrl || !sourceBlobRef.current) {
      setDisplayUrl(null);
      if (remappedUrlRef.current) {
        URL.revokeObjectURL(remappedUrlRef.current);
        remappedUrlRef.current = null;
      }
      return;
    }

    let cancelled = false;
    const blob = sourceBlobRef.current;

    remapBlobToObjectUrl(blob, brightness, contrast, channelGains)
      .then((nextUrl) => {
        if (cancelled) {
          URL.revokeObjectURL(nextUrl);
          return;
        }
        if (remappedUrlRef.current) {
          URL.revokeObjectURL(remappedUrlRef.current);
        }
        remappedUrlRef.current = nextUrl;
        setDisplayUrl(nextUrl);
      })
      .catch((err: unknown) => {
        if (!cancelled) console.error(err);
      });

    return () => {
      cancelled = true;
    };
  }, [
    blobUrl,
    brightness,
    contrast,
    channelGains?.r,
    channelGains?.g,
    channelGains?.b,
    channelGains,
  ]);

  useEffect(() => {
    return () => {
      if (remappedUrlRef.current) {
        URL.revokeObjectURL(remappedUrlRef.current);
        remappedUrlRef.current = null;
      }
    };
  }, []);

  if (channelGains) {
    if (!displayUrl) {
      return (
        <div className={`${className ?? ""} slice-image--loading`} aria-busy="true" />
      );
    }
    return (
      <img
        className={className}
        src={displayUrl}
        alt={alt}
        style={{ opacity }}
      />
    );
  }

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

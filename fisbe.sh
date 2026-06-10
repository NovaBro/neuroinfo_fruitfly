#!/bin/bash

nohup bash -c '
  mkdir -p fisbe &&
  echo "[$(date)] Starting download..." &&
  curl -L -s -w "[$(date)] Download complete: %{size_download} bytes, %{time_total}s\n" \
    https://zenodo.org/api/records/10875063/files-archive -o fisbe/archive.zip &&
  echo "[$(date)] Extracting..." &&
  unzip -o fisbe/archive.zip -d fisbe/ &&
  rm fisbe/archive.zip &&
  echo "[$(date)] Done."
' > download.txt 2>&1 &

echo "PID: $!"
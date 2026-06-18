#!/bin/bash
#SBATCH -J fisbe-download
#SBATCH -n 2
#SBATCH --time=6:00:00
#SBATCH --mem=24g
#SBATCH --account=torch_pr_61_general
#SBATCH --output=fisbe.out
#SBATCH --error=fisbe.err


mkdir -p fisbe
echo "[$(date)] Starting download..."
curl -L -s -w "[$(date)] Download complete: %{size_download} bytes, %{time_total}s\n" \
  https://zenodo.org/api/records/10875063/files-archive -o fisbe/archive.zip
echo "[$(date)] Extracting..."
unzip -o fisbe/archive.zip -d fisbe/
rm fisbe/archive.zip
echo "[$(date)] Done."

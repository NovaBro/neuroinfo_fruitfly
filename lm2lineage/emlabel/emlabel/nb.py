"""Thin client for the public NeuronBridge / FlyLight / FlyEM open data buckets.

Everything is anonymous HTTPS on AWS S3:
    metadata  janelia-neuronbridge-data-prod/<version>/metadata/...
    stacks    janelia-flylight-imagery/Gen1 MCFO/<line>/...aligned_stack.h5j
    EM SWC    janelia-flylight-color-depth/JRC2018_VNC_Unisex_40x_DS/<lib>/SWC/<body>.swc
    PPPM PNG  janelia-ppp-match-prod/...
"""
from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request

import h5py
import numpy as np

META = "https://janelia-neuronbridge-data-prod.s3.amazonaws.com"
IMAGERY = "https://janelia-flylight-imagery.s3.amazonaws.com"
CDM = "https://janelia-flylight-color-depth.s3.amazonaws.com"
PPPM = "https://janelia-ppp-match-prod.s3.amazonaws.com"
VNC_SPACE = "JRC2018_VNC_Unisex_40x_DS"


class NB:
    def __init__(self, cache_dir="cache", version=None):
        self.cache = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.version = version or self._get(f"{META}/current.txt").decode().strip()

    # ---------------------------------------------------------------- fetching
    def _get(self, url, cache_name=None):
        fp = os.path.join(self.cache, cache_name) if cache_name else None
        if fp and os.path.exists(fp):
            return open(fp, "rb").read()
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        if fp:
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, "wb").write(data)
        return data

    def meta(self, rel):
        url = f"{META}/{self.version}/metadata/{urllib.parse.quote(rel)}"
        try:
            return json.loads(self._get(url, f"meta/{self.version}/{rel}"))
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            raise

    def file(self, base, rel, cache_name):
        rel = rel.replace("+", " ")
        return self._get(f"{base}/{urllib.parse.quote(rel)}", cache_name)

    # ---------------------------------------------------------------- queries
    def em_images(self, body_id):
        """NeuronBridge EM image records for a MANC body id (all libraries)."""
        d = self.meta(f"by_body/{int(body_id)}.json")
        return [] if d is None else [r for r in d["results"] if r["anatomicalArea"] == "VNC"]

    def pppm_matches(self, body_id):
        """EM -> LM PatchPerPixMatch results (ranked). Only VNC v0.5 bodies have them."""
        out = []
        for em in self.em_images(body_id):
            f = em["files"].get("PPPMResults")
            if not f:
                continue
            d = self.meta(f"pppmresults/{f}")
            for r in (d or {}).get("results", []):
                out.append(dict(method="PPPM", em_library=em["libraryName"],
                                em_swc=em["files"].get("AlignedBodySWC"),
                                rank=r.get("pppmRank"), score=r.get("pppmScore"),
                                mirrored=bool(r.get("mirrored")), lm=r["image"],
                                pppm_files=r["files"]))
        return out

    def cds_matches_em(self, body_id, library="FlyEM_MANC_v1.2.1"):
        """EM -> LM colour depth search results for Gen1 MCFO."""
        out = []
        for em in self.em_images(body_id):
            if em["libraryName"] != library or not em["files"].get("CDSResults"):
                continue
            d = self.meta(f"cdsresults/{em['files']['CDSResults']}")
            for r in (d or {}).get("results", []):
                if "MCFO" not in r["image"]["libraryName"]:
                    continue
                out.append(dict(method="CDS", em_library=library,
                                em_swc=em["files"].get("AlignedBodySWC"),
                                score=r.get("normalizedScore"), mirrored=bool(r.get("mirrored")),
                                lm=r["image"]))
        return out

    def lm_images(self, line):
        d = self.meta(f"by_line/{line}.json")
        return [] if d is None else [r for r in d["results"] if r["anatomicalArea"] == "VNC"
                                     and "MCFO" in r["libraryName"]]

    def cds_matches_lm(self, line, slide_code):
        """LM -> EM colour depth search results for every channel of one sample."""
        out = []
        for im in self.lm_images(line):
            if im["slideCode"] != slide_code or not im["files"].get("CDSResults"):
                continue
            d = self.meta(f"cdsresults/{im['files']['CDSResults']}")
            for r in (d or {}).get("results", []):
                em = r["image"]
                out.append(dict(method="CDS", lm_channel=im.get("channel"),
                                em_library=em["libraryName"], em_name=em["publishedName"],
                                em_type=em.get("neuronType"), em_swc=em["files"].get("AlignedBodySWC"),
                                score=r.get("normalizedScore"), mirrored=bool(r.get("mirrored")),
                                lm=im))
        return out

    # ---------------------------------------------------------------- data
    def stack(self, lm_record):
        """Aligned 4-channel stack (C, Z, Y, X) uint16 + voxel size (x, y, z) um."""
        rel = lm_record["files"]["VisuallyLosslessStack"]
        name = "stacks/" + os.path.basename(rel.replace("+", " "))
        raw = self.file(IMAGERY, rel, name)
        return read_h5j(io.BytesIO(raw))

    def swc(self, rel):
        data = self.file(CDM, rel, "swc/" + rel.replace("/", "_"))
        a = np.loadtxt(io.StringIO(data.decode()), comments="#", ndmin=2)
        return a  # columns: id type x y z r parent (microns, template space)


def read_h5j(fobj, with_reference=False):
    """Decode a Janelia .h5j (HDF5 + one H.265 stream per channel)."""
    import av
    with h5py.File(fobj, "r") as f:
        ch = f["Channels"]
        h, w, n = int(ch.attrs["height"][0]), int(ch.attrs["width"][0]), int(ch.attrs["frames"][0])
        names = sorted(k for k in ch.keys() if k.startswith("Channel_"))
        kinds = [ch[k].attrs.get("content_type", b"signal").decode() for k in names]
        keep = [i for i, k in enumerate(kinds) if with_reference or k != "reference"]
        out = np.zeros((len(keep), n, h, w), np.uint16)
        for j, i in enumerate(keep):
            c = av.open(io.BytesIO(ch[names[i]][()].tobytes()))
            for z, fr in enumerate(c.decode(video=0)):
                if z >= n:
                    break
                a = fr.to_ndarray()
                out[j, z] = (a if a.ndim == 2 else a[..., 0])[:h, :w]
            c.close()
        vox = np.array(f.attrs.get("voxel_size", [0.461122, 0.461122, 0.7]), float)
        space = f.attrs.get("alignment_space", b"").decode()
    return out, [kinds[i] for i in keep], vox, space

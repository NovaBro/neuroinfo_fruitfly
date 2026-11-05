"""Run the SWC2IMG plugin through Fiji's bundled Java in ImageJ batch mode.

SWC2IMG's dialog adds AWT buttons, so it fails under ``--headless``
(``java.awt.HeadlessException``). ``ij.ImageJ -batch`` with ``setBatchMode(true)`` fills the
dialog from the macro options without showing it, but it still needs a display. On a
display-less Linux node (Greene), wrap the call with ``xvfb-run -a``.
"""

from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from pathlib import Path

DEFAULT_FIJI_DIRS = ("D:/tools/Fiji", "~/Fiji.app", "~/Fiji", "/opt/Fiji.app")


class FijiNotFound(RuntimeError):
    pass


class SWC2IMGError(RuntimeError):
    pass


def find_fiji(fiji_dir: str | os.PathLike | None = None) -> Path:
    """Return the Fiji directory: the argument, else $FIJI_DIR, else a default location."""
    candidates = [fiji_dir, os.environ.get("FIJI_DIR"), *DEFAULT_FIJI_DIRS]
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if (p / "jars").is_dir() and (p / "plugins").is_dir():
            if not glob.glob(str(p / "plugins" / "SWC2IMG_*.jar")):
                raise FijiNotFound(
                    f"{p} has no plugins/SWC2IMG_*.jar. Install SWC2IMG_-1.1.0.jar into plugins/ and "
                    "ImageScience-3.1.0.jar into jars/ (see lm2lineage/swc2img_py/README.md)."
                )
            return p
    raise FijiNotFound("Fiji not found. Pass fiji_dir=... or set FIJI_DIR.")


def _java(fiji: Path) -> str:
    exe = "java.exe" if os.name == "nt" else "java"
    hits = sorted(glob.glob(str(fiji / "java" / "*" / "*" / "bin" / exe)))
    return hits[-1] if hits else "java"


def _macro_path(p: str | os.PathLike) -> str:
    return str(Path(p).resolve()).replace("\\", "/")


def run_swc2img(
    swc_path: str | os.PathLike,
    *,
    voxel_um: float = 0.44,
    axial_um: float | None = None,
    background: float = 10.0,
    correlation_um: float = 0.5,
    snr: float = 10.0,
    fiji_dir: str | os.PathLike | None = None,
    max_heap: str = "5g",
    timeout_s: int = 3600,
) -> tuple[Path, Path, str]:
    """Render one SWC file. Returns (tif path, corrected-SWC path, plugin log).

    SWC2IMG writes ``<base>.swc2img.tif`` (32-bit) and ``<base>.swc2img.swc`` (input shifted
    into image coordinates, micrometres) next to the input file. Coordinates in the input are
    taken as micrometres.
    """
    fiji = find_fiji(fiji_dir)
    swc_path = Path(swc_path).resolve()
    axial_um = voxel_um if axial_um is None else axial_um
    opts = (
        f"file=[{_macro_path(swc_path)}] lateral={voxel_um} axial={axial_um} "
        f"background={background} correlation={correlation_um} snr={snr} save correct log"
    )
    stem = swc_path.with_suffix("")
    tif, corrected = Path(f"{stem}.swc2img.tif"), Path(f"{stem}.swc2img.swc")
    log_file = Path(f"{stem}.swc2img.log")
    for p in (tif, corrected, log_file):
        p.unlink(missing_ok=True)
    # Fiji's stdout is empty when captured from Python on Windows, so the macro saves its own log.
    macro = (
        f'setBatchMode(true);\nrun("SWC2IMG", "{opts}");\n'
        f'File.saveString(getInfo("log"), "{_macro_path(log_file)}");\n'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".ijm", delete=False) as f:
        f.write(macro)
        macro_file = f.name
    cmd = [
        _java(fiji), f"-Xmx{max_heap}", f"-Dij.dir={fiji}", f"-Dplugins.dir={fiji}",
        "-cp", os.pathsep.join([str(fiji / "jars" / "*"), str(fiji / "plugins" / "*")]),
        "ij.ImageJ", "-batch", macro_file,
    ]
    try:
        proc = subprocess.run(cmd, cwd=fiji, capture_output=True, text=True, timeout=timeout_s)
    finally:
        os.unlink(macro_file)
    log = (log_file.read_text() if log_file.exists() else "") + proc.stdout + proc.stderr
    if "Program finished successfully" not in log or not tif.exists() or not corrected.exists():
        tail = "\n".join(log.strip().splitlines()[-25:])
        raise SWC2IMGError(f"SWC2IMG failed on {swc_path}:\n{tail}")
    return tif, corrected, log

"""Review images for proposals: raw | raw + EM skeleton | proposed mask."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def _mip_rgb(vol3, gamma=0.7):
    m = vol3.max(axis=1)                      # (3, Y, X)
    m = np.clip(m, 0, 1) ** gamma
    return np.moveaxis(m, 0, -1)


def review_image(prop, title_lines, max_h=560):
    rgb = prop.crop_rgb.astype(np.float32) / 255.0
    a = _mip_rgb(rgb)
    Y, X = a.shape[:2]
    sk = np.zeros((Y, X), bool)
    sk[prop.skel_vox[:, 1], prop.skel_vox[:, 2]] = True
    b = a * 0.45
    b[sk] = [1.0, 1.0, 1.0]
    masked = rgb * prop.mask_disp[None]
    c = _mip_rgb(masked)
    c[(c.max(-1) == 0) & sk] = [0.35, 0.35, 0.35]
    gap = np.ones((Y, 4, 3))
    panel = np.concatenate([a, gap, b, gap, c], axis=1)
    img = Image.fromarray((panel * 255).astype(np.uint8))
    s = min(3.0, max_h / Y)
    if s != 1:
        img = img.resize((int(img.width * s), int(img.height * s)), Image.NEAREST)
    head = 14 * len(title_lines) + 8
    out = Image.new("RGB", (max(img.width, 520), img.height + head), (20, 20, 20))
    out.paste(img, (0, head))
    d = ImageDraw.Draw(out)
    for i, t in enumerate(title_lines):
        d.text((6, 4 + 14 * i), t, fill=(255, 230, 120))
    return out


def contact_sheet(images, cols=3, pad=6, bg=(255, 255, 255)):
    if not images:
        return None
    w = max(i.width for i in images)
    rows = [images[i:i + cols] for i in range(0, len(images), cols)]
    hs = [max(i.height for i in r) for r in rows]
    sheet = Image.new("RGB", (cols * (w + pad) + pad, sum(hs) + pad * (len(rows) + 1)), bg)
    y = pad
    for r, h in zip(rows, hs):
        x = pad
        for im in r:
            sheet.paste(im, (x, y))
            x += w + pad
        y += h + pad
    return sheet

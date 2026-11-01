"""Vectorized RandomHue gunpowder node (skimage), replacing neurolight's slow colorsys path."""
import numpy as np
import gunpowder as gp
from skimage.color import rgb2hsv, hsv2rgb


class RandomHue(gp.BatchFilter):
    """Random hue shift on a 3-channel CZYX raw array.

    Same API as neurolight.gunpowder.RandomHue: (array, max_change, prob).
    """

    def __init__(self, array, max_change, prob):
        self.array = array
        self.prob = prob
        self.max_change = max_change

    def process(self, batch, request):
        if np.random.rand() >= self.prob:
            return

        array = batch.arrays[self.array]
        dtype = array.data.dtype
        data = np.clip(array.data, 0.0, 1.0)

        # CZYX (3, Z, Y, X) -> channel-last for skimage
        rgb = np.moveaxis(data.astype(np.float64, copy=False), 0, -1)
        hsv = rgb2hsv(rgb)
        hsv[..., 0] = (hsv[..., 0] + np.random.uniform(
            -self.max_change, self.max_change)) % 1.0
        rgb_out = hsv2rgb(hsv)

        array.data = np.moveaxis(rgb_out, -1, 0)
        array.data = np.clip(array.data, 0.0, 1.0).astype(dtype, copy=False)

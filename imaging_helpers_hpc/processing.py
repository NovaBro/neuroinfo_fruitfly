import numpy as np
import logging
# import scipy.ndimage import rotate

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def axis_rotation(image:np.ndarray, rand_axis_idx:int, rand_k_rotations:int):
    """
    Randomly rotates a 3D image by 90, 180, or 270 degrees 
    along a random spatial plane using pure NumPy.
    """
    # 1. Pick a random plane out of the 3 possibilities: (0,1), (0,2), or (1,2)
    spatial_axes = [(1, 2), (1, 3), (2, 3)]
    chosen_axes = spatial_axes[rand_axis_idx]
    
    # 2. Pick a random number of 90-degree rotations (1, 2, or 3 times)
    # k_rotations
    
    # 3. Rotate using numpy's view-based rot90 (Zero memory overhead)
    logger.debug(f"axis_rotation:")
    logger.debug(f"\tBefore roation shape: {image.shape}")
    rotated_image = np.rot90(image, k=rand_k_rotations, axes=chosen_axes)
    logger.debug(f"\tAfter roation shape: {rotated_image.shape}")
    return rotated_image

def channel_flip(image:np.ndarray, random_order):
    """
    Assumes c, z, y, x formmating
    random_order is a np array of indecies of channels
    """
    # random_order = np.random.permutation(3)
    logger.debug(f"channel_flip:")
    logger.debug(f"\tBefore shuffle shape: {image.shape}")
    shuffled_image = image[random_order, ...]
    logger.debug(f"\tAfter shuffle shape: {shuffled_image.shape}")
    return shuffled_image



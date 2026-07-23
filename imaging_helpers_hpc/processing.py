import numpy as np
import logging
# import scipy.ndimage import rotate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def random_90_rotate_3d(image:np.ndarray, rand_axis_int:int, k_rotations:int):
    """
    Randomly rotates a 3D image by 90, 180, or 270 degrees 
    along a random spatial plane using pure NumPy.
    """
    # 1. Pick a random plane out of the 3 possibilities: (0,1), (0,2), or (1,2)
    spatial_axes = [(1, 2), (1, 3), (2, 3)]
    chosen_axes = spatial_axes[rand_axis_int]
    
    # 2. Pick a random number of 90-degree rotations (1, 2, or 3 times)
    # k_rotations
    
    # 3. Rotate using numpy's view-based rot90 (Zero memory overhead)
    logger.debug(f"random_90_rotate_3d:")
    logger.debug(f"\tBefore roation shape: {image.shape}")
    rotated_image = np.rot90(image, k=k_rotations, axes=chosen_axes)
    logger.debug(f"\tAfter roation shape: {rotated_image.shape}")
    return rotated_image

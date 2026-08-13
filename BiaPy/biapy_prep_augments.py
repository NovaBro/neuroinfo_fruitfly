"""Augmentation job generation and application for BiaPy FISBe prep."""
from itertools import permutations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging_helpers_hpc.processing import axis_rotation, channel_flip

NUM_CHANNELS = 3
_IDENTITY_CHANNEL_ORDER = (0, 1, 2)


def _dtype_max(dtype) -> float:
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)
    return float(info.max)


def _scale_clip_cast(raw_f32: np.ndarray, dtype) -> np.ndarray:
    """Clip float32 intensities to ``dtype`` range and cast back."""
    return np.clip(raw_f32, 0, _dtype_max(dtype)).astype(dtype, copy=False)


def per_channel_multiplicative_scale(
    raw: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """Multiply each channel of ``raw`` (C,Z,Y,X) by ``scales`` (C,)."""
    scales = np.asarray(scales, dtype=np.float32)
    if scales.shape != (raw.shape[0],):
        raise ValueError(
            f"channel scales shape {scales.shape} != ({raw.shape[0]},)"
        )
    out = raw.astype(np.float32, copy=False) * scales.reshape(-1, 1, 1, 1)
    return _scale_clip_cast(out, raw.dtype)


def per_instance_intensity_scale(
    raw: np.ndarray,
    instances: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Scale raw voxels per instance mask; overlaps accumulate as a product.

    Parameters
    ----------
    raw : (C, Z, Y, X)
    instances : (I, Z, Y, X)
    scales : (I,)
    """
    scales = np.asarray(scales, dtype=np.float32)
    if scales.shape != (instances.shape[0],):
        raise ValueError(
            f"instance scales shape {scales.shape} != ({instances.shape[0]},)"
        )
    out = raw.astype(np.float32, copy=True)
    for i, mask in enumerate(instances):
        fg = mask > 0
        if not np.any(fg):
            continue
        out[:, fg] *= scales[i]
    return _scale_clip_cast(out, raw.dtype)


def _sample_scalings(lo: float, hi: float, size: int) -> np.ndarray:
    distance = hi - lo
    margin = distance * 0.25
    low_range = np.random.uniform(lo, lo + margin, size=int(np.floor(size / 2))).astype(np.float32)
    high_range = np.random.uniform(hi - margin, hi, size=int(np.ceil(size / 2))).astype(np.float32)
    # return np.random.uniform(lo, hi, size=size).astype(np.float32)
    return np.random.permutation(np.concat([low_range, high_range]))


def generate_augmentation_jobs(
    *,
    num_instances: int = 0,
    enable_channel_flip: bool = False,
    num_channel_flips: int = 1,
    enable_rotate: bool = False,
    num_axes: int = 1,
    num_rotations: int = 1,
    enable_channel_scale: bool = False,
    channel_scale_range: tuple[float, float] | None = (0.25, 1.5),
    num_channel_scales: int = 1,
    enable_instance_scale: bool = False,
    instance_scale_range: tuple[float, float] | None = (0.25, 1.5),
    num_instance_scales: int = 1,
) -> list[dict]:
    """
    Generate jobs for the enabled augmentations.

    Disabled geometry uses identity (channel order 012, a=0, k=0).
    ``aug_id`` only encodes active transforms.
    """
    if not (
        enable_channel_flip
        or enable_rotate
        or enable_channel_scale
        or enable_instance_scale
    ):
        return []

    if enable_channel_flip:
        all_orders = list(permutations([0, 1, 2], 3))
        idx = np.random.choice(len(all_orders), size=num_channel_flips, replace=False)
        channel_orders = [all_orders[i] for i in idx]
    else:
        channel_orders = [_IDENTITY_CHANNEL_ORDER]

    if enable_rotate:
        axis_choices = list(np.random.permutation(3)[0:num_axes])
        k_choices = list(np.random.permutation(4)[0:num_rotations])
    else:
        axis_choices = [0]
        k_choices = [0]

    n_cs = num_channel_scales if enable_channel_scale else 1
    n_is = num_instance_scales if enable_instance_scale else 1

    jobs = []
    for ro in channel_orders:
        for a in axis_choices:
            for k in k_choices:
                for _cs_i in range(n_cs):
                    for is_i in range(n_is):
                        aug_id = ""
                        job: dict = {
                            "c": ro,
                            "a": int(a),
                            "k": int(k),
                        }

                        if enable_channel_flip:
                            aug_id += "_c" + "".join(str(x) for x in ro)
                        if enable_rotate:
                            aug_id += f"_r{a}" + f"_k{k}"

                        if enable_channel_scale and channel_scale_range is not None:
                            lo, hi = channel_scale_range
                            channel_scales = _sample_scalings(lo, hi, NUM_CHANNELS)
                            job["channel_scales"] = channel_scales
                            aug_id += "_cs" + "_".join(f"{s:.2f}" for s in channel_scales)

                        if (
                            enable_instance_scale
                            and instance_scale_range is not None
                            and num_instances > 0
                        ):
                            lo, hi = instance_scale_range
                            instance_scales = _sample_scalings(lo, hi, num_instances)
                            job["instance_scales"] = instance_scales
                            aug_id += f"_is{is_i}" + "_".join(f"{s:.2f}" for s in instance_scales)

                        if not aug_id:
                            # Intensity-only path where instance scale was skipped (0 instances)
                            # and channel scale somehow empty — should not happen; keep stable id.
                            aug_id = "_aug"

                        job["aug_id"] = aug_id
                        jobs.append(job)

    return jobs


def apply_augmentation_set(
    raw: np.ndarray,
    augmentation: dict,
    instances: np.ndarray | None = None,
) -> np.ndarray:
    """Apply intensity then geometry augs to raw (C,Z,Y,X). Labels are not modified."""
    image = raw
    if "instance_scales" in augmentation:
        if instances is None:
            raise ValueError("instance_scales present but instances is None")
        image = per_instance_intensity_scale(
            image, instances, augmentation["instance_scales"]
        )
    if "channel_scales" in augmentation:
        image = per_channel_multiplicative_scale(
            image, augmentation["channel_scales"]
        )
    # Identity c / a / k are no-ops when those augs were disabled at job gen time.
    image = channel_flip(image, augmentation["c"])
    image = axis_rotation(image, augmentation["a"], augmentation["k"])
    return image

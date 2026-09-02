"""Augmentation job generation and application for BiaPy FISBe prep."""
from itertools import permutations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imaging_helpers_hpc.processing import axis_rotation, channel_flip

NUM_CHANNELS = 3
_IDENTITY_CHANNEL_ORDER = (0, 1, 2)
_SCALE_BAND_FRACTION = 0.1
_MIN_SCALE_LOG_L2 = 0.75
# Materialize / shuffle all 2**size masks only up to this size; above, sample ints.
_MASK_ENUM_MAX_SIZE = 12


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


def _band_margin(lo: float, hi: float) -> float:
    return (hi - lo) * _SCALE_BAND_FRACTION


def _scales_from_mask(lo: float, hi: float, mask: np.ndarray) -> np.ndarray:
    """Draw extreme-band scales; False = low band, True = high band."""
    mask = np.asarray(mask, dtype=bool)
    margin = _band_margin(lo, hi)
    out = np.empty(mask.shape[0], dtype=np.float32)
    n_low = int(np.sum(~mask))
    n_high = int(np.sum(mask))
    if n_low:
        out[~mask] = np.random.uniform(lo, lo + margin, size=n_low)
    if n_high:
        out[mask] = np.random.uniform(hi - margin, hi, size=n_high)
    return out


def _mask_from_bits(bits: int, size: int) -> np.ndarray:
    return np.array([bool((bits >> i) & 1) for i in range(size)], dtype=bool)


def _sample_scalings(lo: float, hi: float, size: int) -> np.ndarray:
    """One scale vector: each slot independently low-band or high-band."""
    mask = np.random.randint(0, 2, size=size).astype(bool)
    return _scales_from_mask(lo, hi, mask)


def _scale_log_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.log(np.asarray(a)) - np.log(np.asarray(b))))


def _scale_set_capacity(size: int) -> int:
    """Number of distinct low/high patterns (2**size)."""
    return 1 << size


def _unique_mask_bits(size: int, n: int) -> list[int]:
    """Return up to ``n`` distinct bitmask ints in ``0 .. 2**size - 1``."""
    capacity = _scale_set_capacity(size)
    n_eff = min(n, capacity)
    if size <= _MASK_ENUM_MAX_SIZE:
        return [int(b) for b in np.random.permutation(capacity)[:n_eff]]
    chosen: set[int] = set()
    while len(chosen) < n_eff:
        chosen.add(int(np.random.randint(0, capacity)))
    return list(chosen)


def _plan_scale_set(
    lo: float, hi: float, size: int, n: int
) -> list[np.ndarray]:
    """Plan up to ``n`` diverse scale vectors (capped at 2**size distinct masks)."""
    if size <= 0 or n <= 0:
        return []
    out: list[np.ndarray] = []
    for bits in _unique_mask_bits(size, n):
        mask = _mask_from_bits(bits, size)
        scales = _scales_from_mask(lo, hi, mask)
        for _ in range(3):
            if all(
                _scale_log_l2(scales, prev) >= _MIN_SCALE_LOG_L2 for prev in out
            ):
                break
            scales = _scales_from_mask(lo, hi, mask)
        out.append(scales)
    return out


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

    Channel/instance scale draws are pre-planned with distinct low/high masks so
    jobs look different. Effective draw counts may be lower than requested when
    ``n > 2**size`` (e.g. 2 instances support at most 4 instance-scale patterns).
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

    if enable_channel_scale and channel_scale_range is not None:
        lo_cs, hi_cs = channel_scale_range
        channel_scale_sets: list[np.ndarray | None] = _plan_scale_set(
            lo_cs, hi_cs, NUM_CHANNELS, num_channel_scales
        )
    else:
        channel_scale_sets = [None]

    if (
        enable_instance_scale
        and instance_scale_range is not None
        and num_instances > 0
    ):
        lo_is, hi_is = instance_scale_range
        instance_scale_sets: list[np.ndarray | None] = _plan_scale_set(
            lo_is, hi_is, num_instances, num_instance_scales
        )
    else:
        instance_scale_sets = [None]

    jobs = []
    for ro in channel_orders:
        for a in axis_choices:
            for k in k_choices:
                for cs_i, channel_scales in enumerate(channel_scale_sets):
                    for is_i, instance_scales in enumerate(instance_scale_sets):
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

                        if channel_scales is not None:
                            job["channel_scales"] = channel_scales
                            aug_id += "_cs" + "_".join(
                                f"{s:.2f}" for s in channel_scales
                            )

                        if instance_scales is not None:
                            job["instance_scales"] = instance_scales
                            aug_id += f"_is{is_i}" + "_".join(
                                f"{s:.2f}" for s in instance_scales
                            )

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

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import torch
import zarr

from . import torch_model
from PatchPerPix.visualize import visualize_patches

logger = logging.getLogger(__name__)


def _fg_mask_from_numinst_block(numinst_block, fg_thresh):
    """Build FG mask for one (C, Z, Y, X) numinst block."""
    if numinst_block.shape[0] > 1:
        return (numinst_block[0] < 0.1)
    return numinst_block[0] >= fg_thresh


def _aff_chunk_path(aff_out, bz, by, bx):
    """Filesystem path of the channel-0 spatial chunk for zarr v2 DirectoryStore."""
    store_path = getattr(aff_out.store, "path", None) or getattr(
        aff_out.chunk_store, "path", None)
    if store_path is None:
        return None
    # Channel axis is fully contained in one chunk (chunks[0] == shape[0]).
    return os.path.join(store_path, f"0.{bz}.{by}.{bx}")


def _aff_chunk_exists(aff_out, bz, by, bx):
    path = _aff_chunk_path(aff_out, bz, by, bx)
    if path is None:
        return False
    return os.path.exists(path)


def _block_slices(spatial_shape, block_size):
    """Yield (bz, by, bx, zs, ys, xs) for chunk-aligned spatial blocks."""
    cz, cy, cx = block_size
    nz, ny, nx = spatial_shape
    for bz, z0 in enumerate(range(0, nz, cz)):
        for by, y0 in enumerate(range(0, ny, cy)):
            for bx, x0 in enumerate(range(0, nx, cx)):
                zs = slice(z0, min(z0 + cz, nz))
                ys = slice(y0, min(y0 + cy, ny))
                xs = slice(x0, min(x0 + cx, nx))
                yield bz, by, bx, zs, ys, xs


def _load_block(pred_code_z, numinst_z, zs, ys, xs, fg_thresh):
    """Read one spatial block of code + FG mask from zarr."""
    numinst_block = np.asarray(numinst_z[(slice(None), zs, ys, xs)])
    fg_mask = _fg_mask_from_numinst_block(numinst_block, fg_thresh)
    if not np.any(fg_mask):
        return None, fg_mask
    code_block = np.asarray(
        pred_code_z[(slice(None), zs, ys, xs)], dtype=np.float32)
    return code_block, fg_mask


def _decode_block_gpu(model, device, code_block, fg_mask, batch_size, patch_vol):
    """Decode FG voxels in a block; return float16 (patch_vol, bz, by, bx) buffer."""
    bz, by, bx = fg_mask.shape
    buf = np.zeros((patch_vol, bz, by, bx), dtype=np.float16)
    mz, my, mx = np.nonzero(fg_mask)
    n_fg = len(mz)
    if n_fg == 0:
        return buf, 0

    # code_block: (C, bz, by, bx) -> gather to (N, C)
    codes = code_block[:, mz, my, mx].T
    with torch.no_grad():
        for b in range(0, n_fg, batch_size):
            batch = codes[b:b + batch_size]
            preds = model.decoder(
                torch.as_tensor(batch, device=device))
            preds = preds.reshape(-1, patch_vol).half().cpu().numpy()
            sl = slice(b, b + len(batch))
            buf[:, mz[sl], my[sl], mx[sl]] = preds.T
    return buf, n_fg


def decode_sample(config, model, sample, device, aff_out=None):
    """Decode pred_code -> pred_affs one chunk-aligned spatial block at a time.

    Reads and writes whole blocks (not per-voxel) so GPU work is not drowned
    by zarr read-modify-write thrash. Skips blocks whose aff chunk already
    exists when writing to a zarr DirectoryStore (resume). Prefetches the next
    block on a background thread.
    """
    batch_size = config['decode_batch_size']
    patchshape = config['patchshape']
    if type(patchshape) != np.ndarray:
        patchshape = np.array(patchshape)
    patchshape = patchshape[patchshape > 1]
    patch_vol = int(np.prod(patchshape))
    fg_thresh = config['fg_thresh']

    if "zarr" not in config['output_format']:
        raise NotImplementedError("invalid input format")

    zf_in = zarr.open(sample, 'r')
    pred_code_z = zf_in[config['code_key']]
    numinst_key = config.get('numinst_key', config.get('fg_key'))
    numinst_z = zf_in[numinst_key]
    spatial_shape = pred_code_z.shape[1:]

    if aff_out is not None:
        block_size = tuple(int(c) for c in aff_out.chunks[1:])
        output = None
    else:
        # HDF / in-memory path: use the same block grid as zarr creates.
        block_size = tuple(min(64, s) for s in spatial_shape)
        output = np.zeros((patch_vol,) + spatial_shape, dtype=np.float32)

    blocks = list(_block_slices(spatial_shape, block_size))
    n_blocks = len(blocks)
    msg = (
        f"blockwise decode: shape={spatial_shape} block={block_size} "
        f"n_blocks={n_blocks} batch_size={batch_size}")
    logger.info(msg)
    print(msg, flush=True)

    resume = aff_out is not None
    skipped = 0
    decoded_fg = 0
    t0 = time.time()

    def _prefetch(item):
        bz, by, bx, zs, ys, xs = item
        if resume and _aff_chunk_exists(aff_out, bz, by, bx):
            return item, "skip", None, None
        code_block, fg_mask = _load_block(
            pred_code_z, numinst_z, zs, ys, xs, fg_thresh)
        return item, "ok", code_block, fg_mask

    def _write_block(zs, ys, xs, buf):
        if aff_out is not None:
            aff_out[(slice(None), zs, ys, xs)] = buf
        else:
            output[(slice(None), zs, ys, xs)] = buf.astype(np.float32)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending_write = None
        next_future = pool.submit(_prefetch, blocks[0]) if blocks else None

        for i, item in enumerate(blocks):
            bz, by, bx, zs, ys, xs = item
            block_t0 = time.time()

            # Drain previous write before starting GPU work on this block.
            if pending_write is not None:
                pending_write.result()
                pending_write = None

            pref_item, status, code_block, fg_mask = next_future.result()
            assert pref_item == item

            # Kick off prefetch of the following block.
            if i + 1 < n_blocks:
                next_future = pool.submit(_prefetch, blocks[i + 1])
            else:
                next_future = None

            if status == "skip":
                skipped += 1
                if (i + 1) % 20 == 0 or i + 1 == n_blocks:
                    msg = (
                        f"block {i + 1}/{n_blocks} skipped (resume) "
                        f"bz,by,bx={(bz, by, bx)}")
                    logger.info(msg)
                    print(msg, flush=True)
                continue

            if code_block is None:
                # Empty FG: still materialize a zero chunk for completeness checks.
                buf = np.zeros(
                    (patch_vol,) + fg_mask.shape, dtype=np.float16)
                n_fg = 0
            else:
                buf, n_fg = _decode_block_gpu(
                    model, device, code_block, fg_mask, batch_size, patch_vol)

            pending_write = pool.submit(_write_block, zs, ys, xs, buf)
            decoded_fg += n_fg
            dt = time.time() - block_t0
            # Log every block with FG, otherwise every 20th empty block.
            if n_fg > 0 or (i + 1) % 20 == 0 or i + 1 == n_blocks:
                msg = (
                    f"block {i + 1}/{n_blocks} bz,by,bx={(bz, by, bx)} "
                    f"fg={n_fg} {dt:.2f}s/block")
                logger.info(msg)
                print(msg, flush=True)

        if pending_write is not None:
            pending_write.result()

    elapsed = time.time() - t0
    msg = (
        f"blockwise decode done: fg={decoded_fg} blocks={n_blocks} "
        f"skipped={skipped} in {elapsed:.1f}s "
        f"({elapsed / max(n_blocks, 1):.2f}s/block)")
    logger.info(msg)
    print(msg, flush=True)

    if aff_out is not None:
        return (patch_vol,) + spatial_shape
    return output


def decode(**config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = torch_model.UnetModelWrapper(
        config, device, 0, for_inference=True)
    model.eval()
    try:
        model = model.to(device)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to move model to device. If you are using a child process "
            "to run your model, maybe you already initialized CUDA by sending "
            "your model to device in the main process."
        ) from e

    checkpoint = torch.load(config["checkpoint_file"], map_location=device)
    if config.get("use_swa"):
        logger.info("loading swa checkpoint")
        model = torch.optim.swa_utils.AveragedModel(model)
        model.load_state_dict(checkpoint["swa_model_state_dict"])
    else:  # "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    for idx, sample in enumerate(config['samples']):
        logger.info(
            "decoding sample: %s (%s/%s)", sample, idx, len(config['samples']))

        # Write pred_affs in-place into the prediction sample path so list-mode
        # {dataset}/{stem} zarrs keep codes and affinities together.
        sample_name = os.path.basename(sample).split('.')[0]
        outfn = sample
        mode = 'a' if os.path.exists(outfn) else 'w'

        if config['output_format'] == 'zarr':
            zf = zarr.open(outfn, mode=mode)
            zf_in = zarr.open(sample, 'r')
            spatial_shape = zf_in[config['code_key']].shape[1:]
            patchshape = np.array(config['patchshape'])
            patchshape = patchshape[patchshape > 1]
            patch_vol = int(np.prod(patchshape))

            aff_out = None
            if config['aff_key'] in zf:
                aff_out = zf[config['aff_key']]
            else:
                zf.create(
                    config['aff_key'],
                    shape=(patch_vol,) + spatial_shape,
                    dtype=np.float16,
                    chunks=(patch_vol,) + tuple(
                        min(64, s) for s in spatial_shape))
                zf[config['aff_key']].attrs['offset'] = (
                    [0] * len(config['voxel_size']))
                zf[config['aff_key']].attrs['resolution'] = config['voxel_size']
                aff_out = zf[config['aff_key']]

            with torch.no_grad():
                _ = decode_sample(
                    config, model, sample, device, aff_out=aff_out)

            if config.get('show_patches'):
                if sample_name in config.get('samples_to_visualize', []):
                    prediction = np.array(aff_out)
                    outfn_patched = os.path.join(
                        config['output_folder'], "vis", sample_name + '.hdf')
                    os.makedirs(os.path.dirname(outfn_patched), exist_ok=True)
                    out_key = config['aff_key'] + '_patched'
                    _ = visualize_patches(
                        prediction, config['patchshape'],
                        out_file=outfn_patched, out_key=out_key)

        elif config['output_format'] == 'hdf':
            with torch.no_grad():
                prediction = decode_sample(config, model, sample, device)
            outf = h5py.File(outfn, mode)
            outf.create_dataset(
                config['aff_key'],
                data=prediction,
                compression='gzip'
            )

            if config.get('show_patches'):
                if sample_name in config.get('samples_to_visualize', []):
                    outfn_patched = os.path.join(
                        config['output_folder'], "vis", sample_name + '.hdf')
                    os.makedirs(os.path.dirname(outfn_patched), exist_ok=True)
                    out_key = config['aff_key'] + '_patched'
                    _ = visualize_patches(
                        prediction, config['patchshape'],
                        out_file=outfn_patched, out_key=out_key)
        else:
            raise NotImplementedError

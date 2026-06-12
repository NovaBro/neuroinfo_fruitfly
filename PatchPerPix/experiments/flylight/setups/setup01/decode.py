import logging
import os
import toml
import zarr

import h5py
import numpy as np
import torch

from . import torch_model
from PatchPerPix.visualize import visualize_patches

logger = logging.getLogger(__name__)


def _fg_mask_from_numinst_zarr(numinst_z, fg_thresh):
    """Build foreground mask one z-slice at a time to limit RAM use."""
    spatial_shape = numinst_z.shape[1:]
    pred_fg = np.zeros(spatial_shape, dtype=np.uint8)
    if numinst_z.shape[0] > 1:
        for z in range(spatial_shape[0]):
            pred_fg[z] = (np.array(numinst_z[0, z]) < 0.1).astype(np.uint8)
    else:
        for z in range(spatial_shape[0]):
            pred_fg[z] = (np.array(numinst_z[0, z]) >= fg_thresh).astype(np.uint8)
    return pred_fg


def decode_sample(config, model, sample, device, aff_out=None):
    batch_size = config['decode_batch_size']
    code_units = config['code_units']
    patchshape = config['patchshape']
    if type(patchshape) != np.ndarray:
        patchshape = np.array(patchshape)
    patchshape = patchshape[patchshape > 1]
    patch_vol = int(np.prod(patchshape))

    if "zarr" not in config['output_format']:
        raise NotImplementedError("invalid input format")

    zf_in = zarr.open(sample, 'r')
    pred_code_z = zf_in[config['code_key']]
    numinst_key = config.get('numinst_key', config.get('fg_key'))
    pred_fg = _fg_mask_from_numinst_zarr(
        zf_in[numinst_key], config['fg_thresh'])

    fg_coords = np.transpose(np.nonzero(pred_fg))
    num_batches = int(np.ceil(fg_coords.shape[0] / float(batch_size)))
    logger.info("processing %i fg voxels in %i batches",
                len(fg_coords), num_batches)

    output = None
    if aff_out is None:
        output = np.zeros((patch_vol,) + pred_fg.shape, dtype=np.float32)

    for batch_idx, b in enumerate(range(0, len(fg_coords), batch_size)):
        batch_coords = fg_coords[b:b + batch_size]
        pred_code_batched = []
        for z, y, x in batch_coords:
            code_vec = np.array(
                pred_code_z[(slice(None), int(z), int(y), int(x))],
                dtype=np.float32)
            pred_code_batched.append(code_vec.reshape(1, code_units))
        logger.info(
            '%s/%s: in decode sample: %s',
            batch_idx, num_batches, pred_code_batched[0].shape)
        predictions = model.decoder(
            torch.as_tensor(
                np.stack(pred_code_batched, axis=0).astype(dtype=np.float32),
                device=device))

        logger.info("%s %s", predictions.size(), len(batch_coords))
        for i, (z, y, x) in enumerate(batch_coords):
            z, y, x = int(z), int(y), int(x)
            prediction = predictions[i].cpu().detach().numpy().reshape(patch_vol)
            if aff_out is not None:
                aff_out[(slice(None), z, y, x)] = prediction
            else:
                output[(slice(None), z, y, x)] = prediction

    if aff_out is not None:
        return (patch_vol,) + pred_fg.shape
    return output


def decode(**config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = torch_model.UnetModelWrapper(config, device, 0)
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
    else: # "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    for idx, sample in enumerate(config['samples']):
        logger.info("decoding sample: %s (%s/%s)", sample, idx, len(config['samples']))

        sample_name = os.path.basename(sample).split('.')[0]
        outfn = os.path.join(config['output_folder'],
                             sample_name + '.' + config['output_format'])
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

            prediction_shape = decode_sample(
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

"""Weight loader."""

import numpy as np

from mindspore.train.serialization import load_checkpoint


def load_infer_weights(config):
    """
    Load weights from ckpt or npz.

    Args:
        config (Seq2seqConfig): Config.

    Returns:
        dict, weights.
    """
    model_path = config.existed_ckpt
    if model_path.endswith(".npz"):
        ms_ckpt = np.load(model_path)
        is_npz = True
    else:
        ms_ckpt = load_checkpoint(model_path)
        is_npz = False
    weights = {}
    for param_name in ms_ckpt:
        infer_name = param_name.replace("seq2seq.seq2seq.", "")
        if infer_name.startswith("embedding_lookup."):
            if is_npz:
                weights[infer_name] = ms_ckpt[param_name]
            else:
                weights[infer_name] = ms_ckpt[param_name].data.asnumpy()
            infer_name = "beam_decoder.decoder." + infer_name
            if is_npz:
                weights[infer_name] = ms_ckpt[param_name]
            else:
                weights[infer_name] = ms_ckpt[param_name].data.asnumpy()
            continue
        elif not infer_name.startswith("seq2seq_encoder"):
            if infer_name.startswith("seq2seq_decoder."):
                infer_name = infer_name.replace("seq2seq_decoder.", "decoder.")
            infer_name = "beam_decoder.decoder." + infer_name

        if is_npz:
            weights[infer_name] = ms_ckpt[param_name]
        else:
            weights[infer_name] = ms_ckpt[param_name].data.asnumpy()
    return weights

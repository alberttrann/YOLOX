"""YOLOX TrainTransform plus an explicit letterbox-validity channel.

The 4th image channel is training-only metadata, never a detector input.
For MosaicDetection's final *composite canvas*, the canvas is intentionally
considered valid training background, including synthetic affine fill.
This is NOT a per-source-pixel provenance mask; ignored/crowd regions require
separate labels and should not be inferred from 114-colored pixels.
"""
import numpy as np
from yolox.data import TrainTransform


class PGSMAChannelMaskTransform:
    def __init__(self, max_labels, flip_prob, hsv_prob):
        self.base = TrainTransform(max_labels=max_labels,
                                   flip_prob=flip_prob, hsv_prob=hsv_prob)

    def __call__(self, image, targets, input_dim):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected raw 3-channel BGR image before TrainTransform')
        ih, iw = map(int, input_dim)
        h, w = map(int, image.shape[:2])
        if min(ih, iw, h, w) <= 0:
            raise ValueError('Invalid image or training canvas dimensions')
        # YOLOX preproc() uses top-left letterboxing with int() truncation.
        scale = min(ih / h, iw / w)
        vh, vw = min(ih, int(h * scale)), min(iw, int(w * scale))
        image_chw, labels = self.base(image, targets, input_dim)
        if image_chw.shape != (3, ih, iw):
            raise RuntimeError('Unexpected TrainTransform output geometry; adapt validity tracking')
        mask = np.zeros((1, ih, iw), dtype=np.float32)
        mask[:, :vh, :vw] = 1.0
        image_4c = np.concatenate((image_chw.astype(np.float32, copy=False), mask), axis=0)
        return np.ascontiguousarray(image_4c), labels

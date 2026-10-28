"""Normalize detector quadrilaterals before the pinned OCR crop/result APIs."""

METHOD = 'detector_list_quads_to_arrays_v2'


class ArrayBoxDetector:
    def __init__(self, detector):
        self.detector = detector

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def __call__(self, *args, **kwargs):
        boxes, elapsed = self.detector(*args, **kwargs)
        # Crop indexing and the public result's .tolist() both require arrays.
        # Preserve existing ndarray objects, dtype and coordinates verbatim.
        if isinstance(boxes, list):
            import numpy as np
            boxes = np.asarray(boxes, dtype=np.float32)
        return boxes, elapsed


def adapt_detector(model):
    if not isinstance(model.text_detector, ArrayBoxDetector):
        model.text_detector = ArrayBoxDetector(model.text_detector)

"""Image-only prose can locate a displaced formula beside its unique successor.

This is a placement witness, never a transcription editor. Short glyphs and
mathematical strings cannot establish ownership, even at high OCR confidence.
"""
import re

from book_pipeline.config import digest

METHOD = 'unique_adjacent_vertical_prose_anchor_v1'
MIN_CONFIDENCE = .99
PROSE = re.compile(r'[\u3400-\u9fff]{6,60}')


def policy_signature():
    from importlib.metadata import version
    from pathlib import Path
    from mineru.model.registry import MINERU_4_MODELS_ONNX as repo
    from mineru.model.ocr.resources import PPOCRV6_DICT_PATH
    paths = [repo.ocr_det.local_path(), repo.ocr_rec.local_path(), Path(PPOCRV6_DICT_PATH)]
    from book_pipeline.ingestion.ocr_geometry import METHOD as CROP_GEOMETRY
    return dict(method=METHOD, confidence=MIN_CONFIDENCE, crop_geometry=CROP_GEOMETRY,
                models=[digest(p) if p.is_file() else None for p in paths],
                mineru=version('mineru'), onnxruntime=version('onnxruntime'))


def observe_prose(rgb):
    """Read a cell without being given expected text, proposals or coordinates."""
    import numpy as np
    from book_pipeline.publishing.image_search import local_ocr
    engine = local_ocr()
    if engine is None:
        return []
    model, signature = engine
    rows = []
    h, w = rgb.shape[:2]
    for poly, (text, score) in model.ocr(rgb[:, :, ::-1].copy())[0] or []:
        p = np.asarray(poly, dtype=float)
        if (not PROSE.fullmatch(text) or not MIN_CONFIDENCE <= float(score) <= 1
                or p.shape != (4, 2) or not np.isfinite(p).all()):
            continue
        x, y = p.min(0); z, t = p.max(0)
        pitch = (t-y)/len(text)
        if not (0 <= x < z <= w and 0 <= y < t <= h
                and t-y >= 2*(z-x) and .4*(z-x) <= pitch <= 1.8*(z-x)):
            continue
        rows.append(dict(text=text, score=float(score), bbox=[float(x), float(y), float(z), float(t)],
                         reader_signature=signature))
    return rows


def following_anchor(bounds, original, following, observations, source_texts):
    """Require unique literal successor, same column and immediate adjacency.

    The ordinary-text prefix may end where the detector splits a paragraph.
    Its entire independent reading must match, with no expected-word OCR.
    The original rectangle still owns the column; other geometry vetoes apply
    after this witness, including intervening prose and other formula owners.
    """
    if bounds is None or not following:
        return None
    x, y, z, t = bounds
    matches = []
    for row in observations:
        text, score = row['text'], row['score']
        if (not PROSE.fullmatch(text) or not MIN_CONFIDENCE <= score <= 1
                or not row.get('reader_signature') or not following.startswith(text)
                or sum(s.count(text) for s in source_texts) != 1):
            continue
        a, b, c, d = row['bbox']
        pitch = (d-b)/len(text)
        if not (pitch > 0 and .4*(c-a) <= pitch <= 1.8*(c-a)
                and d-b >= 2*(c-a) and x <= (a+c)/2 <= z
                and -.2*pitch <= b-t <= .7*pitch and t-y <= 3*pitch):
            continue
        common = max(0, min(z, original[2])-max(x, original[0]))
        if common < .35*min(z-x, original[2]-original[0]):
            continue
        matches.append(dict(method=METHOD, observation=row, following_text=following,
                            original_bbox=original, candidate_bbox=bounds))
    return matches[0] if len(matches) == 1 else None

"""Page-level font competition for independently read archive-scan words.

Only independently read words are hypotheses. Several optical sizes and
bounded ink spread compete equally in every style. Weight additionally needs
an aspect-preserving match: stretching each template to a word's rectangle
erases the wider advances that distinguish bold type from scan ink spread.
"""
from functools import lru_cache
from hashlib import sha256
import re
from statistics import median

from book_pipeline.config import ROOT, digest, fingerprint
from book_pipeline.publishing.latin_word_ocr import confirmed_word

METHOD = 'archive_scan_metric_typography_v6'
FONTDIR = ROOT/'.tools/.TinyTeX/texmf-dist/fonts/type1/public/amsfonts/cm'
CM = (('cmr10', 'cmbx10', 'cmti10', 'cmbxti10'),
      ('cmr12', 'cmbx12', 'cmti12', 'cmbxti10'))
FAMILIES = {'computer_modern': CM, 'times': (('tiro', 'tibo', 'tiit', 'tibi'),)}
LABEL_FONTS = ('cmss10', 'cmssbx10', 'cmssi10')
GAINS = (0., .5, 1.)
MIN_WORDS = 12
MIN_FAMILY_SCORE = .8
MIN_FAMILY_MARGIN = .08
MIN_SLANT_SCORE = .75
MIN_SLANT_MARGIN = .18
MIN_METRIC_SCORE = .85
MIN_METRIC_MARGIN = .18
HEIGHT_ERRORS = (-1, 0, 1)


@lru_cache(maxsize=1)
def runtime_signature():
    import pymupdf
    paths = {n: FONTDIR/(n+'.pfb') for n in (*LABEL_FONTS, *(n for group in CM for n in group))}
    if not all(p.is_file() for p in paths.values()):
        return None
    return fingerprint(dict(method=METHOD, fonts={n: digest(p) for n, p in paths.items()},
        times=[sha256(pymupdf.Font(n).buffer).hexdigest() for n in FAMILIES['times'][0]],
        gains=GAINS, min_words=MIN_WORDS, family_score=MIN_FAMILY_SCORE,
        family_margin=MIN_FAMILY_MARGIN, slant_score=MIN_SLANT_SCORE,
        slant_margin=MIN_SLANT_MARGIN, metric_score=MIN_METRIC_SCORE,
        metric_margin=MIN_METRIC_MARGIN, height_errors=HEIGHT_ERRORS))


@lru_cache(maxsize=512)
def templates(text):
    import numpy as np
    import pymupdf
    from book_pipeline.publishing.raster_body_styles import trim_ink
    result = {}
    for groups in (*FAMILIES.values(), (LABEL_FONTS,)):
        for group in groups:
            for name in group:
                if name in result:
                    continue
                path = FONTDIR/(name+'.pfb')
                font = pymupdf.Font(fontfile=str(path)) if name.startswith('cm') else pymupdf.Font(name)
                with pymupdf.open() as pdf:
                    page = pdf.new_page(width=font.text_length(text, fontsize=60)+60, height=140)
                    kwargs = dict(fontname='scan', fontfile=str(path)) if name.startswith('cm') else dict(fontname=name)
                    page.insert_text((25, 100), text, fontsize=60, **kwargs)
                    pix = page.get_pixmap(colorspace=pymupdf.csGRAY)
                    result[name] = trim_ink(255-np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width))
    return result


def shape_scores(ink, text):
    import cv2
    import numpy as np
    from book_pipeline.publishing.raster_body_styles import prose_ink, trim_ink
    # Tight grayscale cropping uses observed dark strokes, not paper grain.
    support = ink > 64
    y, x = np.where(support)
    if not len(x):
        return {}
    ink = ink[y.min():y.max()+1, x.min():x.max()+1]
    ink, text = prose_ink(ink, text)
    ink = trim_ink(ink)
    if ink is None or min(ink.shape) < 8 or not re.fullmatch(r"[A-Za-z]+(?:['-][A-Za-z]+)*[.,;:!?]?", text):
        return {}
    scores = {}
    for name, template in templates(text).items():
        base = cv2.resize(template, (ink.shape[1], ink.shape[0]))
        spread = cv2.dilate(base, np.ones((3, 3), np.uint8))
        scores[name] = max(float(cv2.matchTemplate(ink.astype(np.float32),
            cv2.addWeighted(base, 1-gain, spread, gain, 0).astype(np.float32),
            cv2.TM_CCOEFF_NORMED)[0, 0]) for gain in GAINS)
    return {family: [max(scores[g[k]] for g in groups) for k in range(4)]
            for family, groups in FAMILIES.items()}


def select_family(samples):
    """Require diverse words, line locations and a dominant family advantage."""
    unique = {}
    for sample in samples:
        if len(sample['text'].strip('.,;:!?')) < 4 or not sample['scores']:
            continue
        key = sample['text'].strip('.,;:!?').lower()
        if key not in unique or max(map(max, sample['scores'].values())) > max(map(max, unique[key]['scores'].values())):
            unique[key] = sample
    words = list(unique.values())
    report = dict(status='abstained_insufficient_font_evidence', unique_words=len(words),
                  weight='abstained_scan_gain_ambiguous')
    if len(words) < MIN_WORDS or len({round(w['bbox'][1]/10) for w in words}) < 4:
        return None, report
    scores = {f: median(max(w['scores'][f]) for w in words) for f in FAMILIES}
    family = max(scores, key=scores.get)
    others = [f for f in FAMILIES if f != family]
    margins = [max(w['scores'][family])-max(max(w['scores'][f]) for f in others) for w in words]
    report.update(scores=scores, median_margin=median(margins),
                  supported_fraction=sum(m >= .05 for m in margins)/len(words))
    if (scores[family] < MIN_FAMILY_SCORE or median(margins) < MIN_FAMILY_MARGIN
            or report['supported_fraction'] < .65):
        report['status'] = 'abstained_ambiguous_font_family'
        return None, report
    report.update(status='calibrated_slant_only', family=family)
    return family, report


def slant_scores(binary, gray, family):
    """Both pixel views must support slant; no score can authorize bold."""
    from book_pipeline.publishing.raster_body_styles import FONTS
    if family not in binary or family not in gray:
        return {}
    pairs = [(max(s[family][0:2]), max(s[family][2:4])) for s in (binary, gray)]
    if any(i < MIN_SLANT_SCORE or i-u < MIN_SLANT_MARGIN for u, i in pairs):
        return {}
    upright, italic = max(p[0] for p in pairs), min(p[1] for p in pairs)
    if italic-upright < MIN_SLANT_MARGIN:
        return {}
    return dict(zip(FONTS, (upright, upright, italic, italic)))


def complete_word_ink(ink):
    """Require a blank boundary before trusting a word's aspect ratio.

    A clipped descender shortens the observed height and makes ordinary type
    look wide. The completed strip may supply evidence later; this crop
    cannot. Use the same visible-stroke threshold in both pixel views.
    """
    import numpy as np
    support = ink > 64
    return (min(ink.shape) >= 8 and np.any(support)
            and not any(np.any(edge) for edge in
                (support[0], support[-1], support[:, 0], support[:, -1])))


def metric_scores(ink, text, family):
    """Fit uniform scale, bounded gain and translation, retaining word width.

    A one-pixel height uncertainty covers raster phase and ink thresholds.
    No independent horizontal scaling or tracking can absorb bold advances.
    All styles get exactly the same nuisance parameters and blank padding.
    """
    if family not in FAMILIES:
        return []
    scores = metric_face_scores(ink, text, [n for group in FAMILIES[family] for n in group])
    return ([max(scores[g[k]] for g in FAMILIES[family]) for k in range(4)]
            if scores else [])


def metric_face_scores(ink, text, names, *, strip_terminal=True):
    """Give every candidate font the same scale, scan gain and translation."""
    import cv2
    import numpy as np
    from book_pipeline.publishing.raster_body_styles import prose_ink
    if not complete_word_ink(ink):
        return {}
    y, x = np.where(ink > 64)
    ink = ink[y.min():y.max()+1, x.min():x.max()+1]
    if strip_terminal:
        ink, text = prose_ink(ink, text)
    if not re.fullmatch('[A-Za-z]+[.,;:!?]?', text):
        return {}
    y, x = np.where(ink > 64)
    ink = ink[y.min():y.max()+1, x.min():x.max()+1]
    h, w = ink.shape
    if min(h, w) < 8:
        return {}
    scores = {}
    for name in dict.fromkeys(names):
        temp = templates(text)[name]
        y, x = np.where(temp > 64)
        temp = temp[y.min():y.max()+1, x.min():x.max()+1]
        options = []
        for error in HEIGHT_ERRORS:
            height = h+error
            width = max(1, round(temp.shape[1]*height/temp.shape[0]))
            base = cv2.resize(temp, (width, height))
            base = cv2.copyMakeBorder(base, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
            spread = cv2.dilate(base, np.ones((3, 3), np.uint8))
            for gain in GAINS:
                shape = cv2.addWeighted(base, 1-gain, spread, gain, 0)
                size = (max(h, shape.shape[0])+8, max(w, shape.shape[1])+8)
                observed, candidate = np.zeros(size, np.uint8), np.zeros(size, np.uint8)
                observed[4:4+h, 4:4+w] = ink
                candidate[4:4+shape.shape[0], 4:4+shape.shape[1]] = shape
                padded = cv2.copyMakeBorder(observed, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
                options.append(float(cv2.matchTemplate(padded.astype(np.float32),
                    candidate.astype(np.float32), cv2.TM_CCOEFF_NORMED).max()))
        scores[name] = max(options)
    return scores


def label_bold_scores(binary, gray, text):
    """Establish a local sans family before comparing its weight hypotheses.

    A serif face cannot veto weight after losing the independent family
    competition. Each pixel view must establish both the family margin and
    the usual strong weight margin, retaining natural advances throughout.
    No label semantics authorize weight or resolve an ambiguous font family.
    Unsupported bold slant is never inferred from the available upright face.
    """
    from book_pipeline.publishing.raster_body_styles import FONTS
    names = (*LABEL_FONTS, *(n for groups in FAMILIES.values() for g in groups for n in g))
    views = [metric_face_scores(ink, text, names, strip_terminal=False) for ink in (binary, gray)]
    if any(not s or s['cmssbx10'] < MIN_METRIC_SCORE
           or s['cmssbx10']-max(s[n] for n in LABEL_FONTS if n != 'cmssbx10') < MIN_METRIC_MARGIN
           or s['cmssbx10']-max(v for n, v in s.items() if n not in LABEL_FONTS) < MIN_FAMILY_MARGIN
           for s in views):
        return {}
    regular = max(max(s[n] for n in LABEL_FONTS if n != 'cmssbx10') for s in views)
    bold = min(s['cmssbx10'] for s in views)
    if bold-regular < MIN_METRIC_MARGIN:
        return {}
    return dict(zip(FONTS, (regular, bold, regular, bold)))


def metric_style_scores(binary, gray):
    """Authorize each style axis only with two strong competing pixel fits."""
    from book_pipeline.publishing.raster_body_styles import FONTS
    if len(binary) != 4 or len(gray) != 4:
        return {}
    def supported(positive, negative):
        return all(max(s[k] for k in positive) >= MIN_METRIC_SCORE
            and max(s[k] for k in positive)-max(s[k] for k in negative) >= MIN_METRIC_MARGIN
            for s in (binary, gray))
    italic = supported((2, 3), (0, 1))
    bold = supported((1, 3), (0, 2))
    if not italic and not bold:
        return {}
    scores = [min(a, b) for a, b in zip(binary, gray)]
    # An undecided axis must not become a decision downstream. Flatten it,
    # retaining the independent evidence for the other axis.
    if not italic:
        scores = [max(scores[0], scores[2]), max(scores[1], scores[3])]*2
    if not bold:
        scores = [max(scores[0:2])]*2+[max(scores[2:4])]*2
    return dict(zip(FONTS, scores))


def calibrate_observations(rgb, observed, *, metric_families=None):
    import cv2
    from book_pipeline.publishing.raster_body_styles import supported_words, is_bold, is_italic, anchored_word_span, NUMBERED_LABEL
    samples, scored, scored_cache = [], [], {}
    for line in observed:
        words = []
        if line['score'] < .99:
            continue
        for word in line['words']:
            if not confirmed_word(word):
                continue
            x, y, z, t = word['bbox']
            if (not all(type(v) is int for v in (x, y, z, t))
                    or not (0 <= x < z <= rgb.shape[1] and 0 <= y < t <= rgb.shape[0])):
                continue
            # Repeated tokens need a touching independently read neighbor and
            # a unique literal phrase; a cached location alone is insufficient.
            if anchored_word_span(line['text'], word, line['words'], require_style=False) is None:
                continue
            key = (word['text'], x, y, z, t)
            if key not in scored_cache:
                gray = 255-cv2.cvtColor(rgb[y:t, x:z], cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                scored_cache[key] = shape_scores(binary, word['text']), shape_scores(gray, word['text']), binary, gray
            shape, soft, binary, gray = scored_cache[key]
            words.append((word, shape, soft, binary, gray))
            samples.append(dict(text=word['text'], bbox=word['bbox'], scores=shape))
        scored.append((line, words))
    family, report = select_family(samples)
    result = []
    if family is None:
        return result, report
    if metric_families is not None and family not in metric_families:
        report['status'] = 'calibrated_font_family'
        return result, report
    metric_cache, weighted = {}, set()
    for line, words in scored:
        selected = []
        for word, shape, soft, binary, gray in words:
            key = (word['text'], *word['bbox'])
            if key not in metric_cache:
                metric_cache[key] = metric_style_scores(metric_scores(binary, word['text'], family),
                                                        metric_scores(gray, word['text'], family))
            scores = metric_cache[key] or slant_scores(shape, soft, family)
            label = re.match(NUMBERED_LABEL, line['text'])
            located = anchored_word_span(line['text'], word, line['words'], require_style=False)
            if (not scores and label and located is not None
                    and located[1] <= label.end()
                    and len(re.sub('[^A-Za-z]', '', word['text'])) >= 4):
                scores = label_bold_scores(binary, gray, word['text'])
            if scores:
                selected.append(dict(word, scores=scores))
                if is_bold(scores):
                    weighted.add(key)
        selected = supported_words(selected, line['text'])
        if selected:
            result.append(dict(line, words=selected))
    report['slanted_word_observations'] = sum(is_italic(w['scores']) for r in result for w in r['words'])
    report.update(status='calibrated_metric_typography',
                  weight='requires_dual_view_aspect_preserving_match', bold_word_crops=len(weighted))
    return result, report

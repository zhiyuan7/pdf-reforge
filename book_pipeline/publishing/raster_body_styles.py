"""Conservative CPU typography for unchanged, independently read prose.

Exact line and word OCR establish ownership; raster font comparisons establish
typography. Slant and weight each need their own independent score margin.
Unmatched fonts, unsupported short tokens and gray notes remain unchanged.
Mixed math lines require a complete unchanged prose fragment as context.
"""
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
import re

from book_pipeline.config import digest, fingerprint, read, save
from book_pipeline.ingestion.panels import intersection_fraction
from book_pipeline.publishing.latin_word_ocr import confirmed_word, read_word, runtime_signature

METHOD = 'exact_cpu_scan_body_typography_v35'
# Ownership is revalidated on every run. Both collectors now independently
# read bounded numbered-label crops beside low-confidence formulas.
OBSERVATION_METHOD = 'cpu_scan_body_word_observations_v18'
CALIBRATION_OBSERVATION_METHOD = 'cpu_scan_body_word_observations_v20'
DPI = 300
FONTS = ('tiro', 'tibo', 'tiit', 'tibi')
MIN_SCORE = .65
MIN_MARGIN = .18
MIN_SUPPORTED_SCORE = .55
MIN_BOLD_SCORE = .70
NUMBERED_LABEL = (r'(?:[A-Za-z]{4,}(?: [A-Za-z]{3,}){0,2} \d+(?:\.\d+)*[.:](?=\s|$)'
                  r'|\d+(?:\.\d+)* [A-Za-z]{4,}\.)')
NUMBERED_PREFIX = NUMBERED_LABEL+r'(?: [A-Za-z]{2})?'


def literal_span(text, quote):
    """Ignore only OCR whitespace, and require one complete literal match."""
    compact = re.sub(r'\s+', '', quote)
    if not compact:
        return None
    pattern = r'\s*'.join(re.escape(c) for c in compact)
    hits = list(re.finditer(pattern, text))
    return hits[0].span() if len(hits) == 1 else None


def trim_ink(mask):
    import numpy as np
    y, x = np.where(mask > 0)
    return mask[y.min():y.max()+1, x.min():x.max()+1] if len(x) else None


@lru_cache(maxsize=2048)
def templates(text):
    import numpy as np
    import pymupdf
    result = {}
    for name in FONTS:
        font = pymupdf.Font(name)
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=font.text_length(text, fontsize=36)+40, height=100)
            page.insert_text((15, 65), text, fontname=name, fontsize=36)
            pix = page.get_pixmap(dpi=72, colorspace=pymupdf.csGRAY)
            result[name] = trim_ink(255-np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width))
    return result


def font_scores(mask, text):
    import cv2
    import numpy as np
    mask = trim_ink(mask)
    if mask is None or min(mask.shape) < 8:
        return {}
    scores = {name: float(cv2.matchTemplate(mask.astype(np.float32),
                cv2.resize(template, (mask.shape[1], mask.shape[0])).astype(np.float32),
                cv2.TM_CCOEFF_NORMED)[0, 0])
            for name, template in templates(text).items()}
    # Printed ligatures and tracking change horizontal placement inside a
    # word. Uniformly resizing a tightly spaced template cannot model that.
    # Retry only already italic-favoring shapes; require stronger evidence
    # and give upright fonts exactly the same alternatives.
    italic, upright = max(scores['tiit'], scores['tibi']), max(scores['tiro'], scores['tibo'])
    if (not is_italic(scores) and italic >= MIN_SUPPORTED_SCORE
            and italic-upright >= MIN_MARGIN and len(re.findall('[A-Za-z]', text)) >= 3):
        expanded = {name: max(scores[name], *(float(cv2.matchTemplate(mask.astype(np.float32),
            cv2.resize(template, (mask.shape[1], mask.shape[0])).astype(np.float32),
            cv2.TM_CCOEFF_NORMED)[0, 0]) for template in variants))
            for name, variants in spaced_templates(text).items()}
        if max(expanded['tiit'], expanded['tibi']) >= .75 and is_italic(expanded):
            return expanded
    return scores


@lru_cache(maxsize=256)
def spaced_templates(text):
    """Compare common print ligatures and bounded tracking in every style.

    These are font hypotheses, never replacements for the independently read
    text. A ligature changes only the painted template, not source offsets.
    """
    import numpy as np
    import pymupdf
    ligatures = text
    for letters, glyph in (('ffi', 'ﬃ'), ('ffl', 'ﬄ'), ('ff', 'ﬀ'), ('fi', 'ﬁ'), ('fl', 'ﬂ')):
        ligatures = ligatures.replace(letters, glyph)
    result = {}
    for name in FONTS:
        font = pymupdf.Font(name)
        variants = []
        for word in dict.fromkeys((text, ligatures)):
            for tracking in (0, .02, .04, .06, .08):
                with pymupdf.open() as pdf:
                    page = pdf.new_page(width=font.text_length(word, fontsize=60)+len(word)*tracking*60+40,
                                        height=140)
                    writer = pymupdf.TextWriter(page.rect)
                    x = 15
                    for char in word:
                        writer.append((x, 100), char, font=font, fontsize=60)
                        x += font.text_length(char, fontsize=60)+tracking*60
                    writer.write_text(page)
                    pix = page.get_pixmap(colorspace=pymupdf.csGRAY)
                    variants.append(trim_ink(255-np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)))
        result[name] = variants
    return result


def is_italic(scores):
    if set(scores) != set(FONTS):
        return False
    italic = max(scores['tiit'], scores['tibi'])
    upright = max(scores['tiro'], scores['tibo'])
    return italic >= MIN_SCORE and italic-upright >= MIN_MARGIN


def is_bold(scores):
    if set(scores) != set(FONTS):
        return False
    bold = max(scores['tibo'], scores['tibi'])
    regular = max(scores['tiro'], scores['tiit'])
    return bold >= MIN_BOLD_SCORE and bold-regular >= MIN_MARGIN


def word_boxes(mask, gap_fraction=.15):
    """Split only independently blank columns; never cut a connected glyph."""
    edges, start = [0], None
    for n, occupied in enumerate((mask > 0).any(0)):
        if not occupied and start is None:
            start = n
        if occupied and start is not None:
            if n-start >= max(4, mask.shape[0]*gap_fraction):
                edges.append((start+n)//2)
            start = None
    edges.append(mask.shape[1])
    return [(a, z) for a, z in zip(edges, edges[1:]) if z-a >= 12]


def prose_ink(mask, text):
    """A detached terminal mark need not share the preceding word's font.

    Compare only the letters when the final, independently blank-separated
    component is small enough to be the observed punctuation. Never remove
    a connected glyph, an apostrophe, a hyphen or a full-height component.
    """
    import numpy as np
    ink = trim_ink(mask)
    if ink is None or text[-1:] not in {'.', ',', ';', ':', '!', '?'}:
        return mask, text
    columns = (ink > 0).any(0)
    start = int(np.where(columns)[0][-1])
    while start and columns[start-1]:
        start -= 1
    mark = trim_ink(ink[:, start:])
    if (start >= .6*ink.shape[1] and mark.shape[0] <= .55*ink.shape[0]
            and mark.shape[1] <= .4*ink.shape[0]):
        return ink[:, :start], text[:-1]
    return mask, text


def scored_word(mask, text):
    """Preserve a reliable complete match; retry only detached punctuation.

    A low-resolution terminal letter may correlate better in the complete
    word. Removing punctuation must not discard that existing evidence.
    """
    if text[-1:] in {')', ']'}:
        ink, letters = bracket_ink(mask, text)
        # A parenthesis can be upright beside italic letters. Never apply a
        # composite font decision to punctuation whose font is independent.
        return (letters, font_scores(ink, letters)) if letters != text else (text, {})
    scores = font_scores(mask, text)
    if not is_italic(scores):
        ink, letters = prose_ink(mask, text)
        if letters != text:
            alternative = font_scores(ink, letters)
            if is_italic(alternative):
                return letters, alternative
    return text, scores


def scored_scan_word(mask, gray_ink, text):
    """Keep proven binary matches; retry ambiguous italic ink in grayscale.

    Thresholding can erase low-resolution strokes and their anti-aliased
    boundaries. The unthresholded source supplies an independent shape view,
    never a new reading. It must pass a stronger score with the same upright
    competition, and the original binary shape must already favor italic.
    """
    letters, scores = scored_word(mask, text)
    if is_italic(scores) or is_bold(scores) or set(scores) != set(FONTS):
        return letters, scores
    italic, upright = max(scores['tiit'], scores['tibi']), max(scores['tiro'], scores['tibo'])
    if italic >= MIN_SUPPORTED_SCORE and italic-upright >= MIN_MARGIN:
        alternative, soft = scored_word(gray_ink, text)
        if (is_italic(soft) and max(soft['tiit'], soft['tibi']) >= .75
                and alternative.rstrip('.,;:!?') == letters.rstrip('.,;:!?')):
            return alternative, soft
    return letters, scores


def bracket_ink(mask, text):
    """Separate a closing bracket only with blank columns and shape evidence."""
    import numpy as np
    ink = trim_ink(mask)
    if ink is None or text[-1:] not in {')', ']'} or len(text) < 2:
        return mask, text
    occupied = (ink > 0).any(0)
    start = int(np.where(occupied)[0][-1])
    while start and occupied[start-1]:
        start -= 1
    mark = trim_ink(ink[:, start:])
    if not (start >= .6*ink.shape[1] and .6*ink.shape[0] <= mark.shape[0]
            and mark.shape[1] <= .45*ink.shape[0]):
        return mask, text
    bracket = max(font_scores(mark, text[-1]).values(), default=0)
    letter = max(font_scores(mark, text[-2]).values(), default=0)
    if bracket >= .8 and bracket-letter >= MIN_MARGIN:
        return ink[:, :start], text[:-1]
    return mask, text


def numbered_short_span(line, word):
    """Locate a complete two-letter prose token immediately after a label.

    This only locates a candidate. Its reading, font fit, label evidence and
    source ownership still require independent validation by the caller.
    Single-letter mathematical variables never qualify.
    """
    match = re.match(NUMBERED_LABEL+r' ([A-Za-z]{2})(?!\w)', line)
    if match is None or match.group(1) != word['text']:
        return None
    span = literal_span(line, word['text'])
    return span if span == match.span(1) else None


def supported_words(words, line):
    """Require each glyph comparison to favor italic, including short words.

    An adjacent independently read word may support a borderline match, but
    cannot turn a normal or unknown shape italic. The combined score must
    still pass the original threshold. Exact line ownership is checked later.
    """
    selected = {n for n, w in enumerate(words) if confirmed_word(w)
                and len(re.sub('[^A-Za-z]', '', w['text'])) >= 3
                and (is_italic(w['scores']) or is_bold(w['scores']))}
    # A short word beside a formula lacks an ordinary multiword context.
    # A complete numbered label can anchor its location, but supplies no
    # style: both it and the short word must have their own strong ink fit.
    label = re.match(NUMBERED_LABEL, line)
    if label:
        label_words = [words[n] for n in selected
                       if len(re.sub('[^A-Za-z]', '', words[n]['text'])) >= 4
                       and (span := literal_span(line, words[n]['text'])) is not None
                       and span[1] <= label.end()]
        if label_words:
            selected.update(n for n, w in enumerate(words)
                            if confirmed_word(w) and numbered_short_span(line, w) is not None
                            and (is_italic(w['scores']) or is_bold(w['scores'])))
    for n, left in enumerate(words):
        for k in range(n+1, len(words)):
            right = words[k]
            if abs(left['bbox'][2]-right['bbox'][0]) > 2:
                continue
            if not re.search(r'(?<!\w)'+re.escape(left['text'])+r'\W+'
                             +re.escape(right['text'])+r'(?!\w)', line):
                continue
            pair = [left, right]
            if any(not confirmed_word(w) or set(w['scores']) != set(FONTS) or
                   max(w['scores']['tiit'], w['scores']['tibi']) < MIN_SUPPORTED_SCORE or
                   max(w['scores']['tiit'], w['scores']['tibi'])-
                   max(w['scores']['tiro'], w['scores']['tibo']) < MIN_MARGIN for w in pair):
                continue
            weights = [len(re.sub('[^A-Za-z]', '', w['text'])) for w in pair]
            scores = {font: sum(w['scores'][font]*weight for w, weight in zip(pair, weights))/sum(weights)
                      for font in FONTS}
            if is_italic(scores):
                selected.update((n, k))
    return [w for n, w in enumerate(words) if n in selected]


def context_windows(mask, words):
    """Bound retries by observed styled ink; cut only at blank columns.

    No source text supplies a crop, an expected answer or an OCR correction.
    Full line confidence remains mandatory unless a separately read context
    passes the same threshold and contains enough literal prose of its own.
    """
    import numpy as np
    height, width = mask.shape
    cuts = np.where(~(mask > 0).any(0))[0]
    cuts = np.unique(np.concatenate(([0], cuts, [width])))
    result = set()
    for a, z in words:
        for reach in (3, 6, 10):
            left = int(cuts[cuts <= max(0, a-reach*height)][-1])
            right = int(cuts[cuts >= min(width, z+reach*height)][0])
            for start, end in ((left, z), (a, right), (left, right)):
                if end-start >= 4*height and (start, end) != (0, width):
                    result.add((start, end))
    return sorted(result)


def enough_context(text):
    return (len(re.findall(r'[A-Za-z]+', text)) >= 4
            and len(re.sub('[^A-Za-z]', '', text)) >= 16
            or len(re.findall(r'[\u3400-\u9fff]', text)) >= 8)


def label_windows(mask):
    """Read short prefixes at observed blank gaps, without a transcript.

    A formula can lower the confidence of an otherwise clear numbered label.
    Bound retries to the first four ink-separated strips and sixteen row
    heights; only a complete independently read label can use this context.
    """
    boxes = word_boxes(mask)
    return [(boxes[0][0], end) for _, end in boxes[1:4]
            if end-boxes[0][0] <= 16*mask.shape[0]] if boxes else []


def complete_line_box(rgb, box):
    """Complete clipped glyphs on all sides, retaining neighboring-ink vetoes.

    Sideways italic strokes need the same completion as descenders. Both
    axes use a search bounded by the original line height, never its width.
    """
    vertical = complete_vertical_box(rgb, box)
    x, y, z, t = vertical
    horizontal = complete_vertical_box(rgb.transpose(1, 0, 2), [y, x, t, z],
                                       max(3, round((box[3]-box[1])*.2)), soft=True)
    return [horizontal[1], horizontal[0], horizontal[3], horizontal[2]]


def complete_vertical_box(rgb, box, pad=None, soft=False, blank_border=False):
    """Expand only to complete
    components mostly owned by the original strip, within 20% of its height.
    A component reaching the search edge or unrelated ink in the added rows
    vetoes expansion, so a neighboring line cannot become word evidence.
    """
    import cv2
    import numpy as np
    x, y, z, t = box
    pad = max(3, round((t-y)*.2)) if pad is None else pad
    top, bottom = max(0, y-pad), min(rgb.shape[0], t+pad)
    crop = rgb[top:bottom, x:z]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    threshold, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if soft:
        # Antialiased italic tips can be disconnected by Otsu even though
        # the original gray pixels connect them. Keep that observed bridge;
        # search bounds, ownership and unrelated-ink vetoes remain identical.
        _, ink = cv2.threshold(gray, max(threshold, 192), 255, cv2.THRESH_BINARY_INV)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    owned, low, high = set(), y-top, t-top
    for n, (_, cy, _, ch, area) in enumerate(stats[1:], 1):
        if area < 3:
            continue
        overlap = np.count_nonzero(labels[y-top:t-top] == n)
        if overlap >= .5*area:
            if cy == 0 or cy+ch == bottom-top:
                return box
            owned.add(n)
            low, high = min(low, int(cy)), max(high, int(cy+ch))
            if blank_border:
                # A complete glyph ending exactly on the strip edge still
                # needs visible blank pixels before its aspect ratio is safe.
                low, high = min(low, max(0, int(cy)-2)), max(high, min(bottom-top, int(cy+ch)+2))
    # Recognition needs a blank border around a complete glyph, just as the
    # original detector strip has. The added border is subject to the same
    # neighboring-ink veto as the completed components themselves.
    low = max(0, low-2) if low < y-top else low
    high = min(bottom-top, high+2) if high > t-top else high
    added = np.concatenate((labels[low:y-top].ravel(), labels[t-top:high].ravel()))
    if any(n and n not in owned and stats[n, cv2.CC_STAT_AREA] >= 3 for n in np.unique(added)):
        return box
    return [x, top+low, z, top+high]


def completed_word_observation(rgb, model, word):
    """Retry an edge-touching word only after a fresh image-only reading.

    The component and neighboring-ink guards remain in force. The existing
    observation is retained, and no transcript or font hypothesis is supplied
    to the reader. A changed or uncertain reading cannot supply new metrics.
    """
    box = complete_vertical_box(rgb, word['bbox'], blank_border=True)
    if box == word['bbox']:
        return None
    x, y, z, t = box
    rows = model.ocr(rgb[y:t, x:z, ::-1].copy(), det=False)[0] or []
    if len(rows) != 1:
        return None
    text, score = rows[0]
    if text != word['text'] or not .99 <= score <= 1:
        return None
    return dict(text=text, score=float(score), bbox=box, scores={})


def analyze_page(rgb, model, *, calibration=False):
    """Read a line and each isolated word independently; cache observations."""
    import cv2
    rows = model.ocr(rgb[:, :, ::-1].copy())[0] or []
    observed = []
    for polygon, (text, score) in rows:
        if not re.search(r'[A-Za-z]{3}', text):
            continue
        x, y = [round(min(p[k] for p in polygon))-2 for k in (0, 1)]
        z, t = [round(max(p[k] for p in polygon))+2 for k in (0, 1)]
        x, y, z, t = max(0, x), max(0, y), min(rgb.shape[1], z), min(rgb.shape[0], t)
        if x >= z or y >= t:
            continue
        # Retain reliable original-strip observations. A completed strip can
        # add evidence, never erase an independently verified earlier match.
        line_boxes = dict.fromkeys(((x, y, z, t), tuple(complete_vertical_box(rgb, [x, y, z, t])),
                                   tuple(complete_line_box(rgb, [x, y, z, t]))))
        for x, y, z, t in line_boxes:
            crop = rgb[y:t, x:z]
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            gray_ink = 255-gray
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            words = []
            # Narrow italic spaces can be shorter than 15% of the row height.
            # Keep both partitions: finer splits cannot authorize partial words
            # because independent OCR and literal token boundaries remain required.
            boxes = sorted(set(word_boxes(mask)+word_boxes(mask, .10)+word_boxes(mask, .06)))
            for a, b in boxes:
                # Recognition without detection sees exactly this isolated strip,
                # independently of the OCR that supplied the surrounding line.
                result = model.ocr(crop[:, a:b, ::-1].copy(), det=False)[0] or []
                if len(result) != 1:
                    continue
                word, confidence = result[0]
                alternative = None
                # A ligature can be read as a different short word (fi -> r).
                # A second engine gets pixels only. It cannot correct the
                # transcript and must agree with a unique full line token.
                if (confidence >= .99 and re.fullmatch(r"[A-Za-z][A-Za-z'-]{0,79}", word)
                        and not re.search(r'(?<!\w)'+re.escape(word)+r'(?!\w)', text)):
                    alternative = read_word(crop[:, a:b, ::-1].copy())
                    if alternative and len(re.findall(r'(?<!\w)'+re.escape(alternative['text'])+r'(?!\w)', text)) == 1:
                        word, confidence = alternative['text'], alternative['score']
                    else:
                        continue
                if (confidence < .99 or not re.fullmatch(r"[A-Za-z][A-Za-z'-]*[,.:;!?)\]]?", word)
                        or len(word) > 80) and alternative is None:
                    continue
                word, scores = (word, {}) if calibration else scored_scan_word(mask[:, a:b], gray_ink[:, a:b], word)
                observation = dict(text=word, score=float(confidence), scores=scores, bbox=[x+a, y, x+b, t])
                if alternative:
                    # Punctuation trimming changes the font quote only; the
                    # independent reader still attests the original token.
                    if word != alternative['text']:
                        continue
                    observation['recognition'] = alternative['recognition']
                words.append(observation)
                if calibration:
                    completed = completed_word_observation(rgb, model, observation)
                    if completed:
                        words.append(completed)
            selected = ([w for w in words if confirmed_word(w)
                         and (len(w['text']) >= 3 or numbered_short_span(text, w) is not None)
                         and re.fullmatch('[A-Za-z]+[.,;:!?]?', w['text'])
                         and anchored_word_span(text, w, words, require_style=False) is not None]
                        if calibration else supported_words(words, text))
            if selected and score >= .99:
                observed.append(dict(text=text, score=float(score), bbox=[x, y, z, t], words=selected))
            elif selected:
                # Math, quotation marks or another mixed-script token can
                # lower a whole line's confidence. A fresh local context
                # must independently pass; the original score is not raised.
                windows = dict.fromkeys([(0, mask.shape[1]), *context_windows(mask,
                    [(w['bbox'][0]-x, w['bbox'][2]-x) for w in selected]), *label_windows(mask)])
                for a, b in windows:
                    result = model.ocr(crop[:, a:b, ::-1].copy(), det=False)[0] or []
                    if len(result) != 1:
                        continue
                    context, confidence = result[0]
                    if (not .99 <= confidence <= 1 or not (enough_context(context)
                            or re.fullmatch(NUMBERED_PREFIX, context))):
                        continue
                    local = [w for w in selected if x+a <= w['bbox'][0] < w['bbox'][2] <= x+b]
                    if local:
                        observed.append(dict(text=context, score=float(confidence),
                            bbox=[x+a, y, x+b, t], words=local))
    return observed


def mixed_fragments(index, original, items, excluded=()):
    """Authorize context only beside unchanged math in one source group.

    A rewritten, excluded or unsupported sibling blocks the entire fallback.
    Mathematical bytes are retained as blockers, never candidates for styling.
    """
    result = {}
    for item in items:
        atom = index.atoms[index.ids[item['atom']]]
        siblings = [index.atoms[n] for n in index.groups[atom['group']]]
        if not any(a['kind'] == 'math' for a in siblings):
            continue
        for sibling in siblings:
            source = original.atoms[original.ids[sibling['original_atom']]] if sibling.get('original_atom') in original.ids else None
            if (sibling['kind'] not in {'text', 'math'} or sibling['id'] in excluded
                    or sibling.get('recovery_issue') or sibling.get('writing_direction')
                    or sibling.get('bbox') != atom['bbox'] or sibling['page'] != atom['page']
                    or not source or source['content'][sibling['original_start_offset']:sibling['original_end_offset']] != sibling['content']):
                break
        else:
            result[item['atom']] = [a['content'] for a in siblings if a['kind'] == 'math']
    return result


def edge_prose_contexts(source, observed, *, min_words=4):
    """Match a literal run cut only by source or observed line boundaries.

    A wrapped line can contain only a prefix/suffix of a long prose atom.
    Keep at least four complete words and sixteen letters, and never skip a
    conflicting word inside both strings. Repeated phrases stay ambiguous.
    """
    left, right = [list(re.finditer(r'\S+', s)) for s in (source, observed)]
    if len(left) > 400 or len(right) > 100:
        return []
    result = []
    for s, token in enumerate(left):
        for o, other in enumerate(right):
            if token.group() != other.group() or (s and o):
                continue
            length = 0
            while s+length < len(left) and o+length < len(right) and left[s+length].group() == right[o+length].group():
                length += 1
            if s+length != len(left) and o+length != len(right):
                continue
            a, z = token.start(), left[s+length-1].end()
            quote = source[a:z]
            sufficient = (enough_context(quote) if min_words == 4 else
                          length >= min_words and len(re.sub('[^A-Za-z]', '', quote)) >= 16)
            if (not sufficient
                    or literal_span(source, quote) != (a, z)
                    or literal_span(observed, quote) != (other.start(), right[o+length-1].end())):
                continue
            result.append(((a, z), (other.start(), right[o+length-1].end())))
    return result


def verified_short_edge_contexts(source, line):
    """A three-word line edge needs two touching, independently styled words.

    Called only after validating an entire unchanged mixed source group.
    Keep the sixteen-letter floor, literal boundary alignment and all font
    thresholds. Neither an isolated word nor formula text supplies context.
    """
    if line['score'] < .99:
        return []
    result = []
    for span, context in edge_prose_contexts(source, line['text'], min_words=3):
        if len(source[slice(*span)].split()) != 3:
            continue
        words = []
        for word in line['words']:
            located = anchored_word_span(line['text'], word, line['words'])
            if (confirmed_word(word) and located is not None
                    and context[0] <= located[0] < located[1] <= context[1]
                    and (is_italic(word['scores']) or is_bold(word['scores']))):
                words.append((word, located))
        for left, a in words:
            for right, b in words:
                if (a[1] < b[0] and line['text'][a[1]:b[0]].isspace()
                        and abs(left['bbox'][2]-right['bbox'][0]) <= 2
                        and (is_italic(left['scores']),is_bold(left['scores']))
                            == (is_italic(right['scores']),is_bold(right['scores']))):
                    result.append((span, context))
                    break
            else:
                continue
            break
    return result


def fragment_owner(line, nearby, mixed):
    """Use boundary-aligned multiword prose, without interpreting math.

    Verified complete-group contexts can join short prose across formulas.
    Other fragments need four words and sixteen letters of literal context;
    a matching mathematical token remains ineligible.
    """
    owners = []
    for item in nearby:
        contexts = [c for c in item.get('line_contexts', []) if c['line'] == line['text']]
        if contexts:
            owners.extend((item, c['source'], c['observed']) for c in contexts)
            continue
        if item['atom'] not in mixed:
            continue
        quote = item['text'].strip()
        if len(re.findall(r'[A-Za-z]+', quote)) < 4 or len(re.sub('[^A-Za-z]', '', quote)) < 16:
            continue
        span = literal_span(line['text'], quote)
        if span is None:
            owners.extend((item, source, observed) for source, observed in
                          edge_prose_contexts(item['text'], line['text']))
            continue
        a, z = span
        if ((a and line['text'][a-1].isalnum() and quote[0].isalnum())
                or (z < len(line['text']) and line['text'][z].isalnum() and quote[-1].isalnum())):
            continue
        owners.append((item, (0, len(item['text'])), span))
    return owners


def anchored_word_span(text, word, words, *, require_style=True):
    """Resolve repetition only with a uniquely read, touching neighbor.

    A repeated short word can have an upright occurrence elsewhere on the
    line. Both the geometric neighbor and their exact adjacent source phrase
    must agree. Conflicting anchors or repeated phrases remain ambiguous.
    During OCR collection and font calibration the neighbor need not have a
    style yet, but must independently contain at least three letters. Final
    ownership additionally requires a verified bold or italic neighbor.
    """
    token = r'(?<!\w)'+re.escape(word['text'])+r'(?!\w)'
    hits = [m.span() for m in re.finditer(token, text)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return None
    anchors = set()
    for neighbor in words:
        if neighbor is word or not confirmed_word(neighbor):
            continue
        if (require_style and not (is_italic(neighbor['scores']) or is_bold(neighbor['scores']))
                or not require_style and len(re.sub('[^A-Za-z]', '', neighbor['text'])) < 3):
            continue
        if abs(neighbor['bbox'][2]-word['bbox'][0]) <= 2:
            pattern = r'(?<!\w)'+re.escape(neighbor['text'])+r'\s+('+re.escape(word['text'])+r')(?!\w)'
        elif abs(word['bbox'][2]-neighbor['bbox'][0]) <= 2:
            pattern = r'(?<!\w)('+re.escape(word['text'])+r')\s+'+re.escape(neighbor['text'])+r'(?!\w)'
        else:
            continue
        matches = list(re.finditer(pattern, text))
        if len(matches) == 1:
            anchors.add(matches[0].span(1))
    return next(iter(anchors)) if len(anchors) == 1 else None


def owned_ranges(observed, items, dimensions, gray=(), mixed=None):
    """A full observed line and unique word must agree with one source atom."""
    result = {}
    for line in observed:
        box = [v/d for v, d in zip(line['bbox'], dimensions)]
        if line['score'] < .99 or any(intersection_fraction(box, r['bbox']) > .01 for r in gray):
            continue
        # Source OCR rounds normalized boxes to three decimals. Detection's
        # ink unclip and our 2px border can extend just beyond that box; exact
        # line bytes still decide ownership, never the nearest rectangle.
        nearby = [i for i in items
                  if intersection_fraction(box, [i['bbox'][0]-.003, i['bbox'][1]-.003,
                                                 i['bbox'][2]+.003, i['bbox'][3]+.003]) >= .9]
        owners = [(i, literal_span(i['text'], line['text'])) for i in nearby]
        owners = [(i, span) for i, span in owners if span is not None]
        fallback = not owners
        if fallback:
            owners = fragment_owner(line, nearby, mixed or {})
        if not owners:
            continue
        # Recompute adjacency support rather than trusting a cached flag.
        words = supported_words(line['words'], line['text'])
        for word in words:
            selected = owners
            if fallback:
                # Neither repeated line words nor matching math can acquire a
                # prose style merely because the paragraph rectangle agrees.
                token = r'(?<!\w)'+re.escape(word['text'])+r'(?!\w)'
                observed_span = anchored_word_span(line['text'], word, words)
                if observed_span is None:
                    continue
                a, z = observed_span
                selected = [(i, span) for i, span, context in owners
                            if context[0] <= a < z <= context[1]
                            and not any(re.search(token, math) for math in (mixed or {}).get(i['atom'], []))]
                # Duplicated overlapping text must remain ambiguous even if
                # only one duplicate was eligible for the context fallback.
                def contains_word(item):
                    contexts = [c for c in item.get('line_contexts', []) if c['line'] == line['text']]
                    text = ' '.join(item['text'][c['source'][0]:c['source'][1]] for c in contexts
                        if c['observed'][0] <= a < z <= c['observed'][1]) if contexts else item['text']
                    return bool(re.search(token, text))
                # A validated complete mixed group can distinguish two prose
                # atoms sharing a word through their independently matched
                # contexts. Overlapping atoms outside that group still veto.
                groups = {i.get('context_group') for i, _ in selected}
                if len(selected) == 1 and None not in groups:
                    group = next(iter(groups))
                    conflicts = [i for i in nearby if i.get('context_group') != group]
                    conflicts += [i for i, _ in selected]
                else:
                    conflicts = nearby
                if sum(contains_word(i) for i in conflicts) != 1:
                    continue
            if len(selected) != 1:
                continue
            item, (lo, hi) = selected[0]
            # Exact token boundaries prevent substring matches. Repetition
            # needs an independently observed, adjacent italic anchor.
            span = anchored_word_span(item['text'][lo:hi], word, words)
            if span is None:
                continue
            a, z = span; a += lo; z += lo
            # A complete observed prefix may be a substring of a longer
            # source word ("In"/"Inside"). The short-label path must own a
            # complete source token even in the full-line matching branch.
            if (numbered_short_span(line['text'], word) is not None
                    and numbered_short_span(item['text'], word) != (a, z)):
                continue
            # Adjacent shape support can authorize a borderline italic word;
            # bold always requires its individual full score and margin.
            bold = is_bold(word['scores'])
            row = dict(start=a, end=z, italic=is_italic(word['scores']) or not bold,
                       bold=bold, quote=item['text'][a:z],
                       observed_bbox=[v/d for v, d in zip(word['bbox'], dimensions)], scores=word['scores'])
            previous = result.get(item['atom'], [])
            overlapping = [r for r in previous if a < r['end'] and r['start'] < z]
            # Finer partitions can recognize both a word and its complete
            # punctuated form. Preserve the most complete independently
            # confirmed range, regardless of detector/partition order.
            # Partially intersecting observations do not authorize a union.
            if any(not (a <= r['start'] and r['end'] <= z) for r in overlapping):
                continue
            result[item['atom']] = [r for r in previous if r not in overlapping]+[row]
    return {atom: sorted(rows, key=lambda r: r['start']) for atom, rows in result.items()}


def caption_candidates(index, original, excluded=()):
    """Keep unchanged image/caption groups; their inherited box is not text.

    A caption often inherits only the illustration's rectangle. Validate the
    entire group, including math and the original image, before using nearby
    independently read prose to find its real line bounds. Never edit atoms.
    """
    from book_pipeline.structure.headings import valid_box
    pages = {}
    if index.source.get('translation') or original.source.get('translation'):
        return pages
    for positions in index.groups.values():
        atoms = [index.atoms[n] for n in positions]
        images = [a for a in atoms if a['kind'] == 'image']
        if len(images) != 1 or not valid_box(images[0].get('bbox')):
            continue
        image = images[0]
        sources = []
        for atom in atoms:
            source = original.atoms[original.ids[atom['original_atom']]] if atom.get('original_atom') in original.ids else None
            if (not source or atom['id'] in excluded or atom.get('recovery_issue')
                    or atom.get('writing_direction') or atom.get('source_style_runs')
                    or atom.get('extraction_method') != 'ocr' or atom['page'] != image['page']
                    or atom['kind'] != source['kind'] or atom['role'] != source['role']
                    or atom.get('bbox') != source.get('bbox')
                    or (atom is not image and (atom['kind'] not in {'text', 'math'}
                        or atom['role'] not in {'image_caption', 'chart_caption'}))
                    or (atom is image and (not atom.get('asset_sha256')
                        or atom['asset_sha256'] != source.get('asset_sha256')))
                    or (atom is not image and source['content'][atom['original_start_offset']:
                        atom['original_end_offset']] != atom['content'])):
                break
            sources.append(source)
        else:
            if len({a['group'] for a in sources}) != 1 or len({a['id'] for a in sources}) != len(atoms):
                continue
            if {a['id'] for a in sources} != {original.atoms[n]['id'] for n in original.groups[sources[0]['group']]}:
                continue
            prose = [dict(atom=a['id'], text=a['content']) if a['kind'] == 'text'
                     else dict(math=True) for a in atoms if a['kind'] != 'image']
            if not enough_context(' '.join(a.get('text', '') for a in prose)):
                continue
            for atom in atoms:
                if atom['kind'] == 'text' and atom['content'].strip():
                    pages.setdefault(atom['page'], []).append(dict(atom=atom['id'], text=atom['content'],
                        image_bbox=image['bbox'], prose_group=prose))
    return pages


def caption_line_contexts(text, group):
    """Align a whole observed line across opaque math, never reading the math.

    Literal prose must bracket each formula with at least two words on each
    side. At least one complete text atom and sufficient total prose are
    required. Partial tokens, ambiguous alignments and math-only lines cannot
    authorize styles. Returned offsets cover prose only, excluding formulas.
    """
    observed = list(re.finditer(r'\S+', text))
    source = []
    for atom in group:
        if atom.get('math'):
            source.append(None)
        else:
            source.extend((atom['atom'], m.start(), m.end(), m.group())
                          for m in re.finditer(r'\S+', atom['text']))
    if not 4 <= len(observed) <= 100 or len(source) > 200 or None not in source:
        return []
    answers = set()
    attempts = 0

    def walk(s, o, matches, gaps):
        nonlocal attempts
        attempts += 1
        if len(answers) > 1 or attempts > 4096:
            return
        if o == len(observed):
            if not gaps or not matches or matches[-1][1] != o-1:
                return
            # Each opaque formula needs independently matching prose on both
            # sides; a wildcard cannot swallow a second phrase or a label.
            indices = {n for n, _ in matches}
            if any(not {n-2, n-1, n+1, n+2} <= indices for n in gaps):
                return
            prose = ' '.join(source[n][3] for n, _ in matches)
            complete = any(all(n in indices for n, token in enumerate(source)
                               if token is not None and token[0] == atom['atom'])
                           for atom in group if not atom.get('math'))
            if complete and enough_context(prose):
                answers.add(tuple(matches))
            return
        if s == len(source):
            return
        token = source[s]
        if token is None:
            for end in range(o+1, min(len(observed), o+8)+1):
                walk(s+1, end, matches, [*gaps, s])
        elif token[3] == observed[o].group():
            walk(s+1, o+1, [*matches, (s, o)], gaps)

    for start, token in enumerate(source):
        if token is not None and token[3] == observed[0].group():
            walk(start, 0, [], [])
    if len(answers) != 1 or attempts > 4096:
        return []
    contexts = []
    for n, o in next(iter(answers)):
        atom, start, end, _ = source[n]
        if contexts and contexts[-1]['atom'] == atom:
            contexts[-1]['source'][1] = end
            contexts[-1]['observed'][1] = observed[o].end()
        else:
            contexts.append(dict(atom=atom, source=[start, end],
                observed=[observed[o].start(), observed[o].end()], line=text))
    return contexts


def wrapped_line_contexts(observed, group):
    """Join two adjacent OCR lines only for unchanged mixed-group alignment.

    Both lines retain the normal confidence and context floors. Aligned left
    edges, horizontal overlap and a bounded empty line gap prevent joining
    columns or skipping a line. Formula gaps still require two literal prose
    words on each side. Project only those literal offsets back to each line.
    """
    lines = [r for r in observed if r['score'] >= .99 and enough_context(r['text'])]
    if len(lines) > 500:
        return []
    result = []
    seen = set()
    for upper in lines:
        x, y, z, t = upper['bbox']
        for lower in lines:
            a, b, c, d = lower['bbox']
            height = max(t-y, d-b)
            if (height <= 0 or not 0 <= b-t <= .6*height
                    or abs(x-a) > .5*height
                    or min(z, c)-max(x, a) < .7*min(z-x, c-a)
                    or any(t <= row['bbox'][1] < row['bbox'][3] <= b for row in observed)):
                continue
            text = upper['text']+' '+lower['text']
            contexts = caption_line_contexts(text, group)
            for row, shift in ((upper, 0), (lower, len(upper['text'])+1)):
                for context in contexts:
                    lo = max(context['observed'][0], shift)
                    hi = min(context['observed'][1], shift+len(row['text']))
                    if lo >= hi:
                        continue
                    atom = next(a for a in group if a.get('atom') == context['atom'])
                    start, end = context['source']
                    source = literal_span(atom['text'][start:end], text[lo:hi])
                    if source is None:
                        continue
                    source = [start+source[0], start+source[1]]
                    key = (context['atom'], *source, lo-shift, hi-shift, row['text'])
                    if key not in seen:
                        seen.add(key)
                        result.append(dict(atom=context['atom'], source=source,
                            observed=[lo-shift, hi-shift], line=row['text']))
    return result


def numbered_prose_contexts(line, group):
    """Use an exact numbered row label to own short prose in that row.

    The caller has verified the complete unchanged mixed group. A label's
    letters, digits and delimiter must agree at both line starts; its styled
    word is read independently. Only that prefix and complete literal
    multiword prose atoms qualify. Formula bytes never supply a text anchor.
    """
    if not .99 <= line['score'] <= 1 or not group or group[0].get('math'):
        return []
    first = group[0]
    match = re.match(NUMBERED_LABEL, first['text'])
    if match is None:
        return []
    label = match.group()
    span = literal_span(line['text'], label)
    if (span != (0, len(label)) or literal_span(first['text'], label) != span
            or len(line['text']) > len(label) and not line['text'][len(label)].isspace()):
        return []
    words = supported_words(line['words'], line['text'])
    anchored = [(w, anchored_word_span(line['text'], w, words)) for w in words]
    if not any(s is not None and 0 <= s[0] < s[1] <= len(label)
               and len(re.sub('[^A-Za-z]', '', w['text'])) >= 4
               and (is_italic(w['scores']) or is_bold(w['scores'])) for w, s in anchored):
        return []
    prefix_end = len(label)
    for word, located in anchored:
        short = numbered_short_span(line['text'], word)
        if (short is not None and located == short
                and numbered_short_span(first['text'], word) == short):
            prefix_end = short[1]
    contexts = [dict(atom=first['atom'], source=[0, prefix_end],
                     observed=[0, prefix_end], line=line['text'])]
    for n, atom in enumerate(group[1:], 1):
        if not atom.get('math'):
            terminal = not any(a.get('math') or a['text'].strip() for a in group[n+1:])
            context = numbered_fragment_context(line, atom, anchored, len(label), terminal=terminal)
            if context is not None:
                contexts.append(context)
    return contexts


def numbered_fragment_context(line, atom, anchored, label_end, *, terminal):
    """Match one complete prose fragment after a verified numbered label."""
    tail = atom['text'].strip()
    tokens = list(re.finditer(r'[A-Za-z]+[.,;:!?]?', tail))
    # A single trailing token or mathematical variable cannot acquire style.
    if (len(tokens) < 2 or len(re.sub('[^A-Za-z]', '', tail)) < 8
            or re.sub(r'[A-Za-z]+[.,;:!?]?|\s+', '', tail)):
        return None
    source = literal_span(atom['text'], tail)
    observed = literal_span(line['text'], tail)
    if (source is None or observed is None or observed[0] <= label_end
            or terminal and observed[1] != len(line['text'].rstrip())
            or observed[1] < len(line['text']) and not line['text'][observed[1]].isspace()
            or not line['text'][observed[0]-1].isspace()):
        return None
    # Every tail word needs its own strong fit, with touching pixel strips
    # and one consistent style. The label does not authorize new glyph fits.
    candidates = []
    for token in tokens:
        target = (observed[0]+token.start(), observed[0]+token.end())
        found = [w for w, s in anchored if s == target
                 and (is_italic(w['scores']) or is_bold(w['scores']))]
        if not found:
            return None
        candidates.append(found)
    # Keep alternate detector strips separate; their rectangular coordinates
    # may differ, but at least one complete adjacent chain must agree.
    reachable = candidates[0]
    for choices in candidates[1:]:
        reachable = [w for w in choices if any(
            abs(previous['bbox'][2]-w['bbox'][0]) <= 2
            and previous['bbox'][1::2] == w['bbox'][1::2]
            and (is_italic(previous['scores']), is_bold(previous['scores']))
                == (is_italic(w['scores']), is_bold(w['scores'])) for previous in reachable)]
        if not reachable:
            return None
    return dict(atom=atom['atom'], source=list(source),
                observed=list(observed), line=line['text'])


def anchored_mixed_prose(observed, index, original, items, mixed):
    """Share literal line anchors only inside a complete unchanged source group.

    Formula gaps use the same two-sided prose checks as image captions.
    Wrapped prose keeps its four-word edge context. No source token supplies
    an OCR hypothesis, and competing groups never share ownership evidence.
    """
    prepared = [dict(item) for item in items]
    groups = {}
    for item in prepared:
        if item['atom'] in mixed:
            atom = index.atoms[index.ids[item['atom']]]
            groups.setdefault(atom['group'], []).append(item)
    for group, members in groups.items():
        atoms = [index.atoms[n] for n in index.groups[group]]
        source_ids = [a.get('original_atom') for a in atoms]
        if any(a not in original.ids for a in source_ids):
            continue
        sources = [original.atoms[original.ids[a]] for a in source_ids]
        # The original group must be preserved in full, in order. A removed,
        # split or moved sibling cannot license disambiguation of duplicates.
        if (len({a['group'] for a in sources}) != 1
                or source_ids != [original.atoms[n]['id'] for n in original.groups[sources[0]['group']]]
                or any(a['kind'] != b['kind'] or a['role'] != b['role']
                       or a['page'] != b['page'] or a.get('bbox') != b.get('bbox')
                       or a['content'] != b['content'] or a.get('original_start_offset') != 0
                       or a.get('original_end_offset') != len(b['content'])
                       or a.get('source_style_runs') for a, b in zip(atoms, sources))):
            continue
        prose = [dict(math=True) if a['kind'] == 'math' else dict(atom=a['id'],text=a['content'])
                 for a in atoms]
        wrapped = wrapped_line_contexts(observed, prose)
        numbered = [c for line in observed for c in numbered_prose_contexts(line, prose)]
        for item in members:
            contexts = []
            for line in observed:
                aligned = [c for c in caption_line_contexts(line['text'], prose)
                           if c['atom'] == item['atom']]
                edges = edge_prose_contexts(item['text'],line['text'])
                if not aligned and not edges:
                    edges = verified_short_edge_contexts(item['text'],line)
                contexts.extend(aligned or [dict(source=list(source), observed=list(span), line=line['text'])
                    for source, span in edges])
            for context in numbered:
                if context['atom'] == item['atom'] and not any(
                        c['line'] == context['line']
                        and c['observed'][0] < context['observed'][1]
                        and context['observed'][0] < c['observed'][1] for c in contexts):
                    contexts.append(context)
            for context in wrapped:
                if context['atom'] == item['atom']:
                    projected = {k: v for k, v in context.items() if k != 'atom'}
                    if not any(c['line'] == projected['line']
                               and c['observed'][0] < projected['observed'][1]
                               and projected['observed'][0] < c['observed'][1] for c in contexts):
                        contexts.append(projected)
            item.update(context_group=group, line_contexts=contexts)
    return prepared


def anchored_captions(observed, candidates, dimensions):
    """Locate caption prose below its own image using literal OCR lines.

    A bounded, centered strip prevents internal labels or a nearby column
    from becoming a caption. Geometry alone never authorizes text or style;
    full-line agreement, independent word OCR and font scores still apply.
    Mixed lines additionally require unique, ordered literal prose on both
    sides of opaque math; formulas themselves never receive prose styles.
    """
    result = []
    for item in candidates:
        image = item['image_bbox']
        boxes = []
        contexts = []
        for line in observed:
            if line['score'] < .99 or not enough_context(line['text']):
                continue
            x, y, z, t = [v/d for v, d in zip(line['bbox'], dimensions)]
            if (image[3]-.003 <= y < t <= min(1., image[3]+.06) and t-y <= .035
                    and image[0]-.06 <= x < z <= image[2]+.06
                    and image[0] <= (x+z)/2 <= image[2]):
                matching = [c for c in caption_line_contexts(line['text'], item.get('prose_group', []))
                            if c['atom'] == item['atom']]
                if literal_span(item['text'], line['text']) is not None or matching:
                    boxes.append([x, y, z, t])
                    contexts.extend(c for c in matching if c not in contexts)
        if boxes:
            result.append(dict(atom=item['atom'], text=item['text'],
                bbox=[min(b[0] for b in boxes), min(b[1] for b in boxes),
                      max(b[2] for b in boxes), max(b[3] for b in boxes)],
                line_contexts=contexts))
    return result


def competing_times_observations(rgb, observed):
    """Retain independent Times fits unless another family contradicts them.

    Sparse Latin text on a Chinese page cannot establish a page-wide font.
    It can still supply the existing word-level evidence. Compare both pixel
    views against Computer Modern too; a strong opposing axis vetoes the word.
    This fallback never grants an axis absent from the original Times fit.
    """
    import cv2
    from book_pipeline.publishing.scan_font_calibration import (shape_scores, metric_scores,
        metric_style_scores, MIN_SLANT_SCORE, MIN_SLANT_MARGIN)
    result = []
    for line in observed:
        words = []
        for word in line['words']:
            x, y, z, t = word['bbox']
            if (not confirmed_word(word) or not all(type(v) is int for v in (x,y,z,t))
                    or not (0 <= x < z <= rgb.shape[1] and 0 <= y < t <= rgb.shape[0])):
                continue
            gray = 255-cv2.cvtColor(rgb[y:t,x:z], cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            competing = [shape_scores(ink,word['text']).get('computer_modern') for ink in (binary,gray)]
            if any(not scores for scores in competing):
                continue
            italic, bold = is_italic(word['scores']), is_bold(word['scores'])
            # Shape fits stretch each font to the observed rectangle. Ink
            # spread can then masquerade as bold. Require natural advances in
            # both pixel views for weight, including the Times fallback.
            if bold:
                metric = metric_style_scores(metric_scores(binary, word['text'], 'times'),
                                             metric_scores(gray, word['text'], 'times'))
                if not is_bold(metric):
                    if not italic:
                        continue
                    scores = word['scores']
                    regular = max(scores['tiro'], scores['tibo'])
                    slanted = max(scores['tiit'], scores['tibi'])
                    word = dict(word, scores=dict(zip(FONTS,(regular,regular,slanted,slanted))))
                    bold = False
            def contradicts(scores):
                for enabled, positive, negative in ((italic,(2,3),(0,1)),(bold,(1,3),(0,2))):
                    if enabled and max(scores[k] for k in negative) >= MIN_SLANT_SCORE and (
                            max(scores[k] for k in negative)-max(scores[k] for k in positive) >= MIN_SLANT_MARGIN):
                        return True
                return False
            if not any(contradicts(scores) for scores in competing):
                words.append(word)
        if words:
            result.append(dict(line,words=words))
    return result


def recover_body_styles(index, original, directory, excluded=()):
    import numpy as np
    import pymupdf
    from book_pipeline.publishing.image_search import local_ocr
    from book_pipeline.publishing.raster_decorations import gray_regions, page_pixels
    from book_pipeline.publishing.raster_styles import eligible_items, METHOD as STYLE_METHOD

    source = index.source.get('source_pdf')
    candidates = eligible_items(index, original, excluded)
    captions = caption_candidates(index, original, excluded)
    for number in captions:
        candidates.setdefault(number, [])
    report = dict(method=METHOD, status='not_needed', records=[])
    if not candidates or not source or not Path(source).is_file():
        return {}, report
    runtime = local_ocr()
    if runtime is None:
        report['status'] = 'unavailable_local_ocr'
        return {}, report
    model, signature = runtime
    source_sha = digest(source)
    font_signature = fingerprint([digest_bytes(pymupdf.Font(name).buffer) for name in FONTS])
    result = {}
    with pymupdf.open(source) as pdf:
        for number, items in candidates.items():
            from book_pipeline.ingestion.scan_panels import raster_page
            if not 0 <= number < len(pdf) or not raster_page(pdf[number]):
                continue
            page = pdf[number]
            # A hidden OCR layer says nothing about the printed font. Removing
            # it must not switch identical scan pixels to legacy Times fits.
            from book_pipeline.publishing.scan_font_calibration import runtime_signature as scan_signature, calibrate_observations
            calibration_signature = scan_signature()
            if calibration_signature is None:
                report['status'] = 'requires_scan_font_calibration'
                report['records'].append(dict(page=number+1, status='abstained_missing_scan_fonts'))
                continue
            pix = page.get_pixmap(dpi=DPI, alpha=False, colorspace=pymupdf.csRGB)
            inputs = dict(method=OBSERVATION_METHOD, pdf=source_sha, page=number, dpi=DPI,
                ocr=signature, fonts=font_signature, pymupdf=version('PyMuPDF'),
                latin_word_reader=runtime_signature(),
                score=MIN_SCORE, margin=MIN_MARGIN, supported_score=MIN_SUPPORTED_SCORE,
                bold_score=MIN_BOLD_SCORE)
            calibrated_inputs = dict(inputs, method=CALIBRATION_OBSERVATION_METHOD,
                archive_collection='independent_all_words_v1', calibration=calibration_signature)
            key = fingerprint(calibrated_inputs)[:24]
            cache = Path(directory)/'body-style-ocr'/(key+'.json')
            if cache.is_file():
                observed = read(cache)
            else:
                rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
                observed = analyze_page(rgb, model, calibration=True)
                save(cache, observed)
            rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            observed, calibration_report = calibrate_observations(rgb, observed, metric_families={'computer_modern'})
            if calibration_report.get('family') != 'computer_modern':
                # Keep the established Times word/line collector, including
                # short adjacent words. Collection signatures are unchanged;
                # the new cross-family veto is re-evaluated from source ink.
                times_cache = Path(directory)/'body-style-ocr'/(fingerprint(inputs)[:24]+'.json')
                if times_cache.is_file():
                    times_observed = read(times_cache)
                else:
                    times_observed = analyze_page(rgb, model)
                    save(times_cache, times_observed)
                observed = competing_times_observations(rgb,times_observed)
                calibration_report['fallback'] = 'independent_times_with_cross_family_veto'
            report['status'] = ('requires_scan_weight_calibration'
                if calibration_report['status'] == 'calibrated_slant_only' else
                'requires_visual_review' if calibration_report['status'] in {'calibrated_metric_typography','calibrated_font_family'} else
                'requires_scan_font_calibration')
            caption_items = anchored_captions(observed, captions.get(number, []), [pix.width, pix.height]*2)
            mixed = mixed_fragments(index, original, items, excluded)
            items = anchored_mixed_prose(observed, index, original, items, mixed)
            ranges = owned_ranges(observed, [*items, *caption_items], [pix.width, pix.height]*2,
                                  gray_regions(page_pixels(page)), mixed)
            result.update({atom: [dict(method=STYLE_METHOD, **r) for r in runs] for atom, runs in ranges.items()})
            report['records'].append(dict(page=number+1, cache=str(cache), ranges=ranges,
                                          caption_anchors=caption_items,
                                          font_calibration=calibration_report))
    if result:
        report['status'] = 'requires_visual_review'
    return result, report


def digest_bytes(value):
    from hashlib import sha256
    return sha256(value).hexdigest()

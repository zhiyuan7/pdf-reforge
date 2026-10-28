"""Report conflicting scan reference digits without choosing a transcription.

An image-only CPU reader and a unique ordinary word locate the reason. Two
scales must witness a different digit sequence. Even then this is a quality
issue, never permission to change digits or infer a mathematical reference.
"""
import csv
from io import BytesIO, StringIO
from pathlib import Path
import re
import subprocess

from book_pipeline.config import digest

METHOD = 'cpu_scan_reference_digit_disagreement_v1'
MIN_CONFIDENCE = .80
REFERENCE = re.compile(r'\((\d{1,4})([.·-])(\d{1,4})\)[.,;:]?')


def parse_reference(text, score):
    match = REFERENCE.fullmatch(text)
    if match is None or not MIN_CONFIDENCE <= score <= 1:
        return None
    return dict(text=text, digits=[match[1], match[3]], separator=match[2], score=score)


def observe_reasons(rows, directory):
    """Read physical right-side reasons without any proposed reference."""
    from PIL import Image, ImageOps
    from book_pipeline.publishing.latin_word_ocr import runtime_signature
    signature = runtime_signature()
    if signature is None:
        return []
    directory = Path(directory)
    observations = []
    for row in rows:
        words = row['words']
        candidates = [n for n, w in enumerate(words) if w['text'] == 'by'
                      and MIN_CONFIDENCE <= w['score'] <= 1
                      and w['bbox'][0] > .65*row['pixel_bbox'][2]]
        if len(candidates) != 1 or candidates[0] != len(words)-2:
            continue
        by, suffix = words[-2:]
        initial = parse_reference(suffix['text'], suffix['score'])
        path = Path(row['image'])
        if initial is None or not path.is_file() or digest(path) != row['image_sha256']:
            continue
        readings = [{**initial, 'scale': 1, 'image': str(path), 'image_sha256': digest(path)}]
        with Image.open(path) as original:
            x, y, w, h = suffix['bbox']
            left = by['bbox'][0]-4
            if not (0 <= left < x < x+w+4 <= original.width and 0 <= y < y+h <= original.height):
                continue
            crop = original.crop((left, 0, x+w+4, original.height))
            for scale in (2, 3):
                view = ImageOps.expand(crop.resize((crop.width*scale, crop.height*scale)),
                                       border=12*scale, fill='white')
                stream = BytesIO()
                view.save(stream, format='PNG')
                try:
                    output = subprocess.run(['tesseract', 'stdin', 'stdout', '-l', 'eng', '--psm', '7', 'tsv'],
                        input=stream.getvalue(), capture_output=True, timeout=10, check=True)
                    tokens = [t for t in csv.DictReader(StringIO(output.stdout.decode()), delimiter='\t')
                              if t['level'] == '5']
                    if (len(tokens) != 2 or tokens[0]['text'] != 'by'
                            or not MIN_CONFIDENCE <= float(tokens[0]['conf'])/100 <= 1):
                        continue
                    observed = parse_reference(tokens[1]['text'], float(tokens[1]['conf'])/100)
                except (OSError, subprocess.SubprocessError, UnicodeError, KeyError, ValueError, csv.Error):
                    continue
                if observed is not None:
                    image = directory/f"row-{row['id']:03d}-s{scale}.png"
                    directory.mkdir(parents=True, exist_ok=True)
                    view.save(image)
                    readings.append({**observed, 'scale': scale, 'image': str(image), 'image_sha256': digest(image)})
        if len(readings) >= 2:
            observations.append(dict(row=row['id'], readings=readings, reader_signature=signature))
    return observations


def reference_conflicts(answer, rows, observations):
    """A unique literal prose anchor owns a conflict; mathematics cannot."""
    from book_pipeline.ingestion.scan_reference_punctuation import prose_words
    words = [prose_words(item) for item in answer['items']]
    row_words = [{w['text'].strip('.,;:()').lower() for w in row['words']
                  if .95 <= w['score'] <= 1 and re.fullmatch(r'[A-Za-z]{4,}[.,;:]?', w['text'])}
                 for row in rows]
    conflicts = []
    for n, item in enumerate(answer['items']):
        if item['kind'] != 'formula':
            continue
        matches = [m for span in re.finditer(r'\\text\{[^{}]*\}', item['latex'])
                   for m in re.finditer(r'\bby\s+\((\d{1,4})\.(\d{1,4})\)', span[0])]
        if len(matches) != 1:
            continue
        digits = [matches[0][1], matches[0][2]]
        owners = []
        for obs in observations:
            indexes = [k for k, row in enumerate(rows) if row['id'] == obs['row']]
            if len(indexes) != 1 or not obs.get('reader_signature'):
                continue
            k = indexes[0]
            anchors = {w for w in words[n] & row_words[k]
                       if sum(w in ws for ws in words) == 1 and sum(w in ws for ws in row_words) == 1}
            if anchors:
                owners.append((obs, anchors))
        if len(owners) != 1:
            continue
        observation, anchors = owners[0]
        readings = observation['readings']
        supported = []
        for reading in readings:
            parsed = parse_reference(reading['text'], reading['score'])
            if parsed is None or parsed['digits'] == digits:
                continue
            scales = {r['scale'] for r in readings if parse_reference(r['text'], r['score'])
                      and parse_reference(r['text'], r['score'])['digits'] == parsed['digits']}
            if len(scales & {1, 2, 3}) >= 2 and parsed['digits'] not in supported:
                supported.append(parsed['digits'])
        if supported:
            conflicts.append(dict(item=n, row=observation['row'], reference='.'.join(digits),
                observed_digits=supported, anchors=sorted(anchors), method=METHOD,
                reason='scan_numeric_reference_disagreement', observation=observation))
    return conflicts

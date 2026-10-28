"""Optional, image-only second reader for isolated Latin print ligatures.

Neither the source transcript nor a proposed word is supplied to the engine.
Two segmentation modes must read the complete crop identically at >=95.
"""
import csv
from functools import lru_cache
from io import BytesIO, StringIO
from pathlib import Path
import re
import shutil
import subprocess

from book_pipeline.config import digest, fingerprint

METHOD = 'dual_tesseract_latin_word_v1'
MIN_CONFIDENCE = .95
WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,79}[,.:;!?)\]]?")


@lru_cache(maxsize=1)
def runtime_signature():
    executable = shutil.which('tesseract')
    if not executable:
        return None
    try:
        version = subprocess.run([executable, '--version'], capture_output=True, timeout=5, check=True).stdout.decode()
        listing = subprocess.run([executable, '--list-langs'], capture_output=True, timeout=5, check=True).stdout.decode()
        match = re.search(r'"([^"\n]+)"', listing)
        trained = Path(match[1])/'eng.traineddata' if match else None
        if trained is None or not trained.is_file():
            return None
        return fingerprint(dict(method=METHOD, version=version, model=digest(trained),
                                modes=[7, 8], min_confidence=MIN_CONFIDENCE))
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def single_word(tsv):
    """Do not merge separate tokens or hide low-confidence output."""
    try:
        rows = [r for r in csv.DictReader(StringIO(tsv), delimiter='\t') if r['level'] == '5']
        if len(rows) != 1 or not WORD.fullmatch(rows[0]['text']):
            return None
        score = float(rows[0]['conf'])/100
        if not MIN_CONFIDENCE <= score <= 1:
            return None
        return rows[0]['text'], score
    except (KeyError, TypeError, ValueError, csv.Error):
        return None


def read_word(bgr):
    signature = runtime_signature()
    if signature is None:
        return None
    from PIL import Image
    stream = BytesIO()
    Image.fromarray(bgr[:, :, ::-1]).save(stream, format='PNG')
    readings = []
    try:
        for mode in (7, 8):
            output = subprocess.run(['tesseract', 'stdin', 'stdout', '-l', 'eng', '--psm', str(mode), 'tsv'],
                                    input=stream.getvalue(), capture_output=True, timeout=5, check=True)
            result = single_word(output.stdout.decode())
            if result is None:
                return None
            readings.append(result)
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    if readings[0][0] != readings[1][0]:
        return None
    return dict(text=readings[0][0], score=min(r[1] for r in readings),
                recognition=dict(method=METHOD, signature=signature,
                                 texts=[r[0] for r in readings], scores=[r[1] for r in readings]))


def confirmed_word(word):
    """Retain each engine's confidence scale and explicit reading provenance."""
    evidence = word.get('recognition')
    if evidence is None:
        return .99 <= word['score'] <= 1
    return (evidence.get('method') == METHOD and bool(evidence.get('signature'))
            and evidence.get('texts') == [word['text']]*2
            and len(evidence.get('scores', [])) == 2
            and all(MIN_CONFIDENCE <= s <= 1 for s in evidence['scores'])
            and word['score'] == min(evidence['scores']))

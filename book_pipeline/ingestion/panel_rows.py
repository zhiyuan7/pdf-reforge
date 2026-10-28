"""Bound a terminal display by independently read prose above it."""
from collections import defaultdict
from copy import deepcopy
import csv
from io import StringIO
from pathlib import Path
import re
import subprocess

from book_pipeline.config import digest

METHOD = 'cpu_panel_prose_anchors_terminal_display_v1'


def tokens(text):
    return re.findall(r'[a-z0-9]+(?:\.[0-9]+)*', text.lower())


def anchor_rows(lines, observations):
    """Match literal leading prose, never mathematical guesses or row numbers alone."""
    result = []
    for parts in lines[:-1]:
        prefix = ''
        for part in parts:
            if part['atom']['kind'] != 'text':
                break
            prefix += part['content']
        expected = tokens(prefix)[:3]
        if len(expected) < 2 or not any(any(c.isalpha() for c in t) for t in expected):
            return None
        found = []
        for row in observations:
            words = row['words']
            if len(words) < len(expected):
                continue
            reading = tokens(' '.join(w['text'] for w in words[:len(expected)]))
            if reading[:len(expected)] == expected and min(w['score'] for w in words[:len(expected)]) >= .7:
                found.append(row)
        if len(found) != 1:
            return None
        result.append(found[0])
    if any(a['bbox'][3] >= b['bbox'][1] for a, b in zip(result, result[1:])):
        return None
    return result


def terminal_display_bounds(page, panel, lines, answer, directory):
    """Reject guessed geometry when a monochrome definition ends in a display.

    The final formula receives all pixels below the last witnessed prose row,
    including unattached upper limits. No formula character is changed here.
    """
    if (panel['title_bbox'] or answer['classification'] != 'prose_panel'
            or [r['kind'] for r in answer['lines']] != ['text']*(len(lines)-1)+['display']):
        return answer, None
    from book_pipeline.publishing.latin_word_ocr import runtime_signature
    runtime = runtime_signature()
    evidence = dict(method=METHOD, cpu=runtime, status='unverified')
    if runtime is None:
        return dict(classification='uncertain', reason='CPU panel row reader unavailable', lines=[]), evidence
    import pymupdf
    rect = pymupdf.Rect(*(v*(page.rect.width if k%2==0 else page.rect.height) for k,v in enumerate(panel['bbox'])))
    rect += (rect.width*.012, rect.height*.025, -rect.width*.012, -rect.height*.025)
    image = Path(directory)/'prose-row-witness.png';image.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(dpi=300, clip=rect, alpha=False)
    pix.save(image)
    try:
        output = subprocess.run(['tesseract',str(image),'stdout','-l','eng','--psm','6','tsv'],
                                capture_output=True, check=True, timeout=20)
        grouped = defaultdict(list)
        for word in csv.DictReader(StringIO(output.stdout.decode()),delimiter='\t'):
            if word['level'] == '5' and word['text'].strip():
                x,y,w,h = [int(word[k]) for k in ('left','top','width','height')]
                grouped[word['block_num'],word['par_num'],word['line_num']].append(
                    dict(text=word['text'],score=float(word['conf'])/100,bbox=[x,y,x+w,y+h]))
        observed = [dict(words=words,bbox=[min(w['bbox'][0] for w in words),min(w['bbox'][1] for w in words),
                                          max(w['bbox'][2] for w in words),max(w['bbox'][3] for w in words)])
                    for words in grouped.values()]
    except (OSError,subprocess.SubprocessError,ValueError,KeyError,UnicodeError,csv.Error):
        observed = []
    anchors = anchor_rows(lines,observed)
    evidence.update(image=str(image),image_sha256=digest(image),rows=observed)
    if anchors is None:
        return dict(classification='uncertain', reason='Prose row ownership not independently confirmed', lines=[]), evidence
    # Pixmap origins include the integer rounding of the clipped PDF rectangle.
    def ycoord(y): return (pix.y+y)*72/300/page.rect.height
    x0,x1 = rect.x0/page.rect.width,rect.x1/page.rect.width
    changed=deepcopy(answer)
    for n, row in enumerate(changed['lines'][:-1]):
        bottom = anchors[n+1]['bbox'][1]-1 if n+1 < len(anchors) else anchors[n]['bbox'][3]+1
        row['bbox']=[x0,ycoord(anchors[n]['bbox'][1]-1),x1,ycoord(bottom)]
    changed['lines'][-1]['bbox']=[x0,ycoord(anchors[-1]['bbox'][3]+2),x1,rect.y1/page.rect.height]
    evidence.update(status='bounded_by_cpu_prose',original_rows=answer['lines'],corrected_rows=changed['lines'])
    return changed,evidence

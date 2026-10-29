"""Recover native prose fonts only with complete, unambiguous text ownership.

PDF flags omit italic/bold for some Computer Modern Type1 subsets. Font names
and live spans supply the evidence; math wildcards never acquire prose styles.
"""
from pathlib import Path
import re

from book_pipeline.ingestion.native import spans_for
from book_pipeline.ingestion.panels import intersection_fraction
from book_pipeline.publishing.native_lateral import native_groups
from book_pipeline.publishing.raster_styles import METHOD as STYLE_METHOD

METHOD = 'native_complete_group_prose_fonts_v6'
LIGATURES = str.maketrans({'ﬀ':'ff','ﬁ':'fi','ﬂ':'fl','ﬃ':'ffi','ﬄ':'ffl'})
CJK_PUNCTUATION = str.maketrans('，。．：；（）', ',..:;()')


def prose_letters(text):
    return ''.join(text.split()).translate(LIGATURES)


def font_style(name):
    name = re.sub(r'^[A-Z]{6}\+', '', name)
    # Mathematical alphabets must not become prose emphasis.
    if re.match(r'(?:CM|LM)(?:MI|SY|EX)', name, re.I):
        return False, False
    italic = bool(re.search(r'Italic|Oblique|(?:CM|LM)(?:TI|SL|BXTI|BXSL)\d', name, re.I))
    bold = bool(re.search(r'Bold|(?:CM|LM)(?:BX|SSBX|BXTI|BXSL)\d', name, re.I))
    # Older URW PostScript families use ReguItal/Medi rather than the
    # modern Italic/Bold suffixes. Restrict those aliases to their families.
    if re.fullmatch(r'Nimbus(?:RomNo9L|SansL|MonL)-(?:Regu|Medi)(?:Ital)?',name):
        italic = name.endswith('Ital')
        bold = '-Medi' in name
    if re.fullmatch(r'CenturySchL-(?:Roma|Ital|Bold|BoldItal)',name):
        italic = name.endswith('Ital')
        bold = 'Bold' in name
    return italic, bold


def matched_group_ranges(atoms, spans):
    """Both extreme wildcard alignments must give every prose byte its owner."""
    from book_pipeline.publishing.native_note_styles import font_family
    cjk = any(a['kind']=='text' and re.search(r'[\u3400-\u9fff]',a['content']) for a in atoms)
    # Modern Chinese PDF fonts commonly encode fullwidth punctuation while
    # OCR emits its narrow equivalent. Only one-codepoint punctuation aliases
    # may align; Chinese characters and mathematical wildcards stay unchanged.
    key = lambda t: prose_letters(t).translate(CJK_PUNCTUATION) if cjk else prose_letters(t)
    observed, fonts = '', []
    for span in spans:
        text = key(span['text'])
        observed += text
        italic,bold = font_style(span['font'])
        for char in text:
            chinese = bool(re.fullmatch(r'[\u3400-\u9fff]',char))
            family = 'calligraphic' if chinese and font_family(span['font'])=='calligraphic' else None
            fonts.append((False if chinese else italic,bold,family))
    pieces, owners = [], []
    for atom in atoms:
        if atom['kind'] == 'text':
            offsets = [n for n,c in enumerate(atom['content']) if not c.isspace()
                       for _ in c.translate(LIGATURES)]
            if not offsets:
                continue
            text = key(atom['content'])
            pieces.append('('+re.escape(text)+')')
            owners.append((atom, offsets))
        else:
            # Adjacent mathematics forms a single ownership gap.
            if not pieces or pieces[-1] != '.+?':
                pieces.append('.+?')
    pattern = ''.join(pieces)
    if not owners or not observed:
        return None
    first = re.fullmatch(pattern, observed)
    last = re.fullmatch(pattern.replace('.+?', '.+'), observed)
    if not first or not last or first.regs != last.regs:
        return None
    result = {}
    for k,(atom,offsets) in enumerate(owners, 1):
        a,z = first.span(k)
        runs = []
        for offset,style in zip(offsets, fonts[a:z]):
            if not any(style):
                continue
            previous = runs[-1] if runs else None
            if previous and previous['end']==offset+1:
                continue  # Multiple source letters can own one ligature byte.
            previous_style = (previous['italic'],previous['bold'],previous.get('cjk_family')) if previous else None
            if (previous and previous_style == style and not style[2]
                    and atom['content'][previous['end']:offset].isspace()):
                previous.update(end=offset+1)
            elif previous and previous['end'] == offset and previous_style == style:
                previous.update(end=offset+1)
            else:
                runs.append(dict(method=STYLE_METHOD,evidence=METHOD,start=offset,end=offset+1,
                                 italic=style[0],bold=style[1]))
                if style[2]:runs[-1]['cjk_family']=style[2]
        for run in runs:
            run['quote'] = atom['content'][run['start']:run['end']]
        if runs:
            result[atom['id']] = runs
    return result


def group_ranges(atoms, spans):
    return matched_group_ranges(atoms, spans) or {}


def flat_math_key(text):
    """Compare only flat native expressions; scripts and fractions abstain."""
    symbols={'lor':'∨','vee':'∨','land':'∧','wedge':'∧','sim':'∼',
             'Leftrightarrow':'⇔','Rightarrow':'⇒','Leftarrow':'⇐'}
    text=re.sub(r'\\([A-Za-z]+)',lambda m:symbols.get(m[1],'' if m[1] in
        {'boldsymbol','mathbf','mathrm','text','bf'} else m[0]),text)
    text=re.sub(r'[\s{}]','',text)
    return text if re.fullmatch(r'[A-Za-z0-9(),.;:=+*/<>∧∨∼⇔⇒⇐−-]+',text) else None


def native_math_evidence(atoms, spans):
    """Recover quoted prose and simple math weight from unique live owners.

    Only complete literal prose anchors may align the wildcards. A quotation
    must consist of ordinary text-font glyphs; it cannot consume math symbols,
    scripts or a neighboring formula. No mathematical expression is inferred.
    """
    observed, owners = '', []
    for span in spans:
        value = prose_letters(span['text'])
        observed += value
        owners.extend([span]*len(value))
    pieces, maths = [], []
    for atom in atoms:
        if atom['kind']=='text':pieces.append(re.escape(prose_letters(atom['content'])))
        elif atom['kind']=='math':
            pieces.append('(.+?)');maths.append(atom)
        else:return {},{}
    pattern=''.join(pieces)
    first=re.fullmatch(pattern,observed)
    last=re.fullmatch(pattern.replace('(.+?)','(.+)'),observed)
    if not maths or not first or not last or first.regs!=last.regs:return {},{}
    literals,weights={},{}
    for n,atom in enumerate(maths,1):
        a,z=first.span(n);value=observed[a:z];live=owners[a:z]
        if (re.fullmatch('[“‘][A-Za-z0-9]{1,4}[”’]',value)
                and len({id(s) for s in live})==1
                and all(not any(font_style(s['font']))
                        and re.search(r'(?:Regular|Roman|Roma|CMR\d|LMRoman)',s['font'],re.I)
                        and not re.search('Math',s['font'],re.I) for s in live)):
            literals[atom['id']]=dict(text=value,method='complete_live_quoted_prose',source_fonts=sorted({s['font'] for s in live}))
        key=flat_math_key(atom['content'])
        if (not key or key.rstrip('.,:;')!=value.rstrip('.,:;')
                or len({s.get('line',0) for s in live})!=1):continue
        if all(s.get('size') and s.get('bbox') for s in live):
            sizes=[s['size'] for s in live]
            centers=[(s['bbox'][1]+s['bbox'][3])/2 for s in live]
            heights=[s['bbox'][3]-s['bbox'][1] for s in live]
            if (min(sizes)<.85*max(sizes)
                    or max(centers)-min(centers)>.4*min(heights)):continue
        rendered=atom['content']
        # Only terminal punctuation may differ from the flat native expression.
        tail=re.search(r'[.,:;]*$',value)[0]
        raw_tail=re.search(r'[.,:;]*$',key)[0]
        if raw_tail!=tail:
            # Punctuation inside a TeX group cannot be safely removed with a
            # suffix edit. Keep that expression unchanged instead of doubling
            # or moving the glyph outside its original owner.
            if raw_tail:
                suffix=re.escape(raw_tail)+r'\s*$'
                if not re.search(suffix,rendered):continue
                rendered=re.sub(suffix,'',rendered)
            rendered=rendered.rstrip()+tail
        match=re.fullmatch(r'\s*\\boldsymbol\s*\{\s*([A-Za-z])\s*\}[.,:;]?\s*',rendered)
        if match and all(font_style(s['font'])==(True,False) for s in live[:1]):
            rendered=match[1]+tail
        # Explicitly observed roman words inside a flat inline formula remain
        # words, rather than spaced italic variables or OCR-invented bold text.
        roman=''.join(c if font_style(s['font'])==(False,False) and
            re.search(r'(?:Regular|Roman|Roma|CMR\d|LMRoman)',s['font'],re.I)
            and not re.search('Math',s['font'],re.I) else ' ' for c,s in zip(value,live))
        for word in re.finditer(r'[A-Za-z]{2,}',roman):
            letters=r'\s*'.join(word[0])
            pattern=(r'(?P<bold>\\mathbf\s*\{\s*'+letters+r'\s*\})'
                     r'|\\[A-Za-z]+\s*\{[^{}]*\}'
                     r'|(?P<plain>(?<![A-Za-z\\])'+re.escape(word[0])+r'(?![A-Za-z]))')
            rendered=re.sub(pattern,lambda m:r'\text{ '+word[0]+' }'
                if m.group('bold') or m.group('plain') else m[0],rendered)
        if rendered!=atom['content']:
            weights[atom['id']]=dict(latex=rendered,method='complete_live_flat_math_presentation',
                                     source_fonts=sorted({s['font'] for s in live}))
    return literals,weights


def bounded_line_ranges(atoms, spans):
    """Recover ownership inside a broad formula-repair band, on whole lines.

    Every possible complete line window participates, including ordinary font
    matches. Repeated text or competing math ownership therefore abstains.
    """
    if not atoms or atoms[0]['kind']!='text':return {}
    prefix=prose_letters(atoms[0]['content'])
    anchor_size=6 if re.fullmatch(r'[\u3400-\u9fff]{6,}',prefix) else 12
    if len(prefix)<anchor_size:return {}
    lines=[]
    for span in spans:
        if not lines or lines[-1][0]['line']!=span['line']:lines.append([])
        lines[-1].append(span)
    if len(lines)>80:return {}
    matches=[]
    for start,line in enumerate(lines):
        if not prefix.startswith(''.join(prose_letters(s['text']) for s in line)[:anchor_size]):continue
        window=[]
        for end in range(start,len(lines)):
            window.extend(lines[end])
            ranges=matched_group_ranges(atoms,window)
            if ranges is not None:matches.append(ranges)
    return matches[0] if len(matches)==1 else {}


def recover_native_body_styles(index, original, excluded=(), *, corrections=()):
    import pymupdf
    styles = {}
    report = dict(method=METHOD,status='not_needed',records=[])
    source = index.source.get('source_pdf')
    if (index.source.get('translation') or original.source.get('translation')
            or not source or not Path(source).is_file()):
        return styles, report
    with pymupdf.open(source) as pdf:
        pages = {}
        for group in native_groups(index, allow_ocr=True, split_pages=True):
            atoms = group['atoms']
            if (not 0 <= group['page'] < len(pdf)
                    or any(a.get('recovery_issue')
                           or a['kind'] not in {'text','math'} or a.get('writing_direction')
                           or a.get('block_type') not in {'text','paragraph','page_footnote',
                               'image_caption','chart_caption','table_caption'} for a in atoms)):
                continue
            unchanged = True
            for atom in atoms:
                old = original.atoms[original.ids[atom['original_atom']]] if atom.get('original_atom') in original.ids else None
                if (not old or old['content'][atom['original_start_offset']:atom['original_end_offset']] != atom['content']):
                    unchanged = False
                    break
            if not unchanged:
                continue
            page = pdf[group['page']]
            # A hidden text layer on a scan is not independent font evidence.
            # Born-digital pages routed through OCR for a table can still use
            # exact live paragraph ownership, without trusting OCR typography.
            if sum((pymupdf.Rect(i['bbox']) & page.rect).get_area()
                   for i in page.get_image_info()) > page.rect.get_area()*.55:continue
            if group['page'] not in pages:
                pages[group['page']] = spans_for(page)
            spans = pages[group['page']]
            # Permit rounding of the model's normalized box, not neighboring
            # paragraphs or partial spans. Full-group matching guards the crop.
            box = [group['bbox'][0]-.002,group['bbox'][1]-.002,
                   group['bbox'][2]+.002,group['bbox'][3]+.002]
            selected = [s for s in spans if intersection_fraction(s['bbox'],box) >= .98]
            ranges = group_ranges(atoms, selected)
            literals,weights = native_math_evidence(atoms,selected)
            corrected={c['atom'] for c in corrections}
            literals={a:r for a,r in literals.items() if a not in excluded and a not in corrected}
            weights={a:r for a,r in weights.items() if a not in excluded and a not in corrected}
            report.setdefault('math_text',{}).update(literals)
            report.setdefault('math_styles',{}).update(weights)
            end=index.prefix.index(group['hi'])
            if (end<len(index.atoms) and index.atoms[end]['group']==group['group']
                    and index.atoms[end]['page']==group['page']+1
                    and ''.join(prose_letters(s['text']) for s in selected).endswith('.')
                    and matched_group_ranges(atoms,selected) is not None):
                report.setdefault('paragraph_starts',{})[index.atoms[end]['id']]=dict(
                    method='complete_live_period_blocks_cached_page_join',page=group['page']+1)
            if not ranges and group['bbox'][2]-group['bbox'][0]>.9:
                ranges = bounded_line_ranges(atoms, selected)
            # Exact native ownership is established against unchanged saved
            # text. A render correction only removes the overlapping range;
            # punctuation edits need not erase a whole paragraph's fonts.
            ranges = {atom:[r for r in runs if not any(c['atom']==atom
                and c['start_offset']<r['end'] and r['start']<c['end_offset']
                for c in corrections)] for atom,runs in ranges.items() if atom not in excluded}
            ranges = {atom:runs for atom,runs in ranges.items() if runs}
            styles.update(ranges)
            if ranges:
                report['records'].append(dict(group=group['group'],page=group['page']+1,ranges=ranges))
    if styles or any(report.get(k) for k in ('math_text','math_styles','paragraph_starts')):
        report['status'] = 'requires_visual_review'
    return styles, report


def native_minor_headings(index, registry, changed=()):
    """Recognize body-sized italic source headings without changing hierarchy."""
    import pymupdf
    source=index.source.get('source_pdf')
    if not source or not Path(source).is_file():return {}
    result={}
    with pymupdf.open(source) as pdf:
        for item in registry:
            if (item['kind'] not in {'section','subsection','subsubsection','paragraph'}
                    or re.match(r'\s*(?:\d|第|Chapter\b|Part\b)',index.text(item['lo'],item['hi']),re.I)):
                continue
            atoms=[a for a,lo,hi in index.fragments(item['lo'],item['hi'])]
            if (not atoms or len({a['group'] for a in atoms})!=1
                    or any(a['kind']!='text' or a['id'] in changed or a.get('source_panel')
                           or a.get('extraction_method')!='native'
                           or a.get('block_type')!='paragraph_title' or a.get('recovery_issue')
                           or not a.get('bbox') for a in atoms)
                    or not 0<=atoms[0]['page']<len(pdf)):continue
            b=atoms[0]['bbox'];spans=spans_for(pdf[atoms[0]['page']])
            box=[b[0]-.002,b[1]-.002,b[2]+.002,b[3]+.002]
            selected=[s for s in spans if s['text'].strip() and intersection_fraction(s['bbox'],box)>.98]
            if (not selected or len({s['line'] for s in selected})>2
                    or prose_letters(''.join(s['text'] for s in selected))!=prose_letters(index.text(item['lo'],item['hi']))
                    or not all(font_style(s['font'])==(True,False) for s in selected)):continue
            body=[s['size'] for s in spans if len(s['text'].strip())>=30
                  and s['bbox'][1]>.17 and font_style(s['font'])==(False,False)]
            if not body:continue
            size=sorted(body)[len(body)//2]
            if not all(.9*size<=s['size']<=1.15*size for s in selected):continue
            result[item['anchor']]=dict(method='complete_native_body_size_italic_heading',
                anchor=item['anchor'],page=atoms[0]['page']+1,bbox=b,
                source_font_size=max(s['size'] for s in selected),body_font_size=size)
    return result

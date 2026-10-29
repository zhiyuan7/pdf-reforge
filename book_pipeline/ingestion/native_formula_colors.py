"""Constrain native formula color review to inserting observed color wrappers."""
from pathlib import Path
import re

METHOD='native_exact_formula_color_contract_v1'


def color_contracts(index,band):
    import pymupdf
    from book_pipeline.ingestion.formula_bands import source_groups
    from book_pipeline.ingestion.math_syntax import formula_tags
    from book_pipeline.ingestion.panels import intersection_fraction
    source=index.source.get('source_pdf')
    if not source or not Path(source).is_file():return []
    result=[]
    with pymupdf.open(source) as pdf:
        page=pdf[band['page']]
        for group in source_groups(index):
            if 'colored_math' not in band.get('native_features',{}).get(group['group'],[]):continue
            if len(group['atoms'])!=1:continue
            text=group['atoms'][0]['content'];tags=formula_tags(text)
            if len(tags)!=1:continue
            palette=set()
            for b in page.get_text('dict')['blocks']:
                for line in b.get('lines',[]):
                    for s in line['spans']:
                        box=[v/d for v,d in zip(s['bbox'],[page.rect.width,page.rect.height]*2)]
                        rgb=tuple((s['color']>>k)&255 for k in (16,8,0))
                        if max(rgb)-min(rgb)>30 and intersection_fraction(box,group['bbox'])>.98:
                            palette.add(rgb)
            if palette:result.append(dict(number=tags[0]['number'],latex=text[:tags[0]['start']]+text[tags[0]['end']:],
                colors=[[c/255 for c in rgb] for rgb in sorted(palette)]))
    return result


def without_color(text):
    colors=[]
    def strip(value):
        result=[];cursor=0
        pattern=r'\\textcolor\[rgb\]\{([^{}]*)\}\{'
        for match in re.finditer(pattern,value):
            if match.start()<cursor:continue
            try:rgb=tuple(float(v) for v in match[1].split(','))
            except ValueError:raise ValueError('Use three observed RGB numbers') from None
            if len(rgb)!=3 or any(not 0<=c<=1 for c in rgb):raise ValueError('Invalid RGB color')
            depth=1;end=match.end()
            while end<len(value) and depth:
                if value[end]=='\\':end+=2;continue
                if value[end]=='{':depth+=1
                elif value[end]=='}':depth-=1
                end+=1
            if depth:raise ValueError('Unbalanced native formula color')
            colors.append(rgb);result.extend([value[cursor:match.start()],strip(value[match.end():end-1])]);cursor=end
        result.append(value[cursor:]);return ''.join(result)
    plain=strip(text)
    if re.search(r'\\(?:textcolor|color)\b',plain):raise ValueError('Use only complete RGB textcolor wrappers')
    return plain,colors


def validate_colors(answer,contracts):
    for contract in contracts:
        formulas=[i for i in answer['items'] if i['kind']=='formula' and i['number']==contract['number']]
        if len(formulas)!=1:raise ValueError('Preserve each native formula number once')
        plain,colors=without_color(formulas[0]['latex'])
        if re.sub(r'\s','',plain)!=re.sub(r'\s','',contract['latex']):
            raise ValueError('Color-only review must preserve every supplied TeX token, including bold vectors; only insert RGB wrappers')
        close=lambda a,b:max(abs(x-y) for x,y in zip(a,b))<.015
        if (any(not any(close(c,o) for o in contract['colors']) for c in colors)
                or any(not any(close(o,c) for c in colors) for o in contract['colors'])):
            raise ValueError('Preserve all observed formula colors and introduce no new color')

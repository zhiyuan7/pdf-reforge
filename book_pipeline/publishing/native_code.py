"""Preserve native monospaced panels using exact live glyphs and embedded fonts.

No language or code token is inferred. Complete unchanged text ownership,
monospaced source fonts and independent panel geometry must all agree.
"""
from io import BytesIO
from pathlib import Path
import hashlib
import re

from book_pipeline.ingestion.panels import intersection_fraction
from book_pipeline.publishing.native_body_styles import matched_group_ranges

METHOD='complete_native_monospaced_panel_v4'


def mono(name):
    return bool(re.search(r'Mono|Courier|Inconsolata|(?:CM|LM)TT\d|Typewriter',name,re.I))


def key(text):
    return ''.join(text.split())


def recover_native_code(index,directory,changed=(),*,identify_only=False):
    import pymupdf
    from fontTools.ttLib import TTFont
    source=index.source.get('source_pdf')
    if not source or not Path(source).is_file():return {},[]
    layouts={};rows=[];directory=Path(directory)
    groups=[(g,[index.atoms[n] for n in positions],positions)
            for g,positions in index.groups.items()]
    required_text=''.join(a['content'] for a in index.atoms if a['kind']=='text')
    with pymupdf.open(source) as pdf:
        pages={};font_cache={}
        for pos,(group,atoms,positions) in enumerate(groups):
            if (len(atoms)!=1 or atoms[0]['kind']!='text' or atoms[0]['id'] in changed
                    or atoms[0].get('extraction_method')!='native' or atoms[0].get('recovery_issue')
                    or not atoms[0].get('source_panel') or atoms[0]['content'].count('\n')<3
                    or not 0<=atoms[0]['page']<len(pdf)):continue
            atom=atoms[0];page=pdf[atom['page']];box=atom['bbox']
            if atom['page'] not in pages:
                spans=[]
                for b in page.get_text('dict')['blocks']:
                    for line in b.get('lines',[]):
                        for s in line['spans']:
                            if s['text']:
                                spans.append({**s,'bbox':[v/d for v,d in zip(s['bbox'],[page.rect.width,page.rect.height]*2)]})
                pages[atom['page']]=spans
            live=[s for s in pages[atom['page']] if intersection_fraction(s['bbox'],
                  [box[0]-.002,box[1]-.002,box[2]+.002,box[3]+.002])>.98]
            live.sort(key=lambda s:(round(s['origin'][1],1),s['origin'][0]))
            text=''.join(s['text'] for s in live)
            if (not live or key(text)!=key(atom['content'])
                    or sum(len(key(s['text'])) for s in live if mono(s['font']))<.9*len(key(text))
                    or len({round(s['origin'][1],1) for s in live})<4):continue
            if any(not mono(s['font']) for s in live if key(s['text'])):continue
            if identify_only:
                # Literal code ownership does not depend on whether TeX can
                # embed this PDF font format (e.g. CID fonts). Translation
                # needs the former proof even when publication needs fallback.
                layouts[atom['id']]=dict(method=METHOD,atom=atom['id'],page=atom['page']+1,
                    bbox=box,fonts=sorted({s['font'] for s in live}),identity_verified=True)
                continue
            fonts={re.sub(r'^[A-Z]{6}\+','',f[3]):f for f in page.get_fonts()}
            # PDF resource names can differ from the font's PostScript name
            # reported by live spans (e.g. a spaced family plus 'Book').
            missing={s['font'] for s in live}-set(fonts)
            if missing:
                for f in page.get_fonts():
                    if f[1] not in {'ttf','otf'}:continue
                    try:
                        parsed=TTFont(BytesIO(pdf.extract_font(f[0])[3]))
                        name=parsed['name'].getDebugName(6)
                    except Exception:continue
                    if name in missing:fonts[name]=f
            owned=[]
            for s in live:
                font=fonts.get(re.sub(r'^[A-Z]{6}\+','',s['font']))
                if not font or font[1] not in {'ttf','otf'}:break
                if font[0] not in font_cache:
                    data=pdf.extract_font(font[0])[3]
                    try:
                        parsed=TTFont(BytesIO(data));cmap=parsed.getBestCmap() or {}
                        if 'post' not in parsed:
                            # PDF subsets may omit this naming/underline table,
                            # but xdvipdfmx requires it for a standalone font.
                            # Format3 restores metadata, retaining cmap, outlines
                            # and advance widths of every existing glyph.
                            from fontTools.fontBuilder import FontBuilder
                            FontBuilder(font=parsed).setupPost(keepGlyphNames=False,isFixedPitch=1)
                        # Embedded subsets often retain thousands of empty
                        # cmap slots and stale TrueType hint programs. Rebuild
                        # a standalone subset for the actual excerpt characters;
                        # preserve outlines and advances, without hint bytecode.
                        from fontTools import subset
                        options=subset.Options();options.hinting=False
                        options.name_IDs=['*'];options.layout_features=[]
                        subsetter=subset.Subsetter(options=options)
                        subsetter.populate(text=required_text);subsetter.subset(parsed)
                        stream=BytesIO();parsed.save(stream);data=stream.getvalue()
                    except Exception:break
                    digest=hashlib.sha256(data).hexdigest()
                    name='code-'+digest[:24]+'.'+font[1]
                    target=directory/'fonts'/name;target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(data)
                    command='BookCodeFont'+''.join(chr(65+int(c,16)) for c in digest[:16])
                    font_cache[font[0]]=dict(file=name,command=command,cmap=cmap)
                saved=font_cache[font[0]]
                if any(ord(c) not in saved['cmap'] for c in s['text'] if not c.isspace()):break
                owned.append(dict(text=s['text'],x=s['origin'][0],y=s['origin'][1],
                    size=s['size'],color=s['color'],font=saved['file'],command=saved['command']))
            if len(owned)!=len(live):continue
            x0=min(s['bbox'][0]*page.rect.width for s in live)
            y0=min(s['bbox'][1]*page.rect.height for s in live)
            width=max(s['bbox'][2]*page.rect.width for s in live)-x0
            height=max(s['bbox'][3]*page.rect.height for s in live)-y0
            layout=dict(method=METHOD,atom=atom['id'],page=atom['page']+1,
                bbox=box,x0=x0,y0=y0,width=width,height=height,spans=owned,
                lines=len({round(s['y'],1) for s in owned}),panel=atom['source_panel'])
            layouts[atom['id']]=layout
            if pos+1==len(groups):continue
            _,caption,caption_positions=groups[pos+1]
            if (not caption or any(a.get('extraction_method')!='native' or a['page']!=atom['page']
                    or a['kind'] not in {'text','math'} or a['id'] in changed or a.get('source_panel')
                    or a.get('role') not in {'text','paragraph','image_caption','chart_caption'} for a in caption)
                    or not 40<=sum(len(a['content']) for a in caption)<=500):continue
            b=atom['source_panel']['bbox'];c=caption[0]['bbox']
            if (not .008<=c[0]-b[2]<=.12 or abs(b[1]-c[1])>.04
                    or not .08<=c[2]-c[0]<=.3 or c[3]-c[1]>.22):continue
            observed=[s for s in pages[atom['page']] if intersection_fraction(s['bbox'],
                [c[0]-.002,c[1]-.002,c[2]+.002,c[3]+.002])>.98]
            size=sorted(s['size'] for s in live)[len(live)//2]
            if (not observed or any(s['size']>1.2*size for s in observed)
                    or sum(len(key(s['text'])) for s in observed if mono(s['font']))>
                       .5*sum(len(key(s['text'])) for s in observed)):continue
            if matched_group_ranges(caption,observed) is None:
                # Native extraction has already reviewed line-end joins.
                # Verify its unchanged caption against whole live lines with
                # those terminal hyphens elided; never rewrite the caption.
                joined=[dict(s) for s in observed]
                for a,z in zip(joined,joined[1:]):
                    if (re.search(r'[A-Za-z]-$',a['text']) and re.match(r'[a-z]',z['text'])
                            and z['origin'][1]-a['origin'][1]>.5*a['size']):
                        a['text']=a['text'][:-1]
                if matched_group_ranges(caption,joined) is None:continue
            extent=c[2]-b[0]
            rows.append(dict(method='native_code_side_caption',lo=index.prefix[positions[0]],
                hi=index.prefix[caption_positions[-1]+1],code=atom['id'],
                code_range=[index.prefix[positions[0]],index.prefix[positions[-1]+1]],
                caption_range=[index.prefix[caption_positions[0]],index.prefix[caption_positions[-1]+1]],
                widths=[(b[2]-b[0])/extent,(c[2]-c[0])/extent],targets=[]))
    return layouts,rows


def font_commands(layouts):
    fonts={s['command']:s['font'] for l in layouts.values() for s in l['spans']}
    return '\n'.join('\\newfontfamily\\'+command+'[Path=fonts/]{'+file+'}' for command,file in fonts.items())


def code_tex(layout):
    from book_pipeline.publishing.latex import ESCAPES
    # Embedded fonts retain the actual code characters, including Unicode
    # identifiers; they must not become mathematical TeX substitutions.
    literal=lambda t:''.join(r'\char"20{}' if c==' ' else ESCAPES.get(c,c) for c in t)
    size=sorted(s['size'] for s in layout['spans'])[len(layout['spans'])//2]
    scale=8.5/size
    width,height=layout['width']*scale,layout['height']*scale
    chunks=['\n\\par\\noindent\\begin{adjustbox}{max width=\\linewidth}\n',
            '\\setlength{\\unitlength}{1pt}\\begin{picture}('+f'{width:.4f},{height:.4f}'+')\n']
    for s in layout['spans']:
        x=(s['x']-layout['x0'])*scale;y=height-(s['y']-layout['y0'])*scale
        rgb=','.join(f'{((s["color"]>>n)&255)/255:.5f}' for n in (16,8,0))
        chunks.append('\\put('+f'{x:.4f},{y:.4f}'+'){\\hbox{{\\'+s['command']+
            '\\fontsize{'+f'{s["size"]*scale:.4f}'+'}{10}\\selectfont\\textcolor[rgb]{'+rgb+
            '}{'+literal(s['text'])+'}}}}\n')
    chunks.append('\\end{picture}\\end{adjustbox}\\par\n')
    return ''.join(chunks)

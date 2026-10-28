"""Keep consecutive short display rows together without changing their math."""
import math
import re

from book_pipeline.ingestion.math_syntax import formula_tags
from book_pipeline.ingestion.source import corrected_content


def is_display(atom):
    return atom['kind'] == 'display_math' or (
        atom['kind'] == 'math' and bool(formula_tags(atom['content'])))


def short_display(atom, content):
    # Tall/nested layouts and ambiguous tags need their existing standalone
    # renderer (which can preserve an original image). Do not box them in a run.
    display = atom['kind'] == 'display_math' or (atom['kind'] == 'math' and bool(formula_tags(content)))
    return (display and len(content) <= 600
            and not re.search(r'\\(?:begin|substack|dfrac|cfrac)\b|\\\\', content)
            and len(formula_tags(content)) <= 1
            and len(re.findall(r'\\tag\*?\s*\{', content)) == len(formula_tags(content)))


def same_column(left, right):
    a = left.get('bbox') or left.get('block_bbox')
    b = right.get('bbox') or right.get('block_bbox')
    if not a or not b:
        return False
    width = min(a[2]-a[0], b[2]-b[0])
    return width > 0 and min(a[2], b[2])-max(a[0], b[0]) >= .6*width


def formula_image_prompts(index, barriers=(), skipped=()):
    """Keep a short source prompt with an adjacent uncertain formula crop.

    Ordinary illustrations, long prose and ambiguous semantic/page/column
    ownership retain normal pagination. The image itself is still rendered in
    source order, with no extra copy of its content or source destination.
    """
    prompts=[]
    for group,positions in index.groups.items():
        first,last=min(positions),max(positions)
        atoms=[index.atoms[n] for n in positions]
        if (last+1==len(index.atoms) or positions!=list(range(first,last+1))
                or any(a['kind'] not in {'text','math'} or a['block_type'] not in {'text','paragraph'} for a in atoms)
                or len(index.text(index.prefix[first],index.prefix[last+1]))>120):
            continue
        image=index.atoms[last+1]
        end=atoms[-1]
        # A witnessed vertical continuation can span columns/registers. Its
        # union rectangle is not the column where the prompt ends.
        if end.get('source_prose_end_bbox') and end.get('writing_direction')=='vertical_rl':
            from book_pipeline.structure.headings import valid_box
            box=end['source_prose_end_bbox'];outer=end.get('bbox') or end.get('block_bbox')
            if (valid_box(box) and valid_box(outer) and box[0]>=outer[0] and box[1]>=outer[1]
                    and box[2]<=outer[2] and box[3]<=outer[3]):
                end=dict(end,bbox=box)
        lo,hi=index.prefix[first],index.prefix[last+2]
        if (image['kind']!='image' or image.get('recovery_issue')!='formula_source_crop_requires_transcription'
                or image['page']!=atoms[-1]['page'] or not same_column(end,image)
                or atoms[-1].get('source_panel')!=image.get('source_panel')
                or any(lo<b<hi for b in barriers) or any(a<hi and z>lo for a,z,*_ in skipped)):
            continue
        prompts.append(dict(method='short_prompt_kept_with_formula_image',group=group,lo=lo,
                            prompt_hi=index.prefix[last+1],hi=hi,image=image['id']))
    return prompts


def bounded_display_units(index, barriers=(), skipped=()):
    """Keep a short numbered prompt, math and closing prose as one unit.

    Also protect a short numbered introduction with its first labelled
    subquestion and formula. Require source page/column ownership and bound
    the size; ordinary prose and continuing derivation chains stay breakable.
    """
    groups=[]
    for group,positions in index.groups.items():
        first,last=min(positions),max(positions)
        if positions != list(range(first,last+1)):
            continue
        groups.append(dict(lo=index.prefix[first],hi=index.prefix[last+1],
                           atoms=[index.atoms[n] for n in positions]))
    numbered=re.compile(r'^\s*(?:[✓✔]\s*)?\d+(?:\.\d+)+\s+')
    label=re.compile(r'^\s*(?:[✓✔]\s*)?\d|^\s*\([a-zA-Z0-9]+\)|^\s*[•◦▪‣]')
    def prose(row,limit):
        return (row['atoms'][0]['block_type'] in {'text','paragraph'}
                and row['atoms'][0]['role']=='text'
                and all(a['kind'] in {'text','math'} and not is_display(a) for a in row['atoms'])
                and len(index.text(row['lo'],row['hi'])) <= limit)
    def formula(row):
        a=row['atoms'][0]
        box=a.get('bbox')
        return (len(row['atoms'])==1 and a['kind']=='display_math' and box
                and box[3]-box[1] <= .12 and len(a['content']) <= 800
                and not formula_tags(a['content']))
    units=[]
    for n,prompt in enumerate(groups[:-2]):
        if not prose(prompt,160) or not numbered.match(index.text(prompt['lo'],prompt['hi'])):
            continue
        middle,last=groups[n+1:n+3]
        if prompt['hi']!=middle['lo'] or middle['hi']!=last['lo']:
            continue
        if n+3<len(groups) and any(is_display(a) for a in groups[n+3]['atoms']):
            continue  # Do not end a bounded unit inside a continuing derivation.
        if formula(middle) and prose(last,320) and not label.match(index.text(last['lo'],last['hi'])):
            method='numbered_prompt_formula_closing_prose'
        elif (prose(middle,180) and re.match(r'^\s*\([a-zA-Z0-9]+\)\s+',index.text(middle['lo'],middle['hi']))
              and formula(last)):
            method='numbered_intro_first_subquestion_formula'
        else:
            continue
        atoms=[a for row in (prompt,middle,last) for a in row['atoms']]
        first=atoms[0];lo,hi=prompt['lo'],last['hi']
        boxes=[a.get('block_bbox') or a.get('bbox') for a in atoms]
        if (any(a['page']!=first['page'] or a.get('source_panel')!=first.get('source_panel')
                for a in atoms)
                or not all(boxes) or max(b[3] for b in boxes)-min(b[1] for b in boxes)>.3
                or any(lo<b<hi for b in barriers)
                or any(a<hi and z>lo for a,z,*_ in skipped)
                or (units and units[-1]['hi']>lo)):
            continue
        # A short left-aligned prompt need not overlap its centered formula.
        # Require a real prose column enclosing the entire unit instead of
        # treating the prompt's ink width as a column boundary.
        left,right=min(b[0] for b in boxes),max(b[2] for b in boxes)
        columns=[a.get('block_bbox') or a.get('bbox') for a in index.atoms
                 if a['page']==first['page'] and a['kind']=='text' and a['role']=='text'
                 and a['block_type'] in {'text','paragraph'}
                 and a.get('source_panel')==first.get('source_panel')]
        columns=[b for b in columns if b and b[0]<=left+.005 and b[2]>=right-.005]
        column=max(columns,key=lambda b:b[2]-b[0],default=None)
        if not column:
            continue
        units.append(dict(method=method,lo=lo,hi=hi,page=first['page'],
                          column_bbox=column,groups=[row['atoms'][0]['group'] for row in (prompt,middle,last)]))
    return units


def display_runs(index, decisions, barriers=(), skipped=()):
    """Group only adjacent source atoms in one page/column/semantic scope.

    Six short rows bound the indivisible TeX display; longer sequences have
    safe breaks between chunks. A source paragraph, exclusion or object edge
    always terminates a run. Formula numbers remain per-row tags.
    """
    barriers = set(barriers)
    corrections = {}
    for edit in decisions.get('corrections', []):
        corrections.setdefault(edit['atom'], []).append(edit)
    repairs = decisions.get('formula_repairs', {})
    runs, pending = [], []

    def flush():
        if len(pending) > 1:
            first, last = pending[0], pending[-1]
            runs.append(dict(method='consecutive_short_display_rows',
                             lo=index.prefix[first], hi=index.prefix[last+1],
                             atoms=[index.atoms[n]['id'] for n in pending],
                             page=index.atoms[first]['page'], rows=len(pending)))
        pending.clear()

    for n, atom in enumerate(index.atoms):
        lo, hi = index.prefix[n:n+2]
        content = repairs.get(atom['id'], corrected_content(atom, corrections.get(atom['id'], [])))
        eligible = (short_display(atom, content)
                    and not any(lo < b < hi for b in barriers)
                    and not any(a < hi and z > lo for a, z, *_ in skipped))
        if not eligible:
            flush()
            continue
        if pending:
            previous = index.atoms[pending[-1]]
            if (len(pending) == 6 or lo in barriers
                    or previous['page'] != atom['page']
                    or previous.get('source_panel') != atom.get('source_panel')
                    or not same_column(previous, atom)):
                flush()
        pending.append(n)
    flush()
    return runs


def prompt_space(index, style, runs):
    """Reserve a short source list prompt and the first display beneath it."""
    if style['continuation'] or style['hi'] == index.prefix[-1]:
        return None
    n = index.ids[index.address(style['hi'])['atom']]
    following = index.atoms[n]
    if not is_display(following) or following['page'] != style['page']:
        return None
    prompt = index.text(style['lo'], style['hi'])
    if len(prompt) > 240:
        return None
    row_count = next((r['rows'] for r in runs if r['lo'] == style['hi']), 1)
    lines = sum(2 if ord(c) > 127 else 1 for c in prompt)
    # Allow for wrapped source text, normal display skips and short math rows.
    return min(14, 2 + math.ceil(lines/65) + math.ceil(1.5*row_count))


def list_display_scope(index, style, runs, barriers=(), decisions=None, skipped=()):
    """Keep a prompt's bounded derivation inside its source list paragraph.

    Closing a TeX list before the display starts a new empty paragraph. Keeping
    the same list scope avoids that line and retains the item's math indent.
    Long prose retains its own paragraph boundary and widow/orphan behavior.
    Ambiguous ownership or a long derivation split into chunks stays unchanged.
    """
    if len(index.text(style['lo'], style['hi'])) > 240:
        return None
    run = next((r for r in runs if r['lo'] == style['hi']), None)
    if run is None and style['hi'] < index.prefix[-1]:
        n = index.ids[index.address(style['hi'])['atom']]
        atom = index.atoms[n]
        decisions = decisions or {}
        content = decisions.get('formula_repairs', {}).get(atom['id'], corrected_content(
            atom, [e for e in decisions.get('corrections', []) if e['atom'] == atom['id']]))
        if short_display(atom, content):
            run = dict(lo=style['hi'], hi=index.prefix[n+1], atoms=[atom['id']], page=atom['page'], rows=1)
    if (not run or any(style['lo'] < b < run['hi'] for b in barriers)
            or any(a < run['hi'] and z > style['hi'] for a, z, *_ in skipped)):
        return None
    prompt = index.atoms[index.groups[style['group']][-1]]
    first = index.atoms[index.ids[run['atoms'][0]]]
    if (run['page'] != style['page']
            or prompt.get('source_panel') != first.get('source_panel')
            or not same_column(prompt, first)):
        return None
    if run['hi'] < index.prefix[-1] and run['hi'] not in barriers:
        following = index.atoms[index.ids[index.address(run['hi'])['atom']]]
        last = index.atoms[index.ids[run['atoms'][-1]]]
        if (is_display(following) and following['page'] == last['page']
                and following.get('source_panel') == last.get('source_panel')
                and same_column(last, following)):
            return None
    return run

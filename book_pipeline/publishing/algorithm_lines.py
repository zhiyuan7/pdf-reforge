"""Keep explicit algorithm text and inline mathematics on their source lines."""


def algorithm_line_rows(index, registry=(), exclusions=(), changed=(), native_code=()):
    rows=[]
    for group, positions in index.groups.items():
        atoms=[index.atoms[n] for n in positions]
        lo,hi=index.prefix[positions[0]],index.prefix[positions[-1]+1]
        if (positions!=list(range(positions[0],positions[-1]+1))
                or len({a['page'] for a in atoms})!=1
                or any(a['kind'] not in {'text','math'} or a.get('role')!='algorithm_body'
                       or a.get('block_type')!='code' or a.get('source_panel')
                       or a['id'] in changed or a['id'] in native_code for a in atoms)
                or sum(a['content'].count('\n') for a in atoms)<3
                or any(r['lo']<hi and lo<r['hi'] for r in [*registry,*exclusions])):
            continue
        rows.append(dict(method='explicit_algorithm_source_lines',lo=lo,hi=hi,
                         group=group,page=atoms[0]['page']+1,targets=[]))
    return rows

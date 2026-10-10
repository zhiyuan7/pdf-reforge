"""Source TeX compatibility without mathematical content rewriting."""
import re


def math_compatibility(text):
    """Preserve formula tokens while making simple OCR arrays valid TeX."""
    changes=[]
    def operator_alias(match):
        changes.append('nonstandard OCR operator rendered with operatorname; original operator spelling preserved')
        return r'\operatorname{' + match[1] + '}'
    text=re.sub(r'\\(ctan)\b',operator_alias,text)
    def circled(match):
        number=int(match[1])
        if not 1<=number<=20:return match[0]
        changes.append('circled numeric footnote marker rendered by the bundled Unicode font in math mode')
        return chr(0x2460+number-1)
    text=re.sub(r'\\textcircled\{(\d{1,2})\}',circled,text)
    def broken_tag(match):
        label=match['label']
        prime=re.fullmatch(r'\(?(\d+)\s*\^\s*\{\s*\\prime\s*\}\)?',label)
        stars=re.fullmatch(r'\\left\((\*+)\\right\)',label)
        if not prime and not stars:return match[0]
        value=prime[1]+"'" if prime else stars[1]
        changes.append('restored malformed OCR tag delimiters; original prime/star number preserved')
        return r'\tag{'+value+'}'
    text=re.sub(r'\\tag\{\\\((?P<label>.*?)\\\}',broken_tag,text)
    def array(match):
        spec,body=match['spec'],match['body']
        # Nested environments and complex column specifications need their own
        # parser or source review; a flat c/l/r array is sufficient evidence here.
        if r'\begin{' in body:return match[0]
        def bracket(row):
            value=row['value']
            if '\\' in value or re.fullmatch(r'\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:pt|em|ex|cm|mm|in|bp|sp|pc|dd|cc|mu)\s*|\s*0\s*',value):
                return row[0]
            changes.append('array cell starts with [ after row break; inserted empty group to avoid TeX row-spacing argument')
            return row['break']+'{}'+row['space']+'['+value+']'
        body=re.sub(r'(?P<break>\\\\)(?P<space>\s*)\[(?P<value>[^\[\]]+)\]',bracket,body)
        depth,i,cells,max_cells=0,0,1,1
        while i<len(body):
            if body[i]=='\\':
                if i+1<len(body) and body[i+1]=='\\' and depth==0:
                    max_cells=max(max_cells,cells);cells=1
                i+=2;continue
            if body[i]=='{':depth+=1
            elif body[i]=='}':depth-=1
            elif body[i]=='&' and depth==0:cells+=1
            i+=1
        max_cells=max(max_cells,cells)
        columns=sum(c in 'clr' for c in spec)
        if max_cells>columns:
            spec+=' c'*(max_cells-columns)
            changes.append(f'array declares {columns} columns but source rows contain {max_cells} cells; extended column layout without changing cells')
        return r'\begin{array}{'+spec+'}'+body+r'\end{array}'
    text=re.sub(r'\\begin\{array\}\{(?P<spec>[clr\s]+)\}(?P<body>.*?)\\end\{array\}',array,text,flags=re.S)
    return text,changes


def formula_tags(text):
    """Extract complete brace-balanced tags, including original prime numbers."""
    text,_=math_compatibility(text)
    tags=[]
    for match in re.finditer(r'\\tag\*?\s*\{',text):
        start=match.end();depth=1;i=start
        while i<len(text) and depth:
            if text[i]=='\\':i+=2;continue
            if text[i]=='{':depth+=1
            elif text[i]=='}':depth-=1
            i+=1
        if depth:continue
        label=text[start:i-1]
        label=re.sub(r'\^\s*\{\s*\\prime\s*\}',"'",label)
        label=re.sub(r'\\prime\b',"'",label)
        label=label.removeprefix(r'\(').removesuffix(r'\)').strip('$() ')
        tags.append({'start':match.start(),'end':i,'tex':text[match.start():i],'number':label})
    return tags

"""Native GoTo destinations drive Zotero hover previews, with no PDF JavaScript."""
from pathlib import Path
import pymupdf

from book_pipeline.config import save


class NavigationBuilder:
    def finalize(self,pdf,directory,registry):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        targets={o['anchor']:o for o in registry}
        samples,links,missing=[],[],[]
        with pymupdf.open(pdf) as doc:
            names=doc.resolve_names()
            for page in doc:
                for link in page.get_links():
                    name=link.get('nameddest')
                    if not name or not name.startswith('book-'):
                        continue
                    target=names.get(name)
                    if not target or not 0<=target['page']<len(doc):
                        missing.append(name);continue
                    raw=doc.xref_object(link['xref'])
                    if '/GoTo' not in raw:
                        raise ValueError('Internal reference is not a native GoTo link')
                    links.append({'page':page.number+1,'rect':list(link['from']),
                                  'target':name,'target_page':target['page']+1,
                                  'target_pdf_point':list(target['to'])})
            # Export representative real target crops as verification evidence.
            for kind in ('chapter','section','equation','figure','table','algorithm','definition','example','theorem'):
                row=next((r for r in links if targets.get(r['target'],{}).get('kind')==kind),None)
                if row is None:continue
                page=doc[row['target_page']-1]
                # resolve_names uses PDF bottom-left coordinates; pixmaps use top-left.
                y=max(0,page.rect.height-row['target_pdf_point'][1]-8)
                clip=pymupdf.Rect(25,y,page.rect.width-25,min(page.rect.height,y+210))
                path=directory/f'preview-{kind}.png'
                page.get_pixmap(dpi=120,clip=clip,alpha=False).save(path)
                samples.append({**row,'kind':kind,'preview':str(path.relative_to(directory.parent))})
            result={'mechanism':'native_pdf_goto_and_named_xyz_destination','reader':'Zotero',
                    'links':len(links),'missing_targets':missing,'javascript_required':False,
                    'samples':samples,'reference_book':'LADR4eChinese.pdf',
                    'reader_ui_verified':False,'link_index':links}
        if missing:raise ValueError('Missing hover-preview destinations')
        save(directory.parent/'reports/navigation.json',result)
        return result

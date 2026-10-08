"""Local PDF geometry/navigation checks and actual raster previews, without AI."""
from pathlib import Path
import pymupdf


class LocalPDFReviewer:
    def review(self,pdf,directory,coverage=None,progress=lambda *args:None):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for old in directory.glob('page-*.png'):
            old.unlink()
        issues,previews,mapping=[],[],{}
        with pymupdf.open(pdf) as doc:
            names=doc.resolve_names()
            title_pages={names[h['anchor']]['page'] for h in (coverage or {}).get('heading_levels',[])
                         if h.get('standalone') and h['anchor'] in names}
            break_starts={names[h['anchor']]['page'] for h in (coverage or {}).get('heading_levels',[])
                          if (h.get('standalone') or h['level']==2) and h['anchor'] in names}
            forced_end_pages={p-1 for p in break_starts if p>0}
            for atom,anchor in (coverage or {}).get('source_anchors',{}).items():
                if anchor in names:
                    mapping.setdefault(str(names[anchor]['page']+1),[]).append(atom)
            sample={0,1,len(doc)-1}
            sample.update(title_pages)
            sample.update(range(0,len(doc),30))
            sample.update(value['page'] for name,value in names.items() if name.startswith('book-chapter-'))
            whitespace=[]
            # Every page is checked; preview samples supplement, rather than replace, geometry.
            for n,page in enumerate(doc):
                page_blocks=page.get_text('dict',clip=pymupdf.INFINITE_RECT())['blocks']
                for block in page_blocks:
                    rect=pymupdf.Rect(block['bbox'])
                    if rect.x0 < -1 or rect.y0 < -1 or rect.x1 > page.rect.width+1 or rect.y1 > page.rect.height+1:
                        issues.append({'page':n+1,'reason':'content_outside_page','bbox':list(rect)})
                body=[pymupdf.Rect(b['bbox']) for b in page_blocks if b['bbox'][3]>50 and b['bbox'][1]<page.rect.height-60]
                if body and n not in title_pages|forced_end_pages|{len(doc)-1}:
                    bottom=max(r.y1 for r in body)
                    gap=(page.rect.height-60-bottom)/(page.rect.height-110)
                    if gap>0.30:whitespace.append(dict(page=n+1,bottom_gap_fraction=round(gap,3)))
                if n in sample:
                    path=directory/f'page-{n+1:06d}.png'
                    page.get_pixmap(dpi=100,alpha=False).save(path)
                    previews.append(str(path))
                if (n+1)%50==0 or n+1==len(doc):
                    progress('本地PDF检查',n+1,len(doc))
            return {'method':'local_pdf_geometry_and_sample_rendering','issues':issues,'repairs':[],
                    'checked_pages':len(doc),'rendered_pages':len(previews),'previews':previews,
                    'source_mapping':mapping,'all_current_pages_model_reviewed':False,'model_requests':0,
                    'standalone_title_pages':[p+1 for p in sorted(title_pages)],
                    'forced_heading_end_pages':[p+1 for p in sorted(forced_end_pages)],
                    'large_bottom_gap_pages':whitespace}

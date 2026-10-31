"""Offline draft from verified source blocks; no model calls or quality promotion."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from book_pipeline.config import load_config,read,save
from book_pipeline.application.pipeline import load_book
from book_pipeline.ingestion.source import SourceIndex
from book_pipeline.structure.headings import group_blocks,span_for
from book_pipeline.structure.recognized_headings import recognized_heading_registry
from book_pipeline.publishing.latex import write_project
from book_pipeline.publishing.environment import doctor
from book_pipeline.publishing.compiler import compile_project

parser=argparse.ArgumentParser()
for key in ('pdf','source','config','output'):parser.add_argument('--'+key,type=Path,required=True)
parser.add_argument('--pages')
parser.add_argument('--title',required=True)
parser.add_argument('--author',default='')
parser.add_argument('--syntax-only',action='store_true')
parser.add_argument('--footnote-pages',help='Physical source pages with reviewed note roles/callouts')
parser.add_argument('--cross-figure-pages',help='Reviewed physical source pages allowing native sentence recovery across page and figure boundaries')
args=parser.parse_args()
cfg=load_config(args.config)
index=load_book(args.pdf,args.source)
selected={int(v) for v in args.pages.split(',')} if args.pages else None
if selected:
    index=SourceIndex({**index.source,'atoms':[a for a in index.atoms if a['page']+1 in selected]})
exclusions=[]
toc_pages=set(cfg.get('toc_pages',[])) if not selected else set()
cfg['float_assets']=True
cfg.setdefault('start_page_levels',[1])
for block in group_blocks(index):
    lo,hi=index.span(span_for(index,block))
    if block['type'] in {'header','footer','page_number'} or block['page'] in toc_pages:
        exclusions.append(dict(lo=lo,hi=hi,reason='original_toc' if block['page'] in toc_pages else 'source_marginal'))
        continue
from book_pipeline.structure.running_headers import running_header_rows
running_headers=running_header_rows(index)
exclusions.extend(running_headers)
registry=recognized_heading_registry(index,toc_pages,running_headers=running_headers)
print(f'Draft source: {len(index.atoms)} atoms; {len(registry)} recognized headings',flush=True)
output=args.output.resolve()
if output.exists():raise ValueError('Use a new draft directory')
project=output/'latex'
project.mkdir(parents=True)
save(output/'source-headings.json',dict(method='recognized_source_title_blocks',headings=registry,
    excluded_running_headers=running_headers,
    semantic_enrichment='not_run',quality_status='not_reviewed'))
decisions=dict(corrections=[],joins=[],exclusions=exclusions,bold_spans=[])
from book_pipeline.publishing.source_footnotes import source_page_footnotes
reviewed=read(args.source/'manifest.json').get('reviewed_layout', {})
decisions['reviewed_ink_pages']=reviewed.get('ink_pages', [])
note_pages=({int(p) for p in args.footnote_pages.split(',')} if args.footnote_pages is not None
            else set(reviewed.get('footnote_pages', [])))
decisions['reviewed_cross_figure_pages']=([int(p) for p in args.cross_figure_pages.split(',')]
    if args.cross_figure_pages is not None else reviewed.get('cross_figure_pages', []))
decisions['page_footnotes']=[n for n in source_page_footnotes(index,exclusions)
                           if n['status']=='matched' and n['page'] in note_pages] if note_pages else []
audit=write_project(index,decisions,registry,[],dict(title=args.title,author=args.author),cfg,project,
    heading_layout=dict(standalone_levels=[],centered_levels=[1],page_break_after_levels=[]),sample_mode=bool(selected))
save(output/'draft-scope.json',dict(source=str(args.source),source_pages=sorted(selected) if selected else list(range(1,index.source['source_page_count']+1)),
    source_units=audit['source_units'],source_coverage_complete=audit['coverage_complete'],quality_status='not_reviewed',
    translation_status=index.source.get('translation',{}).get('status','not_translated'),
    formula_review='pending',heading_source='recognized source blocks only',semantic_enrichment='not_run',api_requests=0))
print('LaTeX written; compiling offline',flush=True)
result=compile_project(project,doctor(cfg,require_key=False),timeout=1800,syntax_only=args.syntax_only)
save(output/'compilation.json',result)
print(f"Compiled: {result['success']}; warnings: {len(result['warnings'])}",flush=True)
raise SystemExit(0 if result['success'] else 1)

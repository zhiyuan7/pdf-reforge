"""HTTP boundary for the local workspace; no pipeline work runs on the event loop."""
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import os

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
import pymupdf

from book_pipeline.config import executable, load_config, read
from book_pipeline.publishing.colors import COLOR_KINDS, validate_colors
from book_pipeline.structure.headings import validate_metadata
from book_pipeline.structure.keyword_matching import validate_candidates
from .jobs import JobBusy, JobStore

ASSETS = Path(__file__).parent / 'static'
MAX_UPLOAD = 1024 * 1024 * 1024


def create_app(config_path=None, output=None):
    config = load_config(config_path)
    store = JobStore(output or Path('output/web'), config)
    app = FastAPI(title='PDF 回炉所', docs_url=None, redoc_url=None)
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', '[::1]', 'testserver'])

    @app.middleware('http')
    async def origin_check(request: Request, call_next):
        origin = request.headers.get('origin')
        if request.method not in {'GET', 'HEAD', 'OPTIONS'} and origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
            return Response('Cross-origin request rejected', status_code=403)
        return await call_next(request)

    def get_job(identifier):
        try:
            return store.get(identifier)
        except KeyError:
            raise HTTPException(404, '书稿不存在。')

    @app.get('/api/environment')
    def environment():
        python = Path(config['mineru_python']).expanduser()
        from book_pipeline.config import ROOT
        if not python.is_absolute():
            python = ROOT / python
        return dict(mineru=python.is_file(), tex=all(executable(k, config[k]) for k in ('xelatex', 'latexmk')),
                    api_key=bool(os.environ.get(config['api_key_env'])))

    @app.get('/api/jobs')
    def list_jobs():
        return store.list()

    @app.get('/api/keyword-candidates')
    def keyword_candidates():
        return config['keyword_candidates']

    @app.post('/api/keyword-candidates/validate')
    def check_keyword_candidates(candidates: dict):
        try:
            return validate_candidates(candidates)
        except ValueError as error:
            raise HTTPException(422, str(error))

    @app.put('/api/jobs/{identifier}/keyword-candidates')
    def update_keyword_candidates(identifier: str, candidates: dict):
        get_job(identifier)
        try:
            store.set_keyword_candidates(identifier, candidates)
        except JobBusy as error:
            raise HTTPException(409, str(error))
        except ValueError as error:
            raise HTTPException(422, str(error))
        return store.snapshot(identifier)

    @app.get('/api/jobs/{identifier}')
    def job(identifier: str):
        get_job(identifier)
        return store.snapshot(identifier)

    @app.get('/api/jobs/{identifier}/blocks')
    def blocks(identifier: str):
        get_job(identifier)
        return store.blocks(identifier)

    @app.delete('/api/jobs/{identifier}/block-types')
    def remove_block_type(identifier: str, selector: dict):
        get_job(identifier)
        try:
            store.remove_block_type(identifier, selector)
        except JobBusy as error:
            raise HTTPException(409, str(error))
        except ValueError as error:
            raise HTTPException(422, str(error))
        except KeyError:
            raise HTTPException(404, '特殊块类型不存在或已删除。')
        return store.snapshot(identifier)

    @app.delete('/api/jobs/{identifier}/blocks/{block_id}')
    def remove_block(identifier: str, block_id: str):
        get_job(identifier)
        try:
            store.remove_block(identifier, block_id)
        except JobBusy as error:
            raise HTTPException(409, str(error))
        except KeyError:
            raise HTTPException(404, '特殊块不存在或已取消。')
        return store.snapshot(identifier)

    @app.post('/api/jobs', status_code=201)
    def upload(title: str = Form(..., max_length=240), author: str = Form(..., max_length=240), file: UploadFile = File(...),
               keyword_candidates: str | None = Form(None, max_length=150000)):
        try:
            metadata = validate_metadata({'book': {'title': title.strip(), 'author': author.strip()}})
            candidates = validate_candidates(json.loads(keyword_candidates)) if keyword_candidates is not None else config['keyword_candidates']
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error))
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(422, '请选择 PDF 文件。')
        temporary = None
        try:
            with NamedTemporaryFile(dir=store.directory, suffix='.pdf', delete=False) as stream:
                temporary = Path(stream.name)
                size = 0
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, '文件不能超过 1 GB。')
                    stream.write(chunk)
            try:
                with pymupdf.open(temporary) as document:
                    if document.is_encrypted or not document.page_count:
                        raise ValueError('请选择未加密且有页面的 PDF。')
                    pages = document.page_count
            except (RuntimeError, ValueError) as error:
                raise HTTPException(422, '无法读取 PDF：' + str(error))
            identifier = store.create(metadata, temporary, Path(file.filename).name, pages, candidates)
            return store.snapshot(identifier)
        finally:
            file.file.close()
            if temporary and temporary.exists():
                temporary.unlink()

    @app.post('/api/jobs/{identifier}/process', status_code=202)
    def process(identifier: str):
        get_job(identifier)
        try:
            store.process(identifier)
        except ValueError as error:
            raise HTTPException(409, str(error))
        return store.snapshot(identifier)

    @app.post('/api/jobs/{identifier}/render', status_code=202)
    def render(identifier: str, colors: dict):
        get_job(identifier)
        try:
            selected, typography = presentation(colors)
            store.render(identifier, selected, typography)
        except ValueError as error:
            raise HTTPException(422, str(error))
        return store.snapshot(identifier)

    @app.post('/api/jobs/{identifier}/preview', status_code=202)
    def preview(identifier: str, colors: dict):
        get_job(identifier)
        try:
            selected, typography = presentation(colors)
            store.preview(identifier, selected, typography)
        except ValueError as error:
            raise HTTPException(422, str(error))
        return store.snapshot(identifier)

    def presentation(value):
        if 'colors' in value or 'typography' in value:
            if set(value) - {'colors', 'typography'}:
                raise ValueError('不支持的排版设置。')
            return validate_colors(value.get('colors', {})), value.get('typography')
        return validate_colors(value), None

    @app.post('/api/jobs/{identifier}/extract', status_code=202)
    def extract(identifier: str):
        get_job(identifier)
        try:
            store.extract(identifier)
        except ValueError as error:
            raise HTTPException(422, str(error))
        return store.snapshot(identifier)

    @app.get('/api/jobs/{identifier}/samples/{kind}/{number}.png')
    def sample(identifier: str, kind: str, number: int):
        get_job(identifier)
        try:
            path = store.sample_path(identifier, kind, number)
        except ValueError as error:
            raise HTTPException(404, str(error))
        return FileResponse(path, media_type='image/png', headers={'Cache-Control': 'no-cache'})

    @app.get('/api/jobs/{identifier}/pdf')
    def pdf(identifier: str):
        job = get_job(identifier)
        try:
            path = store.pdf_path(identifier)
        except ValueError as error:
            raise HTTPException(409, str(error))
        return FileResponse(path, media_type='application/pdf', filename=job['metadata']['book']['title'] + '.pdf')

    @app.get('/api/jobs/{identifier}/pages/{number}.png')
    def page(identifier: str, number: int):
        job = get_job(identifier)
        if not 1 <= number <= job['render_pages']:
            raise HTTPException(404, '页码超出范围。')
        path = store.pdf_path(identifier)
        cache = store.directory / identifier / 'preview' / str(job['revision']) / f'{number}.png'
        if not cache.is_file():
            with store.lock:
                if not cache.is_file():
                    with pymupdf.open(path) as document:
                        pixmap = document[number - 1].get_pixmap(matrix=pymupdf.Matrix(1.7, 1.7), alpha=False)
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        pixmap.save(cache)
        return FileResponse(cache, media_type='image/png', headers={'Cache-Control': 'no-cache'})

    @app.get('/api/jobs/{identifier}/navigation')
    def navigation(identifier: str):
        job = get_job(identifier)
        try:
            with pymupdf.open(store.pdf_path(identifier)) as document:
                names = document.resolve_names()
                targets = {name.removeprefix('book-'): value['page'] + 1 for name, value in names.items()
                           if name.startswith('book-') and 'page' in value}
                registry = Path(job.get('render_directory', '')) / 'reports' / 'render_registry.json'
                if registry.is_file():
                    # A box can move past its preceding PDF destination at a page break.
                    for item in read(registry):
                        if item['kind'] in COLOR_KINDS:
                            atom = item.get('body_start', item['start'])['atom']
                            destination = names.get('source-' + atom)
                            if destination and 'page' in destination:
                                targets[item['anchor'].removeprefix('book-')] = destination['page'] + 1
                return dict(toc=document.get_toc(), targets=targets)
        except ValueError:
            return dict(toc=[], targets={})

    @app.get('/')
    def index():
        return FileResponse(ASSETS / 'index.html')

    app.mount('/static', StaticFiles(directory=ASSETS), name='static')
    return app

"""Validate automatic OCR reuse against the configured runtime and export bytes."""
import json
from pathlib import Path
import subprocess
import shutil
import tempfile

from book_pipeline.config import ROOT, digest, read
from mineru_checkpoint import export_signature


def runtime_versions(config):
    python = Path(config['mineru_python']).expanduser()
    if not python.is_absolute():
        python = ROOT / python
    result = subprocess.run([str(python), '-c',
        'import json; from importlib.metadata import version; '
        'print(json.dumps({k+"_version":version(k) for k in ("mineru","docvortex")}))'],
        check=True, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def validate_ocr_cache(pdf, directory, config, pages, *, versions=None):
    """Reject unknown/changed exports; never rewrite an old review or checkpoint."""
    directory = Path(directory).resolve()
    manifest = read(directory/'manifest.json')
    versions = runtime_versions(config) if versions is None else versions
    expected = export_signature(digest(pdf), pages, config['mineru_tier'],
        config['mineru_ocr_mode'], config['mineru_model_source'], versions)
    if (manifest.get('export_signature') != expected
            or any(manifest.get(k) != expected[k] for k in expected if k != 'method')):
        raise ValueError('OCR 缓存签名与输入、页码、设置或运行版本不符，请使用独立输出目录')
    hashes = manifest.get('export_hashes')
    if not isinstance(hashes, dict) or 'middle_json.json' not in hashes:
        raise ValueError('OCR 缓存缺少导出文件校验信息，请使用独立输出目录')
    for name, sha in hashes.items():
        target = (directory/name).resolve()
        if (not target.is_relative_to(directory) or not target.is_file()
                or digest(target) != sha):
            raise ValueError('OCR 缓存导出文件已改变或缺失，请使用独立输出目录')
    middle = read(directory/'middle_json.json')
    if middle.get('schema')!='docvortex.middle' or middle.get('schema_version')!='2.0':
        raise ValueError('OCR 缓存正文格式不兼容')
    if [p['page_idx']+1 for p in middle['pages']] != list(pages):
        raise ValueError('OCR 缓存正文页码与请求不符')
    # Referenced assets must be covered too, rather than merely present.
    def assets(node):
        if isinstance(node, list):
            for child in node:
                assets(child)
        elif isinstance(node, dict):
            if node.get('image_path') not in (None, '') and node['image_path'] not in hashes:
                raise ValueError('OCR 缓存图片未纳入校验')
            for child in node.values():
                if isinstance(child, (list, dict)):
                    assets(child)
    assets(middle)
    return manifest


def find_ocr_cache(pdf, root, config, pages):
    """Only automatic discovery; an explicit mismatched cache still fails loudly."""
    sha, versions = digest(pdf), None
    for path in Path(root).rglob('manifest.json'):
        try:
            manifest = read(path)
            if (manifest.get('input_sha256') != sha or not manifest.get('export_signature')
                    or manifest.get('source_page_numbers') != list(pages)
                    or any(manifest.get(k) for k in
                        ('translation','corrected_source_sha256','table_prose_recovery'))):
                continue
            if versions is None:
                versions = runtime_versions(config)
            validate_ocr_cache(pdf,path.parent,config,pages,versions=versions)
            return path.parent
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def copy_ocr_cache(pdf, source, destination, config, pages):
    """Validate the copied bytes before publishing, keeping the old export intact."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ocr-reuse-',dir=destination.parent) as temporary:
        copied = Path(temporary)/'export'
        shutil.copytree(source,copied)
        validate_ocr_cache(pdf,copied,config,pages)
        copied.rename(destination)

"""Keep costly inference independent of exports and optional SDK JSON fields."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def expand_empty_pages(middle):
    """SDK skip_defaults omits blocks=[]; preserve page indices and all content."""
    empty=[]
    for page in middle['pages']:
        if 'blocks' not in page:page['blocks']=[]
        if not isinstance(page['blocks'],list):
            raise ValueError(f"MinerU page {page.get('page_idx')} has invalid blocks")
        if not page['blocks']:empty.append(page['page_idx']+1)
    return empty


def checkpoint_path(root,signature):
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    return Path(root)/key


def write_checkpoint(folder,signature,payload):
    folder.mkdir(parents=True,exist_ok=True)
    temporary=folder/'inference.json.gz.tmp'
    with gzip.open(temporary,'wt',encoding='utf-8',compresslevel=1) as stream:
        json.dump(payload,stream,ensure_ascii=False)
    temporary.replace(folder/'inference.json.gz')
    metadata={'signature':signature,'payload_sha256':sha256(folder/'inference.json.gz')}
    temp_metadata=folder/'checkpoint.json.tmp'
    temp_metadata.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp_metadata.replace(folder/'checkpoint.json')


def read_checkpoint(folder,signature):
    metadata_path=folder/'checkpoint.json'
    if not metadata_path.exists():return None
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    data=folder/'inference.json.gz'
    if metadata.get('signature')!=signature or not data.is_file() or sha256(data)!=metadata.get('payload_sha256'):
        raise ValueError(f'Inference checkpoint is changed/incomplete: {folder}; use --no-resume or --refresh-inference')
    with gzip.open(data,'rt',encoding='utf-8') as stream:
        return json.load(stream)

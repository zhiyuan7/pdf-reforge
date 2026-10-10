"""Keep costly inference independent of exports and optional SDK JSON fields."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from copy import deepcopy
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
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(folder/'inference.json.gz')
    metadata={'signature':signature,'payload_sha256':sha256(folder/'inference.json.gz')}
    temp_metadata=folder/'checkpoint.json.tmp'
    temp_metadata.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with temp_metadata.open('rb') as stream:
        os.fsync(stream.fileno())
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


def validate_page_payload(payload, page_idx):
    """A completed page includes its original index and embedded image bytes."""
    middle = payload['middle_json']
    if [p['page_idx'] for p in middle['pages']] != [page_idx]:
        raise ValueError('Single-page inference returned different source pages')
    model = payload.get('model_output')
    if model is not None:
        mapping = model.get('page_index_map', [])
        if len(model['pages']) != 1 or not (mapping == [page_idx] or (page_idx == 0 and mapping == [])):
            raise ValueError('Single-page model output has a different page mapping')


def merge_page_payloads(payloads, page_indices, source_page_count):
    """Join SDK page trees and per-page layout metadata without losing assets."""
    if not payloads or len(payloads) != len(page_indices):
        raise ValueError('Missing recognized pages')
    for payload, page_idx in zip(payloads, page_indices):
        validate_page_payload(payload, page_idx)
    full = page_indices == list(range(source_page_count))
    result = {}
    for name in ('middle_json', 'model_output'):
        documents = [p.get(name) for p in payloads]
        if all(d is None for d in documents):
            result[name] = None
            continue
        if any(d is None for d in documents):
            raise ValueError('Some pages are missing model output')
        merged = deepcopy(documents[0])
        merged['pages'] = [deepcopy(page) for d in documents for page in d['pages']]
        if name == 'middle_json':
            merged['is_full_document'] = full
        else:
            merged['page_index_map'] = [] if full else list(page_indices)
        # SDK extensions carry the physical page dimensions used during export.
        layout = merged.get('extensions', {}).get('docvortex_layout')
        if layout is not None:
            layout['pages'] = [deepcopy(page) for d in documents
                               for page in d['extensions']['docvortex_layout']['pages']]
        result[name] = merged
    return result


def recognize_pages(page_indices, source_page_count, signature, folder, parse_page,
                    *, refresh=False, progress=None, check_source=None):
    """Commit each finished page before starting the next; resume verified pages."""
    if not page_indices or page_indices != sorted(set(page_indices)) or any(
            type(p) is not int or not 0 <= p < source_page_count for p in page_indices):
        raise ValueError('Requested page indices must be unique, ordered and in the PDF')
    payloads, reused = [], []
    for ordinal, page_idx in enumerate(page_indices):
        page_signature = {**signature, 'page_idx': page_idx}
        page_folder = Path(folder) / f'page-{page_idx + 1:06d}'
        payload = None
        if not refresh:
            try:
                payload = read_checkpoint(page_folder, page_signature)
                if payload is not None:
                    validate_page_payload(payload, page_idx)
            except (OSError, ValueError, KeyError, TypeError, EOFError):
                # An interrupted/corrupt page is redone independently. Other
                # verified page files and previous complete exports are kept.
                payload = None
        if payload is None:
            if progress:
                progress('识别单页', ordinal, len(page_indices), page_idx + 1)
            payload = parse_page(page_idx)
            validate_page_payload(payload, page_idx)
            if check_source:
                check_source()
            write_checkpoint(page_folder, page_signature, payload)
            phase = '逐页保存 MinerU 结果'
        else:
            reused.append(page_idx + 1)
            phase = '复用 MinerU 页面缓存'
        payloads.append(payload)
        if progress:
            progress(phase, ordinal + 1, len(page_indices), page_idx + 1)
    return merge_page_payloads(payloads, page_indices, source_page_count), reused

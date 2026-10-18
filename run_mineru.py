#!/usr/bin/env python3
"""使用 MinerU 4 本地识别 PDF，并保存完整的 Markdown/JSON/图片结果包。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

from pipeline_progress import emit
from mineru_checkpoint import checkpoint_path, expand_empty_pages, read_checkpoint, write_checkpoint, sha256, recognize_pages, normalize_middle_json


ROOT = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="结果目录，默认 output/mineru/<PDF文件名>")
    parser.add_argument("--tier", choices=("flash", "basic", "standard", "advanced"), default="standard")
    parser.add_argument("--pages", default="all", help="全部页面用 all；部分页面如 1-3,5（从 1 开始）")
    parser.add_argument("--ocr-mode", choices=("auto", "txt", "ocr"), default="auto")
    parser.add_argument('--checkpoint-dir',type=Path,help='逐页持久缓存目录；中断后复用已完成的页面')
    parser.add_argument('--refresh-inference',action='store_true',help='重新推理，忽略已有推理缓存')
    parser.add_argument(
        "--model-source", choices=("auto", "huggingface", "modelscope", "local"),
        default=os.environ.get("MINERU_MODEL_SOURCE", "auto"),
        help="模型下载来源；local 仅使用已下载的模型，PDF 始终在本地识别",
    )
    return parser.parse_args()


def write_paged_markdown(middle: dict, destination: Path) -> None:
    """用 MinerU 完整视图保留每页及页眉页脚，附上公式/表格原图供核对。"""
    from mineru.parser import ParseResult
    from mineru.render import RenderMode

    sections = []
    for page in middle["pages"]:
        # 分别渲染页面，防止非连续书页被默认 renderer 合并为一个段落。
        page_result = ParseResult.from_dict({**middle, "pages": [page], "is_full_document": False})
        body = page_result.markdown(mode=RenderMode.FULL)
        section = f"# PDF 第 {page['page_idx'] + 1} 页\n\n{body}"
        originals = []
        for block in page.get("blocks", []):
            if block["type"] == "equation" and block.get("image_path"):
                tag = re.search(r"\\tag\{([^}]+)\}", block.get("content", ""))
                label = f"公式（{tag.group(1)}）" if tag else f"公式块 {block['index']}"
                originals.append(f"- [{label}原图]({block['image_path']})")
            elif block["type"] == "table":
                for child in block.get("content", []):
                    if child.get("image_path"):
                        originals.append(f"- [表格块 {block['index']} 原图]({child['image_path']})")
        if originals:
            section += "\n\n公式与表格原图（用于核对自动识别）：\n\n" + "\n".join(originals)
        sections.append(section)
    destination.write_text("\n\n---\n\n".join(sections) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    pdf = args.pdf.expanduser().resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError(f"找不到 PDF 文件：{pdf}")
    with pdf.open("rb") as stream:
        if b"%PDF-" not in stream.read(1024):
            raise ValueError(f"文件不是有效的 PDF：{pdf}")

    output = (args.output or ROOT / "output" / "mineru" / pdf.stem).expanduser().resolve()
    archive = output.with_name(output.name + "_results.zip")
    # 避免把旧结果误认为本次识别成功；重跑时指定新的 -o 目录。
    if output.exists() or archive.exists():
        raise FileExistsError(f"结果目录或压缩包已存在，请使用 -o 指定新目录：{output}")

    # 必须在导入 MinerU 前配置，避免模块初始化时读取到其他设置。
    os.environ["MINERU_MODEL_SOURCE"] = args.model_source
    from mineru.parser import parse
    from mineru.parser.page_range import parse_page_range
    from mineru.parser.writer import FileBasedDataWriter
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(str(pdf)) as document:
        source_page_count = len(document)
    print(f"输入：{pdf}（{source_page_count} 页）", flush=True)
    print(f"MinerU {version('mineru')} / tier={args.tier} / pages={args.pages}", flush=True)
    started = time.monotonic()
    input_sha256=sha256(pdf)
    legacy_signature={'schema_version':1,'input_sha256':input_sha256,'source_page_count':source_page_count,
        'tier':args.tier,'ocr_mode':args.ocr_mode,'page_range':args.pages,'model_source':args.model_source,
        'mineru_version':version('mineru'),'docvortex_version':version('docvortex')}
    signature={**legacy_signature,'schema_version':2,'inference_mode':'pagewise_v1'}
    page_indices=parse_page_range(args.pages,source_page_count)
    cache_root=getattr(args,'checkpoint_dir',None) or output.parent/'.cache/mineru_inference'
    cache_root=Path(cache_root).expanduser().resolve()
    checkpoint=checkpoint_path(cache_root,signature)
    page_signature={k:v for k,v in signature.items() if k!='page_range'}
    page_cache=checkpoint_path(cache_root/'pages',page_signature)
    payload=None
    refresh=getattr(args,'refresh_inference',False)
    if not refresh:
        for path, stamp in ((checkpoint,signature),(checkpoint_path(cache_root,legacy_signature),legacy_signature)):
            try:
                payload=read_checkpoint(path,stamp)
            except (OSError,ValueError,EOFError):
                print(f'整本缓存未通过校验，改用逐页缓存恢复：{path}',flush=True)
            if payload is not None:
                checkpoint=path
                break
    inference_reused=payload is not None
    reused_pages=[p+1 for p in page_indices] if inference_reused else []
    if inference_reused:
        from mineru.parser import ParseResult
        from mineru.types import ModelJson
        payload['middle_json']=normalize_middle_json(payload['middle_json'])
        parsed=ParseResult.from_dict(payload['middle_json'])
        model=ModelJson.from_dict(payload['model_output']) if payload.get('model_output') is not None else None
        result=ParseResult(parsed.middle_json,model)
        print(f'已校验并复用 MinerU 推理缓存，继续导出：{checkpoint}',flush=True)
        emit('复用已完成的 MinerU 推理',len(page_indices),len(page_indices))
    else:
        emit('MinerU 加载与识别',0,len(page_indices))
        source_stamp=pdf.stat()
        def check_source():
            current=pdf.stat()
            if (current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns)!=(source_stamp.st_ino,source_stamp.st_size,source_stamp.st_mtime_ns,source_stamp.st_ctime_ns):
                raise RuntimeError('输入 PDF 在推理期间发生变化，未提交该页缓存')
        def parse_page(page_idx):
            parsed=parse(str(pdf),tier=args.tier,ocr_mode=args.ocr_mode,page_range=str(page_idx+1))
            return {'middle_json':normalize_middle_json(parsed.middle_json.to_dict(skip_defaults=False)),
                    'model_output':parsed._model_output.to_dict(skip_defaults=False) if parsed._model_output is not None else None}
        def page_progress(phase,done,total,page_number):
            emit(phase,done,total,detail=f'原 PDF 第 {page_number} 页',page_number=page_number)
            if phase!='识别单页':
                print(f'{phase}：第 {page_number} 页（已保存 {done}/{total} 页）',flush=True)
        try:
            payload,reused_pages=recognize_pages(page_indices,source_page_count,page_signature,page_cache,parse_page,
                                                refresh=refresh,progress=page_progress,check_source=check_source)
        except BaseException:
            print(f'识别已中断；已完成页面保留在 {page_cache}，以相同参数重跑即可继续。',file=sys.stderr,flush=True)
            raise
        if sha256(pdf)!=input_sha256:raise RuntimeError('输入 PDF 在推理期间发生变化，未提交整本缓存')
        from mineru.parser import ParseResult
        from mineru.types import ModelJson
        payload['middle_json']=normalize_middle_json(payload['middle_json'])
        parsed=ParseResult.from_dict(payload['middle_json'])
        model=ModelJson.from_dict(payload['model_output']) if payload.get('model_output') is not None else None
        result=ParseResult(parsed.middle_json,model)
        emit('持久保存 MinerU 推理',len(page_indices),len(page_indices))
        write_checkpoint(checkpoint,signature,payload)
        print(f'MinerU 推理已持久保存：{checkpoint}',flush=True)
        del payload
    emit('保存 MinerU 结果',len(page_indices),len(page_indices))

    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep unfinished exports on disk as well. Only a validated directory is
    # promoted to output; exceptions never delete the expensive inference cache.
    work=output.with_name(output.name+'.work')
    work.mkdir(parents=True,exist_ok=True)
    try:
        staging = work / "result"
        if staging.exists():shutil.rmtree(staging)
        result.save(FileBasedDataWriter(str(staging)))
        middle = json.loads((staging / "middle_json.json").read_text(encoding="utf-8"))
        # Retain exact SDK serialization and expand only its missing empty lists
        # in the pipeline input; later font/layout consumers also expect blocks.
        shutil.copy2(staging/'middle_json.json',staging/'middle_json.raw.json')
        empty_pages=expand_empty_pages(middle)
        (staging/'middle_json.json').write_text(json.dumps(middle,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        structured = json.loads((staging / "structured_content.json").read_text(encoding="utf-8"))
        write_paged_markdown(middle, staging / "markdown_paged.md")
        pages = middle["pages"]
        page_indices = [page["page_idx"] for page in pages]
        if args.pages == "all":
            if sorted(page_indices) != list(range(source_page_count)) or not middle["is_full_document"]:
                raise RuntimeError(f"识别页数不完整：输入 {source_page_count} 页，输出页号 {page_indices}")
        if [page["page_idx"] for page in structured["pages"]] != page_indices:
            raise RuntimeError("两份 JSON 的页面索引不一致")
        markdown = (staging / "markdown.md").read_text(encoding="utf-8")
        block_counts = Counter(block["type"] for page in pages for block in page["blocks"])
        image_files = sorted(path for path in (staging / "images").rglob("*") if path.is_file())
        manifest = {
            "input_pdf": str(pdf),
            "input_sha256": input_sha256,
            "mineru_version": version("mineru"),
            "tier": args.tier,
            "ocr_mode": args.ocr_mode,
            "model_source": args.model_source,
            "requested_pages": args.pages,
            "source_page_count": source_page_count,
            "recognized_page_count": len(pages),
            "pages_without_recognized_blocks":empty_pages,
            "inference_checkpoint":str(checkpoint),
            "inference_reused":inference_reused,
            "page_checkpoint_directory":str(page_cache),
            "reused_page_numbers":reused_pages,
            "source_page_numbers": [index + 1 for index in page_indices],
            "is_full_document": middle["is_full_document"],
            "block_counts": dict(block_counts),
            "image_count": len(image_files),
            "markdown_characters": len(markdown),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": sorted(str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # 在临时目录中完成打包，防止出错时留下不完整的交付文件。
        temporary_archive = Path(shutil.make_archive(str(work / "package"), "zip", staging))
        staging.rename(output)
        temporary_archive.rename(archive)
    except BaseException:
        print(f'导出未完成；推理缓存和导出中间文件保留在 {checkpoint} 与 {work}',file=sys.stderr,flush=True)
        raise
    else:
        shutil.rmtree(work)

    print(f"完成：{len(pages)} 页，{len(image_files)} 张图片，{len(markdown)} 个 Markdown 字符", flush=True)
    print(f"分页 Markdown：{output / 'markdown_paged.md'}")
    print(f"MinerU 原始 Markdown：{output / 'markdown.md'}")
    print(f"结构化 JSON：{output / 'structured_content.json'}")
    print(f"完整结果包：{archive}")
    return output


def main() -> int:
    try:
        run(arguments())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"识别失败：{exc}", file=sys.stderr)
        return 1
    finally:
        # 在 Python 等待工作线程之前关闭模型；仅靠 atexit 可能滞留后台线程。
        runtime = sys.modules.get("mineru.model.vlm.runtime")
        if runtime is not None:
            runtime.shutdown_cached_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

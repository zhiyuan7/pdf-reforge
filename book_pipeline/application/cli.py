"""PDF 回炉所：按目录重建正文结构，再生成可导航的书稿 PDF。"""
import argparse
import json
from pathlib import Path
import sys

from book_pipeline.config import ROOT, load_config, read


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, help='原始 PDF；使用 --mineru-dir 时可从识别清单读取')
    parser.add_argument('--input-mode', choices=['auto', 'ocr'], help='逐页自动分流（默认）或强制 OCR')
    parser.add_argument('--translation', choices=['auto', 'off'], help='自动将英文翻译为中文（默认）或保留原文')
    parser.add_argument('--source-dir', '--mineru-dir', dest='mineru_dir', type=Path,
                        help='复用完整的原始/校正/翻译提取目录或 MinerU 结果')
    parser.add_argument('--metadata', type=Path, help='书名、作者等 YAML 配置，默认与 PDF 同名的 .book.yaml')
    parser.add_argument('--config', type=Path, default=ROOT / 'book_config.json')
    parser.add_argument('-o', '--output', type=Path, help='独立输出目录，默认 output/<PDF 文件名>')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--match-headings', action='store_true', help='第一步：读取原书目录并匹配标题')
    modes.add_argument('--process-sections', action='store_true', help='第二步：清理、跨页配对和特殊块抽取')
    modes.add_argument('--map-references', action='store_true', help='第三步：本地类型＋编号引用映射')
    modes.add_argument('--render-processed-book', action='store_true', help='第四步：排版已保存结果并生成 PDF')
    modes.add_argument('--correct-formulas', action='store_true', help='按整页宽横带二次识别与文字重叠的公式')
    parser.add_argument('--formula-corrections', type=Path, help='已完成的 formula_corrections.json；各后续阶段复用同一版本')
    parser.add_argument('--heading-report', type=Path, help='第一步结果目录')
    parser.add_argument('--processed-book-dir', type=Path, help='第二步完整结果目录')
    parser.add_argument('--mapping-report', type=Path, help='第三步结果目录')
    parser.add_argument('--section-seed', type=int, default=20261008, help='抽取六个标题范围的随机种子')
    parser.add_argument('--all-sections', action='store_true', help=argparse.SUPPRESS)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--check', action='store_true', help='检查排版环境和 API 密钥是否存在')
    checks.add_argument('--check-local', action='store_true', help='只检查本地排版环境')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.input_mode: config['input_mode'] = args.input_mode
        if args.translation: config['translation'] = args.translation
        if args.check or args.check_local:
            from book_pipeline.publishing.environment import doctor
            result = {'status': 'complete', 'tools': doctor(config, require_key=args.check)}
        else:
            if args.pdf is None and args.mineru_dir:
                args.pdf = Path(read(args.mineru_dir / 'manifest.json')['input_pdf'])
            if args.pdf is None:
                parser.error('请提供 --pdf 或 --mineru-dir')
            from .pipeline import load_book, load_metadata, match_headings, process_content, run_workflow
            progress = lambda stage, n, total: print(f'{stage}: {n}/{total}', flush=True)
            stage = ('headings' if args.match_headings else 'content' if args.process_sections else
                     'references' if args.map_references else 'pdf' if args.render_processed_book else
                     'formula_corrections' if args.correct_formulas else None)
            output = args.output or ROOT / 'output' / args.pdf.stem
            if stage and not args.output:
                output = output / stage
            if stage:
                if not args.mineru_dir:
                    parser.error('单独运行阶段时需要 --mineru-dir')
                index = load_book(args.pdf, args.mineru_dir, args.formula_corrections)
                if args.correct_formulas:
                    if index.source.get('translation'):
                        parser.error('请先对原始提取来源校正公式，再翻译；不能对中文译文重新识别英文正文')
                    if args.formula_corrections:
                        parser.error('--correct-formulas 使用原始 OCR，不能同时提供 --formula-corrections')
                    from book_pipeline.ingestion.formula_bands import correct_formula_bands
                    correction = correct_formula_bands(index, config, output, progress)
                    result = dict(status=correction['status'], regions=len(correction['records']), usage=correction['usage'])
                elif args.match_headings:
                    result = match_headings(index, load_metadata(args.pdf, args.metadata), config, output, progress)
                elif args.process_sections:
                    if not args.heading_report:
                        parser.error('--process-sections 需要 --heading-report')
                    result = process_content(index, args.heading_report, config, output, args.section_seed, progress)
                elif args.map_references:
                    if not args.processed_book_dir:
                        parser.error('--map-references 需要 --processed-book-dir')
                    from book_pipeline.structure.rule_mapping import run_rule_mapping
                    mapped = run_rule_mapping(index, args.processed_book_dir, output, heading_directory=args.heading_report, progress=progress)
                    result = {'status': 'complete', 'method': mapped['method'], **mapped['statistics'], 'usage': mapped['usage']}
                else:
                    if not args.processed_book_dir or not args.mapping_report:
                        parser.error('--render-processed-book 需要 --processed-book-dir 和 --mapping-report')
                    from book_pipeline.publishing.processed_book import render_saved_book
                    result = render_saved_book(index, args.processed_book_dir, args.mapping_report,
                                               load_metadata(args.pdf, args.metadata), config, output, progress=progress)
            else:
                result = run_workflow(args.pdf, args.mineru_dir, args.metadata, config, output, args.section_seed, progress)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['status'] in {'complete', 'complete_with_unmatched'} else 1
    except KeyboardInterrupt:
        print('已中断；有效缓存和已保存结果可供下次复用。', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print('PDF 重构失败：' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

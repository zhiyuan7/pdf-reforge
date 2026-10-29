#!/usr/bin/env python3
"""Assemble disjoint native recovery and CPU exports without API calls."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from book_pipeline.config import load_config
from book_pipeline.ingestion.source_assembly import assemble_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pdf', 'native', 'output', 'config'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--ocr', type=Path, action='append', default=[])
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = assemble_source(args.pdf, args.native, args.ocr, args.output, load_config(args.config),
                             allow_partial=args.allow_partial)
    print(f"Assembled {len(result['source_page_numbers'])}/{result['source_page_count']} pages; "
          'formula review, translation and quality acceptance pending.', flush=True)


if __name__ == '__main__':
    main()

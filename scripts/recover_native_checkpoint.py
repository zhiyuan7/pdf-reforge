#!/usr/bin/env python3
"""Recover interrupted native extraction using validated response caches only."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from book_pipeline.config import load_config
from book_pipeline.ingestion.native_checkpoint import recover_native_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    def progress(label, done, total):
        if done % 25 == 0 or done == total:
            print(f'{label}: {done}/{total}', flush=True)
    manifest = recover_native_checkpoint(args.pdf, args.cache, args.output, load_config(args.config), progress)
    print(f"Recovered {len(manifest['source_page_numbers'])}/{manifest['source_page_count']} pages; "
          'translation and quality review pending.', flush=True)


if __name__ == '__main__':
    main()

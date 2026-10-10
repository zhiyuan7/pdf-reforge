#!/usr/bin/env python3
"""Local PDF Reforge workspace."""
import argparse
from pathlib import Path

from book_pipeline.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--config', type=Path, default=ROOT / 'book_config.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'web')
    args = parser.parse_args()
    import uvicorn
    from book_pipeline.web.server import create_app
    uvicorn.run(create_app(args.config, args.output), host=args.host, port=args.port)


if __name__ == '__main__':
    main()

"""Bind image review evidence to the files actually included by generated TeX."""
from pathlib import Path

from book_pipeline.config import digest


def asset_manifest(project, assets):
    project = Path(project)
    return [dict(source=source, file=relative, source_sha256=digest(source),
                 sha256=digest(project / relative))
            for source, relative in sorted(assets.items())]


def verify_asset_manifest(project, records):
    """Report changed intermediates and included files without modifying either.

    Searchable image PDFs can intentionally differ from their raster sources;
    each side is checked against its own saved digest. An exported project may
    no longer have its original source files, but must retain all included files.
    """
    project = Path(project)
    issues = []
    for row in records:
        rendered = project / row['file']
        if not rendered.is_file():
            issues.append(dict(file=row['file'], reason='included_asset_missing'))
        elif digest(rendered) != row['sha256']:
            issues.append(dict(file=row['file'], reason='included_asset_changed'))
        source = Path(row['source'])
        if source.is_file() and digest(source) != row['source_sha256']:
            issues.append(dict(file=row['file'], source=str(source), reason='source_asset_changed'))
    return issues

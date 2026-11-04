#!/usr/bin/env python3
"""Check source syntax, dependencies and portability without CUDA or third-party Python modules."""
from pathlib import Path
import ast
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
# Mirrors src/Makefile PROJECT_CPPFLAGS; quoted includes also search their own directory.
LOCAL_INCLUDE_ROOTS = (ROOT/'src', ROOT/'include')
QUANTIZER_INCLUDE_ROOTS = (
    ROOT/'tools/quantization',
    ROOT/'third_party/cuvs_rabitq/src/neighbors',
    ROOT/'third_party/cuvs_rabitq/src',
)


def main():
    errors = []
    counts = {'python': 0, 'json': 0, 'local_includes': 0, 'package_imports': 0}
    code_extensions = {'.h', '.hpp', '.cuh', '.cpp', '.cu'}
    for path in ROOT.rglob('*'):
        # Local datasets and build/run outputs are not source files.
        if path.relative_to(ROOT).parts[0] in {'build', 'data', 'artifacts', 'outputs'}:
            continue
        if path.is_symlink():
            errors.append(f'Unexpected symlink: {path.relative_to(ROOT)}')
        if not path.is_file() or any(x in path.parts for x in ['.git', '__pycache__', '.venv']):
            continue
        if path.suffix == '.py':
            try:
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                             else [item.name for item in node.names] if isinstance(node, ast.Import) else [])
                    for name in names:
                        if name.startswith(('scripts.', 'tests.python.', 'tests.tools.')):
                            target = ROOT.joinpath(*name.split('.'))
                            if not target.with_suffix('.py').is_file() and not (target/'__init__.py').is_file():
                                errors.append(f'Unresolved package import {path.relative_to(ROOT)}: {name}')
                            counts['package_imports'] += 1
            except SyntaxError as exc:
                errors.append(str(exc))
            counts['python'] += 1
        if path.suffix == '.json':
            try: json.loads(path.read_text())
            except ValueError as exc: errors.append(f'{path}: {exc}')
            counts['json'] += 1
        if path.suffix in code_extensions:
            for include in re.findall(r'^\s*#\s*include\s+"([^"]+)"', path.read_text(), re.M):
                roots = (QUANTIZER_INCLUDE_ROOTS if path.relative_to(ROOT).parts[0] in {'tools', 'third_party'}
                         else LOCAL_INCLUDE_ROOTS)
                candidates = [path.parent/include, *(base/include for base in roots)]
                if not any(candidate.is_file() for candidate in candidates):
                    errors.append(f'Unresolved include {path.relative_to(ROOT)}: {include}')
                counts['local_includes'] += 1
        if path.suffix in code_extensions | {'.py', '.json'} and path.name != 'verify_release.py':
            if re.search(r'/(?:home/[^/]+|data/[^/]+|soft/(?:mini|ana)conda[0-9]*)/', path.read_text()):
                errors.append(f'Private machine path in {path.relative_to(ROOT)}')
    makefile = (ROOT/'src/Makefile').read_text()
    if 'PROJECT_CPPFLAGS := -I. -I$(ROOT_DIR)/include' not in makefile:
        errors.append('Native include roots changed; update the release checker with the Makefile')
    for relative in re.findall(r'\.\./tests/native/[A-Za-z0-9_]+\.(?:cpp|cu|py|h)', makefile):
        if not (ROOT/'src'/relative).is_file():
            errors.append(f'Missing Makefile test source: {relative}')
    for match in re.finditer(r'^(?:CUDA_SOURCES|HOST_SOURCES) (?:\+?=|:=) (.+)$', makefile, re.M):
        for relative in match.group(1).split():
            if not (ROOT/'src'/relative).is_file():
                errors.append(f'Missing production translation unit: {relative}')
    print(json.dumps({'checks': counts, 'errors': errors}, indent=2))
    return bool(errors)

if __name__ == '__main__':
    sys.exit(main())

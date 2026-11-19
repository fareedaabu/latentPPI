#!/usr/bin/env python3
"""Resolve one AF3 command; emit NUL-separated paths, job name, and arguments.

Arguments: command, command-base directory, input override, output override.
Shell syntax is tokenized with shlex, never executed.
"""
import json
import re
import shlex
import sys
from pathlib import Path

raw, base, input_override, output_override = sys.argv[1:]
tokens = shlex.split(raw)
if not tokens or Path(tokens.pop(0)).name != 'run_alphafold.py':
    raise SystemExit('Each command must start with run_alphafold.py (an absolute script path is also accepted)')
paths = {}
args = []
i = 0
forced = {'run_inference', 'save_embeddings', 'compress_large_output_files', 'force_output_dir'}
while i < len(tokens):
    token = tokens[i]
    key, sep, value = token.partition('=')
    key = key.removeprefix('--')
    if key in {'json_path', 'output_dir', 'model_dir', 'db_dir'}:
        if not sep:
            i += 1
            if i == len(tokens) or tokens[i].startswith('--'):
                raise SystemExit(f'Missing value for --{key}')
            value = tokens[i]
        paths[key] = value
    elif key in forced or (key.startswith('no') and key[2:] in forced):
        if not sep and i + 1 < len(tokens) and tokens[i + 1].lower() in {'true', 'false', '0', '1'}:
            i += 1
    elif key in {'input_dir', 'flagfile'} or token in {';', '&&', '||', '|', '>', '<', '&'}:
        raise SystemExit(f'Unsupported command token: {token}')
    else:
        args.append(token)
    i += 1

def absolute(value):
    path = Path(value).expanduser()
    return (Path(base) / path).resolve() if not path.is_absolute() else path.resolve()

if not paths.get('json_path') or not (output_override or paths.get('output_dir')):
    raise SystemExit('Command needs --json_path and --output_dir (or AF3_OUTPUT_DIR override)')
json_path = absolute(paths['json_path'])
if input_override:
    json_path = absolute(input_override) / json_path.name
with json_path.open() as handle:
    data = json.load(handle)
name = re.sub(r'[^a-z0-9_.-]', '', data['name'].lower().replace(' ', '_'))
if not name or name in {'.', '..'}:
    raise SystemExit('Invalid AF3 JSON name')
output = absolute(output_override or paths['output_dir'])
# Bind the whole inputs parent at its original absolute path. This preserves
# relative MSA/template paths (../msa_cache/...) and absolute paths within it.
for value in [str(json_path), str(output), name, *args]:
    sys.stdout.buffer.write(value.encode() + b'\0')

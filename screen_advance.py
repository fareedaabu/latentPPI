"""Advance completed gate passes to independent full AF3 batches."""
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import shlex
import time
import uuid

import pandas as pd
import yaml

import screen_workflow as workflow


def batch_claims(config):
    root = Path(config.inputs.output_base)
    identity = root / 'screen_identity.json'
    screen_id = json.loads(identity.read_text())['screen_id'] if identity.exists() else None
    for path in sorted((root / 'advance').glob('batch-*/selection.json')):
        claim = json.loads(path.read_text())
        if claim['screen_id'] == screen_id:
            yield path.parent, claim


def candidate_report(config, passing):
    table = passing.copy()
    table['advance_status'] = 'eligible'
    table['advance_batch'] = ''
    table['advance_recycles'] = pd.Series(index=table.index, dtype='Int64')
    table = table.set_index('system_name')
    for batch, claim in batch_claims(config):
        names = table.index.intersection(claim['systems'])
        if names.empty:
            continue
        submitted = any((batch / 'latentppi_runs').glob('run-*/submissions.jsonl'))
        table.loc[names, 'advance_status'] = 'submitted' if submitted else 'prepared'
        table.loc[names, 'advance_batch'] = batch.name
        table.loc[names, 'advance_recycles'] = claim['recycles']
        generation = (batch / 'metrics/current').resolve()
        results = generation / 'all_results.tsv'
        if results.is_file():
            scores = pd.read_csv(results, sep='\t', comment='#', dtype={'system_name': str}).set_index('system_name')
            matched = names.intersection(scores.index)
            table.loc[matched, 'advance_status'] = scores.loc[matched, 'status'].replace('pending', 'submitted')
            # Add the child measurements alongside the first-pass measurements.
            metadata = json.loads((generation / 'screen_status.json').read_text())['results']
            for col in metadata['measured_metrics']:
                if col in scores:
                    table.loc[matched, 'advance_' + col] = scores.loc[matched, col]
    if config.output.rank_confident_iptm:
        from screen_results import confidence_ranks
        for col in ('advance_ipTM_af', 'advance_plddt'):
            if col not in table:
                table[col] = float('nan')
        table['advance_rank'], table['advance_rank_group'] = confidence_ranks(
            table.advance_ipTM_af, table.advance_plddt, table.advance_status.eq('completed'))
    return table.reset_index()


def batch_command(line, source, output, recycles):
    """Retain per-task scientific flags while replacing the paths and recycle count."""
    tokens = iter(shlex.split(line))
    args = []
    for token in tokens:
        key, sep, _ = token.partition('=')
        if key in ('--json_path', '--output_dir', '--num_recycles'):
            if not sep and next(tokens, None) is None:
                raise ValueError('Missing command value for ' + key)
        else:
            args.append(token)
    args.extend([f'--json_path={source}', f'--output_dir={output}', f'--num_recycles={recycles}'])
    return shlex.join(args)


def submit_batch(config, source_report, selected, recycles, root):
    from latentppi import Paths, Report, Scorer, load_config, render, validate
    from run_pipeline import command_json_path

    rows = workflow.read_pairs(config, require_commands=True)
    lines = Path(config.paths.cmd_file).read_text().splitlines()
    names = set(selected.system_name)
    rows = [dict(row) for row in rows if row['system_name'] in names]
    if len(rows) != len(names):
        raise ValueError('Source manifest changed during advance')
    batch = root / ('batch-' + uuid.uuid4().hex[:12])
    raw = asdict(config)
    raw['runner'] = 'full'
    raw['inputs'].update(output_base=str(batch), af3_num_recycles=recycles)
    raw['paths'] = asdict(Paths(script_dir=config.paths.script_dir, af3_input_dir=config.paths.af3_input_dir,
                                summary_tsv=config.paths.summary_tsv))
    raw['scorer'] = asdict(Scorer(rank_by=source_report.rank_by))
    raw['cluster']['offset'] = 0
    raw['output'].update(metrics_dir='metrics', known_interactors='')
    raw['stages']['run_ipsae_parse'] = True
    cfg_file = batch / 'advance.yaml'
    report = Report()
    child = load_config(cfg_file, report, raw=raw)
    commands = []
    for index, row in enumerate(rows, 1):
        line = lines[row['line_number'] - 1]
        path = command_json_path(line, Path(config.inputs.output_base))
        if path is None:
            raise ValueError('Missing --json_path in source command')
        path = Path(config.paths.af3_input_dir) / path.name
        if json.loads(path.read_text())['name'] != row['system_name']:
            raise ValueError(f'Input name does not match the source manifest: {path}')
        row.update(line_number=index, json_path=str(path))
        commands.append(batch_command(line, path, child.paths.af3_output_dir, recycles))

    workflow.atomic_write(cfg_file, yaml.safe_dump(raw, sort_keys=False))
    workflow.atomic_write(child.paths.cmd_file, '\n'.join(commands) + '\n')
    with (batch / 'af3_lengths.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected.to_csv(batch / 'screen_candidates.tsv', sep='\t', index=False)
    validate(child, report)
    if report.errors:
        raise ValueError('; '.join(report.errors))
    identity = Path(config.inputs.output_base) / 'screen_identity.json'
    claim = dict(screen_id=json.loads(identity.read_text())['screen_id'], systems=sorted(names),
                 recycles=recycles)
    # Reserve the selection before sbatch. A lost scheduler reply cannot duplicate a batch.
    workflow.atomic_write(batch / 'selection.json', json.dumps(claim) + '\n')
    print(f'Advancing {len(rows)} pairs with {recycles} recycles in {batch}')
    workflow.launch(child, report, cfg_file, render)


def advance(config, report, recycles, preview, watch):
    # Hold a shared lock through sleeps as well as submissions so cleanup
    # cannot remove the source files between watcher polls.
    with workflow.submission_lock(config.inputs.output_base, filename=".latentppi_watch.lock", shared=True):
        return _advance(config, report, recycles, preview, watch)


def _advance(config, report, recycles, preview, watch):
    from screen_results import ensure_compatible_screen
    if config.runner != 'latent':
        raise ValueError('advance requires a latent source screen')
    root = Path(config.inputs.output_base) / 'advance'
    interval = max(1, config.output.refresh_seconds or 300)
    if not preview:
        ensure_compatible_screen(config, report)
    while True:
        # Query before collection so a final gate pass is collected even if
        # the source collector exits during this scan.
        jobs, warning = workflow.active_jobs(config) if watch else ([], None)
        if warning:
            raise RuntimeError(warning)
        if watch and jobs and not (root.parent / 'af3_lengths.csv').exists():
            time.sleep(interval)
            continue
        with workflow.submission_lock(root):
            busy = False
            if not preview:
                for batch, _ in batch_claims(config):
                    child = replace(config, inputs=replace(config.inputs, output_base=str(batch)))
                    active, warning = workflow.active_jobs(child)
                    if warning:
                        raise RuntimeError(warning)
                    busy |= bool(active)
            workflow.screen_state(config, write_metrics=True, rank_by=report.rank_by)
            candidates = pd.read_csv(Path(config.output.metrics_dir) / 'advanced_candidates.csv',
                                     dtype={'system_name': str}, keep_default_na=False)
            selected = candidates.loc[candidates.advance_status.eq('eligible')]
            if preview:
                print(f'{len(selected)} newly eligible pairs for {recycles} recycles')
                for name in selected.system_name:
                    print(name)
            elif busy:
                print('An advance batch is active. New candidates will be included in the next batch.')
            elif not selected.empty:
                submit_batch(config, report, selected, recycles, root)
                busy = True
                workflow.screen_state(config, write_metrics=True, rank_by=report.rank_by)
            else:
                print('No new gate passes to advance.')
        if not watch or not (jobs or busy):
            return
        time.sleep(interval)

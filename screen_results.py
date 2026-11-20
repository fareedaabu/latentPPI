"""Public screen tables. One collected master supplies every result view."""
from contextlib import contextmanager
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

import pandas as pd

from collect_metrics import finite_number, seed_key

PUBLIC_FILES = ('all_results.tsv', 'ranked_survivors.tsv', 'known_interactors.tsv',
                'advanced_candidates.csv', 'summary.txt', 'screen_status.json', 'metrics_all.csv', 'metrics_pairs.csv')
FINAL_METRICS = ('plddt', 'ipTM_af', 'ipSAE', 'pDockQ2', 'LIS', 'iLIS')


def truth(value):
    return value is True or str(value).lower() == 'true'


def value_or_none(value):
    return None if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)) else value


@contextmanager
def publication_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.publish.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def pair_candidates(config, pairs):
    path = Path(config.inputs.output_base) / 'screen_candidates.tsv'
    candidates = {}
    if path.is_file():
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                if row['system_name'] in candidates:
                    raise ValueError('Duplicate candidate: ' + row['system_name'])
                candidates[row['system_name']] = row
    for pair in pairs:
        name = pair['system_name']
        row = candidates.setdefault(name, dict(system_name=name, preprocessing_status='ready'))
        row.update(pair)
        # Older screens have no accession columns. Recover explicit MSA references,
        # since target names can contain underscores.
        source = Path(pair.get('json_path', ''))
        if not source.is_absolute():
            source = Path(config.inputs.output_base) / source
        if source.is_file():
            data = json.loads(source.read_text())
            row['expected_seeds'] = [str(s) for s in data.get('modelSeeds', [])]
            proteins = [s['protein'] for s in data.get('sequences', []) if 'protein' in s]
            if len(proteins) == 2:
                for label, protein in zip(('a', 'b'), proteins):
                    row.setdefault('chain_' + label + '_accession',
                                   Path(protein.get('unpairedMsaPath', '')).stem)
                    row.setdefault('chain_' + label + '_length', len(protein.get('sequence', '')))
                row.setdefault('bait_accession', Path(config.inputs.fasta_file).stem)
                # Screen inputs historically put the bait on chain A.
                row.setdefault('target_accession', row.get('chain_b_accession', ''))
                row.setdefault('homodimer', proteins[0].get('sequence') == proteins[1].get('sequence'))
    return candidates


def outcome(candidate, group, attempt, legacy_status, final=False):
    preprocessing = candidate.get('preprocessing_status', 'ready')
    if preprocessing != 'ready':
        return preprocessing
    if legacy_status == 'skipped_length':
        return 'skipped_length'
    if attempt and attempt.get('exit_code') not in (None, 0):
        return 'failed_oom' if attempt.get('failure_reason') == 'oom' else 'failed_other'
    if attempt and attempt.get('exit_code') is None:
        return 'incomplete' if final else 'running'
    if legacy_status in ('completed', 'gated'):
        return 'killed_at_gate' if legacy_status == 'gated' else 'completed'
    if group is not None and 'latentppi_scorer_gate_status' in group:
        errors = group.latentppi_scorer_gate_status.dropna().astype(str)
        if errors.str.lower().isin(['error', 'failed', 'fail']).any():
            return 'gate_error'
    if attempt or (group is not None and group.status.ne('not_run').any()):
        return 'incomplete'
    return 'pending' if legacy_status == 'never_run' else 'incomplete'


def confidence_ranks(values, plddt, completed):
    plddt = pd.to_numeric(plddt, errors='coerce')
    tier = pd.Series('missing_score', index=values.index)
    measured = plddt.notna() & values.notna()
    tier.loc[measured] = 'plddt_le_70'
    tier.loc[measured & plddt.gt(70)] = 'plddt_gt_70'
    ranks = pd.Series(pd.NA, index=values.index, dtype='Int64')
    offset = 0
    for label in ('plddt_gt_70', 'plddt_le_70', 'missing_score'):
        scores = values.loc[completed & tier.eq(label)]
        ranks.loc[scores.index] = offset + scores.rank(
            method='min', ascending=False, na_option='bottom').astype('Int64')
        offset += len(scores)
    return ranks, tier.where(completed)


def build_master(config, pairs, frame, attempts, state, gate_metric, *, final=False):
    reserved = {'system_name', 'status', 'rank', 'rank_pct', 'gate_score', 'gate_rank',
                'attempt_id', 'run_id', 'gate_metric', 'final_rank_by'}
    if gate_metric in reserved or gate_metric.endswith(('_pred', '_true')):
        raise ValueError('Use a metric name without a column suffix for scorer.rank_by')
    final_metric = 'ipTM_af' if config.output.rank_confident_iptm else gate_metric
    candidates = pair_candidates(config, pairs)
    groups = {name: group for name, group in frame.groupby('system_name', sort=False)}
    final_metrics = sorted(set(FINAL_METRICS) | {c[:-5] for c in frame if c.endswith('_true')})
    predicted = sorted(c for c in frame if c.endswith('_pred'))
    rows = []
    for name, candidate in candidates.items():
        group = groups.get(name)
        attempt = attempts.get(name, {})
        status = outcome(candidate, group, attempt, state['systems'].get(name, 'never_run'), final)
        row = {k: value_or_none(candidate.get(k)) for k in (
            'system_name', 'bait_accession', 'target_accession', 'chain_a_accession',
            'chain_b_accession', 'pair_key', 'chain_a_length', 'chain_b_length', 'total_length')}
        row.update(status=status, max_total_residues=config.cluster.max_total_residues,
                   homodimer=truth(candidate.get('homodimer')),
                   homodimer_rule_applied=truth(candidate.get('homodimer_rule_applied')),
                   gate_metric=gate_metric, final_rank_by=final_metric,
                   attempt_id=attempt.get('attempt_id'), run_id=attempt.get('run_id'),
                   requested_recycles=attempt.get('num_recycles'),
                   gpu_retry=attempt.get('gpu_retry', False), gpu_type=attempt.get('gpu_type'),
                   gate_input_recycle=None, gate_target_recycle=None,
                   gate_outcome='unknown', gate_score=None, gate_rank=None, rank=None, rank_pct=None,
                   n_seeds=0, n_seeds_gated=0, n_seeds_passed=0, failure_stage=attempt.get('stage') if status.startswith('failed') else None)
        checkpoint = attempt.get('checkpoint_metadata', {}).get(gate_metric, {})
        row['checkpoint_input_recycle'] = checkpoint.get('input_recycle')
        row['checkpoint_target_recycle'] = checkpoint.get('target_recycle')
        for column in predicted:
            row[column] = None
        for metric in final_metrics:
            row[metric] = None
        if group is not None:
            row['n_seeds'] = int(group.seed.notna().sum())
            row['n_seeds_gated'] = int(group.gated.fillna(False).sum())
            gates = group.gated.fillna(False).astype(bool)
            if 'latentppi_scorer_gate_status' in group and 'latentppi_scorer_gate_triggered' in group:
                passed = (group.latentppi_scorer_gate_status.eq('ok')
                          & group.latentppi_scorer_gate_triggered.eq(False))
                for key in ('latentppi_scorer_gate_enabled', 'latentppi_scorer_gate_applied'):
                    if key in group:
                        passed &= ~group[key].eq(False)
                if 'diffusion_skipped' in group:
                    passed &= ~group.diffusion_skipped.eq(True)
                row['n_seeds_passed'] = int(passed.sum())
            if gates.any():
                row['gate_outcome'] = 'killed' if gates.all() else 'mixed'
            elif 'latentppi_scorer_gate_status' in group:
                statuses = group.latentppi_scorer_gate_status.dropna().astype(str).str.lower()
                row['gate_outcome'] = 'error' if statuses.isin(['error', 'fail', 'failed']).any() else 'passed' if row['n_seeds_passed'] else 'unknown'
            for column in predicted:
                row[column] = finite_number(pd.to_numeric(group[column], errors='coerce').mean())
            row['gate_score'] = row.get(gate_metric + '_pred')
            for dest, keys in (
                ('requested_recycles', ('requested_recycles', 'num_recycles')),
                ('gate_input_recycle', ('latentppi_scorer_input_recycle',)),
                ('gate_target_recycle', ('latentppi_scorer_target_recycle',))):
                for key in keys:
                    if key in group:
                        values = group[key].dropna().astype(str).unique()
                        if len(values) == 1:
                            row[dest] = values[0]
                            break
            if status == 'completed':
                for metric in final_metrics:
                    if metric + '_true' in group:
                        row[metric] = finite_number(pd.to_numeric(group[metric + '_true'], errors='coerce').mean())
        if status == 'killed_at_gate' and finite_number(row['gate_input_recycle']) == 1:
            row['status'] = 'killed_r1'
        row['missing_final_metrics'] = ','.join(m for m in final_metrics if row[m] is None)
        rows.append(row)
    columns = list(rows[0]) if rows else [
        'system_name', 'status', 'gate_score', 'gate_rank', 'rank', 'rank_pct', 'final_rank_by', *final_metrics]
    master = pd.DataFrame(rows, columns=columns)
    master['gate_score'] = pd.to_numeric(master.gate_score, errors='coerce')
    master['gate_rank'] = master.gate_score.rank(method='min', ascending=False).astype('Int64')
    if final_metric not in master:
        master[final_metric] = float('nan')
    values = pd.to_numeric(master[final_metric], errors='coerce').where(master.status.eq('completed'))
    master['rank'] = values.rank(method='min', ascending=False).astype('Int64')
    if config.output.rank_confident_iptm:
        master['rank'], master['rank_group'] = confidence_ranks(values, master.plddt, master.status.eq('completed'))
    count = master['rank'].count()
    master['rank_pct'] = master['rank'] / count if count else None
    master.attrs['measured_metrics'] = final_metrics
    return master.sort_values('system_name', kind='stable').reset_index(drop=True)


def normalize_accession(text):
    key = text.strip().upper()
    # Preserve the existing version-insensitive lookup for RefSeq NP accessions.
    return re.sub(r'^(NP_\d+)\.\d+$', r'\1', key)


def known_interactors(path, master):
    fields = ['input_line', 'input_accession', 'normalized_accession', 'outcome', 'system_name',
              'status', 'rank', 'gate_score', 'gate_rank', 'homodimer_rule_applied', 'details']
    records, seen = [], set()
    lookup = {}
    for _, result in master.iterrows():
        target = value_or_none(result.get('target_accession'))
        accessions = [target] if target else [value_or_none(result.get(k)) for k in
                                             ('chain_a_accession', 'chain_b_accession')]
        for accession in {normalize_accession(str(a)) for a in accessions if a}:
            lookup.setdefault(accession, []).append(result)
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        if not raw.strip():
            continue
        key = normalize_accession(raw)
        row = dict(input_line=number, input_accession=raw, normalized_accession=key,
                   outcome='input problem', details='')
        if key in seen:
            row['details'] = 'Duplicate target identifier after normalization.'
        else:
            seen.add(key)
            matches = lookup.get(key, [])
            if not matches:
                row.update(outcome='not found in screened proteome', details='No matching candidate accession.')
            elif len(matches) > 1:
                row['details'] = 'Multiple screened targets match this normalized identifier.'
            else:
                result = matches[0]
                for column in ('system_name', 'status', 'rank', 'gate_score', 'gate_rank', 'homodimer_rule_applied'):
                    row[column] = value_or_none(result.get(column))
                status = result['status']
                row['outcome'] = ('killed at gate' if status in ('killed_at_gate', 'killed_r1') else
                                  'skipped for length' if status == 'skipped_length' else
                                  'failed (out of memory)' if status == 'failed_oom' else
                                  'failed (other)' if status == 'failed_other' else status.replace('_', ' '))
                if truth(result.get('homodimer_rule_applied')):
                    row['details'] = 'Homodimer rule applied; outcome shown separately.'
        records.append(row)
    return pd.DataFrame(records, columns=fields)


def summary_metadata(config, master, state, rank_by, generation):
    identity = Path(config.inputs.output_base) / 'screen_identity.json'
    screen_id = json.loads(identity.read_text()).get('screen_id') if identity.is_file() else None
    snapshots = sorted((Path(config.inputs.output_base) / 'latentppi_runs').glob('*/resolved_config.json'))
    model_info = []
    run_ids = set(master.get('run_id', pd.Series(dtype=str)).dropna())
    for path in snapshots:
        if path.parent.name in run_ids:
            data = json.loads(path.read_text())
            model_info.append(dict(run_id=path.parent.name, checkpoint_sha256=data.get('checkpoint_sha256'),
                                   checkpoint_metadata=data.get('checkpoint_metadata'),
                                   models=data.get('effective_scorer', {}).get('model_paths')))
    return dict(generation_id=generation, screen_id=screen_id, collected_at=time.time(), mode=config.runner,
                counts=dict(Counter(master.status)), final_rank_by='ipTM_af' if config.output.rank_confident_iptm else rank_by,
                rank_plddt_threshold=70 if config.output.rank_confident_iptm else None,
                measured_metrics=master.attrs.get('measured_metrics', []), gate_rank_by=rank_by, seed_aggregation='mean', sample_selection='max_ipSAE_per_seed',
                requested_recycles=config.inputs.af3_num_recycles, gate_threshold=config.scorer.threshold,
                max_total_residues=config.cluster.max_total_residues,
                checkpoint_records=model_info,
                configured_checkpoints={m: model.path for m, model in config.scorer.models.items()},
                checkpoint_recycle_metadata='See per-row gate_input_recycle/gate_target_recycle; blank means unavailable.',
                bait=config.inputs.fasta_file, proteome=config.inputs.proteome_msa_dir,
                partial=bool(master.status.isin(['pending', 'running', 'incomplete']).any()),
                excluded_unplanned_records=state.get('excluded_unplanned_records', 0))


def write_tsv(frame, path, comment):
    with path.open('w') as handle:
        handle.write('# ' + comment + '\n')
        frame.to_csv(handle, sep='\t', index=False)


def publish(config, master, state, rank_by, legacy=None):
    """Caller holds publication_lock from before collection through publication.

    Pin current.resolve() once when reading multiple files. The pointer swap is
    atomic; each retained generation is immutable and internally consistent.
    """
    directory = Path(config.output.metrics_dir)
    generations = directory / '.generations'
    current = directory / 'current'
    if current.is_symlink() and not re.fullmatch(r'\.generations/[0-9a-f]{32}', os.readlink(current)):
        raise ValueError('Refusing to replace an unrelated current symlink')
    if generations.is_symlink():
        raise ValueError('Results generation directory must not be a symlink')
    generations.mkdir(exist_ok=True)
    generation = uuid.uuid4().hex
    stage = generations / generation
    stage.mkdir()
    metadata = summary_metadata(config, master, state, rank_by, generation)
    final_metric = metadata['final_rank_by']
    cutoff = ', pLDDT > 70 first, missing scores last' if config.output.rank_confident_iptm else ''
    tag = f'generation={generation}; final_sort={final_metric} descending{cutoff}; seed_aggregation=mean; sample_selection=max_ipSAE_per_seed'
    try:
        write_tsv(master, stage / 'all_results.tsv', tag)
        survivors = master.loc[master.status.eq('completed')].sort_values(
            ['rank', 'system_name'], na_position='last', kind='stable')
        write_tsv(survivors, stage / 'ranked_survivors.tsv', tag)
        if config.runner == 'latent':
            from screen_advance import candidate_report
            passing = survivors.loc[survivors.get('n_seeds_passed', pd.Series(0, index=survivors.index)).gt(0)]
            candidate_report(config, passing).to_csv(stage / 'advanced_candidates.csv', index=False)
        if config.output.known_interactors:
            write_tsv(known_interactors(config.output.known_interactors, master),
                      stage / 'known_interactors.tsv', tag)
        if config.output.write_summary:
            lines = [f'Generation: {generation}', f'Screen ID: {metadata["screen_id"] or "unavailable for older screen"}', f'Mode: {config.runner}',
                     f'Final ranking: measured {final_metric}, descending{cutoff}',
                     f'Gate ranking: predicted {rank_by}, descending',
                     'Across seeds: mean. Within each seed: the sample with highest ipSAE.',
                     f'Requested recycles: {config.inputs.af3_num_recycles} (not a claim that every result is R10)',
                     f'Gate threshold: {config.scorer.threshold}', f'Bait: {config.inputs.fasta_file}',
                     f'Max total residues: {config.cluster.max_total_residues}',
                     f'Proteome MSA source: {config.inputs.proteome_msa_dir}',
                     f'Unfinished pairs remain: {metadata["partial"]}',
                     'Counts: ' + json.dumps(metadata['counts'], sort_keys=True),
                     'Checkpoints: ' + json.dumps(metadata['checkpoint_records'] or metadata['configured_checkpoints'], sort_keys=True),
                     'Checkpoint recycle metadata: ' + metadata['checkpoint_recycle_metadata'],
                     'Blank scores/ranks mean unavailable. Gate kills are not low final AF3 scores.',
                     'iLIS is blank unless supplied by the score source.',
                     'Full-mode gate scores are shown only if the runtime supplies them.']
            (stage / 'summary.txt').write_text('\n'.join(lines) + '\n')
        (stage / 'screen_status.json').write_text(json.dumps({**state, 'results': metadata}, indent=2, sort_keys=True) + '\n')
        if legacy is not None:
            from collect_metrics import write_csv
            write_csv(legacy[0], stage / 'metrics_all.csv', False)
            write_csv(legacy[1], stage / 'metrics_pairs.csv', False)
        # Install public aliases before the single switch; existing regular files
        # are replaceable generated outputs, never arbitrary directories.
        for name in PUBLIC_FILES:
            target = directory / name
            if (stage / name).exists():
                if target.is_symlink() and os.readlink(target) != 'current/' + name:
                    raise ValueError(f'Refusing to replace unrelated result symlink: {target}')
                link = directory / ('.link-' + generation)
                link.symlink_to('current/' + name)
                os.replace(link, target)
        pointer = directory / ('.current-' + generation)
        pointer.symlink_to('.generations/' + generation)
        if (directory / 'current').exists() and not (directory / 'current').is_symlink():
            raise ValueError('Results current path must be a generation symlink')
        os.replace(pointer, directory / 'current')
        # Optional files from a previous generation must not look current.
        for name in PUBLIC_FILES:
            target = directory / name
            if not (stage / name).exists() and target.is_symlink() and os.readlink(target) == 'current/' + name:
                target.unlink()
    except BaseException:
        # Never remove a generation that has already been published.
        if not (directory / 'current').is_symlink() or (directory / 'current').resolve() != stage.resolve():
            shutil.rmtree(stage)
        raise
    # Keep the current and previous complete sets. Periodic refreshes should not
    # accumulate another full screen table every few minutes indefinitely.
    complete = sorted((p for p in generations.iterdir() if p.is_dir() and not p.is_symlink()
                       and re.fullmatch(r'[0-9a-f]{32}', p.name)), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in complete[2:]:
        if old != stage:
            shutil.rmtree(old)
    return metadata


def ensure_compatible_screen(config, report):
    """Prevent reusing a result directory for different scientific settings."""
    from dataclasses import asdict
    from screen_workflow import atomic_write
    root = Path(config.inputs.output_base)
    path = root / 'screen_identity.json'

    def settings(raw, checkpoint_hashes):
        inputs, af3, scorer = raw['inputs'], raw['af3'], raw['scorer']
        return dict(runner=raw['runner'], fasta_file=inputs['fasta_file'],
                    proteome_msa_dir=inputs['proteome_msa_dir'],
                    af3_num_recycles=inputs['af3_num_recycles'],
                    af3_num_diffusion_samples=inputs['af3_num_diffusion_samples'],
                    max_identity=inputs['max_identity'], filter_non_standard=inputs['filter_non_standard'],
                    af3_image=af3['image'], model_parameters_dir=af3['model_parameters_dir'],
                    gate_metric=scorer['rank_by'] if raw['runner'] == 'latent' else '',
                    gate_threshold=scorer['threshold'] if raw['runner'] == 'latent' else None,
                    checkpoints=checkpoint_hashes)

    hashes = {metric: hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for metric, p in report.model_paths.items()}
    current = settings(asdict(config), hashes)
    if config.runner == 'latent':
        current['gate_metric'] = report.rank_by
    if path.exists():
        previous = json.loads(path.read_text())
        old_settings = previous['settings']
    else:
        previous = None
        snapshots = list((root / 'latentppi_runs').glob('run-*/resolved_config.json'))
        if snapshots:
            latest = max(snapshots, key=lambda p: p.stat().st_mtime)
            old = json.loads(latest.read_text())
            old_settings = settings(old['config'], old.get('checkpoint_sha256', {}))
            if old['config']['runner'] == 'latent':
                old_settings['gate_metric'] = old.get('effective_scorer', {}).get('rank_by', old_settings['gate_metric'])
        else:
            old_settings = current
    changed = [key for key in current if old_settings.get(key) != current[key]]
    fasta = Path(config.inputs.fasta_file)
    fasta_hash = hashlib.sha256(fasta.read_bytes()).hexdigest() if fasta.is_file() else None
    if previous and previous.get('fasta_sha256') != fasta_hash:
        changed.append('bait sequence contents')
    if changed:
        raise ValueError('This output directory belongs to different screen settings: ' + ', '.join(changed)
                         + '. Use a new inputs.output_base for the changed screen.')
    if not path.exists():
        atomic_write(path, json.dumps(dict(screen_id=uuid.uuid4().hex, settings=current,
                                           fasta_sha256=fasta_hash), indent=2, sort_keys=True) + '\n')

"""One short scheduler invocation per batch, with durable handoffs."""
import json
import os
from pathlib import Path
import re
import time

from gpu_routing import resource_config, tier_errors, tier_index


def dispatch(config, snapshot, env, *, force=False, preview=False):
    import screen_workflow as sw
    cluster = config.cluster
    if cluster.screen_cluster != cluster.inputs_cluster:
        raise ValueError('Chained dispatch requires matching screen_cluster and inputs_cluster; '
                         'use dispatch_mode: persistent for separate clusters')
    errors = tier_errors(cluster.gpu_tiers)
    if errors:
        raise ValueError('; '.join(errors))
    snapshot = Path(snapshot)
    path = snapshot / 'dispatch_state.json'
    token = int(env.get('LATENTPPI_DISPATCH_STEP', '0'))
    if token < 0:
        raise ValueError('Invalid dispatch step')
    state = json.loads(path.read_text()) if path.exists() else None
    acknowledged = False
    if state:
        if state.get('done') or token < state['step']:
            print('Dispatch step already handed off; no jobs submitted.')
            return []
        if token != state['step']:
            raise RuntimeError('Unexpected dispatch step; inspect dispatch_state.json')
        pending = state.get('pending')
        if pending:
            # A successfully scheduled successor may arrive after its predecessor
            # died before recording the response. Its unique step acknowledges that handoff.
            if pending['kind'] == 'continuation' and token == pending['step']:
                state['pending'] = None
                acknowledged = True
            else:
                raise RuntimeError('Unresolved submission in dispatch_state.json; inspect the queue '
                                   'before resuming with latentppi.py run. This step will not be replayed.')
    elif token:
        raise RuntimeError('Missing dispatch checkpoint for continuation')

    def save():
        if not preview:
            sw.atomic_write(path, json.dumps(state, sort_keys=True) + '\n')

    if acknowledged:
        save()

    parent = os.environ.get('SLURM_JOB_ID', '')
    parent = parent if re.fullmatch(r'\d+', parent) else None

    def continuation(after=(), delayed=False):
        deps = list(dict.fromkeys([*after, *([parent] if parent else [])]))
        next_step = state['step'] + 1
        state['step'] = next_step
        state['pending'] = dict(kind='continuation', step=next_step, depends_on=deps)
        save()  # Reserve the handoff before sbatch; ambiguous responses are never replayed.
        begin = 'now+' + str(max(1, (cluster.poll_interval + 59) // 60)) + 'minutes' if delayed else None
        try:
            job = sw.submit('dispatch', snapshot / 'dispatch.sh', None, None,
                            {**env, 'LATENTPPI_DISPATCH_STEP': str(next_step)}, deps,
                            config=config, artifact_dir=snapshot, preview=preview,
                            dependency='afterany', begin=begin)
        except sw.SubmitQuotaExceeded as exc:
            state['pending'] = None
            state['needs_resume'] = 'Continuation rejected by submission quota'
            save()
            raise RuntimeError('No queue slot for a continuation. Submitted GPU jobs continue; '
                               'rerun latentppi.py run after they finish and quota is available.') from exc
        state['pending'] = None
        state['continuation_job'] = job
        save()
        if preview:
            print(job)
            print('# This continuation wakes after the batch and dispatcher terminate; no CPU waits in between.')
        return job

    queue = [] if preview else sw.user_queue(config) if cluster.max_submit_jobs or (state and state.get('last_array')) else []
    if state and not preview:
        if state.get('last_array') in queue:
            continuation([state['last_array']])
            return []
        if state['remaining'] and cluster.max_submit_jobs:
            capacity = cluster.max_submit_jobs - cluster.submit_reserve - len(queue) - 1
            if capacity < 1:
                if len(queue) >= cluster.max_submit_jobs:
                    raise RuntimeError('Queue budget has no slot for a continuation; resume when quota is available')
                continuation(delayed=True)
                return []
    pairs = sw.read_pairs(config, require_commands=True)
    excluded = sw.length_exclusions(config, pairs)
    sw.report_length_exclusions(config, excluded)
    names = {p['line_number']: p['system_name'] for p in pairs}
    retry_path = Path(config.inputs.output_base) / 'latentppi_gpu_retries.json'
    retries = json.loads(retry_path.read_text()) if retry_path.exists() else {}
    if config.output.refresh_seconds and not preview:
        try:
            observed = sw.screen_state(config, write_metrics=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f'WARNING: results refresh failed; keeping previous results: {exc}')
            observed = sw.screen_state(config)
    else:
        observed = sw.screen_state(config)
    if state is None:
        outstanding = list(names) if force else observed['outstanding_indices']
        eligible = [i for i in outstanding if i > cluster.offset and i not in excluded]
        remaining = eligible
        if cluster.retry_failed_once:
            remaining = [i for i in eligible if names[i] not in retries
                         and (force or observed['systems'].get(names[i]) != 'failed')]
        state = dict(version=1, step=0, phase='initial', eligible=eligible,
                     remaining=remaining, pending=None, done=False)
        save()
    eligible = set(state['eligible'])
    if not eligible.issubset(names):
        raise RuntimeError('Manifest changed during dispatch; restore the original inputs')
    if eligible.intersection(excluded):
        raise RuntimeError('Length cutoff or manifest changed during dispatch; resume with a new run snapshot')
    routes = {p['line_number']: tier_index(cluster.gpu_tiers, p) for p in pairs
              if p['line_number'] in eligible} if cluster.gpu_tiers else {}
    if not state['remaining'] and state['phase'] == 'initial' and cluster.retry_failed_once:
        state['phase'] = 'retry'
        state['remaining'] = [i for i in state['eligible'] if names[i] not in retries
                              and observed['systems'].get(names[i]) not in ('completed', 'gated')]
        save()
    if not state['remaining']:
        state['pending'] = dict(kind='collector')
        save()
        job = sw.submit('collector', snapshot / 'collector.sh', None, None, env,
                        [parent] if parent else [], config=config, artifact_dir=snapshot,
                        preview=preview, dependency='afterany')
        state.update(done=True, pending=None, collector_job=job)
        save()
        if preview:
            print(job)
        return []

    capacity = len(state['remaining'])
    if cluster.max_submit_jobs:
        # Keep one extra slot for the successor while this dispatcher still runs.
        capacity = cluster.max_submit_jobs - cluster.submit_reserve - len(queue) - 1
        if capacity < 1:
            if not preview and len(queue) >= cluster.max_submit_jobs:
                raise RuntimeError('Queue budget has no slot for a continuation; resume when quota is available')
            continuation(delayed=True)
            return []
    retry = state['phase'] == 'retry'
    tier = None
    submit_config = config
    candidates = state['remaining'][:capacity]
    if cluster.gpu_tiers:
        route = len(cluster.gpu_tiers) - 1 if retry else routes[state['remaining'][0]]
        tier = cluster.gpu_tiers[route]
        submit_config = resource_config(config, tier)
        candidates = [i for i in state['remaining'] if retry or routes[i] == route][:capacity]
    elif retry:
        submit_config = resource_config(config, dict(partition=cluster.retry_partition, gres=cluster.retry_gres))
    offset, spec = sw.chunk_indices(candidates, sw.array_limit(config), cluster.array_spec_limit,
                                   cluster.array_chunk, cluster.offset)[0]
    count = sum(int(b[-1]) - int(b[0]) + 1 for b in
                (part.split('-') for part in spec.split('%')[0].split(',')))
    selected = candidates[:count]
    if retry and not preview:
        for i in selected:
            retries[names[i]] = dict(state='reserved', reserved_at=time.time(),
                                    partition=submit_config.cluster.partition, gres=submit_config.cluster.gres)
        sw.atomic_write(retry_path, json.dumps(retries, sort_keys=True) + '\n')
    state['pending'] = dict(kind='array', indices=selected, retry=retry)
    save()
    runner = Path(config.paths.script_dir) / (
        'run_af3_latent_compress_pool.sh' if config.runner == 'latent' else 'run_af3_compress_pool.sh')
    route_env = (dict(LATENTPPI_GPU_TIER=tier['name'], LATENTPPI_GPU_ROUTE_REASON='retry' if retry else 'total_length')
                 if tier else {})
    try:
        job = sw.submit('array', runner, Path(config.paths.cmd_file), spec,
                        {**env, **route_env, 'OFFSET': str(offset), 'LATENTPPI_GPU_RETRY': str(int(retry))}, [],
                        config=submit_config, artifact_dir=snapshot, preview=preview)
    except sw.SubmitQuotaExceeded:
        if retry and not preview:
            for i in selected:
                retries.pop(names[i], None)
            sw.atomic_write(retry_path, json.dumps(retries, sort_keys=True) + '\n')
        state['pending'] = None
        save()
        continuation(delayed=True)
        return []
    # All other submission errors leave the reservation intact for operator inspection.
    state['pending']['job_id'] = job
    save()
    if retry and not preview:
        for i in selected:
            retries[names[i]].update(state='submitted', job_id=job)
        sw.atomic_write(retry_path, json.dumps(retries, sort_keys=True) + '\n')
    accepted = set(selected)
    state['remaining'] = [i for i in state['remaining'] if i not in accepted]
    state['last_array'] = job
    if preview:
        print(job)
    continuation(['<array-job-id>' if preview else job])
    return [job]

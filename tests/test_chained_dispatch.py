"""Dependency handoffs without a live scheduler or GPU inference."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import latentppi as lp
import screen_workflow as sw


class ChainedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / 'run-test'
        self.snapshot.mkdir()
        self.config = lp.Config(runner='full')
        self.config.inputs.output_base = str(self.root)
        self.config.inputs.fasta_file = str(self.root / 'bait.fasta')
        self.config.inputs.proteome_msa_dir = str(self.root / 'msas')
        self.config.paths.cmd_file = str(self.root / 'commands.txt')
        self.config.paths.script_dir = str(Path(__file__).resolve().parents[1])
        self.config.output.refresh_seconds = 0
        self.config.cluster.gpu_tiers = [
            dict(name='regular', max_total_residues=3000, partition='regular', gres='gpu:l40:1'),
            dict(name='large', partition='large', gres='gpu:big:1')]
        self.config.cluster.retry_failed_once = True
        self.config.paths.af3_output_dir = str(self.root / 'predictions')
        with (self.root / 'af3_lengths.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=['line_number', 'system_name', 'total_length'])
            writer.writeheader()
            for i, length in enumerate([2999, 3000, 3001, 1000, 7000], 1):
                writer.writerow(dict(line_number=i, system_name=f'pair{i}', total_length=length))
        Path(self.config.paths.cmd_file).write_text('command\n' * 5)
        self.observed = dict(completed=0, outstanding_indices=list(range(1, 6)),
                             systems={f'pair{i}': 'never_run' for i in range(1, 6)})
        self.calls = []
        patches = [patch.object(sw, 'screen_state', side_effect=lambda *a, **k: self.observed),
                   patch.object(sw, 'array_limit', return_value=1001),
                   patch.object(sw, 'user_queue', return_value=[]),
                   patch.object(sw, 'submit', side_effect=self.submit),
                   patch.object(sw.time, 'sleep', side_effect=AssertionError('dispatcher must not sleep')),
                   patch.object(sw, 'wait_for_array', side_effect=AssertionError('dispatcher must not wait')),
                   patch.dict(os.environ, SLURM_JOB_ID='900')]
        self.mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)

    def submit(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return str(100 + len(self.calls))

    def dispatch(self, step=0, **kwargs):
        return sw.dispatch(self.config, self.snapshot, {'LATENTPPI_DISPATCH_STEP': str(step)}, **kwargs)

    def state(self):
        return json.loads((self.snapshot / 'dispatch_state.json').read_text())

    def test_length_cap_applies_to_force_continuations_and_retry(self):
        self.config.cluster.max_total_residues = 3000
        for step in range(3):
            self.dispatch(step, force=True)
        arrays = [c for c in self.calls if c[0][0] == 'array']
        self.assertEqual([c[0][3] for c in arrays], ['1-2,4%4', '1-2,4%4'])
        self.assertEqual([c[0][4]['LATENTPPI_GPU_RETRY'] for c in arrays], ['0', '1'])
        self.assertEqual(self.state()['eligible'], [1, 2, 4])
        self.assertTrue(self.state()['done'])

    def test_length_cap_all_excluded_only_collects(self):
        self.config.cluster.max_total_residues = 1
        self.dispatch(force=True)
        self.assertEqual([c[0][0] for c in self.calls], ['collector'])
        self.assertTrue(self.state()['done'])

    def test_length_cap_rejects_missing_length_before_submission(self):
        self.config.cluster.max_total_residues = 3000
        path = self.root / 'af3_lengths.csv'
        path.write_text(path.read_text().replace('2999', ''))
        with self.assertRaisesRegex(ValueError, 'total_length'):
            self.dispatch()
        self.assertEqual(self.calls, [])

    def test_tier_batches_then_exactly_one_retry_then_collection(self):
        for step in range(4):
            self.dispatch(step)
        self.assertEqual([c[0][0] for c in self.calls],
                         ['array', 'dispatch', 'array', 'dispatch', 'array', 'dispatch', 'collector'])
        arrays = [c for c in self.calls if c[0][0] == 'array']
        self.assertEqual([c[0][3] for c in arrays], ['1-2,4%4', '3,5%4', '1-5%4'])
        self.assertEqual([c[1]['config'].cluster.partition for c in arrays], ['regular', 'large', 'large'])
        self.assertEqual([c[0][4]['LATENTPPI_GPU_RETRY'] for c in arrays], ['0', '0', '1'])
        first_continuation = self.calls[1]
        self.assertEqual(first_continuation[0][5], ['101', '900'])
        self.assertEqual(first_continuation[1]['dependency'], 'afterany')
        self.assertEqual(first_continuation[0][4]['LATENTPPI_DISPATCH_STEP'], '1')
        self.assertTrue(self.state()['done'])
        before = len(self.calls)
        self.dispatch(3)
        self.dispatch(0)
        self.assertEqual(len(self.calls), before)
        retries = json.loads((self.root / 'latentppi_gpu_retries.json').read_text())
        self.assertEqual(len(retries), 5)

    def test_gate_exclusion_and_force_applies_only_at_initialization(self):
        self.dispatch(0, force=True)
        self.dispatch(1, force=True)
        self.observed['systems'].update(pair1='completed', pair2='gated', pair4='completed')
        self.dispatch(2, force=True)
        self.assertEqual(self.calls[-2][0][3], '3,5%4')
        self.dispatch(3, force=True)
        self.assertTrue(self.state()['done'])

    def test_restart_uses_saved_remaining_indices(self):
        self.dispatch()
        self.dispatch(0)  # Duplicate old invocation must not submit again.
        self.assertEqual(len(self.calls), 2)
        self.dispatch(1)
        self.assertEqual(self.calls[2][0][3], '3,5%4')

    def test_ambiguous_array_response_is_never_replayed(self):
        self.mocks[3].side_effect = RuntimeError('lost response')
        with self.assertRaisesRegex(RuntimeError, 'lost response'):
            self.dispatch()
        self.assertEqual(self.state()['pending']['kind'], 'array')
        self.mocks[3].side_effect = self.submit
        with self.assertRaisesRegex(RuntimeError, 'Unresolved submission'):
            self.dispatch()
        self.assertFalse(self.calls)

    def test_ambiguous_continuation_can_be_acknowledged_by_successor(self):
        def submit(*a, **kw):
            if a[0] == 'dispatch':
                raise RuntimeError('lost continuation response')
            return self.submit(*a, **kw)
        self.mocks[3].side_effect = submit
        with self.assertRaises(RuntimeError):
            self.dispatch()
        self.assertEqual(self.state()['pending']['kind'], 'continuation')
        self.mocks[3].side_effect = self.submit
        self.dispatch(1)
        self.assertEqual(self.calls[1][0][3], '3,5%4')

    def test_full_queue_defers_without_sleeping_or_reserving_retry(self):
        self.config.cluster.max_submit_jobs = 6
        self.config.cluster.submit_reserve = 2
        self.mocks[2].return_value = ['900', '800', '800']
        self.dispatch()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0][0], 'dispatch')
        self.assertEqual(self.calls[0][1]['begin'], 'now+1minutes')
        self.assertEqual(self.state()['remaining'], [1, 2, 3, 4, 5])
        self.assertFalse((self.root / 'latentppi_gpu_retries.json').exists())

    def test_capacity_reserves_continuation_slot(self):
        self.config.cluster.max_submit_jobs = 6
        self.config.cluster.submit_reserve = 2
        self.mocks[2].return_value = ['900']
        self.dispatch()
        self.assertEqual(self.calls[0][0][3], '1-2%4')

    def test_explicit_quota_rejection_schedules_delayed_continuation(self):
        def submit(*a, **kw):
            if a[0] == 'array':
                raise sw.SubmitQuotaExceeded('quota')
            return self.submit(*a, **kw)
        self.mocks[3].side_effect = submit
        self.dispatch()
        self.assertEqual(self.calls[0][0][0], 'dispatch')
        self.assertEqual(self.state()['remaining'], [1, 2, 3, 4, 5])

    def test_retry_rejection_releases_budget(self):
        self.observed['systems'] = {f'pair{i}': 'failed' for i in range(1, 6)}
        def submit(*a, **kw):
            if a[0] == 'array':
                raise sw.SubmitQuotaExceeded('quota')
            return self.submit(*a, **kw)
        self.mocks[3].side_effect = submit
        self.dispatch()
        self.assertEqual(json.loads((self.root / 'latentppi_gpu_retries.json').read_text()), {})

    def test_no_continuation_slot_reports_resume(self):
        def submit(*a, **kw):
            if a[0] == 'dispatch':
                raise sw.SubmitQuotaExceeded('quota')
            return self.submit(*a, **kw)
        self.mocks[3].side_effect = submit
        with self.assertRaisesRegex(RuntimeError, 'rerun latentppi.py run'):
            self.dispatch()
        self.assertEqual(self.state()['last_array'], '101')
        self.assertEqual(self.state()['remaining'], [3, 5])
        self.assertIn('needs_resume', self.state())

    def test_running_previous_batch_does_not_allow_next_array(self):
        self.dispatch()
        self.mocks[2].return_value = ['101']
        self.dispatch(1)
        self.assertEqual(self.calls[-1][0][0], 'dispatch')
        self.assertEqual(self.calls[-1][0][5], ['101', '900'])
        self.assertEqual(self.state()['remaining'], [3, 5])

    def test_preview_does_not_persist_or_submit(self):
        self.dispatch(preview=True)
        self.assertFalse((self.snapshot / 'dispatch_state.json').exists())
        self.assertFalse((self.root / 'latentppi_gpu_retries.json').exists())
        self.assertTrue(all(c[1]['preview'] for c in self.calls))

    def test_old_snapshot_preserves_mode_and_resources(self):
        old = dict(cluster=dict(cpus_per_task=4, dispatcher_time_limit='7-00:00:00'))
        raw = lp.snapshot_config(old)
        self.assertNotIn('dispatch_mode', old['cluster'])
        self.assertEqual(raw['cluster']['dispatch_mode'], 'persistent')
        self.assertEqual(raw['cluster']['dispatcher_cpus_per_task'], 4)
        self.assertEqual(raw['cluster']['dispatcher_time_limit'], '7-00:00:00')
        self.assertEqual(lp.Config().cluster.dispatch_mode, 'chained')

    def test_unknown_submission_blocks_new_run(self):
        path = self.root / 'latentppi_runs' / 'run-old' / 'dispatch_state.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(pending=dict(kind='continuation', step=1))))
        with self.assertRaisesRegex(RuntimeError, 'Unresolved continuation'):
            sw.check_chained_handoffs(self.root)
        path.write_text(json.dumps(dict(pending=dict(kind='array', job_id='123'))))
        sw.check_chained_handoffs(self.root)  # Known jobs are checked by active_jobs.

    def test_repeat_quota_deferral_does_not_rescan_results(self):
        self.config.cluster.max_submit_jobs = 6
        self.config.cluster.submit_reserve = 2
        self.mocks[2].return_value = ['900', '800', '800']
        self.dispatch()
        calls = self.mocks[0].call_count
        self.dispatch(1)
        self.assertEqual(self.mocks[0].call_count, calls)

    def test_refresh_failure_keeps_scheduling(self):
        self.config.output.refresh_seconds = 300
        self.mocks[0].side_effect = [OSError('publication failed'), self.observed]
        self.dispatch()
        self.assertEqual([c[0][0] for c in self.calls], ['array', 'dispatch'])

    def test_internal_command_passes_continuation_step(self):
        from dataclasses import asdict
        (self.snapshot / 'resolved_config.json').write_text(json.dumps(dict(
            config=asdict(self.config), environment={}, effective_scorer=dict(rank_by=''))))
        (self.snapshot / 'run_options.json').write_text(json.dumps(dict(force=False)))
        with patch.dict(os.environ, LATENTPPI_DISPATCH_STEP='2'), patch.object(sw, 'dispatch') as dispatch:
            self.assertEqual(lp.main(['_dispatch', str(self.snapshot)]), 0)
        self.assertEqual(dispatch.call_args.args[2]['LATENTPPI_DISPATCH_STEP'], '2')
        self.assertEqual(dispatch.call_args.args[0].cluster.dispatch_mode, 'chained')

    def test_submit_requests_one_cpu_and_begin(self):
        with patch.object(sw.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '555', '')) as run:
            submit_impl(self.config, self.snapshot, run)


def submit_impl(config, snapshot, run):
    # Saved before tests install their mocks.
    ORIGINAL_SUBMIT('dispatch', snapshot / 'dispatch.sh', None, None, {}, ['123'],
                    config=config, artifact_dir=snapshot, dependency='afterany', begin='now+1minutes')
    args = run.call_args.args[0]
    assert '--cpus-per-task=1' in args
    assert '--begin=now+1minutes' in args
    assert '--dependency=afterany:123' in args


ORIGINAL_SUBMIT = sw.submit

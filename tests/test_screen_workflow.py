"""Exercise orchestration without submitting jobs or running model inference."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import latentppi as lp
import screen_workflow as sw
import query_msa
import record_run_env


class WorkflowTests(unittest.TestCase):
    def test_cpu_submission_bypasses_gpu_directives(self):
        runner = REPO / 'run_af3_compress_pool.sh'
        for cpu in (True, False):
            with self.subTest(cpu=cpu):
                self.config.cluster.cpu_only = cpu
                self.config.cluster.partition = 'your_cpu,your_cpu_alt' if cpu else 'your_gpu,your_gpu_alt'
                with patch.object(sw.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '123', '')) as run:
                    sw.submit('array', runner, Path(self.config.paths.cmd_file), '1-2%2', {}, [],
                              config=self.config, artifact_dir=self.snapshot)
                args = run.call_args.args[0]
                self.assertIn('--partition=' + self.config.cluster.partition, args)
                self.assertEqual(run.call_args.kwargs['env']['AF3_CPU_ONLY'], str(int(cpu)))
                self.assertEqual(any(a.startswith('--gres=') for a in args), not cpu)
                if cpu:
                    wrapper = self.snapshot / 'cpu_array.sh'
                    self.assertIn(str(wrapper), args)
                    self.assertNotIn('#SBATCH', wrapper.read_text())
                    self.assertIn(str(runner), wrapper.read_text())
                    self.assertIn('"$@"', wrapper.read_text())
                else:
                    self.assertIn(str(runner), args)

    def test_quota_batches_wait_and_preserve_indices(self):
        self.config.cluster.max_submit_jobs = 6
        self.config.cluster.submit_reserve = 2
        events = []
        def submit(stage, script, cmd, spec, env, dependencies, **kwargs):
            events.append((stage, spec, env.get("OFFSET")))
            return str(100 + len(events))
        def wait(config, job):
            events.append(("wait", job))
        with patch.object(sw, "array_limit", return_value=1001), \
             patch.object(sw, "user_queue", side_effect=[["9"] * 4, ["9"] * 2, ["9"]]), \
             patch.object(sw.time, "sleep") as sleep, \
             patch.object(sw, "submit", side_effect=submit), \
             patch.object(sw, "wait_for_array", side_effect=wait):
            sw.dispatch(self.config, self.snapshot, {})
        sleep.assert_called_once_with(60)
        self.assertEqual([e[0] for e in events], ["array", "wait", "array", "wait", "collector"])
        self.assertEqual(events[0][1:], ("1-2%4", "0"))
        self.assertEqual(events[2][1:], ("3-5%4", "0"))

    def test_queue_expands_tasks_and_fails_closed(self):
        self.config.cluster.screen_cluster = "gpu_site"
        with patch.object(sw.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, "100\n100\n200\n", "")) as run:
            self.assertEqual(sw.user_queue(self.config), ["100", "100", "200"])
        self.assertIn("--array", run.call_args.args[0])
        self.assertIn("--clusters=gpu_site", run.call_args.args[0])
        for result in (subprocess.CompletedProcess([], 1, "", "unavailable"),
                       subprocess.CompletedProcess([], 0, "unexpected", "")):
            with patch.object(sw.subprocess, "run", return_value=result):
                with self.assertRaises(RuntimeError):
                    sw.user_queue(self.config)

    def test_waits_until_entire_array_leaves_queue(self):
        with patch.object(sw, "user_queue", side_effect=[["100", "100"], ["100"], ["200"]]), \
             patch.object(sw.time, "sleep") as sleep:
            sw.wait_for_array(self.config, "100")
        self.assertEqual(sleep.call_count, 2)

    def test_quota_rejection_retries_but_ambiguous_response_does_not(self):
        self.config.cluster.max_submit_jobs = 50
        for failure, retry in [(sw.SubmitQuotaExceeded("quota"), True),
                               (RuntimeError("unrecognized submission response"), False)]:
            with self.subTest(retry=retry), \
                 patch.object(sw, "array_limit", return_value=1001), \
                 patch.object(sw, "user_queue", return_value=[]), \
                 patch.object(sw, "wait_for_array"), \
                 patch.object(sw.time, "sleep") as sleep, \
                 patch.object(sw, "submit", side_effect=[failure, "100", "101"]) as submit:
                if retry:
                    sw.dispatch(self.config, self.snapshot, {})
                    self.assertEqual(submit.call_count, 3)
                    self.assertEqual(submit.call_args_list[0].args, submit.call_args_list[1].args)
                    sleep.assert_called_once()
                else:
                    with self.assertRaises(RuntimeError):
                        sw.dispatch(self.config, self.snapshot, {})
                    self.assertEqual(submit.call_count, 1)

    def test_interrupted_wait_collects_after_active_array(self):
        with patch.object(sw, "array_limit", return_value=1001), \
             patch.object(sw, "submit", return_value="100") as submit, \
             patch.object(sw, "wait_for_array", side_effect=RuntimeError("queue unavailable")):
            with self.assertRaisesRegex(RuntimeError, "queue unavailable"):
                sw.dispatch(self.config, self.snapshot, {})
        self.assertEqual(submit.call_args.args[0], "collector")
        self.assertEqual(submit.call_args.args[5], ["100"])

    def test_dispatch_has_separate_wall_time(self):
        self.config.cluster.dispatcher_time_limit = "2-00:00:00"
        command = sw.submit("dispatch", "dispatch.sh", None, None, {}, [],
                            config=self.config, artifact_dir=self.snapshot, preview=True)
        self.assertIn("--time=2-00:00:00", command)
        self.assertIn("--no-requeue", command)

    def test_submission_errors_only_retry_explicit_quota_rejections(self):
        for error, expected in [
                ("sbatch: error: QOSMaxSubmitJobPerUserLimit", sw.SubmitQuotaExceeded),
                ("Job violates accounting/QOS policy", RuntimeError),
                ("connection lost", RuntimeError)]:
            with patch.object(sw.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 1, "", error)):
                with self.assertRaises(expected) as caught:
                    sw.submit("array", "runner.sh", "cmds", "1-5%4", {}, [],
                              config=self.config, artifact_dir=self.snapshot)
                self.assertIs(type(caught.exception), expected)
        self.assertFalse((self.snapshot / "submissions.jsonl").exists())

    def test_submit_uses_isolated_environment_without_export_file(self):
        env = {'PATH': os.environ['PATH'], 'VALUE': "spaces,quotes'\n$literal"}
        with patch.dict(os.environ, UNRECORDED_OVERRIDE='bad'), \
             patch.object(sw.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '123', '')) as run:
            sw.submit('inputs', 'runner.sh', None, None, env, [],
                      config=self.config, artifact_dir=self.snapshot)
        self.assertEqual(run.call_args.kwargs['env'], env)
        self.assertIn('--export=ALL', run.call_args.args[0])
        self.assertFalse(any(arg.startswith('--export-file=') for arg in run.call_args.args[0]))

    def test_job_name_prefix_and_legacy_default(self):
        for prefix in ("", "crebrf-wt"):
            self.config.cluster.job_name = prefix
            for stage in ('inputs', 'dispatch', 'array', 'collector'):
                with self.subTest(prefix=prefix, stage=stage):
                    with patch.object(sw.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '123', '')) as run:
                        sw.submit(stage, 'runner.sh', None, '1-2' if stage == 'array' else None,
                                  {}, [], config=self.config, artifact_dir=self.snapshot)
                    job_arg = next(arg for arg in run.call_args.args[0] if arg.startswith('--job-name='))
                    if prefix:
                        self.assertEqual(job_arg, '--job-name=crebrf-wt-' + stage)
                    else:
                        self.assertRegex(job_arg, r'^--job-name=latentppi-[0-9a-f]{12}-' + stage + '$')

    def test_cluster_routing_and_ledger(self):
        self.config.cluster.screen_cluster = 'gpu_site'
        self.config.cluster.inputs_cluster = 'cpu_site'
        for stage in ('inputs', 'dispatch', 'array', 'collector'):
            target = 'gpu_site' if stage == 'array' else 'cpu_site'
            with patch.object(sw.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '123;' + target, '')) as run:
                sw.submit(stage, 'runner.sh', None, '1-2' if stage == 'array' else None,
                          {}, [], config=self.config, artifact_dir=self.snapshot)
            self.assertIn('--clusters=' + target, run.call_args.args[0])
        entries = [json.loads(line) for line in (self.snapshot/'submissions.jsonl').read_text().splitlines()]
        self.assertEqual([e['cluster'] for e in entries], ['cpu_site', 'cpu_site', 'gpu_site', 'cpu_site'])
        self.config.cluster.inputs_cluster = ''
        preview = sw.submit('inputs', 'runner.sh', None, None, {}, [], config=self.config,
                            artifact_dir=self.snapshot, preview=True)
        self.assertNotIn('--clusters', preview)

    def test_status_queries_recorded_clusters_even_after_config_changes(self):
        folder = self.root/'latentppi_runs'/'run-old'
        folder.mkdir(parents=True)
        (folder/'submissions.jsonl').write_text('\n'.join(json.dumps(e) for e in [
            dict(job_id='123', cluster='cpu_site'), dict(job_id='123', cluster='gpu_site'),
            dict(job_id='456')]))
        self.config.cluster.screen_cluster = 'new_site'
        self.config.cluster.job_name = 'renamed-screen'  # Tracking still follows recorded IDs.
        with patch.object(sw.shutil, 'which', return_value='squeue'), \
             patch.object(sw.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '123|RUNNING|screen\n', '')) as run:
            jobs, warning = sw.active_jobs(self.config)
        self.assertIsNone(warning)
        self.assertEqual(len(jobs), 3)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn('--clusters=cpu_site', commands[0])
        self.assertIn('--clusters=gpu_site', commands[1])
        self.assertFalse(any(arg.startswith('--clusters') for arg in commands[2]))

    def test_array_limit_uses_screen_cluster(self):
        self.config.cluster.screen_cluster = 'gpu_site'
        with patch.object(sw.shutil, 'which', return_value='scontrol'), \
             patch.object(sw.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'MaxArraySize = 100', '')) as run:
            self.assertEqual(sw.array_limit(self.config), 100)
        self.assertIn('--clusters=gpu_site', run.call_args.args[0])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=REPO, prefix='.workflow-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = lp.Config(runner='full')
        self.config.cluster.dispatch_mode = 'persistent'
        self.config.inputs.output_base = str(self.root)
        self.config.paths.script_dir = str(REPO)
        for key, value in [('cmd_file', 'af3_cmds.txt'), ('af3_input_dir', 'af3_inputs'),
                           ('af3_output_dir', 'predictions'), ('ipsae_summary', 'ipsae.csv')]:
            setattr(self.config.paths, key, str(self.root / value))
        self.config.output.metrics_dir = str(self.root / 'metrics')
        self.config.output.legacy_csv = True  # Exercise the explicitly retained legacy exports.
        self.output = Path(self.config.paths.af3_output_dir)
        self.names = ['target_with_name_complete', 'target_with_name_gated', 'failed', 'never', 'multi_seed']
        self.snapshot = self.root / 'snapshot'
        self.snapshot.mkdir()
        self.prepare()

    def prepare(self):
        inputs = Path(self.config.paths.af3_input_dir)
        inputs.mkdir(exist_ok=True)
        with (self.root / 'af3_lengths.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=['system_name','line_number','json_path'])
            writer.writeheader()
            for index, name in enumerate(self.names, 1):
                path = inputs / (name + '.json')
                path.write_text(json.dumps(dict(name=name, modelSeeds=[1,2] if name=='multi_seed' else [1])))
                writer.writerow(dict(system_name=name,line_number=index,json_path=str(path)))
        Path(self.config.paths.cmd_file).write_text('command\n' * len(self.names))

    def timing(self, name, seed=1, **extra):
        folder = self.output / name
        folder.mkdir(parents=True, exist_ok=True)
        record = dict(input_name=name,seed=seed,trunk_time_seconds=10,diffusion_time_seconds=2,
                      trunk_per_recycle_seconds=[5,5])
        record.update(extra)
        with (folder/'prediction_timings.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')

    def partial(self):
        self.timing(self.names[0])
        self.timing(self.names[1], latentppi_scorer_gate_triggered=True,latentppi_scorer_gate_status='ok')
        (self.output/'failed').mkdir()
        (self.output/'failed/prediction_timings.jsonl').write_text('{"input_name":"failed","seed":1}\n')
        self.timing('multi_seed')

    def test_status_resume_and_expected_seeds(self):
        self.partial()
        state = sw.screen_state(self.config, write_metrics=True)
        self.assertEqual({k:state[k] for k in ['total','completed','gated','failed','never_run']},
                         dict(total=5,completed=2,gated=1,failed=2,never_run=1))
        self.assertEqual(state['never_run_indices'],[4])
        self.assertEqual(state['outstanding_indices'],[3,4,5])
        self.assertTrue((self.root/'metrics/metrics_all.csv').is_file())
        self.timing('multi_seed',seed=2)
        self.assertEqual(sw.screen_state(self.config)['completed'],3)

    def test_show_status_light_mode_skips_metric_recompute_and_warns(self):
        state = dict(total=5, completed=2, gated=1, failed=2, never_run=1, skipped_length=0,
                     never_run_indices=[4], outstanding_indices=[3, 4, 5],
                     timing_warnings={}, systems={}, collector={})
        with patch.object(sw, "screen_state", return_value=state) as compute, \
             patch.object(sw, "active_jobs", return_value=([], None)), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            sw.show_status(self.config)
        self.assertFalse(compute.call_args.kwargs["write_metrics"])
        self.assertIn("light mode (default)", stderr.getvalue())

    def test_show_status_heavy_mode_recomputes_metrics_and_announces(self):
        state = dict(total=5, completed=2, gated=1, failed=2, never_run=1, skipped_length=0,
                     never_run_indices=[4], outstanding_indices=[3, 4, 5],
                     timing_warnings={}, systems={}, collector={})
        with patch.object(sw, "screen_state", return_value=state) as compute, \
             patch.object(sw, "active_jobs", return_value=([], None)), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            sw.show_status(self.config, heavy=True)
        self.assertTrue(compute.call_args.kwargs["write_metrics"])
        self.assertIn("heavy mode", stderr.getvalue())

    def test_chunks_preserve_exact_indices_with_limits_and_offset(self):
        indices = [1,3,5,8,10,11,14,18,21,30]
        batches = sw.chunk_indices(indices, max_array_size=8,max_spec_length=8,concurrency=2,offset=2)
        actual=[]
        for offset,spec in batches:
            self.assertLessEqual(len(spec),8)
            for group in spec.split('%')[0].split(','):
                bounds=list(map(int,group.split('-')))
                for local in range(bounds[0],bounds[-1]+1):
                    self.assertTrue(1<=local<8)
                    actual.append(offset+local)
        self.assertEqual(actual,[i for i in indices if i>2])
        with self.assertRaises(ValueError):
            sw.chunk_indices([1],max_array_size=1)

    def test_timing_warnings_do_not_retry_successful_results(self):
        for gated in [True, False]:
            for missing in [True, False]:
                with self.subTest(gated=gated, missing=missing):
                    name = self.names[0]
                    self.timing(name, trunk_per_recycle_seconds=None,
                                diffusion_skipped=gated,
                                latentppi_scorer_gate_triggered=gated,
                                latentppi_scorer_gate_status='ok',
                                latentppi_scorer_rank_by='ipTM_af',
                                latentppi_scorer_pred_metrics={'ipTM_af': 0.14})
                    path = self.output/name/'prediction_timings.jsonl'
                    record = json.loads(path.read_text().splitlines()[-1])
                    if missing:
                        del record['trunk_time_seconds']
                    path.write_text(json.dumps(record)+'\n')
                    attempts = self.output/'.latentppi_attempts'
                    attempts.mkdir(exist_ok=True)
                    attempt = attempts/'task.json'
                    attempt.write_text(json.dumps(dict(system_name=name, exit_code=0)))
                    state = sw.screen_state(self.config, write_metrics=True)
                    self.assertEqual(state['systems'][name], 'gated' if gated else 'completed')
                    self.assertNotIn(1, state['outstanding_indices'])
                    status = 'missing_timing' if missing else 'invalid_timing'
                    self.assertEqual(state['timing_warnings'][name], [{'seed':'1','status':status}])
                    self.assertIn(status, (self.root/'metrics/metrics_all.csv').read_text())
                    for code in [1, None]:
                        attempt.write_text(json.dumps(dict(system_name=name, exit_code=code)))
                        self.assertEqual(sw.screen_state(self.config)['systems'][name], 'failed')

    def test_timing_warning_requires_result_and_all_seeds(self):
        name = 'multi_seed'
        attempts = self.output/'.latentppi_attempts'
        attempts.mkdir(parents=True)
        (attempts/'task.json').write_text(json.dumps(dict(system_name=name, exit_code=0)))
        self.timing(name, trunk_per_recycle_seconds=None, diffusion_skipped=False)
        self.assertEqual(sw.screen_state(self.config)['systems'][name], 'failed')
        self.timing(name, seed=2, trunk_per_recycle_seconds=None, diffusion_skipped=False)
        self.assertEqual(sw.screen_state(self.config)['systems'][name], 'completed')
        path = self.output/name/'prediction_timings.jsonl'
        for record in [dict(input_name=name, seed=1),
                       dict(input_name=name, diffusion_skipped=False)]:
            path.write_text(json.dumps(record)+'\n')
            self.assertEqual(sw.screen_state(self.config)['systems'][name], 'failed')

    def test_dispatch_dependencies_and_force(self):
        self.partial()
        self.config.cluster.max_array_size=4
        calls=[]
        def scheduler(args,**kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args,0,str(100+len(calls))+';cluster\n','')
        with patch.object(sw,'wait_for_array'), patch.object(sw.shutil,'which',return_value=None),patch.object(sw.subprocess,'run',side_effect=scheduler):
            jobs=sw.dispatch(self.config,self.snapshot,{},inputs_job='99')
        self.assertEqual(jobs,['101','102'])
        self.assertIn('--array=3%4',calls[0])
        self.assertIn('--array=1-2%4',calls[1])
        self.assertFalse(any(x.startswith('--dependency=') for x in calls[0]))
        self.assertFalse(any(x.startswith('--dependency=') for x in calls[-1]))
        self.assertIn('--partition=your_cpu',calls[-1])
        self.assertFalse(any(x.startswith('--gres=') for x in calls[-1]))
        entries = [json.loads(line) for line in (self.snapshot/'submissions.jsonl').read_text().splitlines()]
        exports=[Path(entry['export_file']).read_bytes() for entry in entries[:2]]
        self.assertIn(b'OFFSET=0\0',exports[0])
        self.assertIn(b'OFFSET=3\0',exports[1])
        with patch.object(sw,'wait_for_array'), patch.object(sw.shutil,'which',return_value=None),patch.object(sw.subprocess,'run',side_effect=scheduler):
            sw.dispatch(self.config,self.snapshot,{},force=True)
        self.assertIn('--array=1-3%4',calls[3])

    def tier_pairs(self):
        pairs = sw.read_pairs(self.config)
        for pair, length in zip(pairs, [2999, 3000, 3001, 1000, 7000]):
            pair['total_length'] = str(length)
        with (self.root / 'af3_lengths.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
        self.config.cluster.gpu_tiers = [
            dict(name='regular', max_total_residues=3000, partition='regular', gres='gpu:l40:1'),
            dict(name='large', partition='large', gres='gpu:big:1')]
        self.config.output.refresh_seconds = 0
        return pairs

    def test_gpu_tiers_boundary_sparse_arrays_and_retry_resume(self):
        self.tier_pairs()
        self.config.cluster.retry_failed_once = True
        calls = []
        def scheduler(args, **kwargs):
            calls.append((args, kwargs['env']))
            return subprocess.CompletedProcess(args, 0, str(100 + len(calls)), '')
        with patch.object(sw, 'wait_for_array'), patch.object(sw, 'array_limit', return_value=1001), \
             patch.object(sw.subprocess, 'run', side_effect=scheduler):
            sw.dispatch(self.config, self.snapshot, {})
            arrays = [(args, env) for args, env in calls if any(a.startswith('--array=') for a in args)]
            self.assertEqual(len(arrays), 3)
            self.assertIn('--array=1-2,4%4', arrays[0][0])
            self.assertIn('--partition=regular', arrays[0][0])
            self.assertIn('--array=3,5%4', arrays[1][0])
            self.assertIn('--gres=gpu:big:1', arrays[1][0])
            self.assertIn('--array=1-5%4', arrays[2][0])
            self.assertIn('--partition=large', arrays[2][0])
            self.assertEqual([env['LATENTPPI_GPU_RETRY'] for _, env in arrays], ['0', '0', '1'])
            ledger = json.loads((self.root / 'latentppi_gpu_retries.json').read_text())
            self.assertEqual(len(ledger), 5)
            self.assertTrue(all(r['gres'] == 'gpu:big:1' for r in ledger.values()))
            entries = [json.loads(line) for line in (self.snapshot / 'submissions.jsonl').read_text().splitlines()]
            self.assertEqual([e.get('gpu_tier') for e in entries[:3]], ['regular', 'large', 'large'])
            calls.clear()
            sw.dispatch(self.config, self.snapshot, {})
            self.assertFalse(any(any(a.startswith('--array=') for a in args) for args, _ in calls))

    def test_length_cap_preserves_commands_and_filters_force_and_retries(self):
        self.tier_pairs()
        self.config.cluster.max_total_residues = 3000
        self.config.cluster.retry_failed_once = True
        originals = {p: p.read_bytes() for p in (self.root / 'af3_lengths.csv',
                                                Path(self.config.paths.cmd_file))}
        with patch.object(sw, 'array_limit', return_value=1001), patch.object(sw, 'wait_for_array'), \
             patch.object(sw, 'submit', return_value='123') as submit:
            sw.dispatch(self.config, self.snapshot, {}, force=True)
        arrays = [c for c in submit.call_args_list if c.args[0] == 'array']
        self.assertEqual([c.args[3] for c in arrays], ['1-2,4%4', '1-2,4%4'])
        self.assertEqual([c.args[4]['LATENTPPI_GPU_RETRY'] for c in arrays], ['0', '1'])
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_length_cap_disabled_missing_lengths_and_preview(self):
        pairs = sw.read_pairs(self.config)
        self.assertEqual(sw.length_exclusions(self.config, pairs), {})
        self.config.cluster.max_total_residues = 3000
        for raw in (None, '', 'unknown', '0', '-1', '3000.5', True):
            with self.subTest(length=raw):
                pairs[0]['total_length'] = raw
                with self.assertRaisesRegex(ValueError, 'positive integer total_length'):
                    sw.length_exclusions(self.config, pairs)
        with patch.object(sw, 'submit') as submit:
            with self.assertRaisesRegex(ValueError, 'total_length'):
                sw.dispatch(self.config, self.snapshot, {})
            submit.assert_not_called()
        self.tier_pairs()
        self.config.cluster.gpu_tiers = []
        with patch.object(sw, 'array_limit', return_value=1001), \
             patch.object(sw.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()) as output:
            sw.dispatch(self.config, self.snapshot, {}, preview=True)
        run.assert_not_called()
        self.assertIn('2 command(s) excluded above 3,000', output.getvalue())
        self.assertIn('--array=1-2,4%4', output.getvalue())
        self.config.cluster.max_total_residues = 1
        with patch.object(sw, 'submit', return_value='123') as submit:
            sw.dispatch(self.config, self.snapshot, {}, force=True)
        self.assertEqual([c.args[0] for c in submit.call_args_list], ['collector'])

    def test_three_tiers_preview_offset_and_no_retry_budget(self):
        self.tier_pairs()
        self.config.cluster.gpu_tiers.insert(0, dict(name='small', max_total_residues=1500,
                                                    partition='small', gres='gpu:small:1'))
        self.config.cluster.max_array_size = 4
        with patch.object(sw, 'array_limit', return_value=4), patch.object(sw.subprocess, 'run') as run, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            sw.dispatch(self.config, self.snapshot, {}, preview=True)
        run.assert_not_called()
        text = output.getvalue()
        for tier in ('small', 'regular', 'large'):
            self.assertIn('--partition=' + tier, text)
        self.assertIn('OFFSET=3', text)
        self.assertFalse((self.root / 'latentppi_gpu_retries.json').exists())

    def test_tiers_preserve_gate_exclusion_and_quota_retries(self):
        self.tier_pairs()
        self.partial()
        self.config.cluster.retry_failed_once = True
        calls = []
        rejected = False
        def submit(stage, script, cmd, spec, env, dependencies, **kwargs):
            nonlocal rejected
            if stage == 'array' and env['LATENTPPI_GPU_RETRY'] == '1' and not rejected:
                rejected = True
                raise sw.SubmitQuotaExceeded('quota')
            calls.append((stage, spec, env))
            return str(100 + len(calls))
        with patch.object(sw, 'array_limit', return_value=1001), patch.object(sw, 'wait_for_array'), \
             patch.object(sw.time, 'sleep'), patch.object(sw, 'submit', side_effect=submit):
            sw.dispatch(self.config, self.snapshot, {})
        arrays = [call for call in calls if call[0] == 'array']
        self.assertEqual([a[1] for a in arrays], ['4%4', '3-5%4'])
        self.assertEqual([a[2]['LATENTPPI_GPU_TIER'] for a in arrays], ['regular', 'large'])
        ledger = json.loads((self.root / 'latentppi_gpu_retries.json').read_text())
        self.assertEqual(set(ledger), set(self.names[2:]))

    def test_initial_large_route_does_not_reserve_retry(self):
        self.tier_pairs()
        self.config.cluster.retry_failed_once = False
        with patch.object(sw, 'array_limit', return_value=1001), patch.object(sw, 'wait_for_array'), \
             patch.object(sw, 'submit', return_value='123') as submit:
            sw.dispatch(self.config, self.snapshot, {})
        arrays = [c for c in submit.call_args_list if c.args[0] == 'array']
        self.assertEqual(len(arrays), 2)
        self.assertTrue(all(c.args[4]['LATENTPPI_GPU_RETRY'] == '0' for c in arrays))
        self.assertFalse((self.root / 'latentppi_gpu_retries.json').exists())

    def test_tiers_reject_missing_lengths_before_submission(self):
        self.config.cluster.gpu_tiers = [dict(name='large', partition='big', gres='gpu:big:1')]
        with patch.object(sw, 'submit') as submit:
            with self.assertRaisesRegex(ValueError, 'total_length'):
                sw.dispatch(self.config, self.snapshot, {})
        submit.assert_not_called()

    def test_one_gpu_retry_survives_resume(self):
        self.partial()
        self.config.cluster.retry_failed_once = True
        self.config.cluster.retry_partition = 'big'
        self.config.cluster.retry_gres = 'gpu:a100:1'
        self.config.output.refresh_seconds = 0
        calls = []
        def scheduler(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, str(100 + len(calls)), '')
        with patch.object(sw, 'wait_for_array'), patch.object(sw.shutil, 'which', return_value=None), \
             patch.object(sw.subprocess, 'run', side_effect=scheduler):
            sw.dispatch(self.config, self.snapshot, {})
            arrays = [args for args in calls if any(a.startswith('--array=') for a in args)]
            self.assertEqual(len(arrays), 2)
            self.assertIn('--gres=gpu:1', arrays[0])
            self.assertIn('--partition=big', arrays[1])
            self.assertIn('--gres=gpu:a100:1', arrays[1])
            ledger = json.loads((Path(self.config.inputs.output_base) / 'latentppi_gpu_retries.json').read_text())
            self.assertEqual(len(ledger), 3)
            calls.clear()
            sw.dispatch(self.config, self.snapshot, {})
            self.assertFalse(any(any(a.startswith('--array=') for a in args) for args in calls))

    def test_retry_reservation_handles_submission_errors(self):
        self.partial()
        self.config.cluster.retry_failed_once = True
        self.config.cluster.retry_partition = 'big'
        self.config.cluster.retry_gres = 'gpu:a100:1'
        self.config.output.refresh_seconds = 0
        ledger = self.root / 'latentppi_gpu_retries.json'
        # An uncertain sbatch response consumes the reservation, avoiding duplicates.
        with patch.object(sw, 'array_limit', return_value=1001), \
             patch.object(sw, 'wait_for_array'), \
             patch.object(sw, 'submit', side_effect=['101', RuntimeError('lost response'), '103']):
            with self.assertRaisesRegex(RuntimeError, 'lost response'):
                sw.dispatch(self.config, self.snapshot, {})
        self.assertEqual(len(json.loads(ledger.read_text())), 3)
        self.assertTrue(all(r['state'] == 'reserved' for r in json.loads(ledger.read_text()).values()))
        ledger.unlink()
        # A definite quota rejection releases the reservation and allows submission.
        with patch.object(sw, 'array_limit', return_value=1001), \
             patch.object(sw, 'wait_for_array'), patch.object(sw.time, 'sleep'), \
             patch.object(sw, 'submit', side_effect=['101', sw.SubmitQuotaExceeded('quota'), '102', '103']):
            sw.dispatch(self.config, self.snapshot, {})
        self.assertTrue(all(r['job_id'] == '102' for r in json.loads(ledger.read_text()).values()))

    def test_fallback_retries_only_unresolved_pairs(self):
        pairs = sw.read_pairs(self.config)
        self.config.cluster.retry_failed_once = True
        self.config.cluster.retry_partition = 'big'
        self.config.cluster.retry_gres = 'gpu:a100:1'
        self.config.output.refresh_seconds = 0
        initial = dict(outstanding_indices=[1, 2, 3], completed=0,
                       systems={p['system_name']: 'never_run' for p in pairs})
        after = dict(systems={pairs[0]['system_name']: 'completed', pairs[1]['system_name']: 'gated',
                              pairs[2]['system_name']: 'never_run'})
        with patch.object(sw, 'screen_state', side_effect=[initial, after]), \
             patch.object(sw, 'wait_for_array'), patch.object(sw, 'array_limit', return_value=1001), \
             patch.object(sw, 'submit', side_effect=['101', '102', '103']) as submit:
            sw.dispatch(self.config, self.snapshot, {})
        retry = submit.call_args_list[1]
        self.assertEqual(retry.args[3], '3%4')
        self.assertEqual(retry.args[4]['LATENTPPI_GPU_RETRY'], '1')

    def test_stage_zero_chain_and_no_submit(self):
        Path(self.config.paths.cmd_file).unlink()
        calls=[]
        def scheduler(args,**kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args,0,str(200+len(calls)), '')
        # render uses git for provenance; isolate that read from mocked scheduler calls.
        with patch.object(lp,'run_check',return_value=subprocess.CompletedProcess([],0,'','')):
            with patch.object(sw.subprocess,'run',side_effect=scheduler):
                sw.launch(self.config,lp.Report(),self.root/'config.yaml',lp.render)
            self.assertEqual(len(calls),2)
            self.assertIn('--partition=your_cpu',calls[0])
            self.assertIn('--dependency=afterok:201',calls[1])
            with patch.object(sw,'active_jobs',return_value=([],None)),patch.object(sw.subprocess,'run') as run:
                sw.launch(self.config,lp.Report(),self.root/'config.yaml',lp.render,preview=True)
                run.assert_not_called()

    def test_prepared_skips_inputs_and_active_jobs_prevent_duplicates(self):
        with patch.object(lp,'run_check',return_value=subprocess.CompletedProcess([],0,'','')), \
             patch.object(lp, 'submission_commands', return_value={}), \
             patch.object(sw,'submit', return_value='preview') as submit:
            sw.launch(self.config,lp.Report(),self.root/'config.yaml',lp.render,preview=True)
            submit.assert_called_once()
            self.assertEqual(submit.call_args.args[0], 'dispatch')
        with patch.object(sw,'active_jobs',return_value=([{'job_id':'123'}],None)):
            with self.assertRaisesRegex(RuntimeError,'active jobs'):
                sw.launch(self.config,lp.Report(),self.root/'config.yaml',lp.render)

    def test_failed_postprocessing_overrides_good_timing(self):
        name=self.names[0]
        self.timing(name)
        source=Path(self.config.paths.af3_input_dir)/(name+'.json')
        with patch.object(record_run_env.subprocess,'run',side_effect=FileNotFoundError):
            record_run_env.record('start',self.output,name,source)
        record_run_env.record('finish',self.output,name,source,7)
        self.assertEqual(sw.screen_state(self.config)['systems'][name],'failed')
        data=json.loads((self.output/name/'run_env.json').read_text())
        self.assertEqual(data['exit_code'],7)
        self.assertIn('gpu_type',data)

    def test_hook_preserves_cleanup_exit_status(self):
        source=Path(self.config.paths.af3_input_dir)/(self.names[0]+'.json')
        for code in [0,7,143]:
            command='''set -Eeuo pipefail
cleanup() { local code=$?; trap - EXIT; echo "cleanup:$code"; exit "$code"; }
source "$SCRIPT_DIR/run_env_capture.sh"
exit "$TEST_EXIT"
'''
            env={**os.environ,'SCRIPT_DIR':str(REPO),'PYTHON_BIN':sys.executable,'DRY_RUN':'0',
                 'AF3_OUTPUT_DIR':str(self.output),'JOB_NAME':self.names[0],'JSON_PATH':str(source),'TEST_EXIT':str(code)}
            result=subprocess.run(['bash','-c',command],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,code,result.stderr)
            self.assertIn(f'cleanup:{code}',result.stdout)

    def test_empty_filtered_screen_collects_without_arrays(self):
        self.names=[]
        self.prepare()
        with patch.object(sw.shutil,'which',return_value=None),patch.object(sw,'submit',return_value='123') as submit:
            sw.dispatch(self.config,self.snapshot,{})
        self.assertEqual(submit.call_count,1)
        self.assertEqual(submit.call_args.args[0],'collector')
        self.assertEqual(sw.screen_state(self.config,write_metrics=True)['total'],0)

    def test_toy_prepares_fifty_pairs_and_status_without_af3(self):
        toy=REPO/'examples/toy_screen'
        result=subprocess.run([sys.executable,str(REPO/'run_pipeline.py'),
            '--fasta-single',str(toy/'target_small.fasta'),'--msa-a',str(toy/'query_msa/target_small.a3m'),
            '--msa-dir',str(toy/'proteome'),'--output-dir',str(self.root),
            '--recreate-manifest','--recreate-commands'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        state=sw.screen_state(self.config,write_metrics=True)
        self.assertEqual(state['total'],50)
        self.assertEqual(state['never_run_indices'],list(range(1,51)))

    def test_cli_force_inputs_and_inputs_only_without_models(self):
        import yaml
        toy=REPO/'examples/toy_screen'
        path=self.root/'config.yaml'
        path.write_text(yaml.safe_dump(dict(inputs=dict(fasta_file=str(toy/'target_small.fasta'),
            output_base=str(self.root),proteome_msa_dir=str(toy/'proteome')),paths=dict(script_dir=str(REPO)))))
        with patch.object(sw,'launch') as launch,contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(lp.main(['inputs',str(path),'--force-inputs','--no-submit']),0)
        config=launch.call_args.args[0]
        self.assertTrue(config.inputs.force and config.inputs.recreate_manifest and config.inputs.recreate_commands)
        self.assertTrue(launch.call_args.kwargs['inputs_only'])

    def test_query_msa_checks_and_sbatch_header(self):
        fasta=self.root/'query.fasta';fasta.write_text('>query\nACDE\n')
        msa=self.root/'query.a3m';msa.write_text('>arbitrary_header\nACDE\n')
        self.assertEqual(query_msa.discover(fasta,self.root),msa)
        other=self.root/'other.a3m';other.write_text('>other\nACDE\n')
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            query_msa.discover(fasta,self.root)
        other.unlink();msa.write_text('>wrong\nAAAA\n')
        with self.assertRaises(ValueError): query_msa.discover(fasta,self.root)
        lines=(REPO/'screen.sh').read_text().splitlines()
        executable=next(i for i,line in enumerate(lines) if line.strip() and not line.startswith('#'))
        self.assertTrue(all(i<executable for i,line in enumerate(lines) if line.startswith('#SBATCH')))

if __name__=='__main__': unittest.main()

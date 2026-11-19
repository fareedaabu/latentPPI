"""Config checks and artifact rendering require no GPU, scheduler, or PyTorch install."""
import contextlib
import io
import json
import os
import pickle
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import latentppi as lp
import screen_workflow as sw


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".config-test-", dir=REPO)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run with spaces [*] quotes'"
        self.models = self.root / "models"
        self.models.mkdir()
        (self.models / "first.pt").write_bytes(b"synthetic checkpoint")
        (self.models / "second.pt").write_bytes(b"second checkpoint")
        (self.root / "target_with_name.fasta").write_text(">target\nACDEFGHIK\n")
        (self.root / "proteome").mkdir()
        (self.root / "proteome/target.a3m").write_text(">target\nLMNPQRSTV\n")
        for name in ["weights", "databases"]:
            (self.root / name).mkdir()
        (self.root / "image.sif").touch()
        self.config_path = self.root / "config.yaml"
        self.data = {
            "inputs": {"fasta_file": "target_with_name.fasta", "output_base": str(self.run),
                       "proteome_msa_dir": "proteome"},
            "paths": {"script_dir": str(REPO)},
            "af3": {"image": "image.sif", "model_parameters_dir": "weights", "databases_dir": "databases"},
            "scorer": {"rank_by": "invented_metric_421", "models": {
                "invented_metric_421": {"path": str(self.models / "first.pt")}}},
        }
        self.which = patch.object(lp.shutil, "which", return_value=None)
        self.which.start()
        self.addCleanup(self.which.stop)
        self.loader = patch.object(lp, "load_checkpoints", return_value={})
        self.load_mock = self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_cpu_mode_and_partition_aliases(self):
        self.data['cluster'] = dict(screen_partition='your_cpu, your_cpu_alt',
                                    inputs_partition='your_cpu, your_cpu_alt', cpu_only=True, gres='')
        config, report = self.checked()
        self.assertFalse(report.errors, report.errors)
        self.assertEqual(config.cluster.partition, 'your_cpu,your_cpu_alt')
        self.assertEqual(config.cluster.inputs_partition, 'your_cpu,your_cpu_alt')
        self.assertEqual(lp.environment(config, report, self.root)['AF3_CPU_ONLY'], '1')
        self.data['cluster']['cpu_only'] = False
        _, report = self.checked()
        self.assertTrue(any('cluster.gres' in e for e in report.errors))
        self.data['cluster']['partition'] = 'legacy'
        _, report = self.load()
        self.assertTrue(any('specify only one' in e for e in report.errors))

    def test_multiple_partition_validation(self):
        self.data['cluster'] = dict(screen_partition='your_gpu,your_gpu_alt',
                                    inputs_partition='your_cpu,your_cpu_alt')
        config, report = self.load()
        result = subprocess.CompletedProcess([], 0,
            'your_gpu*|up\nyour_gpu_alt|up\nyour_cpu|up\nyour_cpu_alt|up\n', '')
        with patch.object(lp.shutil, 'which', side_effect=lambda n: 'sinfo' if n == 'sinfo' else None), \
             patch.object(lp, 'run_check', return_value=result):
            lp.validate(config, report)
            self.assertFalse(report.errors, report.errors)
            config.cluster.partition += ',missing'
            report = lp.Report()
            lp.validate(config, report)
            self.assertTrue(any("partition 'missing'" in e for e in report.errors))
        for value in ('your_cpu,', ',your_cpu', 'your_cpu,,your_cpu_alt'):
            self.data['cluster']['screen_partition'] = value
            _, report = self.checked()
            self.assertTrue(any('empty entries' in e for e in report.errors))

    def test_gpu_tier_schema_and_inventory(self):
        self.data['cluster'] = dict(dispatch_mode='persistent', screen_cluster='gpu_site', inputs_cluster='cpu_site',
            retry_failed_once=True, gpu_tiers=[
                dict(name='regular', max_total_residues=3000, partition='regular', gres='gpu:l40:1'),
                dict(name='large', partition='big', gres='gpu:big:1')])
        config, report = self.load()
        def result(command, **kwargs):
            inventory = ('regular|up|gpu:l40:1\nbig|up|gpu:big:1\n' if '-Mgpu_site' in command
                         else 'your_cpu|up|(null)\n')
            return subprocess.CompletedProcess(command, 0, inventory, '')
        with patch.object(lp.shutil, 'which', side_effect=lambda n: 'sinfo' if n == 'sinfo' else None), \
             patch.object(lp, 'run_check', side_effect=result):
            lp.validate(config, report)
            self.assertFalse(report.errors, report.errors)
            self.assertFalse(any('GPU request' in w for w in report.warnings))
            config.cluster.gpu_tiers[-1]['gres'] = 'gpu:missing:1'
            report = lp.Report()
            lp.validate(config, report)
            self.assertTrue(any('gpu_tiers[1]' in w for w in report.warnings))
        from gpu_routing import tier_errors
        for tiers in ([None], [dict(name='a')],
                      [dict(name='a', partition='p', gres='g', max_total_residues=3000)],
                      [dict(name='a', partition='p', gres='g', max_total_residues=True),
                       dict(name='a', partition='p', gres='g')]):
            self.assertTrue(tier_errors(tiers), tiers)
        config.cluster.cpu_only = True
        report = lp.Report()
        lp.validate(config, report)
        self.assertTrue(any('gpu_tiers requires GPU' in e for e in report.errors))

    def test_old_snapshot_defaults_to_no_tiers(self):
        config, report = self.load()
        self.assertEqual(config.cluster.gpu_tiers, [])
        self.assertFalse(config.cluster.retry_failed_once)
        self.assertIsNone(config.cluster.max_total_residues)

    def test_complex_length_cap_validation_and_snapshot(self):
        for value in (None, 1, 3000):
            with self.subTest(value=value):
                self.data['cluster'] = dict(max_total_residues=value)
                config, report = self.checked()
                self.assertFalse(report.errors, report.errors)
                artifacts, _ = self.render(config, report)
                raw = json.loads((artifacts / 'resolved_config.json').read_text())['config']
                restored_report = lp.Report()
                restored = lp.load_config(self.config_path, restored_report, raw=raw)
                self.assertFalse(restored_report.errors, restored_report.errors)
                self.assertEqual(restored.cluster.max_total_residues, value)
                del raw['cluster']['max_total_residues']
                restored = lp.load_config(self.config_path, lp.Report(), raw=lp.snapshot_config(raw))
                self.assertIsNone(restored.cluster.max_total_residues)
        for value in (0, -1, True, False, 3000.0, '3000', [], {}):
            with self.subTest(invalid=value):
                self.data['cluster'] = dict(max_total_residues=value)
                _, report = self.checked()
                self.assertTrue(any('cluster.max_total_residues' in e for e in report.errors), report.errors)

    def test_gpu_inventory_and_retry_validation(self):
        self.data['cluster'] = dict(dispatch_mode='persistent', screen_cluster='gpu_site', inputs_cluster='cpu_site',
                                    retry_failed_once=True, retry_partition='big', retry_gres='gpu:a100:1',
                                    gres='gpu:small:1')
        config, report = self.load()
        def result(command, **kwargs):
            inventory = ('your_gpu|up|gpu:small:2(S:0-1)\nbig|up|gpu:a100:4(S:0-3)\n'
                         if '-Mgpu_site' in command else 'your_cpu|up|(null)\n')
            return subprocess.CompletedProcess(command, 0, inventory, '')
        with patch.object(lp.shutil, 'which', side_effect=lambda n: 'sinfo' if n == 'sinfo' else None), \
             patch.object(lp, 'run_check', side_effect=result):
            lp.validate(config, report)
            self.assertFalse(report.errors, report.errors)
            self.assertFalse(any('GPU request' in w for w in report.warnings), report.warnings)
            config.cluster.retry_gres = 'gpu:h100:1'
            report = lp.Report()
            lp.validate(config, report)
            self.assertFalse(report.errors, report.errors)
            self.assertTrue(any('retry_gres' in w and 'gpu_site' in w for w in report.warnings))
        config.cluster.retry_gres = ''
        report = lp.Report()
        lp.validate(config, report)
        self.assertTrue(any('requires cluster.retry_partition' in e for e in report.errors))

    def test_gpu_counts_generic_and_multi_type(self):
        for request, warning in [('gpu:4', False), ('gpu:a100:4', False),
                                 ('gpu:a100:5', True), ('gpu:h100:1', True),
                                 ('gpu:a100:1,gpu:h100:1', True)]:
            report = lp.Report()
            lp.check_gpu_request(request, ['gpu:a100:4(S:0-3)'], report, 'test')
            self.assertEqual(bool(report.warnings), warning, request)

    def test_embedding_save_config_and_environment(self):
        config, report = self.load()
        self.assertFalse(config.stages.save_embeddings)
        self.data['stages'] = {'save_embeddings': False, 'run_pool': False}
        config, report = self.load()
        self.assertFalse(config.stages.save_embeddings)
        self.assertEqual(lp.environment(config, report, self.root)['SAVE_EMBEDDINGS'], '0')
        self.data['stages']['save_embeddings'] = True
        config, report = self.load()
        self.assertEqual(lp.environment(config, report, self.root)['SAVE_EMBEDDINGS'], '1')
        self.data['stages']['save_embeddings'] = False
        self.data['stages']['run_pool'] = True
        config, report = self.checked()
        self.assertIn('stages.run_pool requires stages.save_embeddings: true', report.errors)

    def load(self):
        self.config_path.write_text(yaml.safe_dump(self.data))
        report = lp.Report()
        return lp.load_config(self.config_path, report), report

    def checked(self):
        config, report = self.load()
        lp.validate(config, report)
        return config, report

    def render(self, config, report):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            artifacts = lp.render(config, report, self.config_path)
        return artifacts, output.getvalue()

    def test_advance_watcher_dispatches_full_runner_during_source_screen(self):
        import csv
        config, _ = self.load()
        inputs = Path(config.paths.af3_input_dir)
        inputs.mkdir(parents=True)
        source = inputs / 'p.json'
        source.write_text(json.dumps({'name': 'pair', 'modelSeeds': [1]}))
        Path(config.paths.cmd_file).write_text(shlex.join([
            'run_alphafold.py', '--json_path', str(source), '--num_recycles=1']) + '\n')
        with (self.run / 'af3_lengths.csv').open('w') as out:
            writer = csv.writer(out)
            writer.writerow(['line_number', 'system_name', 'json_path'])
            writer.writerow([1, 'pair', source])
        folder = Path(config.paths.af3_output_dir) / 'pair'
        folder.mkdir(parents=True)
        record = dict(input_name='pair', seed=1, trunk_time_seconds=1, diffusion_time_seconds=1,
                      trunk_per_recycle_seconds=[1], latentppi_scorer_gate_status='ok',
                      latentppi_scorer_gate_triggered=False)
        # A pass arrives after the watcher starts, while the parent still has queued work.
        def finish_pair(_):
            (folder / 'prediction_timings.jsonl').write_text(json.dumps(record) + '\n')

        source_polls = iter([['parent'], ['parent'], []])
        def jobs(c):
            return (next(source_polls) if c.inputs.output_base == str(self.run) else []), None

        original_submit = sw.submit
        submitted = []
        def simulate(*args, **kwargs):
            if not kwargs['preview']:
                submitted.append(args[:2])
                return str(len(submitted))
            return original_submit(*args, **kwargs)

        with patch.object(sw, 'active_jobs', side_effect=jobs), patch.object(sw.time, 'sleep', finish_pair), \
             patch.object(sw, 'submit', side_effect=simulate):
            self.assertEqual(lp.main(['advance', str(self.config_path), '--watch']), 0)
            self.assertEqual(len(submitted), 1)
            snapshot = next((self.run / 'advance').glob('batch-*/latentppi_runs/run-*'))
            self.assertEqual(lp.main(['_dispatch', str(snapshot)]), 0)
        array = [script.name for stage, script in submitted if stage == 'array']
        self.assertEqual(array, ['run_af3_compress_pool.sh'])
        resolved = json.loads((snapshot / 'resolved_config.json').read_text())
        self.assertEqual(resolved['environment']['AF3_NUM_RECYCLES'], '10')
        self.assertIsNone(resolved['scorer_manifest'])

    def test_single_invented_metric_defaults_and_pending_inputs(self):
        config, report = self.checked()
        self.assertFalse(report.errors, report.errors)
        self.assertEqual(config.scorer.predict_metrics, ["invented_metric_421"])
        self.assertNotIn("invented_metric_421", (REPO / "latentppi.py").read_text())
        self.assertEqual(config.inputs.max_identity, 95.0)
        self.assertFalse(config.stages.run_pool)
        self.assertIsNone(report.pair_count)
        self.assertEqual(report.proteome_count, 1)
        self.assertEqual(config.paths.cmd_file, str(self.run / "af3_cmds.txt"))
        self.assertTrue((self.run / "slurm_logs").is_dir())
        self.assertTrue(any("conda not" in w for w in report.warnings))

    def test_errors_aggregate_missing_rank_checkpoint_and_paths(self):
        self.data["scorer"]["predict_metrics"] = ["absent"]
        self.data["scorer"]["threshold"] = 1.5
        self.data["inputs"]["fasta_file"] = "missing.fasta"
        self.data["af3"]["image"] = "missing.sif"
        config, report = self.checked()
        text = "\n".join(report.errors)
        for expected in ["absent", "available metrics", "rank_by", "threshold", "fasta_file", "af3.image"]:
            self.assertIn(expected, text)
        self.assertGreaterEqual(len(report.errors), 5)
        self.data["scorer"]["predict_metrics"] = ["invented_metric_421"]
        self.data["scorer"]["models"]["invented_metric_421"]["path"] = "nonexistent.pt"
        _, report = self.checked()
        self.assertTrue(any("checkpoint" in e and "invented_metric_421" in e and "nonexistent.pt" in e
                            for e in report.errors))

    def test_rank_must_be_predicted_even_when_model_exists(self):
        self.data["scorer"]["models"]["second"] = {"path": str(self.models / "second.pt")}
        self.data["scorer"]["predict_metrics"] = ["second"]
        _, report = self.checked()
        self.assertTrue(any("absent from predict_metrics" in e for e in report.errors))

    def test_bad_types_unknown_keys_and_bad_fasta_all_reported(self):
        self.data["inputs"].update(force="false", af3_num_recycles=-1, mystery=True)
        (self.root / "target_with_name.fasta").write_text(">a\nACD\n>b\nEFG\n")
        self.data["cluster"] = {"offset": -2, "array_chunk": 0}
        _, report = self.checked()
        text = "\n".join(report.errors)
        for expected in ["inputs.force", "mystery", "af3_num_recycles", "exactly one", "offset", "array_chunk"]:
            self.assertIn(expected, text)

    def test_job_name_validation_and_snapshot(self):
        for value in ("", "crebrf-wt", "screen_2.test"):
            with self.subTest(value=value):
                self.data["cluster"] = {"job_name": value}
                config, report = self.checked()
                self.assertFalse(report.errors, report.errors)
                artifacts, _ = self.render(config, report)
                raw = json.loads((artifacts / "resolved_config.json").read_text())["config"]
                loaded_report = lp.Report()
                restored = lp.load_config(self.config_path, loaded_report, raw=raw)
                self.assertFalse(loaded_report.errors, loaded_report.errors)
                self.assertEqual(restored.cluster.job_name, value)
                # A queued job's older snapshot has no new optional field.
                del raw["cluster"]["job_name"]
                self.assertEqual(lp.load_config(self.config_path, lp.Report(), raw=raw).cluster.job_name, "")
        for value in ("two words", "a|b", "../name", "-name", "a/b"):
            with self.subTest(value=value):
                self.data["cluster"] = {"job_name": value}
                _, report = self.checked()
                self.assertTrue(any("cluster.job_name" in e for e in report.errors), report.errors)

    def test_public_cluster_names_preserve_snapshot_format(self):
        self.data["cluster"] = dict(max_concurrent_tasks=4, user_queue_budget=20,
                                    reserved_queue_slots=1)
        config, report = self.checked()
        self.assertFalse(report.errors, report.errors)
        self.assertEqual((config.cluster.array_chunk, config.cluster.max_submit_jobs,
                          config.cluster.submit_reserve), (4, 20, 1))
        artifacts, _ = self.render(config, report)
        snapshot = json.loads((artifacts / "resolved_config.json").read_text())["config"]
        self.assertEqual(snapshot["cluster"]["max_submit_jobs"], 20)
        self.assertNotIn("user_queue_budget", snapshot["cluster"])
        before = json.dumps(snapshot, sort_keys=True)
        restored_report = lp.Report()
        restored = lp.load_config(artifacts / "resolved_config.json", restored_report, raw=snapshot)
        self.assertFalse(restored_report.errors, restored_report.errors)
        self.assertEqual(restored.cluster, config.cluster)
        self.assertEqual(json.dumps(snapshot, sort_keys=True), before)

    def test_public_queue_budget_disabled_and_invalid_values(self):
        for settings in ({}, {"user_queue_budget": None}, {"max_submit_jobs": 0}):
            with self.subTest(settings=settings):
                self.data["cluster"] = settings
                config, report = self.checked()
                self.assertFalse(report.errors, report.errors)
                self.assertEqual(config.cluster.max_submit_jobs, 0)
        for value in (0, -1, True, False, 2.5, "null", "20"):
            with self.subTest(value=value):
                self.data["cluster"] = {"user_queue_budget": value}
                _, report = self.load()
                self.assertTrue(any("user_queue_budget" in e for e in report.errors), report.errors)

    def test_cluster_alias_conflicts_and_snapshot_input_not_mutated(self):
        for public, legacy, value in (("max_concurrent_tasks", "array_chunk", 4),
                                      ("user_queue_budget", "max_submit_jobs", 20),
                                      ("reserved_queue_slots", "submit_reserve", 1)):
            with self.subTest(public=public):
                raw = {**self.data, "cluster": {public: value, legacy: value}}
                before = json.dumps(raw, sort_keys=True)
                report = lp.Report()
                lp.load_config(self.config_path, report, raw=raw)
                self.assertTrue(any("specify only one" in e for e in report.errors), report.errors)
                self.assertEqual(json.dumps(raw, sort_keys=True), before)

    def test_public_cluster_names_keep_validation(self):
        for settings, field in ((dict(max_concurrent_tasks=0), "max_concurrent_tasks"),
                                (dict(reserved_queue_slots=-1), "reserved_queue_slots"),
                                (dict(user_queue_budget=2, reserved_queue_slots=1), "user_queue_budget")):
            with self.subTest(settings=settings):
                self.data["cluster"] = settings
                _, report = self.checked()
                self.assertTrue(any(field in e for e in report.errors), report.errors)

    def test_submission_settings_validation_and_snapshot(self):
        self.data["cluster"] = dict(max_submit_jobs=50, submit_reserve=2,
                                    poll_interval=30, array_chunk=4,
                                    dispatcher_time_limit="2-00:00:00")
        config, report = self.checked()
        self.assertFalse(report.errors, report.errors)
        artifacts, _ = self.render(config, report)
        saved = json.loads((artifacts / "resolved_config.json").read_text())["config"]["cluster"]
        for key, value in self.data["cluster"].items():
            self.assertEqual(saved[key], value)
        self.data["cluster"].update(max_submit_jobs=2, poll_interval=0,
                                    dispatcher_time_limit="bad")
        _, report = self.checked()
        for field in ("max_submit_jobs", "poll_interval", "dispatcher_time_limit"):
            self.assertTrue(any(field in error for error in report.errors), report.errors)

    def test_chained_mode_validation_and_budget(self):
        self.data['cluster'] = dict(dispatch_mode='chained', max_submit_jobs=3, submit_reserve=1)
        _, report = self.checked()
        self.assertTrue(any('dispatcher slot' in e for e in report.errors))
        self.data['cluster']['max_submit_jobs'] = 4
        _, report = self.checked()
        self.assertFalse(report.errors, report.errors)
        self.data['cluster'].update(screen_cluster='gpu', inputs_cluster='cpu')
        _, report = self.checked()
        self.assertTrue(any('matching screen_cluster' in e for e in report.errors))
        self.data['cluster'].update(dispatch_mode='invalid', dispatcher_cpus_per_task=0)
        _, report = self.checked()
        self.assertTrue(any('dispatch_mode' in e for e in report.errors))
        self.assertTrue(any('dispatcher_cpus_per_task' in e for e in report.errors))

    def test_submission_budget_accounts_for_dispatcher(self):
        cases = [
            ("", "", 2, True),
            ("site", "site", 2, True),
            ("gpu", "cpu", 2, False),
            ("", "", 3, False),
            ("", "", 20, False),
            ("", "", 0, False),
        ]
        for screen, inputs, budget, invalid in cases:
            with self.subTest(screen=screen, inputs=inputs, budget=budget):
                self.data["cluster"] = dict(dispatch_mode="persistent", screen_cluster=screen, inputs_cluster=inputs,
                                            max_submit_jobs=budget, submit_reserve=1, array_chunk=4)
                _, report = self.checked()
                self.assertEqual(bool(report.errors), invalid, report.errors)
                if invalid:
                    self.assertIn("dispatcher slot", "\n".join(report.errors))
                limited = budget > 0 and not invalid and budget < 6
                self.assertEqual(any("array task slot(s)" in w for w in report.warnings), limited)

    def test_ambiguous_cluster_budget_warns(self):
        for screen, inputs in [("gpu", ""), ("", "cpu")]:
            with self.subTest(screen=screen, inputs=inputs):
                self.data["cluster"] = dict(dispatch_mode="persistent", screen_cluster=screen, inputs_cluster=inputs,
                                            max_submit_jobs=2, submit_reserve=1, array_chunk=1)
                _, report = self.checked()
                self.assertFalse(report.errors, report.errors)
                self.assertTrue(any("default resolves to the same cluster" in w for w in report.warnings))

    def test_array_index_limit_warns_about_unreachable_concurrency(self):
        self.data["cluster"] = dict(max_array_size=4, array_chunk=4)
        _, report = self.checked()
        self.assertFalse(report.errors, report.errors)
        self.assertTrue(any("one active array cannot reach" in w for w in report.warnings))
        self.data["cluster"]["array_chunk"] = 3
        _, report = self.checked()
        self.assertFalse(any("one active array cannot reach" in w for w in report.warnings))

    def test_generated_manifest_subset_resolves_in_bind_mount(self):
        self.data["scorer"]["models"]["not_requested"] = {"path": "missing-unused.pt"}
        self.data["scorer"]["predict_metrics"] = ["invented_metric_421"]
        config, report = self.checked()
        self.assertFalse(report.errors, report.errors)
        artifacts, commands = self.render(config, report)
        manifest = json.loads((artifacts / "scorer_manifest.json").read_text())
        self.assertEqual(set(manifest["models"]), {"invented_metric_421"})
        relative = manifest["models"]["invented_metric_421"]["path"]
        self.assertFalse(Path(relative).is_absolute())
        self.assertNotIn("..", Path(relative).parts)
        self.assertFalse((artifacts / relative).is_symlink())
        self.assertEqual((artifacts / relative).read_bytes(), (self.models / "first.pt").read_bytes())
        env = json.loads((artifacts / "resolved_config.json").read_text())["environment"]
        self.assertEqual(env["AF3_SCORER_DIR"], str(artifacts))
        container_manifest = Path(env["LATENTPPI_MODEL_MANIFEST"])
        self.assertEqual(container_manifest, Path("/root/scorer/scorer_manifest.json"))
        host_equivalent = artifacts / (container_manifest.parent / relative).relative_to("/root/scorer")
        self.assertTrue(host_equivalent.is_file())
        self.assertEqual(env["AF3_SCORER_FILENAME"], relative)
        self.assertEqual(env["SINGULARITYENV_LATENTPPI_MODEL_MANIFEST"], str(container_manifest))
        self.assertIn("--dependency=afterok:<inputs-job-id>", commands)

    def test_manifest_override_is_authoritative_and_unchanged(self):
        manifest = self.models / "manual.json"
        manifest.write_text(json.dumps({"rank_by": "external", "models": {"external": {"path": "first.pt"}}}))
        before = manifest.read_bytes()
        self.data["scorer"] = {"manifest_path": str(manifest), "models": "ignored",
                               "predict_metrics": ["wrong"], "rank_by": 123}
        config, report = self.checked()
        self.assertFalse(report.errors, report.errors)
        self.assertEqual(report.rank_by, "external")
        artifacts, _ = self.render(config, report)
        self.assertFalse((artifacts / "scorer_manifest.json").exists())
        self.assertEqual(manifest.read_bytes(), before)
        env = json.loads((artifacts / "resolved_config.json").read_text())["environment"]
        self.assertEqual(env["AF3_SCORER_DIR"], str(self.models))
        self.assertEqual(env["AF3_SCORER_FILENAME"], "first.pt")

    def test_manifest_rejects_invalid_rank_and_escaping_paths(self):
        manifest = self.models / "manual.json"
        self.data["scorer"] = {"manifest_path": str(manifest)}
        for data in [dict(rank_by="absent", models={"a": {"path": "first.pt"}}),
                     dict(rank_by="a", models={"a": {"path": "../outside.pt"}}),
                     dict(rank_by="a", models={"a": {"path": str(self.models / "first.pt")}})]:
            manifest.write_text(json.dumps(data))
            _, report = self.checked()
            self.assertTrue(report.errors)

    def test_env_quoting_nul_export_and_reproducible_defaults(self):
        self.data["cluster"] = {"mail_user": "a,b+test@example.org"}
        config, report = self.checked()
        artifacts, _ = self.render(config, report)
        dump = json.loads((artifacts / "resolved_config.json").read_text())
        self.assertEqual(dump["config"]["stages"]["pool_seed"], 1)
        self.assertEqual(len(dump["git_commit"]), 40)
        self.assertIn("screen.sh", dump["script_sha256"])
        pairs = (artifacts / "sbatch.env").read_bytes().split(b"\0")
        exported = dict(item.decode().split("=", 1) for item in pairs if item)
        self.assertEqual(exported, dump["environment"])
        result = subprocess.run(["bash", "-c", 'set -Eeuo pipefail; source "$1"; "$PYTHON_BIN" -c '
                                 + shlex.quote('import os,json; print(json.dumps(dict(os.environ)))'),
                                 "bash", str(artifacts / "env.sh")], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(result.stdout)
        for key, value in exported.items():
            self.assertEqual(actual[key], value, key)
        self.assertNotIn("SLURM_SUBMIT_DIR", exported)
        before = (artifacts / "resolved_config.json").read_bytes()
        self.render(config, report)
        self.assertEqual((artifacts / "resolved_config.json").read_bytes(), before)

    def test_array_count_offset_and_stale_msa(self):
        self.run.mkdir()
        (self.run / "af3_cmds.txt").write_text("command1\ncommand2\ncommand3\n")
        inputs = self.run / "af3_inputs"
        inputs.mkdir()
        (inputs / "pair.json").write_text(json.dumps({"sequences": [{"protein": {
            "unpairedMsaPath": str(self.root / "missing.a3m")}}]}))
        (self.run / "target.a3m").write_text(">target\nACDEFG\n")
        self.data["cluster"] = {"offset": 1, "array_chunk": 2}
        config, report = self.checked()
        self.assertEqual(report.pair_count, 3)
        self.assertTrue(any("missing.a3m" in e for e in report.errors))
        self.assertTrue(any("SINGLE-SEQUENCE" in w for w in report.warnings))
        commands = lp.submission_commands(config, report, self.run / "latentppi_config")
        self.assertIn("dispatch.sh", commands["dispatch"])
        self.assertIn("--partition=your_cpu", commands["inputs"])
        self.assertNotIn("--dry-run", commands["inputs"])
        self.assertNotIn("--tools", commands["inputs"])

    def test_conda_and_sinfo_checks(self):
        config, report = self.load()
        def result(command, **kwargs):
            if command[0] == "conda":
                return subprocess.CompletedProcess(command, 0, json.dumps({"envs": ["/envs/colabfold"]}), "")
            return subprocess.CompletedProcess(command, 0, "your_gpu*|up\nyour_cpu|up\n", "")
        with patch.object(lp.shutil, "which", side_effect=lambda name: name), patch.object(lp, "run_check", side_effect=result):
            lp.validate(config, report)
        self.assertFalse(report.errors, report.errors)
        config.cluster.partition = "missing"
        config.inputs.conda_env = "missing"
        report = lp.Report()
        with patch.object(lp.shutil, "which", side_effect=lambda name: name), patch.object(lp, "run_check", side_effect=result):
            lp.validate(config, report)
        self.assertTrue(any("conda_env not found" in e for e in report.errors))
        self.assertTrue(any("partition 'missing'" in e for e in report.errors))

    def test_cluster_names_and_remote_partition_validation(self):
        self.data['cluster'] = dict(dispatch_mode='persistent', screen_cluster='gpu_site', inputs_cluster='cpu_site')
        config, report = self.load()
        def result(command, **kwargs):
            partition = 'your_gpu' if '-Mgpu_site' in command else 'your_cpu'
            return subprocess.CompletedProcess(command, 0, partition + '|up\n', '')
        with patch.object(lp.shutil, 'which', side_effect=lambda name: 'sinfo' if name == 'sinfo' else None), \
             patch.object(lp, 'run_check', side_effect=result) as run:
            lp.validate(config, report)
        self.assertFalse(report.errors, report.errors)
        commands = [call.args[0] for call in run.call_args_list if call.args[0][0] == 'sinfo']
        self.assertEqual(len(commands), 2)
        self.assertTrue(any('-Mgpu_site' in command for command in commands))
        self.assertTrue(any('-Mcpu_site' in command for command in commands))
        for invalid in ['a,b', 'all', 'bad name']:
            config.cluster.screen_cluster = invalid
            report = lp.Report()
            lp.validate(config, report)
            self.assertTrue(any('cluster.screen_cluster' in error for error in report.errors))

    def test_checkpoint_load_failure_names_metric(self):
        self.load_mock.return_value = {"invented_metric_421": "bad checkpoint"}
        _, report = self.checked()
        self.assertTrue(any("invented_metric_421" in e and "bad checkpoint" in e for e in report.errors))

    def test_twenty_metrics_and_full_runner(self):
        self.data["scorer"]["models"] = {f"dynamic_{i}": {"path": str(self.models / "first.pt")} for i in range(20)}
        self.data["scorer"]["rank_by"] = "dynamic_19"
        config, report = self.checked()
        self.assertFalse(report.errors, report.errors)
        self.assertEqual(len(report.model_paths), 20)
        self.data["runner"] = "full"
        del self.data["scorer"]
        config, report = self.checked()
        self.assertFalse(report.errors, report.errors)
        artifacts, commands = self.render(config, report)
        self.assertEqual(json.loads((artifacts / "resolved_config.json").read_text())["config"]["runner"], "full")
        self.assertFalse((artifacts / "scorer_manifest.json").exists())

    def test_cli_never_submits_and_errors_prevent_artifacts(self):
        self.load()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            status = lp.main(["render", str(self.config_path)])
        self.assertEqual(status, 0)
        self.assertIn("would submit", stdout.getvalue())
        self.data["inputs"]["output_base"] = str(self.root / "invalid-run")
        self.data["scorer"]["threshold"] = -1
        self.load()
        with contextlib.redirect_stderr(io.StringIO()):
            status = lp.main(["render", str(self.config_path)])
        self.assertEqual(status, 1)
        self.assertFalse((self.root / "invalid-run/latentppi_config").exists())

    def test_status_defaults_to_light_and_accepts_heavy(self):
        self.load()
        with patch.object(sw, "show_status") as status:
            self.assertEqual(lp.main(["status", str(self.config_path)]), 0)
            self.assertFalse(status.call_args.kwargs["heavy"])
        with patch.object(sw, "show_status") as status:
            self.assertEqual(lp.main(["status", str(self.config_path), "--heavy"]), 0)
            self.assertTrue(status.call_args.kwargs["heavy"])

    def test_bad_config_stops_before_default_validation(self):
        for content in (None, "inputs: [broken", "runner: full\nrunner: latent\n"):
            path = self.root / "bad.yaml"
            if content is not None:
                path.write_text(content)
            stderr = io.StringIO()
            with patch.object(lp, "validate") as validate, contextlib.redirect_stderr(stderr):
                self.assertEqual(lp.main(["run", str(path)]), 1)
            validate.assert_not_called()
            self.assertNotIn("your_cpu", stderr.getvalue())
            self.assertIn("ERROR:", stderr.getvalue())

    def test_stage_zero_honors_env_and_force_with_mock_colabfold(self):
        # Run the actual screen.sh; only external ColabFold/conda and run_pipeline are mocked.
        binary = self.root / "bin"
        binary.mkdir()
        scripts = self.root / "scripts"
        scripts.mkdir()
        (scripts / "query_msa.py").write_text((REPO / "query_msa.py").read_text())
        (scripts / "run_pipeline.py").write_text(
            'import json,os,sys\nfrom pathlib import Path\n'
            'Path(os.environ["OUTPUT_BASE"], "args.json").write_text(json.dumps(sys.argv[1:]))\n'
            'Path(os.environ["OUTPUT_BASE"], "af3_cmds.txt").write_text("one\\n")\n')
        for name, text in {
            "conda": "#!/bin/bash\nexit 0\n",
            "colabfold_batch": '#!/bin/bash\nprintf ">new\\nACDEFGHIK\\n>hit\\nACDEFGHIK\\n" > "$2/new.a3m"\n',
        }.items():
            file = binary / name
            file.write_text(text)
            file.chmod(0o755)
        (binary / "python").symlink_to(sys.executable)
        self.run.mkdir()
        (self.run / "target_with_name.a3m").write_text(">old\nACDEFGHIK\n")
        env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
               "AF3_SCRIPT_DIR": str(scripts), "OUTPUT_BASE": str(self.run),
               "FASTA_FILE": str(self.root / "target_with_name.fasta"),
               "PROTEOME_MSA_DIR": str(self.root / "proteome"), "FORCE": "1",
               "FILTER_NON_STANDARD": "0", "RECREATE_MANIFEST": "1", "RECREATE_COMMANDS": "1",
               "AF3_NUM_RECYCLES": "7", "AF3_NUM_DIFFUSION_SAMPLES": "3", "MAX_IDENTITY": "82"}
        result = subprocess.run(["bash", str(REPO / "screen.sh")], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads((self.run / "args.json").read_text())
        for flag in ["--force", "--no-filter-non-standard", "--recreate-manifest", "--recreate-commands"]:
            self.assertIn(flag, args)
        self.assertEqual(args[args.index("--af3-num-recycles") + 1], "7")
        self.assertEqual(args[args.index("--af3-num-diffusion-samples") + 1], "3")
        self.assertEqual(args[args.index("--msa-a") + 1], str(self.run / "target_with_name.a3m"))
        self.assertTrue((self.run / "target_with_name.a3m").read_text().startswith(">new\n"))
        self.assertEqual(len(list(self.run.glob("*.a3m"))), 1)
        self.assertNotIn("--dry-run", args)
        self.assertNotIn("--tools", args)
        self.assertIn("Prepared pairs: 1; proteome A3Ms: 1; max_identity: 82", result.stdout)

    def test_duplicate_yaml_and_bind_delimiters_are_reported(self):
        self.config_path.write_text("runner: full\nrunner: latent\n")
        report = lp.Report()
        lp.load_config(self.config_path, report)
        self.assertTrue(any("duplicate YAML key" in e for e in report.errors))
        self.data["paths"]["af3_output_dir"] = "comma,directory"
        _, report = self.checked()
        self.assertTrue(any("--bind" in e for e in report.errors))

    def test_printed_commands_execute_only_against_stub_scheduler(self):
        config, report = self.checked()
        artifacts, _ = self.render(config, report)
        (self.run / "af3_cmds.txt").write_text("first\nsecond\n")
        binary = self.root / "stub-bin"
        binary.mkdir()
        stub = binary / "sbatch"
        record = self.root / "sbatch-args.jsonl"
        stub.write_text(f"#!{sys.executable}\nimport json,sys,os\nfrom pathlib import Path\n"
                        "assert '--export=ALL' in sys.argv\n"
                        "assert not any(a.startswith('--export-file=') for a in sys.argv)\n"
                        "assert 'UNRECORDED_OVERRIDE' not in os.environ\n"
                        "assert os.environ['FASTA_FILE'] == \"space , quote' dollar$ semicolon;\"\n"
                        f"with Path({str(record)!r}).open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
        stub.chmod(0o755)
        config.inputs.fasta_file = "space , quote' dollar$ semicolon;"
        with patch.dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"]):
            commands = lp.submission_commands(config, report, artifacts)
        for command in commands.values():
            result = subprocess.run(["bash", "-c", command], text=True, capture_output=True,
                                    env={**os.environ, "UNRECORDED_OVERRIDE": "bad"})
            self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in record.read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        self.assertIn("--dependency=afterok:<inputs-job-id>", calls[1])
        self.assertIn("--export=ALL", calls[1])
        self.assertEqual(calls[1][-1], str(artifacts / "dispatch.sh"))

    def test_latent_dry_run_preserves_requested_recycles(self):
        self.run.mkdir()
        inputs = self.run / "af3_inputs"
        inputs.mkdir()
        input_json = inputs / "pair.json"
        input_json.write_text(json.dumps({"name": "pair", "modelSeeds": [1]}))
        cmd_file = self.run / "af3_cmds.txt"
        cmd_file.write_text(shlex.join(["run_alphafold.py", "--json_path=" + str(input_json),
                                       "--output_dir=" + str(self.run / "predictions"), "--num_recycles=7"]) + "\n")
        env = {**os.environ, "AF3_SCRIPT_DIR": str(REPO), "DRY_RUN": "1", "OFFSET": "0",
               "AF3_INPUT_DIR": str(inputs), "AF3_OUTPUT_DIR": str(self.run / "predictions"),
               "PYTHON_BIN": sys.executable, "TMPDIR": str(self.root), "RUN_POOL": "0",
               "SLURM_ARRAY_TASK_ID": "1", "DEFAULT_NUM_RECYCLES": "1"}
        result = subprocess.run(["bash", str(REPO / "run_af3_latent_compress_pool.sh"), str(cmd_file)],
                                cwd=self.run, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--num_recycles=7", result.stdout)
        self.assertNotIn("--num_recycles 1", result.stdout)


class CheckpointRuntimeTests(unittest.TestCase):
    def test_existing_runtime_uses_safe_cpu_load(self):
        config = lp.Config()
        result = subprocess.CompletedProcess([], 0, '{"novel": "bad archive"}\n', "")
        with patch.object(lp.importlib.util, "find_spec", return_value=object()), \
                patch.object(lp, "run_check", return_value=result) as run:
            errors = lp.load_checkpoints({"novel": "/tmp/model.pt"}, config)
        self.assertEqual(errors, {"novel": "bad archive"})
        self.assertIn('map_location="cpu", weights_only=True', run.call_args.args[0][2])

    def test_old_runtime_supports_numpy_and_rejects_arbitrary_globals(self):
        import numpy as np
        with tempfile.TemporaryDirectory(prefix=".checkpoint-test-", dir=REPO) as temporary:
            root = Path(temporary)
            safe = root / "safe.pt"
            safe.write_bytes(pickle.dumps({"feature_mean": np.array([1, 2], dtype=np.float32)}))
            marker = root / "must-not-exist"
            class Unsafe:
                def __reduce__(self):
                    return os.system, ("touch " + shlex.quote(str(marker)),)
            unsafe = root / "unsafe.pt"
            unsafe.write_bytes(pickle.dumps(Unsafe()))
            def fake_load(path, **kwargs):
                with open(path, "rb") as handle:
                    return kwargs["pickle_module"].load(handle)
            fake_torch = types.SimpleNamespace(serialization=types.SimpleNamespace(), load=fake_load)
            output = io.StringIO()
            with patch.dict(sys.modules, {"torch": fake_torch}), \
                    patch.object(sys, "argv", ["loader", json.dumps({"safe": str(safe), "unsafe": str(unsafe)})]), \
                    contextlib.redirect_stdout(output):
                exec(lp.CHECKPOINT_CODE, {})
            errors = json.loads(output.getvalue())
            self.assertNotIn("safe", errors)
            self.assertIn("unsupported checkpoint global", errors["unsafe"])
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()

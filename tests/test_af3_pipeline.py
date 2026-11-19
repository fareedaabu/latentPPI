"""Exercise real pooling/compression with a mock GPU launcher and ipSAE parser."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pool_embeddings import pool_pair, pool_single


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='af3-pipeline-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'inputs/jsons'
        self.inputs.mkdir(parents=True)
        (self.inputs.parent / 'msa_cache').mkdir()
        self.json = self.inputs / 'filename_different_from_name.json'
        self.json.write_text(json.dumps({'name': 'P12345_Q99999', 'modelSeeds': [1]}))
        self.output = self.root / 'output with spaces'
        self.job = self.output / 'p12345_q99999'
        self.shards = self.root / 'h5_shards'
        self.shard = self.shards / 'recycle_7/embeddings_p12345_q99999.h5'
        self.summary = self.root / 'dataset_summary.tsv'
        self.summary.write_text('uniprot_a\tuniprot_b\tlen_a\tlen_b\tlabel\tevidence_count\nP12345\tQ99999\t1\t2\t1\t3\n')
        self.cmds = self.root / 'commands.txt'
        self.cmds.write_text(shlex.join([
            'run_alphafold.py', '--json_path', str(self.json),
            '--output_dir=' + str(self.output), '--norun_data_pipeline',
            '--num_recycles', '7', '--num_diffusion_samples=2',
        ]) + '\n')
        self.launcher = self.root / 'singularity'
        self.launcher.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys
from pathlib import Path
import numpy as np
args = sys.argv[1:]
Path(os.environ['ARGS_LOG']).write_text(json.dumps(args))
if os.environ.get('FAIL_FOLD'):
    sys.exit(7)
output = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--output_dir=')))
json_path = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--json_path=')))
name = json.loads(json_path.read_text())['name'].lower()
if os.environ.get('PRESERVE_NAME_CASE'):
    name = json.loads(json_path.read_text())['name']
job = output / name
embed = job / ('seed-' + os.environ.get('POOL_SEED', '1') + '_embeddings')
embed.mkdir(parents=True, exist_ok=True)
if not os.environ.get('GATED_FOLD'):
    np.savez_compressed(embed / (name + '_seed-' + os.environ.get('POOL_SEED', '1') + '_embeddings.npz'),
        pair_embeddings=np.arange(3*3*128, dtype=np.float16).reshape(3,3,128),
        single_embeddings=np.arange(3*384, dtype=np.float16).reshape(3,384))
if not os.environ.get('MISSING_PRE'):
    np.savez_compressed(embed / (name + '_seed-' + os.environ.get('POOL_SEED', '1') + '_pre_pairformer_embeddings.npz'),
        pair_embeddings=np.full((3,3,128), 5, dtype=np.float16),
        single_embeddings=np.full((3,384), 6, dtype=np.float16))
if os.environ.get('GATED_FOLD'):
    (job / 'prediction_timings.jsonl').write_text(json.dumps({
        'input_name': name, 'seed': int(os.environ.get('POOL_SEED', '1')),
        'latentppi_scorer_gate_triggered': True,
        'latentppi_scorer_gate_status': 'ok'}) + '\\n')
else:
    for rel in [name + '_model.cif', name + '_confidences.json',
                name + '_summary_confidences.json', 'seed-1_sample-0/model.cif',
                'seed-1_sample-0/confidences.json', 'seed-1_sample-0/summary_confidences.json']:
        p = job / rel
        p.parent.mkdir(exist_ok=True)
        p.write_text(os.environ.get('FOLD_CONTENT', 'new fold contents'))
''')
        self.launcher.chmod(0o755)
        self.parser = self.root / 'parse.py'
        self.parser.write_text("import os, sys\nsys.exit(8 if os.environ.get('FAIL_PARSE') else 0)\n")
        image = self.root / 'image.sif'
        image.touch()
        self.env = {**os.environ,
            'SLURM_SUBMIT_DIR': str(self.root), 'SLURM_ARRAY_TASK_ID': '1',
            'AF3_SCRIPT_DIR': str(REPO), 'PYTHON_BIN': sys.executable,
            'POOL_PYTHON': sys.executable, 'IPSAE_PYTHON': sys.executable,
            'IPSAE_PARSER_PY': str(self.parser), 'IPSAE_PY': str(self.parser),
            'SINGULARITY_BIN': str(self.launcher), 'AF3_IMAGE': str(image),
            'AF3_MODEL_PARAMETERS_DIR': str(self.root), 'AF3_DATABASES_DIR': str(self.root),
            'ZSTD_BIN': shutil.which('zstd') or 'zstd', 'ZSTD_LEVEL': '1',
            'ZSTD_THREADS': '1', 'ARGS_LOG': str(self.root / 'args.json'),
            'RUN_IPSAE_PARSE': '1', 'RUN_POOL': '1', 'RUN_COMPRESS': '1',
            'SAVE_EMBEDDINGS': '1', 'KEEP_NPZ': '0', 'COMPRESS_EMBEDDINGS': '0', 'DRY_RUN': '0', 'OFFSET': '0',
        }
        # Avoid inherited user overrides changing test fixtures.
        for key in ['AF3_INPUT_DIR', 'AF3_OUTPUT_DIR', 'CMD_BASE_DIR', 'SUMMARY_TSV',
                    'SHARD_DIR', 'POOL_PY', 'POOL_SEED', 'AF3_MSA_DIR']:
            self.env.pop(key, None)
        self.env['AF3_INPUT_DIR'] = str(self.inputs)
        self.env['SUMMARY_TSV'] = str(self.summary)

    def run_pipeline(self, expected=0, **env):
        result = subprocess.run(['bash', str(REPO / 'run_af3_compress_pool.sh'), str(self.cmds)],
                                cwd=self.root, env={**self.env, **env}, text=True, capture_output=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_cpu_and_gpu_container_flags_both_runners(self):
        for runner in (self.run_pipeline, self.run_latent_pipeline):
            for cpu in ('0', '1'):
                with self.subTest(runner=runner.__name__, cpu=cpu):
                    result = runner(DRY_RUN='1', AF3_CPU_ONLY=cpu)
                    self.assertEqual('--nv' in result.stdout, cpu == '0')

    def test_legacy_runners_without_save_embeddings_setting(self):
        # Slurm retains the submitted runner, but it sources the current helper.
        for name in ('run_af3_compress_pool.sh', 'run_af3_latent_compress_pool.sh'):
            with self.subTest(runner=name):
                script = self.root / ('legacy_' + name)
                source = (REPO / name).read_text()
                setting = 'SAVE_EMBEDDINGS=${SAVE_EMBEDDINGS:-0}\n'
                self.assertIn(setting, source)
                script.write_text(source.replace(setting, ''))
                env = {**self.env, 'DRY_RUN': '1', 'RUN_POOL': '1'}
                env.pop('SAVE_EMBEDDINGS', None)
                result = subprocess.run(['bash', str(script), str(self.cmds)],
                                        cwd=self.root, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('--save_embeddings', result.stdout)
                self.assertIn('LATENTPPI_SAVE_PRE_PAIRFORMER=1', result.stdout)
                self.assertNotIn('--nosave_embeddings', result.stdout)

    def test_disable_embedding_exports_both_runners(self):
        for runner in (self.run_pipeline, self.run_latent_pipeline):
            with self.subTest(runner=runner.__name__):
                result = runner(DRY_RUN='1', SAVE_EMBEDDINGS='0', RUN_POOL='0')
                self.assertIn('--nosave_embeddings', result.stdout)
                self.assertNotIn('--save_embeddings', result.stdout)
                self.assertIn('LATENTPPI_SAVE_PRE_PAIRFORMER=0', result.stdout)
                result = runner(DRY_RUN='1', SAVE_EMBEDDINGS='1', RUN_POOL='0')
                self.assertIn('--save_embeddings', result.stdout)
                self.assertNotIn('--nosave_embeddings', result.stdout)
                self.assertIn('LATENTPPI_SAVE_PRE_PAIRFORMER=1', result.stdout)
                result = runner(expected=1, DRY_RUN='1', SAVE_EMBEDDINGS='invalid', RUN_POOL='0')
                self.assertIn('SAVE_EMBEDDINGS must be 0 or 1', result.stderr)
                result = runner(expected=1, DRY_RUN='1', SAVE_EMBEDDINGS='0', RUN_POOL='1')
                self.assertIn('RUN_POOL=1 requires SAVE_EMBEDDINGS=1', result.stderr)

    def test_full_flow(self):
        self.run_pipeline()
        self.assertFalse((self.job / 'seed-1_embeddings').exists())
        for rel in ['p12345_q99999_model.cif', 'p12345_q99999_confidences.json',
                    'seed-1_sample-0/model.cif', 'seed-1_sample-0/confidences.json']:
            source = self.job / rel
            self.assertFalse(source.exists())
            archive = Path(str(source) + '.zst')
            data = subprocess.check_output([self.env['ZSTD_BIN'], '-dc', str(archive)])
            self.assertEqual(data, b'new fold contents')
        self.assertTrue((self.job / 'seed-1_sample-0/summary_confidences.json').exists())
        with h5py.File(self.shard) as f:
            self.assertEqual(f['system_name'].asstr()[0], 'P12345_Q99999')
            self.assertEqual(f['pair_row_mean'].shape, (1, 128))
            expected = np.arange(3*3*128).reshape(3, 3, 128)[0, 1:, :].mean(axis=0)
            np.testing.assert_array_equal(f['pair_row_mean'][0], expected)
            self.assertEqual(f['neg_source'].asstr()[0], '')
        with h5py.File(self.shards / 'prepairformer/embeddings_p12345_q99999.h5') as f:
            np.testing.assert_array_equal(f['pair_row_mean'][0], np.full(128, 5))
            self.assertEqual(f['system_name'].asstr()[0], 'P12345_Q99999')
        args = json.loads((self.root / 'args.json').read_text())
        self.assertIn('--num_diffusion_samples=2', args)
        self.assertEqual(args[args.index('--num_recycles') + 1], '7')
        self.assertIn('--save_embeddings', args)
        self.assertIn('LATENTPPI_ENABLE=0', args)
    @unittest.skipUnless((REPO / 'merge_shards.py').is_file(), 'optional local shard merger is not bundled')
    def test_optional_shard_merge(self):
        self.run_pipeline()
        expected = np.arange(3*3*128).reshape(3,3,128)[0,1:,:].mean(axis=0)
        csv = self.root / 'scores.csv'
        csv.write_text('system,ipSAE\nP12345_Q99999,0.75\n')
        for stage in ('prepairformer', 'recycle_7'):
            merged = self.root / (stage + '.h5')
            subprocess.run([sys.executable, str(REPO / 'merge_shards.py'),
                            '--shard-dir', str(self.shards / stage), '--output', str(merged)],
                           check=True, capture_output=True)
            subprocess.run([sys.executable, str(REPO / 'dataset_build/add_ipsae.py'),
                            '--hdf5-path', str(merged), '--ipsae-csv', str(csv)],
                           check=True, capture_output=True)
            with h5py.File(merged) as f:
                self.assertEqual(f['system_name'].asstr()[:].tolist(), ['P12345_Q99999'])
                self.assertEqual(f['ipSAE'][0], 0.75)
                np.testing.assert_array_equal(f['pair_row_mean'][0],
                                              np.full(128, 5) if stage == 'prepairformer' else expected)

    def test_ipsae_disabled(self):
        self.run_pipeline(RUN_IPSAE_PARSE='0', FAIL_PARSE='1')
        self.assertTrue(self.shard.exists())
        self.assertTrue((self.job / 'p12345_q99999_model.cif.zst').exists())

    def test_case_preserving_container(self):
        result = self.run_pipeline(PRESERVE_NAME_CASE='1')
        job = self.output / 'P12345_Q99999'
        self.assertIn('[pool] Saved', result.stdout)
        self.assertFalse((job / 'seed-1_embeddings').exists())
        self.assertTrue((job / 'P12345_Q99999_model.cif.zst').exists())
        with h5py.File(self.shard) as f:
            self.assertEqual(f['system_name'].asstr()[:].tolist(), ['P12345_Q99999'])

    def test_ambiguous_case_folders_stop_postprocessing(self):
        (self.output / 'P12345_Q99999').mkdir(parents=True)
        self.run_pipeline(expected=1)
        self.assertFalse(self.shard.exists())
        self.assertTrue((self.job / 'seed-1_embeddings/p12345_q99999_seed-1_embeddings.npz').exists())

    def test_fold_failure_stops_postprocessing(self):
        self.run_pipeline(expected=7, FAIL_FOLD='1')
        self.assertFalse(self.shard.exists())
        self.assertIn('exit=7', (self.root / 'af3_job_durations.txt').read_text())

    def test_parse_failure_preserves_embeddings(self):
        self.run_pipeline(expected=8, FAIL_PARSE='1')
        self.assertFalse(self.shard.exists())
        self.assertTrue((self.job / 'seed-1_embeddings/p12345_q99999_seed-1_embeddings.npz').exists())
        self.assertFalse(list(self.job.rglob('*.zst')))

    def test_pool_failure_preserves_previous_shard_and_raw_files(self):
        self.run_pipeline()
        original = self.shard.read_bytes()
        self.summary.write_text(self.summary.read_text().replace('\t1\t2\t', '\t2\t2\t'))
        self.run_pipeline(expected=1)
        self.assertEqual(self.shard.read_bytes(), original)
        self.assertTrue((self.job / 'seed-1_embeddings/p12345_q99999_seed-1_embeddings.npz').exists())
        self.assertTrue((self.job / 'p12345_q99999_model.cif').exists())
        self.assertFalse(list(self.shards.glob('*.partial')))

    def test_retry_replaces_shard_and_old_archives(self):
        self.run_pipeline()
        self.run_pipeline(FOLD_CONTENT='second fold')
        with h5py.File(self.shard) as f:
            self.assertEqual(f['system_name'].shape, (1,))
        data = subprocess.check_output([self.env['ZSTD_BIN'], '-dc', str(self.job / 'p12345_q99999_model.cif.zst')])
        self.assertEqual(data, b'second fold')

    def test_compression_failure_preserves_source_and_old_archive(self):
        self.run_pipeline()
        archive = self.job / 'p12345_q99999_model.cif.zst'
        before = archive.read_bytes()
        broken = self.root / 'broken_zstd'
        broken.write_text('#!/bin/bash\nexit 9\n')
        broken.chmod(0o755)
        self.run_pipeline(expected=9, ZSTD_BIN=str(broken), KEEP_NPZ='1')
        self.assertEqual(archive.read_bytes(), before)
        self.assertTrue((self.job / 'p12345_q99999_model.cif').exists())
        self.assertFalse(list(self.job.rglob('*.partial')))

    def test_alternate_seed_and_npz_archives(self):
        self.run_pipeline(POOL_SEED='42', COMPRESS_EMBEDDINGS='1')
        self.assertTrue((self.job / 'seed-42_embeddings/p12345_q99999_seed-42_embeddings.npz.zst').exists())
        self.assertFalse((self.job / 'seed-42_embeddings/p12345_q99999_seed-42_embeddings.npz').exists())

    def test_dry_run(self):
        result = self.run_pipeline(DRY_RUN='1')
        self.assertIn('p12345_q99999', result.stdout)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / 'args.json').exists())

    def test_external_msa_directory_is_mounted_and_missing_msa_fails(self):
        shared = self.root / 'shared proteome msas'
        shared.mkdir()
        msa = shared / 'Q99999.a3m'
        msa.write_text('>Q99999\nACDE\n')
        self.json.write_text(json.dumps({
            'name': 'P12345_Q99999', 'modelSeeds': [1],
            'sequences': [{'protein': {'unpairedMsaPath': str(msa)}}],
        }))
        result = self.run_pipeline(DRY_RUN='1')
        self.assertIn(f'{shared}:{shared}:ro'.replace(' ', '\\ '), result.stdout)
        msa.unlink()
        result = self.run_pipeline(expected=1, DRY_RUN='1')
        self.assertIn('MSA referenced by', result.stderr)

    def test_parser_failure_stops_before_inference(self):
        self.cmds.write_text('run_alphafold.py --json_path\n')
        result = self.run_pipeline(expected=1)
        self.assertIn('Missing value for --json_path', result.stderr)
        self.assertFalse((self.root / 'args.json').exists())
        self.assertFalse(self.shards.exists())

    def test_slurm_dry_run_finds_helpers_and_preserves_flags(self):
        result = self.run_pipeline(DRY_RUN='1', SLURM_JOB_ID='123')
        self.assertIn('LATENTPPI_SAVE_PRE_PAIRFORMER=1', result.stdout)
        self.assertIn('--num_diffusion_samples=2', result.stdout)
        self.assertIn('--num_recycles 7', result.stdout)
        self.assertFalse(self.output.exists())

    def run_latent_pipeline(self, expected=0, **env):
        # The stop command file omits --num_recycles and uses one-based arrays.
        self.cmds.write_text(shlex.join([
            'run_alphafold.py', '--json_path', str(self.json),
            '--output_dir=' + str(self.output), '--num_diffusion_samples=1',
        ]) + '\n')
        (self.root / 'latentppi_max_ipSAE_d0dom_model.pt').touch()
        latent_env = {**self.env,
                      'SLURM_ARRAY_TASK_ID': '1',
                      'AF3_OUTPUT_DIR': str(self.output),
                      'SUMMARY_TSV': str(self.summary),
                      'SHARD_DIR': str(self.shards),
                      'AF3_SCORER_DIR': str(self.root)}
        latent_env.pop('OFFSET', None)  # Exercise the runner's one-based default.
        result = subprocess.run(
            ['bash', str(REPO / 'run_af3_latent_compress_pool.sh'), str(self.cmds)],
            cwd=self.root, env={**latent_env, **env}, text=True, capture_output=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_latent_full_flow_with_one_based_array(self):
        self.run_latent_pipeline()
        args = json.loads((self.root / 'args.json').read_text())
        for flag in ('--num_recycles=1', '--flash_attention_implementation=xla',
                     '--norun_data_pipeline', '--save_embeddings',
                     'LATENTPPI_SAVE_PRE_PAIRFORMER=1',
                     'XLA_FLAGS=--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'):
            self.assertIn(flag, args)
        self.assertTrue((self.shards / 'prepairformer/embeddings_p12345_q99999.h5').exists())
        self.assertTrue((self.shards / 'recycle_1/embeddings_p12345_q99999.h5').exists())
        self.assertTrue((self.job / 'p12345_q99999_model.cif.zst').exists())

    def test_latent_gated_fold_pools_pre_only_and_skips_ipsae(self):
        result = self.run_latent_pipeline(GATED_FOLD='1', FAIL_PARSE='1')
        self.assertIn('Scorer gate skipped diffusion', result.stdout)
        self.assertTrue((self.shards / 'prepairformer/embeddings_p12345_q99999.h5').exists())
        self.assertFalse((self.shards / 'recycle_1').exists())
        self.assertFalse(list(self.job.rglob('*.zst')))

    def test_duplicate_task_lock(self):
        self.output.mkdir()
        with (self.output / '.p12345_q99999.pipeline.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.run_pipeline(expected=1)
        self.assertFalse((self.root / 'args.json').exists())

    def test_missing_stage_preserves_raw_files(self):
        self.run_pipeline(expected=1, MISSING_PRE='1')
        self.assertFalse(self.shard.exists())
        self.assertTrue((self.job / 'seed-1_embeddings/p12345_q99999_seed-1_embeddings.npz').exists())

    def test_second_stage_failure_preserves_both_previous_shards(self):
        self.run_pipeline()
        pre = self.shards / 'prepairformer/embeddings_p12345_q99999.h5'
        originals = [p.read_bytes() for p in (pre, self.shard)]
        broken_pooler = self.root / 'pool.py'
        broken_pooler.write_text(
            'import runpy, sys\n'
            'if "recycle_7" in sys.argv[sys.argv.index("--hdf5-path") + 1]: sys.exit(9)\n'
            f'runpy.run_path({str(REPO / "pool_embeddings.py")!r}, run_name="__main__")\n')
        self.run_pipeline(expected=9, POOL_PY=str(broken_pooler))
        self.assertEqual([p.read_bytes() for p in (pre, self.shard)], originals)
        self.assertEqual(len(list((self.job / 'seed-1_embeddings').glob('*.npz'))), 2)
        self.assertFalse(list(self.shards.rglob('*.partial')))

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            pool_pair(np.ones((3, 4, 128)), 1, 2)
        with self.assertRaises(ValueError):
            pool_single(np.ones((4, 384)), 1, 2)
        with self.assertRaises(ValueError):
            pool_pair(np.ones((3, 3, 128)), 0, 3)


if __name__ == '__main__':
    unittest.main()

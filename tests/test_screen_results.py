"""Results are tested with small synthetic jobs; no AF3 or scheduler is needed."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import latentppi as lp
import screen_workflow as sw
import screen_results as results
import collect_metrics as cm
import record_run_env
import parse_ipsae
import screen_cleanup


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = lp.Config(runner='full')
        self.config.inputs.output_base = str(self.root)
        self.config.inputs.fasta_file = str(self.root/'bait.fasta')
        (self.root/'bait.fasta').write_text('>bait\nAAAA\n')
        self.config.paths.af3_output_dir = str(self.root/'predictions')
        self.config.paths.ipsae_summary = str(self.root/'ipsae_summary_af3.csv')
        self.config.output.metrics_dir = str(self.root/'metrics')
        self.config.scorer.rank_by = 'LIS'
        self.output = Path(self.config.paths.af3_output_dir)
        self.inputs = self.root/'inputs'
        self.inputs.mkdir()
        self.pairs = []
        self.candidates = []

    def table(self, path, rows, delimiter='\t'):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)

    def pair(self, name, accession, *, seeds=(1,), status='ready'):
        path = self.inputs/(name+'.json')
        path.write_text(json.dumps(dict(name=name, modelSeeds=list(seeds))))
        self.pairs.append(dict(system_name=name, line_number=len(self.pairs)+1, json_path=str(path)))
        self.candidates.append(dict(system_name=name, bait_accession='NP_000001.1',
                                   target_accession=accession, preprocessing_status=status))
        self.table(self.root/'af3_lengths.csv', self.pairs, ',')
        self.table(self.root/'screen_candidates.tsv', self.candidates)
        return path

    def timing(self, name, pred=0.5, seed=1, **kw):
        folder = self.output/name
        folder.mkdir(parents=True, exist_ok=True)
        row = dict(input_name=name, seed=seed, trunk_time_seconds=10, diffusion_time_seconds=2,
                   trunk_per_recycle_seconds=[10], latentppi_scorer_pred_metrics={'LIS':pred})
        row.update(kw)
        with (folder/'prediction_timings.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')

    def score(self, rows):
        self.table(Path(self.config.paths.ipsae_summary), rows, ',')

    def read(self, name='all_results.tsv'):
        return pd.read_csv(Path(self.config.output.metrics_dir)/name, sep='\t', comment='#')

    def test_length_cap_reports_skips_preserves_results_and_can_be_raised(self):
        for i, name in enumerate(('complete', 'gate', 'failed', 'pending', 'boundary')):
            self.pair(name, f'NP_{i}')
            self.pairs[-1]['total_length'] = 3000 if name == 'boundary' else 4000
        self.table(self.root / 'af3_lengths.csv', self.pairs, ',')
        self.timing('complete')
        self.timing('gate', latentppi_scorer_gate_triggered=True, latentppi_scorer_gate_status='ok')
        (self.output / 'failed').mkdir()
        self.config.cluster.max_total_residues = 3000
        state = sw.screen_state(self.config, write_metrics=True)
        master = self.read().set_index('system_name')
        self.assertEqual(master.status.to_dict(), dict(complete='completed', gate='killed_at_gate',
            failed='skipped_length', pending='skipped_length', boundary='pending'))
        self.assertEqual(state['skipped_length'], 2)
        self.assertEqual(state['outstanding_indices'], [5])
        self.assertEqual(master.loc['pending', 'total_length'], 4000)
        self.assertEqual(master.loc['pending', 'max_total_residues'], 3000)
        metadata = json.loads((Path(self.config.output.metrics_dir) / 'screen_status.json').read_text())['results']
        self.assertEqual(metadata['max_total_residues'], 3000)
        self.assertEqual(metadata['counts']['skipped_length'], 2)
        for cap in (4000, None):
            self.config.cluster.max_total_residues = cap
            state = sw.screen_state(self.config, write_metrics=True)
            self.assertEqual(state['skipped_length'], 0)
            self.assertEqual(state['outstanding_indices'], [3, 4, 5])
            self.assertEqual(self.read().set_index('system_name').loc['pending', 'status'], 'pending')

    def test_length_cap_does_not_hide_active_attempts(self):
        for name in ('running', 'failed'):
            self.pair(name, 'NP_' + name)
            self.pairs[-1]['total_length'] = 4000
        self.table(self.root / 'af3_lengths.csv', self.pairs, ',')
        attempts = self.output / '.latentppi_attempts'
        attempts.mkdir(parents=True)
        for name, exit_code in (('running', None), ('failed', 1)):
            (attempts / (name + '.json')).write_text(json.dumps(dict(
                system_name=name, attempt_id=name + '-attempt', exit_code=exit_code)))
        self.config.cluster.max_total_residues = 3000
        state = sw.screen_state(self.config, write_metrics=True)
        self.assertEqual(self.read().set_index('system_name').status.to_dict(),
                         dict(running='running', failed='skipped_length'))
        self.assertEqual(state['skipped_length'], 1)

    def test_length_cap_all_skipped_is_not_partial(self):
        self.pair('large', 'NP_1')
        self.pairs[-1]['total_length'] = 4000
        self.table(self.root / 'af3_lengths.csv', self.pairs, ',')
        self.config.cluster.max_total_residues = 3000
        state = sw.screen_state(self.config, write_metrics=True)
        self.assertEqual(state['outstanding_indices'], [])
        self.assertFalse(state['results']['partial'])

    def test_rank_by_selected_measured_metric_including_full_mode(self):
        self.config.output.rank_confident_iptm = False
        for name, acc in [('a','NP_000002.1'),('b','NP_000003'),('gate','NP_000004')]:
            self.pair(name, acc)
        self.timing('a', 0.99)
        self.timing('b', 0.1)
        self.timing('gate', 0.8, latentppi_scorer_gate_triggered=True, latentppi_scorer_gate_status='ok')
        self.score([dict(system='a',seed=1,Type='max',ipSAE=0.9,LIS=0.2,ipTM_af=0.9),
                    dict(system='b',seed=1,Type='max',ipSAE=0.1,LIS=0.8,ipTM_af=0.1),
                    dict(system='gate',seed=1,Type='max',ipSAE=1,LIS=1,ipTM_af=1)])
        sw.screen_state(self.config, write_metrics=True)
        master=self.read().set_index('system_name')
        self.assertEqual(master.loc['b','rank'],1)
        self.assertEqual(master.loc['a','gate_rank'],1)
        self.assertEqual(master.loc['gate','status'],'killed_at_gate')
        self.assertTrue(pd.isna(master.loc['gate','LIS']))
        self.assertTrue(pd.isna(master.loc['gate','rank']))
        self.assertEqual(master.loc['b','ipTM_af'],0.1)  # No model needed for measured ipTM.
        self.assertTrue(master.iLIS.isna().all())
        self.assertEqual(self.read('ranked_survivors.tsv').system_name.tolist(),['b','a'])
        self.assertIn('final_sort=LIS', (self.root/'metrics/ranked_survivors.tsv').read_text())
        self.assertFalse((self.root/'metrics/metrics_pairs.csv').exists())

    def test_advance_ranks_and_submits_every_confirmed_pass_once(self):
        import screen_advance
        self.config.inputs.proteome_msa_dir = str(self.inputs)
        self.config.runner = 'latent'
        self.config.paths.af3_input_dir = str(self.inputs)
        self.config.paths.cmd_file = str(self.root / 'cmds.txt')
        scores = [('low', 70, .99), ('high', 71, .6), ('higher', 75, .8),
                  ('missing', '', .9), ('killed', 80, .99), ('unknown', 80, .99)]
        commands = []
        for name, _, _ in scores:
            path = self.pair(name, 'NP_' + name)
            commands.append(f'run_alphafold.py --json_path={path} --num_recycles 1 --num_diffusion_samples=4')
            gate = dict(latentppi_scorer_gate_triggered=name == 'killed', latentppi_scorer_gate_status='ok')
            self.timing(name, **(gate if name != 'unknown' else {}))
        Path(self.config.paths.cmd_file).write_text('\n'.join(commands) + '\n')
        self.score([dict(system=name, seed=1, Type='max', ipSAE=.3, plddt=p, ipTM_af=i)
                    for name, p, i in scores])
        with patch.object(lp, 'validate'), patch.object(sw, 'launch') as launch:
            screen_advance.advance(self.config, lp.Report(rank_by='LIS'), 10, False, False)
            screen_advance.advance(self.config, lp.Report(rank_by='LIS'), 10, False, False)
        launch.assert_called_once()
        child = launch.call_args.args[0]
        self.assertEqual(child.runner, 'full')
        batch_lines = Path(child.paths.cmd_file).read_text().splitlines()
        self.assertEqual(len(batch_lines), 4)
        self.assertTrue(all('--num_recycles=10' in line and '--num_diffusion_samples=4' in line
                            and '--num_recycles 1' not in line for line in batch_lines))
        path = self.root / 'metrics/advanced_candidates.csv'
        table = pd.read_csv(path)
        self.assertEqual(table.system_name.tolist(), ['higher', 'high', 'low', 'missing'])
        self.assertEqual(table['rank'].tolist(), [2, 3, 4, 5])
        self.assertEqual(table['advance_status'].unique().tolist(), ['prepared'])
        self.assertEqual(table['plddt'].iloc[2], 70)
        self.assertTrue({'LIS_pred', 'ipTM_af', 'ipSAE'}.issubset(table.columns))
        self.assertEqual(len(sw.read_pairs(child, require_commands=True)), 4)
        for name, _, _ in scores[:4]:
            folder = Path(child.paths.af3_output_dir) / name
            folder.mkdir(parents=True)
            (folder / 'prediction_timings.jsonl').write_text(json.dumps(dict(input_name=name, seed=1,
                trunk_time_seconds=10, diffusion_time_seconds=1, trunk_per_recycle_seconds=[10])) + '\n')
        self.table(Path(child.paths.ipsae_summary), [dict(system='high', seed=1, Type='max',
                   ipSAE=.4, plddt=85, ipTM_af=.9)], ',')
        sw.screen_state(child, write_metrics=True)
        sw.screen_state(self.config, write_metrics=True)
        high = pd.read_csv(path).set_index('system_name').loc['high']
        self.assertEqual((high.advance_status, high.advance_plddt, high.advance_ipTM_af),
                         ('completed', 85, .9))

    def test_unknown_final_metric_never_falls_back_to_gate_score(self):
        self.config.output.rank_confident_iptm = False
        self.pair('a','NP_1')
        self.timing('a')
        sw.screen_state(self.config, write_metrics=True)
        self.assertTrue(self.read()['rank'].isna().all())
        self.assertEqual(len(self.read('ranked_survivors.tsv')),1)

    def test_known_entries_retained_with_version_case_and_duplicate_rules(self):
        self.pair('a','NP_000002.3')
        self.pair('gate','NP_000003.1')
        self.timing('a')
        self.timing('gate', latentppi_scorer_gate_triggered=True,latentppi_scorer_gate_status='ok')
        known=self.root/'known.txt'
        known.write_text(' np_000002 \nNP_000003.2\nNP_000002.9\nbad\nNP_999999\n\n')
        self.config.output.known_interactors=str(known)
        sw.screen_state(self.config, write_metrics=True)
        data=self.read('known_interactors.tsv')
        self.assertEqual(len(data),5)
        self.assertEqual(data.outcome.tolist(),['completed','killed at gate','input problem','input problem','not found in screened proteome'])
        self.assertEqual(data.input_line.tolist(),[1,2,3,4,5])

    def test_seed_mean_whole_sample_row_and_incomplete_status(self):
        self.pair('multi','NP_1',seeds=(1,2))
        self.timing('multi',seed=1)
        self.score([dict(system='multi',seed=1,Type='max',ipSAE=0.8,LIS=0.2,sample_num=0),
                    dict(system='multi',seed=1,Type='max',ipSAE=0.7,LIS=0.9,sample_num=1),
                    dict(system='multi',seed=2,Type='max',ipSAE=0.6,LIS=0.6,sample_num=0)])
        sw.screen_state(self.config,write_metrics=True)
        self.assertEqual(self.read().iloc[0].status,'incomplete')
        self.assertTrue(pd.isna(self.read().iloc[0]['rank']))
        self.timing('multi',seed=2)
        sw.screen_state(self.config,write_metrics=True)
        self.assertAlmostEqual(self.read().iloc[0].LIS,0.4)

    def test_attempt_excludes_old_predictions_and_old_structure_scores(self):
        source=self.pair('a','NP_1')
        self.timing('a',0.9)
        # Network filesystem timestamps can be ahead of the worker clock.
        timing=self.output/'a'/'prediction_timings.jsonl'
        future=timing.stat().st_mtime+60
        os.utime(timing,(future,future))
        with patch.object(record_run_env.subprocess,'run', side_effect=FileNotFoundError):
            attempt_id=record_run_env.record('start',self.output,'a',source)
        sw.screen_state(self.config,write_metrics=True)
        master=self.read().iloc[0]
        self.assertEqual(master.status,'running')
        self.assertTrue(pd.isna(master.gate_score))
        self.timing('a',0.2)
        record_run_env.record('finish',self.output,'a',source,0)
        self.score([dict(system='a',seed=1,Type='max',ipSAE=0.9,LIS=0.99,attempt_id='old'),
                    dict(system='a',seed=1,Type='max',ipSAE=0.1,LIS=0.1,attempt_id=attempt_id)])
        sw.screen_state(self.config,write_metrics=True)
        self.assertAlmostEqual(self.read().iloc[0].gate_score,0.2)
        self.assertAlmostEqual(self.read().iloc[0].LIS,0.1)

    def test_outcomes_and_no_oom_guess_from_exit_code(self):
        for name in ['pending','oom','failed','running','gate_error','skip']:
            source=self.pair(name,'NP_'+str(len(self.pairs)+1),status='skipped_length' if name=='skip' else 'ready')
            if name in ('oom','failed','running'):
                with patch.object(record_run_env.subprocess,'run', side_effect=FileNotFoundError):
                    aid=record_run_env.record('start',self.output,name,source)
                if name=='oom':
                    (self.output/'.latentppi_attempts'/(aid+'.stderr')).write_text('CUDA out of memory')
                if name!='running':
                    record_run_env.record('finish',self.output,name,source,137)
        self.timing('gate_error',trunk_per_recycle_seconds=None,latentppi_scorer_gate_status='error')
        sw.screen_state(self.config,write_metrics=True)
        data=self.read().set_index('system_name')
        self.assertEqual(data.status.to_dict(),dict(pending='pending',oom='failed_oom',failed='failed_other',
                         running='running',gate_error='gate_error',skip='skipped_length'))
        sw.screen_state(self.config,write_metrics=True,final=True)
        self.assertEqual(self.read().set_index('system_name').loc['running','status'],'incomplete')

    def test_failed_publication_preserves_previous_complete_set(self):
        self.pair('a','NP_1')
        self.timing('a')
        sw.screen_state(self.config,write_metrics=True)
        before=(self.root/'metrics/current').resolve()
        contents=(before/'all_results.tsv').read_bytes()
        with patch.object(results,'write_tsv',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                sw.screen_state(self.config,write_metrics=True)
        self.assertEqual((self.root/'metrics/current').resolve(),before)
        self.assertEqual((before/'all_results.tsv').read_bytes(),contents)
        for _ in range(3):
            sw.screen_state(self.config,write_metrics=True)
        self.assertEqual(len(list((self.root/'metrics/.generations').iterdir())),2)

    def test_periodic_refresh_obeys_interval(self):
        self.config.output.refresh_seconds=10
        with patch.object(sw,'user_queue',side_effect=[['1'],['1'],[]]), \
             patch.object(sw.time,'monotonic',side_effect=[0,0,11,11]), \
             patch.object(sw.time,'sleep'), patch.object(sw,'refresh_results') as refresh:
            sw.wait_for_array(self.config,'1')
        refresh.assert_called_once()

    def test_changed_scientific_settings_refused_but_output_options_allowed(self):
        report=lp.Report()
        results.ensure_compatible_screen(self.config,report)
        self.config.output.refresh_seconds=0
        results.ensure_compatible_screen(self.config,report)
        self.config.inputs.af3_num_recycles=10
        with self.assertRaisesRegex(ValueError,'new inputs.output_base'):
            results.ensure_compatible_screen(self.config,report)

    def test_new_result_links_can_be_cleaned_without_following_other_links(self):
        self.pair('a','NP_1')
        self.timing('a')
        sw.screen_state(self.config,write_metrics=True)
        for field in ('af3_input_dir','cmd_file','shard_dir','ipsae_outroot'):
            setattr(self.config.paths,field,str(self.root/field))
        self.config.paths.script_dir=str(REPO)
        self.config.inputs.proteome_msa_dir=str(self.root/'proteome')
        self.config.af3.image=str(self.root/'af3.sif')
        self.config.af3.model_parameters_dir=str(self.root/'weights')
        self.config.af3.databases_dir=str(self.root/'db')
        plan=screen_cleanup.build_plan(self.config,self.root/'config.yaml')
        self.assertIn(self.root/'metrics/current',plan)
        self.assertIn(self.root/'metrics/.generations',plan)
        (self.root/'metrics/current').unlink()
        (self.root/'metrics/current').symlink_to(self.inputs)
        with self.assertRaisesRegex(ValueError,'symlink'):
            screen_cleanup.build_plan(self.config,self.root/'config.yaml')

    def test_ipsae_parser_preserves_seed_and_extends_legacy_summary(self):
        source=self.pair('NP_1_NP_2','NP_2',seeds=(1,2))
        folder=self.output/'NP_1_NP_2'
        for seed in [1,2]:
            sample=folder/f'seed-{seed}_sample-0'
            sample.mkdir(parents=True)
            (sample/'model.cif').write_text('fixture')
            (sample/'confidences.json').write_text('{}')
        (folder/'NP_1_NP_2_model.cif').write_text('top')
        (folder/'NP_1_NP_2_confidences.json').write_text('{}')
        scorer=self.root/'mock_ipsae.py'
        scorer.write_text('import sys\nfrom pathlib import Path\np=Path(sys.argv[2])\np.with_name(p.stem+"_15_15.txt").write_text("Chn1 Chn2 Type ipSAE LIS\\nA B max 0.8 0.5\\n")\n')
        summary=Path(self.config.paths.ipsae_summary)
        summary.write_text('ipsae_file,af3_json,af3_cif,system,model_num,seed,sample_num,pae_cutoff,dist_cutoff,plddt,Chn1,Chn2,Type,ipSAE,LIS\nold,,,old,,1,0,15,15,,A,B,max,0.1,0.1\n')
        command=[sys.executable,str(REPO/'parse_ipsae.py'),'--mode','af3','--root',str(folder),
                 '--outroot',str(self.root/'scores'),'--summary',str(summary),'--input-json',str(source),
                 '--prefer-seed-samples','--ipsae-py',str(scorer)]
        completed=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
        data=pd.read_csv(summary)
        self.assertEqual(data.system.tolist(),['old','NP_1_NP_2','NP_1_NP_2'])
        self.assertEqual(data.seed.tolist(),[1,1,2])
        self.assertIn('attempt_id',data)
        self.assertEqual(len(list((self.root/'scores').glob('*.txt'))),3)  # two scores plus done ledger

    def test_new_attempt_recomputes_ipsae_and_full_run_without_timings_completes(self):
        source=self.pair('a','NP_1')
        folder=self.output/'a'
        sample=folder/'seed-1_sample-0'
        scorer=self.root/'mock_ipsae.py'
        scorer.write_text('import sys, os\nfrom pathlib import Path\np=Path(sys.argv[2])\np.with_name(p.stem+"_15_15.txt").write_text("Chn1 Chn2 Type ipSAE LIS\\nA B max 0.8 "+os.environ["TEST_SCORE"]+"\\n")\n')
        for score in ['0.9','0.2']:
            with patch.object(record_run_env.subprocess,'run',side_effect=FileNotFoundError):
                aid=record_run_env.record('start',self.output,'a',source)
            sample.mkdir(parents=True,exist_ok=True)
            (sample/'model.cif').write_text('fresh structure')
            (sample/'confidences.json').write_text('{}')
            cmd=[sys.executable,str(REPO/'parse_ipsae.py'),'--mode','af3','--root',str(folder),
                 '--outroot',str(self.root/'scores'),'--summary',self.config.paths.ipsae_summary,
                 '--input-json',str(source),'--prefer-seed-samples','--attempt-id',aid,'--ipsae-py',str(scorer)]
            completed=subprocess.run(cmd,env={**os.environ,'TEST_SCORE':score},capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
            record_run_env.record('finish',self.output,'a',source,0)
            sw.screen_state(self.config,write_metrics=True)
            row=self.read().iloc[0]
            self.assertEqual(row.status,'completed')
            self.assertAlmostEqual(row.LIS,float(score))
            self.assertEqual(row.attempt_id,aid)
        self.assertEqual(len(pd.read_csv(self.config.paths.ipsae_summary)),2)

    def test_restarted_during_collection_defers_scores(self):
        source=self.pair('a','NP_1')
        self.timing('a',0.9)
        original=cm.collect
        def restarted(*args,**kwargs):
            frame=original(*args,**kwargs)
            with patch.object(record_run_env.subprocess,'run',side_effect=FileNotFoundError):
                record_run_env.record('start',self.output,'a',source)
            return frame
        with patch.object(cm,'collect',side_effect=restarted):
            sw.screen_state(self.config,write_metrics=True)
        self.assertTrue(pd.isna(self.read().iloc[0].gate_score))
        self.assertEqual(self.read().iloc[0].status,'running')


if __name__=='__main__':
    unittest.main()

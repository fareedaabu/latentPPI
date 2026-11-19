"""Synthetic screens exercise collection, scientific selection, and the public CLI."""
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import collect_metrics as cm


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix=".metrics-test-", dir=REPO)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def record(self, name="target_with_underscores_target", seed=1, **extra):
        return dict(input_name=name, seed=seed, trunk_time_seconds=10,
                    diffusion_time_seconds=2, trunk_per_recycle_seconds=[5, 5],
                    latentppi_chain_len_a=10, latentppi_chain_len_b=20, **extra)

    def timing(self, folder, records):
        path = self.root / folder / "prediction_timings.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return path

    def table(self, filename, fields, rows, delimiter=","):
        path = self.root / filename
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def run_cli(self, *args):
        result = subprocess.run([sys.executable, str(REPO / "collect_metrics.py"),
                                 "--root", str(self.root), *args], capture_output=True, text=True)
        return result

    def test_mixed_schemas_configs_manifests_and_sidecar(self):
        name = "target_with_underscores_target"
        old = self.record(latentppi_scorer_pred_metric_name="old_score",
                          latentppi_scorer_pred_metric=0.1, latentppi_scorer_gate_enabled=False,
                          num_recycles=1)
        new = self.record(latentppi_scorer_pred_metrics={"alien_fixture_score": 0.8, "old_score": 0.7},
                          latentppi_scorer_gate_enabled=True, latentppi_scorer_threshold=0.6,
                          latentppi_scorer_rank_by="alien_fixture_score", num_recycles=10,
                          latentppi_scorer_model_ids={"alien_fixture_score": "run1"},
                          latentppi_scorer_model_status={"alien_fixture_score": "ok"},
                          latentppi_tap_point="prepairformer")
        abort = {**new, "latentppi_chain_len_a": 0, "trunk_time_seconds": 0.0002}
        path = self.timing("deep/pair", [old, new, abort])
        (path.parent / "run_env.json").write_text(json.dumps(dict(gpu_type="A100", node="n1",
                                                                 slurm_job_id="123", slurm_array_task_id=7)))
        self.timing("other", [self.record("other", latentppi_scorer_pred_metric_name="old_score",
                                          latentppi_scorer_pred_metric=0.5)])
        jobs = self.table("jobs.tsv", ["system_name", "af3_json", "af3_out_dir"], [
            dict(system_name=name, af3_json="input.json", af3_out_dir="deep/pair")], "\t")
        lengths = self.table("af3_lengths.csv", ["system_name", "line_number", "chain_a_length",
                                                "chain_b_length", "total_length", "json_path"], [
            dict(system_name=name, line_number=7, chain_a_length=100, chain_b_length=200, total_length=300),
            dict(system_name="never_ran", line_number=8, chain_a_length=5, chain_b_length=6, total_length=11)])
        frame, metrics, stats = cm.collect(self.root, jobs, lengths)
        self.assertEqual(metrics, ["alien_fixture_score", "old_score"])
        self.assertNotIn("alien_fixture_score", (REPO / "collect_metrics.py").read_text())
        row = frame.set_index("system_name").loc[name]
        self.assertEqual((row.line_number, row.chain_a_length, row.chain_b_length), (7, 100, 200))
        self.assertEqual((row.num_recycles, row.latentppi_scorer_threshold, row.gpu_type), (10, 0.6, "A100"))
        self.assertTrue(row.latentppi_scorer_gate_enabled)
        self.assertEqual(row.alien_fixture_score_pred, 0.8)
        self.assertEqual(row.total_seconds, 12)
        self.assertTrue(row.recycle_timing_consistent)
        self.assertEqual(frame.set_index("system_name").loc["never_ran", "status"], "not_run")
        self.assertEqual(frame.set_index("system_name").loc["other", "chain_a_length"], 10)
        self.assertTrue(pd.isna(frame.set_index("system_name").loc["other", "alien_fixture_score_pred"]))
        self.assertEqual((stats["incomplete_dropped"], stats["duplicate_keys"], stats["files_parsed"]), (1, 1, 2))
        result = self.run_cli("--rank-by", "alien_fixture_score")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("latentppi_scorer_model_ids", pd.read_csv(self.root / "metrics_all.csv"))

    def test_lenient_invalid_and_incomplete_failures(self):
        aborted = {**self.record("aborted"), "latentppi_chain_len_a": 0, "trunk_time_seconds": 0.0001}
        self.timing("abort", [aborted])
        path = self.timing("lenient", [self.record("nan", latentppi_scorer_pred_metrics={
            "score": float("nan"), "plus": float("inf"), "minus": -float("inf")})])
        with path.open("a") as handle:
            handle.write('header,not,json\n{"input_name":')
        self.timing("broken", [])
        (self.root / "broken/prediction_timings.jsonl").write_text("broken\n")
        jobs = self.table("jobs.tsv", ["system_name", "af3_out_dir"], [
            dict(system_name="failed", af3_out_dir="broken")], "\t")
        frame, metrics, stats = cm.collect(self.root, jobs)
        rows = frame.set_index("system_name")
        self.assertEqual(rows.loc["failed", "status"], "parse_error")
        self.assertEqual(rows.loc["aborted", "status"], "incomplete")
        self.assertTrue(rows.loc["nan", [m + "_pred" for m in metrics]].isna().all())
        self.assertEqual((stats["lenient_parses"], stats["invalid_lines"]), (1, 3))

    def test_last_keyed_failure_and_inconsistent_timing(self):
        first = self.record("failure", latentppi_scorer_pred_metrics={"s": 0.8})
        latest = dict(input_name="failure", seed=1, latentppi_scorer_gate_enabled=False)
        inconsistent = {**self.record("bug"), "trunk_per_recycle_seconds": [0.0001]}
        self.timing("timings", [first, latest, inconsistent])
        frame, _, stats = cm.collect(self.root)
        rows = frame.set_index("system_name")
        self.assertEqual(rows.loc["failure", "status"], "missing_timing")
        self.assertFalse(rows.loc["bug", "recycle_timing_consistent"])
        self.assertEqual(rows.loc["bug", "trunk_time_seconds"], 10)
        self.assertEqual(stats["duplicate_keys"], 1)

    def test_seed_aggregation_ranking_and_null_retention(self):
        records = [self.record("a", 1, latentppi_scorer_pred_metrics={"s": 0.2}),
                   self.record("a", 2, latentppi_scorer_pred_metrics={"s": 0.8}),
                   self.record("b", 1, latentppi_scorer_pred_metrics={"s": 0.5}),
                   self.record("z", 1), self.record("n", 1)]
        self.timing("all", records)
        frame, _, _ = cm.collect(self.root)
        pairs = cm.aggregate_pairs(frame, ["s_pred"], "s", "mean")
        self.assertEqual(pairs.system_name.tolist(), ["a", "b", "n", "z"])
        self.assertEqual(pairs.loc[0, "s_pred"], 0.5)
        self.assertEqual(pairs.loc[0, "s_pred_max"], 0.8)
        self.assertEqual(pairs.loc[0, "n_seeds_ok"], 2)
        self.assertEqual(pairs.loc[0, "rank_pct"], 0.5)
        self.assertTrue(pd.isna(pairs.loc[2, "rank"]))
        maximum = cm.aggregate_pairs(frame, ["s_pred"], "s", "max")
        self.assertEqual(maximum.loc[0, "s_pred"], 0.8)

    def ipsae_fixture(self):
        self.timing("all", [self.record("pair", 1, latentppi_scorer_pred_metrics={"ipSAE": 0.5, "extra": 0.1}),
                            self.record("pair", 2, latentppi_scorer_pred_metrics={"ipSAE": 0.4}),
                            self.record("gate", latentppi_scorer_pred_metrics={"ipSAE": 0.2},
                                        latentppi_scorer_gate_triggered=True, latentppi_scorer_gate_status="ok")])
        rows = [dict(system="pair", seed=1, Type="asym", ipSAE=0.99, model_num=0, sample_num=0, Chn1="A"),
                dict(system="pair", seed=1, Type="max", ipSAE=0.6, model_num=1, sample_num=0, Chn1="A"),
                dict(system="pair", seed=1, Type="max", ipSAE=0.8, model_num=2, sample_num=1, Chn1="A"),
                dict(system="pair", seed=2, Type="max", ipSAE=0.3, model_num=0, sample_num=0, Chn1="A"),
                dict(system="gate", seed=1, Type="max", ipSAE=0.9),
                dict(system="pair", seed="", Type="max", ipSAE=1)]
        return self.table("ipsae_summary_af3.csv", ["system", "seed", "Type", "ipSAE", "model_num",
                                                    "sample_num", "Chn1"], rows)

    def test_owner_rule_join_seed_and_gated_nulls(self):
        self.ipsae_fixture()
        result = self.run_cli("--rank-by", "ipSAE")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = pd.read_csv(self.root / "metrics_all.csv").set_index(["system_name", "seed"])
        self.assertEqual(rows.loc[("pair", 1), "ipSAE_true"], 0.8)
        self.assertEqual(rows.loc[("pair", 2), "ipSAE_true"], 0.3)
        self.assertTrue(pd.isna(rows.loc[("gate", 1), "ipSAE_true"]))
        self.assertNotIn("extra_true", rows)
        stats = json.loads(result.stderr)
        self.assertEqual(stats["systems_matched_ipsae"], 1)
        self.assertEqual(stats["ipsae_rows_missing_join_key"], 1)
        self.assertEqual(stats["predicted_metrics_without_ground_truth"], ["extra"])
        self.assertEqual(stats["pairs_gated"], 1)

    def test_configurable_selection_and_ambiguity(self):
        path = self.ipsae_fixture()
        data = pd.read_csv(path, keep_default_na=False)
        data = data[data.seed != ""]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            cm.select_ipsae_rows(data, "unique")
        rule = json.dumps({"where": {"Type": "max", "Chn1": "A"}, "order_by": ["model_num", "sample_num"]})
        selected = cm.select_ipsae_rows(data, rule)
        self.assertEqual(selected.iloc[0].ipSAE, 0.3)
        self.assertIn(0.6, selected.ipSAE.tolist())
        with self.assertRaisesRegex(ValueError, "absent"):
            cm.select_ipsae_rows(data, '{"where":{"nonexistent":"x"}}')

    def test_max_ipsae_whole_row_and_ties_are_order_independent(self):
        data = pd.DataFrame([
            dict(system="x", seed="1", Type="max", ipSAE=0.8, other=0.2, model_num=2, sample_num=0),
            dict(system="x", seed="1", Type="max", ipSAE=0.8, other=0.3, model_num=1, sample_num=0),
            dict(system="x", seed="1", Type="asym", ipSAE=0.9, other=1.0, model_num=0, sample_num=0)])
        for source in [data, data.iloc[::-1]]:
            row = cm.select_ipsae_rows(source, "max_ipsae").iloc[0]
            self.assertEqual((row.ipSAE, row.other), (0.8, 0.3))

    def test_summary_without_eligible_rows_keeps_predictions(self):
        self.timing("one", [self.record("pair", latentppi_scorer_pred_metrics={"ipSAE": 0.7})])
        for rows in [[], [dict(system="pair", seed="", Type="max", ipSAE=0.9)],
                     [dict(system="pair", seed=1, Type="asym", ipSAE=0.9)],
                     [dict(system="pair", seed=1, Type="max", ipSAE="")]]:
            self.table("ipsae_summary_af3.csv", ["system", "seed", "Type", "ipSAE"], rows)
            result = self.run_cli()
            self.assertEqual(result.returncode, 0, result.stderr)
            frame = pd.read_csv(self.root / "metrics_all.csv")
            self.assertEqual(frame.loc[0, "ipSAE_pred"], 0.7)
            self.assertTrue(pd.isna(frame.loc[0, "ipSAE_true"]))

    def test_all_null_metric_and_large_seed(self):
        seed = 2**63 + 1
        self.timing("one", [self.record("pair", seed, latentppi_scorer_pred_metrics={"ipSAE": None})])
        self.table("ipsae_summary_af3.csv", ["system", "seed", "Type", "ipSAE"], [
            dict(system="pair", seed=seed, Type="max", ipSAE=0.9)])
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        frame = pd.read_csv(self.root / "metrics_all.csv", dtype={"seed": str})
        self.assertEqual(frame.loc[0, "seed"], str(seed))
        self.assertEqual(frame.loc[0, "ipSAE_true"], 0.9)
        self.assertEqual(json.loads(result.stderr)["pairs_null_rank_by"], 1)

    def test_single_metric_gzip_quiet_and_idempotence(self):
        source = self.timing("one", [self.record(latentppi_scorer_pred_metric_name="only",
                                                latentppi_scorer_pred_metric=0.7)])
        original = source.read_bytes()
        for compressed in [False, True]:
            args = ["--quiet"] + (["--gzip"] if compressed else [])
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            path = self.root / ("metrics_pairs.csv.gz" if compressed else "metrics_pairs.csv")
            before = path.read_bytes()
            self.assertEqual(self.run_cli(*args).returncode, 0)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(pd.read_csv(path).loc[0, "only_pred"], 0.7)
            if compressed:
                self.assertIn("only_pred_max", gzip.decompress(before).decode())
        self.assertEqual(source.read_bytes(), original)

    def test_rank_validation_and_twenty_dynamic_metrics(self):
        metrics = {f"fixture_{i:02}": i / 20 for i in range(20)}
        self.timing("twenty", [self.record(latentppi_scorer_pred_metrics=metrics)])
        for args in [[], ["--rank-by", "absent"]]:
            result = self.run_cli(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("available metrics: fixture_00", result.stderr)
        result = self.run_cli("--rank-by", "fixture_19")
        self.assertEqual(result.returncode, 0, result.stderr)
        columns = pd.read_csv(self.root / "metrics_all.csv").columns
        self.assertEqual([c for c in columns if c.endswith("_pred")], [m + "_pred" for m in sorted(metrics)])

    def test_empty_screen_and_no_predictions(self):
        for has_manifest in [False, True]:
            if has_manifest:
                self.table("af3_lengths.csv", ["system_name", "line_number"], [dict(system_name="never", line_number=1)])
            result = self.run_cli()
            self.assertEqual(result.returncode, 0, result.stderr)
            pairs = pd.read_csv(self.root / "metrics_pairs.csv")
            self.assertEqual(len(pairs), int(has_manifest))
        self.assertTrue(pd.isna(pairs.loc[0, "rank"]))

    def test_duplicate_files_glob_and_gate_status(self):
        self.timing("a", [self.record("pair", latentppi_scorer_pred_metrics={"s": 0.1})])
        self.timing("z", [self.record("pair", latentppi_scorer_pred_metrics={"s": 0.9},
                                     latentppi_scorer_gate_triggered=True, latentppi_scorer_gate_status="invalid_bundle")])
        frame, _, stats = cm.collect(self.root)
        self.assertEqual(frame.loc[0, "s_pred"], 0.9)
        self.assertFalse(frame.loc[0, "gated"])
        self.assertEqual(stats["duplicate_keys"], 1)
        frame, _, stats = cm.collect(self.root, pattern="a/*.jsonl")
        self.assertEqual(stats["files_found"], 1)
        self.assertEqual(frame.loc[0, "s_pred"], 0.1)


if __name__ == "__main__":
    unittest.main()

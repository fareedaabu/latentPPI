"""Check screen inputs use the shared proteome MSA store without copying it."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import csv


REPO = Path(__file__).resolve().parents[1]


class ScreenMsaPathsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="screen-msa-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.screen = self.root / "screen"
        self.screen.mkdir()
        self.proteome = self.root / "shared proteome msas"
        self.proteome.mkdir()
        self.fasta = self.screen / "query.fasta"
        self.fasta.write_text(">query\nACDEFGHIK\n")
        self.query_msa = self.screen / "query.a3m"
        self.query_msa.write_text(">query\nACDEFGHIK\n>hit\nACDEFGHIK\n")
        self.target_msa = self.proteome / "target.a3m"
        self.target_msa.write_text(">target\nLMNPQRSTV\n")

    def run_inputs(self, *extra):
        result = subprocess.run([
            sys.executable, str(REPO / "run_pipeline.py"),
            "--fasta-single", str(self.fasta),
            "--msa-dir", str(self.proteome),
            "--output-dir", str(self.screen),
            *extra,
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.screen / "af3_inputs/query_target.json"

    def test_external_proteome_and_query_in_screen(self):
        path = self.run_inputs("--msa-a", str(self.query_msa))
        data = json.loads(path.read_text())
        paths = [entry["protein"]["unpairedMsaPath"] for entry in data["sequences"]]
        self.assertEqual(paths, [str(self.query_msa), str(self.target_msa)])
        self.assertTrue(all(Path(value).is_file() for value in paths))
        self.assertFalse((self.screen / "msa_cache").exists())
        self.assertEqual(len(list(self.proteome.glob("*.a3m"))), 1)

        # A rerun migrates old screen JSON paths without --force.
        for entry in data["sequences"]:
            entry["protein"]["unpairedMsaPath"] = "../msa_cache/old.a3m"
        path.write_text(json.dumps(data))
        self.run_inputs("--msa-a", str(self.query_msa))
        refreshed = json.loads(path.read_text())
        self.assertEqual([entry["protein"]["unpairedMsaPath"]
                          for entry in refreshed["sequences"]], paths)

    def test_query_fallback_stays_in_screen_folder(self):
        self.query_msa.unlink()
        path = self.run_inputs()
        data = json.loads(path.read_text())
        query_path = Path(data["sequences"][0]["protein"]["unpairedMsaPath"])
        self.assertEqual(query_path.parent, self.screen)
        self.assertEqual(query_path.read_text(), ">query\nACDEFGHIK\n")
        self.assertEqual(data["sequences"][1]["protein"]["unpairedMsaPath"],
                         str(self.target_msa))
        self.assertFalse((self.screen / "msa_cache").exists())

    def test_skipped_candidates_remain_visible_and_leave_old_job_list(self):
        self.run_inputs("--msa-a", str(self.query_msa))
        # The same accession now has an identical sequence: preserve its exclusion
        # and remove the earlier prepared job instead of displaying a false pending pair.
        self.target_msa.write_text(">target\nACDEFGHIK\n")
        self.run_inputs("--msa-a", str(self.query_msa))
        with (self.screen / "screen_candidates.tsv").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(rows[0]['preprocessing_status'], 'skipped_identity')
        self.assertEqual(rows[0]['target_accession'], 'target')
        self.assertEqual(rows[0]['homodimer'], 'True')
        self.assertEqual((self.screen / "af3_cmds.txt").read_text(), '')


if __name__ == "__main__":
    unittest.main()

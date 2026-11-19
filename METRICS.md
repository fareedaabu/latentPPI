# Screen results

The screen keeps `ipsae_summary_af3.csv` as an internal source of structure scores. The files to read are in `output.metrics_dir` (default: `output_base/metrics`).

| File | What it tells you |
| --- | --- |
| `all_results.tsv` | Every considered pair, its outcome, gate score, final scores and ranks |
| `advanced_candidates.csv` | Completed latent gate passes, source and advance scores, batch status and ranks |
| `ranked_survivors.tsv` | Completed pairs ordered by final rank |
| `known_interactors.tsv` | An answer for each supplied known-interactor entry; only written when a list is configured |
| `summary.txt` | Settings, score definitions and counts; optional |
| `screen_status.json` | Machine-readable workflow status and result-generation metadata |

The ranked file is written in both `latent` and `full` runs. There is no final-confidence cutoff. Under the default ranking, completed pairs with missing scores still receive a rank in the last group.

## Collect results

With default chained dispatch, results refresh at the start of each active batch-planning step, after the preceding array has terminated. A final collector runs when all batches/retries are resolved. A nonzero `output.refresh_seconds` enables these boundary updates; it does not schedule periodic work while a batch runs. Use `status --heavy` for an on-demand refresh. In persistent mode, results also refresh periodically while waiting (every five minutes by default, approximately following queue polls). Failed refreshes retain the previous published results and report a warning.

Refresh manually after input preparation with:

```bash
python latentppi.py status config.yaml --heavy
```

Default light status prints workflow progress without rewriting tables.

```yaml
output:
  metrics_dir: metrics
  refresh_seconds: 300       # 0 disables dispatcher refreshes; final collection remains
  known_interactors: ''      # Optional .txt path, relative to the YAML
  write_summary: true
  legacy_csv: false          # Enable only if an older downstream script needs the CSVs
```

## Which score controls the rank?

The gate uses the predicted `scorer.rank_by` metric. By default, completed pairs with measured ipTM and mean pLDDT > 70 rank first, followed by measured pairs with pLDDT ≤ 70, then pairs missing either score. Each group sorts by descending measured `ipTM_af`, with missing ipTM last. Every completed pair receives a rank. Set `output.rank_confident_iptm: false` for the previous measured scorer-metric ranking. The pLDDT rule applies after diffusion and does not change gate eligibility.

| Column | Meaning |
| --- | --- |
| `rank` | Final rank within the screen, using the confidence groups above by default |
| `rank_pct` | Rank divided by the number of ranked pairs; smaller is better |
| `gate_score` | Predicted value of the selected metric, averaged across available seeds |
| `gate_rank` | Descending gate-score rank across all pairs with a finite prediction, including killed or failed pairs |
| `final_rank_by`, `gate_metric` | Measured ranking metric and predicted gate metric, respectively |
| `ipTM_af`, `ipSAE`, `pDockQ2`, `LIS`, `iLIS`, other metric names | Final measured values; empty when unavailable |
| `<metric>_pred` | Model prediction, when supplied by the runtime |
| `requested_recycles` | Recorded requested recycle count; not proof that every recycle finished |
| `gate_input_recycle`, `gate_target_recycle` | Values explicitly emitted in the runtime record, when available |
| `checkpoint_input_recycle`, `checkpoint_target_recycle` | Training metadata read from the saved checkpoint; not proof of the runtime's actual tap point |
| `missing_final_metrics` | Names of unavailable final scores |
| `gpu_retry`, `gpu_type` | Whether the recorded attempt is the one-time fallback, and its detected GPU model (when available) |
| `attempt_id`, `run_id` | Which attempt/run supplied the row, where recorded |

Ties receive the same minimum rank; system name breaks display-order ties. Unavailable measurements stay blank. The first comment line in each TSV states the ranking metric, sample-selection rule and generation ID. Read with pandas using `pd.read_csv(path, sep='\t', comment='#')`.

Within a seed, the collector selects the `Type=max` row with the largest finite **ipSAE** across samples. Ties prefer the lower model number, then sample number, then the serialized row. Every measured metric comes from that selected row. Across seeds, measured scores are averaged over available, non-gated seeds. These are the existing sample-selection rules; selecting LIS for ranking does not select a different sample for each metric.

All available observed metrics are collected independently of which model predictions exist. The local scorer does not implement iLIS; its column stays empty unless the score source supplies it. `ipTM_af` is the AF3 chain-pair ipTM imported by ipSAE. A missing AF3 summary gives an unavailable ipTM rather than zero. No output is labelled “R10” solely because that was the intended experiment.

## Pair outcomes

| `status` | Meaning |
| --- | --- |
| `completed` | All expected seeds are resolved, with at least one folded result, and no recorded runner failure |
| `killed_at_gate` | All resolved seeds were killed by the gate; no final structure score is shown |
| `killed_r1` | Same outcome, with an explicitly recorded runtime input recycle of 1 |
| `pending` | No recorded attempt/output yet |
| `running` | An attempt has started and has not recorded an exit; an abruptly interrupted worker can remain here until final collection |
| `incomplete` | Insufficient evidence to call the pair completed; final collection uses this for attempts without an exit record |
| `failed_oom` | Runner failed and its AF3 stderr explicitly reported an out-of-memory error |
| `failed_other` | Runner failed for another or unknown reason; exit code 137 alone is not proof of OOM |
| `gate_error` | A runtime record reports a gate error and no completed outcome is established |
| `skipped_identity`, `skipped_nonstandard` | Input preparation excluded the pair for that reason |
| `skipped_length` | Unfinished pair excluded from submission because its total length exceeds `cluster.max_total_residues`; also supported for candidate records with this preprocessing outcome |

`gate_outcome` also distinguishes killed, passed, error, mixed and unknown records. If some seeds were killed and others completed, final scores use the folded seeds and the row records the mixed gate outcome. Missing expected seeds leave the pair incomplete.

For the optional [complex size cutoff](CONFIG.md#complex-size-cutoff), result rows
include `total_length` and `max_total_residues` (blank when disabled).
`screen_status.json` and `summary.txt` also record the configured cutoff.
Size exclusions do not count as pending work or failures. Existing completed
results and recorded running attempts retain their outcomes. Raising or removing
the cutoff re-evaluates exclusions when status/results are refreshed; it does not
rewrite inputs or erase previous attempts.

`homodimer` records identical input sequences. `homodimer_rule_applied` is a separate flag so a rule application cannot hide a subsequent failure. This output change does not introduce an always-R10 homodimer rule or change the existing sequence-identity filter.

`screen_candidates.tsv` retains candidate accessions, chain roles and exclusions before jobs are generated. Older screens without that file can only report their existing planned pairs; preparation must be rerun to recover skipped candidates. For old inputs, accession matching uses explicit MSA paths where available and never splits compound names at underscores.

## Known interactors

Set `output.known_interactors` to a text file containing one NP accession per line.

- Surrounding whitespace and capitalization are ignored for matching.
- A numeric version suffix is ignored: `NP_000537.3` matches `NP_000537`.
- Blank lines are ignored.
- Every nonblank line gets a result, including duplicates, malformed entries and accessions not found.
- Duplicates are identified after normalization. Different listed versions of one accession are duplicates.
- If several proteome candidates match after removing versions, the entry is reported as an input problem rather than arbitrarily choosing one.

The output retains original text and line numbers. Killed entries explicitly read “killed at gate” and retain their gate score/rank. Completed entries show their rank among successfully folded pairs. Target accessions are stored separately from chain order. An old screen with insufficient accession information can report no match; it does not guess from an ambiguous name.

## Interrupted and resumed screens

New attempts record which timing data and structure files already existed before AF3 starts. Collection ignores older records and attaches ipSAE rows only from the selected attempt. Comparing files against their earlier state avoids relying on agreement between worker and filesystem clocks. The ipSAE success ledger includes the attempt ID, so a forced rerun recomputes structure scores instead of reusing an old success. Historical attempt metadata is retained under `.latentppi_attempts/history/`.

If a task changes attempts during collection, that pair's scores are deferred until the next refresh. A failed new attempt has no final rank, even if an older attempt succeeded. Old summary rows remain in the internal CSV, but are excluded from a newer attempt's results.

Older screens lacking attempt IDs remain readable, with their original limitations. Top-level old ipSAE rows with no seed are excluded. New runners prefer seed/sample structures and use the original AF3 input to preserve the exact system name. If only a top-level structure exists, a seed is assigned only when the original input has exactly one seed.

`screen_identity.json` prevents reuse of a directory with changed bait contents, model checkpoint contents, runner, gate settings or requested inference settings. Use a new `inputs.output_base` for a different experiment. `--force` repeats compatible work; it does not bypass this check. The guard records resource paths; it does not hash every proteome MSA or the AF3 image/weights. Preserve those sources during a screen.

## Consistent file updates

Every public table is derived from the same collected master. A writer lock serializes collection/publication. A complete set is written in `metrics/.generations/<id>/`, then the `metrics/current` link is switched atomically. Public filenames point through `current`.

For a script reading several files, resolve `current` once and read that directory so a refresh between file opens cannot mix versions:

```python
from pathlib import Path
import pandas as pd

snapshot = Path('screen/metrics/current').resolve()
all_pairs = pd.read_csv(snapshot / 'all_results.tsv', sep='\t', comment='#')
ranked = pd.read_csv(snapshot / 'ranked_survivors.tsv', sep='\t', comment='#')
```

The current and previous sets are retained. Copy a set elsewhere for long-lived analysis; later refreshes remove older sets. An interrupted write before publication leaves the previous complete set available. Cleanup recognizes these generated links and directories and still rejects unrelated symlinks.

## Legacy CSVs and standalone collector

`output.legacy_csv: true` additionally writes the old `metrics_all.csv` and `metrics_pairs.csv` alongside each new set. **The legacy pair CSV retains its predicted-score ranking.** Use `ranked_survivors.tsv` for final-score ranking. Old regular CSVs already on disk are not automatically deleted when legacy output is disabled; they may be stale.

The standalone diagnostic collector remains available:

```bash
python collect_metrics.py --root /path/to/screen --out-dir /path/to/diagnostics \
  --rank-by ipSAE_d0dom --agg mean
```

It writes the two legacy CSVs, optionally compressed with `--gzip`. It supports `--jobs-tsv`, `--lengths-csv`, `--ipsae-summary`, `--glob` and `--ipsae-row-select` overrides. Use workflow `status --heavy` to produce the new master, survivor and known-interactor files.

The detailed seed CSV preserves timing/record validity in `status` (`ok`, `missing_timing`, `invalid_timing`, `missing_seed`, `not_run`, `incomplete`, and read/parse errors). These describe the timing record, whereas `all_results.tsv` describes the pair's outcome. Nested timing fields are JSON cells. Malformed/partial JSONL lines are skipped and counted; later collection can read a completed append.

## GPU routing provenance

Optional `cluster.gpu_tiers` selects initial resources using total complex length.
The recommended example caps the regular tier at 3,000 residues, conservatively
chosen from L40 screening runs. Size routing is independent of the retry budget:
a large-tier first attempt can still receive one retry on that tier.

For tier-enabled submissions, `latentppi_runs/<run>/submissions.jsonl` includes
`gpu_tier` and `gpu_route_reason` (`total_length` or `retry`), alongside the array
indices, offset and requested Slurm resources. The saved config records cutoffs.
These describe requested routing; the attempt's `gpu_type` records observed hardware.
Existing metrics and worker record formats are unchanged by tier routing.

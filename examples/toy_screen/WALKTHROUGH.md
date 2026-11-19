# Your first 50-pair screen

Run a complete screen with one synthetic **target protein** and 50 candidates. The artificial sequences and alignments test the workflow, not biological interactions. Run all commands from the repository root.

[README](../../README.md) · [Configuration reference](../../CONFIG.md) · [Metrics reference](../../METRICS.md)

## 1. Set your site resources

Follow the [installation steps](../../README.md#install), then edit [full.yaml](full.yaml):

| Fields | Set to |
| --- | --- |
| `af3.*` | Your compatible patched AF3 image, weights, and databases |
| `inputs.conda_env` | Your preparation environment (`latentppi` if using the supplied environment) |
| `cluster.inputs_partition` / `cluster.screen_partition` | Preparation / AF3 screening partitions; comma-separated names accepted |
| `cluster.gres` | GPU request for your cluster |

The checkout, submitting Python environment, and resources must be available on compute nodes. The AF3 image must write `prediction_timings.jsonl` for completion tracking. This config disables latent scoring, structure scoring, pooling, and compression, so it needs no scorer checkpoint or ipSAE installation.

Keep the example's input and output paths for this walkthrough. See [CONFIG.md](../../CONFIG.md) when adapting them to your own proteins.

## 2. Copy the target MSA and preview

```bash
mkdir -p examples/toy_screen/run
cp examples/toy_screen/query_msa/target_small.a3m examples/toy_screen/run/target_small.a3m
python latentppi.py validate examples/toy_screen/full.yaml
python latentppi.py run examples/toy_screen/full.yaml --no-submit
```

The supplied MSA avoids a ColabFold search. Fix any validation errors before submitting. Missing command files and input JSON directories are expected before preparation.

The preview writes configuration and submission artifacts under `run/latentppi_runs/preview-*`, but submits no jobs. Use `run --no-submit` to preview scheduler submissions.

## 3. Submit and check progress

```bash
python latentppi.py run examples/toy_screen/full.yaml
# Once input preparation finishes:
python latentppi.py status examples/toy_screen/full.yaml
```

Preparation should produce **50 pairs**, with `af3_inputs/`, `af3_cmds.txt`, `jobs.tsv`, and `af3_lengths.csv` under `run/`. GPU arrays start after preparation succeeds; the collector runs after all arrays terminate, including failures. Existing prepared inputs skip the preparation stage.

Status shows completion counts, outstanding indices, and recorded Slurm jobs. Before preparation finishes, the manifest may be unavailable. During execution, unfinished attempts can appear as failed/incomplete; also check the queue state.

Each run records submission IDs, settings, and a job ledger under `latentppi_runs/run-*`. Keep the checkout scripts and any external scorer manifest unchanged while jobs are queued. The CLI refuses overlapping recorded active jobs for the same screen. If a scheduler response is ambiguous, inspect the queue before retrying; submission is not automatically retried.

## 4. Read the results

Results refresh at batch boundaries with default chained dispatch. Each one-CPU dispatcher submits a batch and a dependent continuation, then exits; it does not wait on an allocated CPU throughout the GPU run. To refresh the tables manually:

```bash
python latentppi.py status examples/toy_screen/full.yaml --heavy
```

Look under `examples/toy_screen/run/`:

| Path | What to inspect |
| --- | --- |
| `metrics/ranked_survivors.tsv` | Successful pairs share the last rank group in this example because structure scoring is disabled |
| `metrics/all_results.tsv` | Every pair and its outcome |
| `metrics/summary.txt` | Settings and counts |
| `metrics/screen_status.json` | Completion counts and outstanding indices |
| `af3_predictions/` | Structures, timing records, and GPU/job metadata |
| `slurm_logs/` | Job output and errors |
| `latentppi_runs/` | Submission records and config snapshots |

This example also has no observed ipSAE scores because structure scoring is disabled. See [METRICS.md](../../METRICS.md) for collection options, column definitions, and ranking rules.

## 5. Resume or change a screen

After existing jobs terminate, run the same command to retry unfinished pairs:

```bash
python latentppi.py run examples/toy_screen/full.yaml
```

 A recorded failed runner attempt remains outstanding even if AF3 finished before postprocessing failed. Pre-output failures are tracked under `.latentppi_attempts`; older runs without these markers cannot distinguish such failures from never-run pairs.

| Goal | Action |
| --- | --- |
| Retry unfinished pairs | Re-run `run` with the same YAML |
| Repeat all predictions | Add `--force` |
| Regenerate preparation, including the target MSA | Add `--force-inputs`; requires ColabFold and still skips completed predictions |
| Regenerate preparation and repeat predictions | Combine `--force-inputs --force` |
| Add candidates | Run `inputs` with the YAML, wait for it to finish, then run `run` |
| Compare runners, models, or scientific settings | Use a new `inputs.output_base` |

Resume uses system identity, not configuration hashes. Editing the YAML alone does not invalidate completed results. For deliberate reruns in the same directory, use the appropriate force flags above, including input regeneration when prepared commands need to change.

## 6. Clean up

Preview the deletion plan, then clean the example when you no longer need its results:

```bash
python latentppi.py clean examples/toy_screen/full.yaml --dry-run
python latentppi.py clean examples/toy_screen/full.yaml
```

| Option | What it removes |
| --- | --- |
| Default | Predictions and attempt records, ipSAE outputs, embedding shards, generated metrics, logs, run snapshots, rendered config, and temporary query-MSA work folders |
| `--all` | Everything above, plus AF3 input JSONs, `af3_cmds.txt`, `jobs.tsv`, and `af3_lengths.csv` |
| `--all --remove-query-msa` | Everything above, plus the target MSA cache and query-only fallback cache |

Default cleanup retains prepared inputs and the cached target MSA. `--all` makes the next run prepare fresh inputs; removing the target MSA also requires a new MSA search. Cleanup preserves the source FASTA, YAML, shared candidate MSAs, original model resources, supplied dataset summary, and unrelated files outside dedicated output directories.

Cleanup prints the plan and asks for confirmation; `--yes` skips only that prompt. `--dry-run` writes nothing and does not query Slurm. Cleanup checks recorded parent and advance-batch jobs before deletion. Stop the advance watcher and wait for active jobs to finish. An unreadable queue or a corrupt submission ledger also blocks deletion. It never cancels jobs. Finish or cancel manually submitted jobs yourself first; CLI job checks do not cover them.

Output directories must be dedicated to the screen.  Only current YAML paths and known artifacts are covered; clean external outputs and earlier configurations' output locations separately. The submission lock file stays in place. Missing artifacts are harmless; deletion errors return a nonzero status, and submission history is removed last.

`delete` is an alias for `clean`; `clean_screen.sh config.yaml [options]` is a compatibility wrapper. Cleanup needs Python with PyYAML, but no working AF3 resources or checkpoints.

## Local check without a cluster

After copying the MSA in step 2, prepare inputs locally:

```bash
python run_pipeline.py --fasta-single examples/toy_screen/target_small.fasta \
  --msa-a examples/toy_screen/run/target_small.a3m \
  --msa-dir examples/toy_screen/proteome --output-dir examples/toy_screen/run \
  --af3-num-recycles 1 --af3-num-diffusion-samples 1
python latentppi.py status examples/toy_screen/full.yaml
```

A fresh directory should report **50 total, 50 never-run, indices 1–50**. No AF3 inference runs. Once site resources are configured, `run` can submit these prepared inputs.

## Try latent scoring

Use [config.yaml](config.yaml) with a compatible latent AF3 image and the bundled scorer. Copy the supplied target MSA into the configured `./run_latent` directory before validation.

Configure the checkpoints and gate using the [scorer reference](../../CONFIG.md#latent-scorer). This YAML enables ipSAE parsing and compression; supply ipSAE and `zstd` or disable those [optional stages](../../CONFIG.md#optional-stages).

Repeat steps 2–4 using `config.yaml` and your new output directory. The [metrics guide](../../METRICS.md) explains predicted scores and the missing observed values for gated pairs.

## Optional complex size cutoff

Add `max_total_residues: 3000` directly under `cluster` in either toy YAML to
exclude complexes with more than 3,000 combined residues from folding submission.
The default is `null` (no cutoff). The toy complexes are below this example limit.
Inputs and command files keep all pairs; dispatch skips oversized indices and
status reports unfinished excluded pairs as `skipped_length`. The cutoff applies
to retries and forced runs too. See the [cutoff reference](../../CONFIG.md#complex-size-cutoff)
for existing-run behavior and the distinction from GPU tier boundaries.

## Optional size-based GPU routing

For larger screens, see the [GPU tier configuration](../../CONFIG.md#size-based-gpu-tiers).
The recommended regular/large example uses `max_total_residues: 3000`, inclusive
of both chains. **We chose 3,000 conservatively based on our L40 runs**; users with
smaller GPUs should lower that cutoff or add a smaller tier. A large first attempt
still receives only one retry on the final tier when retries are enabled.

The toy configs leave tiers disabled. Add tiers only after choosing actual cluster
resources; existing screens without this setting retain their original routing.

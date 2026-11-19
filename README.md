# 🧬 latentPPI: efficient whole-proteome screening

> **Early release:** latentPPI is under active development. Features, configuration options, and documentation may change. AF3 image download and build-repository links are **coming soon**.

Screen a **target protein** against a proteome using full AF3 or LatentPPI-gated inference. One YAML config handles input preparation, Slurm submission, resume, and ranking. Here, *target* means the query protein; *candidates* are the proteins screened against it.

[First screen](examples/toy_screen/WALKTHROUGH.md) · [Configuration reference](CONFIG.md) · [Metrics reference](METRICS.md)

## 🧰 Install

The dependencies for latentPPI can be installed by the following:

## 🐍 Conda environment
```bash
conda env create -f environment.yml
conda activate latentppi
```

You need Python 3.10+ with PyYAML, pandas, and NumPy; Slurm, Bash, flock, and Singularity; and the patched AF3 image, weights, and databases. AF3 image links are **coming soon**; see the AF3 SIF image section below. Obtain weights through the [AlphaFold3 repository](https://github.com/google-deepmind/alphafold3), following its terms of use. The `ipsae.py` and information can be found from its [repository](https://github.com/DunbrackLab/IPSAE).

Optional stages need additional tools:

| Stage | Requirements |
| --- | --- |
| Generate a target MSA | ColabFold (`colabfold_batch`) in `inputs.conda_env` |
| Latent scoring | Compatible AF3 image and a [bundled scorer checkpoint](models/checkpoint/) |
| Structure scoring | `ipsae.py` and its dependencies |
| Pool / compress outputs | `h5py` / `zstd`, respectively |

### 📦 AF3 SIF image

- Prebuilt AF3 SIF image download: **coming soon**.
- `af3-latentppi` build repository: **coming soon**. Building the image requires `Docker`, followed by conversion to a `.sif` with `Singularity`.

Set `AF3_IMAGE` to the resulting SIF path when running the screen. 

## 👶🏻👩🏽‍💻 Run your first screen

The [toy example](examples/toy_screen/WALKTHROUGH.md) pairs one synthetic target with 50 candidates. Its supplied MSA avoids an MSA search; its full-run config disables scoring, pooling, and compression.

**First edit [full.yaml](examples/toy_screen/full.yaml)** with your AF3 paths, conda environment, CPU/GPU partitions, and GPU request. Then run from the repository root:

```bash
mkdir -p examples/toy_screen/run
cp examples/toy_screen/query_msa/target_small.a3m examples/toy_screen/run/target_small.a3m

python latentppi.py validate examples/toy_screen/full.yaml
python latentppi.py run examples/toy_screen/full.yaml --no-submit
python latentppi.py run examples/toy_screen/full.yaml

# Once input preparation finishes:
python latentppi.py status examples/toy_screen/full.yaml
python latentppi.py status examples/toy_screen/full.yaml --heavy
```

`run` prepares inputs, submits GPU tasks after preparation succeeds, and collects metrics after the tasks finish—even if some fail. Re-run the same command after jobs terminate to submit only unfinished pairs.

The toy sequences and alignments test the workflow, not biological interactions. For preparation without Slurm or AF3, follow the [local check](examples/toy_screen/WALKTHROUGH.md#local-check-without-a-cluster).

**To try a gated screen**, edit the [latent toy config](examples/toy_screen/config.yaml) with your AF3 paths and Slurm settings. It uses the bundled ipSAE_d0dom scorer and writes to `./run_latent`.

> **Prerelease:** The AF3 SIF image does not yet advance passing pairs automatically. After submitting the screen, start `advance --watch` and leave it running. It submits each batch of passing pairs for a fresh, ungated 10-recycle run.

```bash
mkdir -p examples/toy_screen/run_latent
cp examples/toy_screen/query_msa/target_small.a3m examples/toy_screen/run_latent/
python latentppi.py run examples/toy_screen/config.yaml
python latentppi.py advance examples/toy_screen/config.yaml --watch
```

## Complex size cutoff (optional)

To skip folding complexes above a chosen combined length, add:

```yaml
cluster:
  max_total_residues: 3000
```

This allows up to **3,000 total residues across both chains** and excludes larger
complexes from submission. Omit the option or set it to `null` to disable it.
The limit applies to both runners and dispatch modes, including retries and
`--force`, before any GPU tier routing.

Large complexes remain in generated inputs and command files, preserving task
indices. Status and results label unfinished excluded pairs `skipped_length`;
completed results are retained. Raise the limit and rerun to include previously
excluded pairs without rebuilding inputs. Existing submitted jobs keep their
saved settings. See the [cutoff reference](CONFIG.md#complex-size-cutoff) for
details, including preview behavior and manually submitted runners.

## Size-based GPU routing (optional)

New screens can route complexes by **total residues across both chains**. Our
recommended two-tier example keeps **up to and including 3,000 residues** on the
regular GPU and sends larger complexes directly to the large GPU:

```yaml
cluster:
  gpu_tiers:
    - name: regular
      max_total_residues: 3000
      partition: your_regular_gpu_partition
      gres: gpu:your_regular_gpu_type:1
    - name: large
      partition: your_large_gpu_partition
      gres: gpu:your_large_gpu_type:1
  retry_failed_once: true
```

**We chose 3,000 conservatively based on our L40 screening runs**, with additional
A100 40GB OOM evidence. It is a routing heuristic, not a guaranteed memory limit.
Users with smaller GPUs should lower the cutoff or add a smaller tier; GPU memory,
AF3 padding and inference settings affect capacity. The GPU investigation report is coming soon.

Replace the example resources with your cluster's actual partition/GRES names.
Tiers are ordered by increasing size, with an unbounded final tier. Any failed
pair gets at most one retry on the final tier, including pairs initially routed
there. Gate-rejected pairs are not failures. The retry budget persists on resume.

Omitting `gpu_tiers` (or using `[]`) preserves the existing single-resource behavior;
this feature does not change existing screen configs or queued jobs. With tiers,
omit `retry_partition`/`retry_gres`: the final tier provides the retry resources.
See [configuration details](CONFIG.md#size-based-gpu-tiers).

## 🫵 Use your own target

Start from [full.yaml](examples/toy_screen/full.yaml) or the [latent config](examples/toy_screen/config.yaml). Set:

- `inputs.fasta_file`: one target sequence.
- `inputs.proteome_msa_dir`: candidate `.a3m` files in a shared directory.
- `inputs.output_base`: the screen's output directory.
- `paths.script_dir`, `af3.*`, and `cluster.*`: your checkout, resources, and scheduler settings.

Input/resource paths resolve relative to the YAML; output paths resolve relative to `output_base`. Checkpoint paths resolve relative to `script_dir`. Update relative paths if you move the YAML.

A supplied target MSA must be the only root-level `.a3m` in the output directory. Its filename must match the FASTA stem (`target.fasta` → `target.a3m`), and its first sequence must match the target. Candidate MSAs stay in their shared directory; keep them readable on compute nodes.

## 💾 Create your Own 'Proteome'
The project proteome download is **coming soon**. To screen your own proteome, generate an MSA for each candidate sequence. See the [ColabFold setup guide](https://colabfold.mmseqs.com/) for local search requirements.


### 🎯 Enable latent scoring

Set `runner: latent` and supply compatible checkpoints:

```yaml
runner: latent
scorer:
  rank_by: ipSAE_d0dom
  threshold: 0.5
  models:
    ipSAE_d0dom: {path: models/checkpoint/latentppi_max_ipSAE_d0dom__emb-r1_score-r1_model.pt}
```

`rank_by` selects the predicted gate metric. Completed pairs rank by measured ipTM within pLDDT groups (above 70 first, then 70 or below, then missing scores). Gate predictions have a separate `gate_rank`. Add models to predict more metrics; all configured models are used unless `predict_metrics` selects a subset. Pairs gated below the threshold skip diffusion and have no structure-derived ipSAE. See [CONFIG.md](CONFIG.md) for manifests and optional stages.

## 📌 Everyday commands

Replace `config.yaml` with your config path.

| Command | Action |
| --- | --- |
| `python latentppi.py validate config.yaml` | Check settings and resources |
| `python latentppi.py clean config.yaml --dry-run` | Preview removal of generated results |
| `python latentppi.py clean config.yaml` | Remove only results and keep prepared inputs and the query FASTA |
| `python latentppi.py clean config.yaml --all` | Remove results and prepared inputs after confirmation; keep the query MSA |
| `python latentppi.py run config.yaml --no-submit` | Preview submissions without submitting |
| `python latentppi.py run config.yaml` | Start or resume the first-pass screen |
| `python latentppi.py advance config.yaml --watch` | During a latent screen, automatically submit passing pairs for a full 10-recycle run as they become available |
| `python latentppi.py advance config.yaml` | After a latent screen finishes, submit all newly eligible pairs in one batch |
| `python latentppi.py advance config.yaml --no-submit` | Refresh the candidate CSV and list newly eligible pairs without submitting jobs |
| `python latentppi.py status config.yaml` | Light status (default): skip metrics recomputation; show completion counts and recorded Slurm jobs |
| `python latentppi.py status config.yaml --heavy` | Heavy status: recompute metrics files, then show completion counts and recorded Slurm jobs |
| `python latentppi.py inputs config.yaml` | Prepare or update inputs only |
| `python latentppi.py render config.yaml` | Write environment/config artifacts only |

### Advance passing pairs to 10 recycles

**Advancement starts when you run `advance`.** For a latent screen, submit the first pass and then start the watcher with the same config on a host that can submit Slurm jobs.

```bash
python latentppi.py run config.yaml
python latentppi.py advance config.yaml --watch
```

Leave the watcher running. It checks for completed pairs with at least one passing seed and submits one advance batch at a time. Each pair gets a fresh, ungated AF3 run with its original input and seeds, **10 recycles by default**, and structure scoring. The default polling interval is 300 seconds. Restarting the watcher picks up new passes and skips pairs already assigned to a batch.

**If the first pass has already finished**, run `python latentppi.py advance config.yaml` once. To preview the selection first, add `--no-submit`. To request a different recycle count, use `--recycles N` (1–10).

Read `metrics/advanced_candidates.csv` for first-pass scores and the latest published advance status. The `advance_` score columns and `advance_rank` use the second run. Refresh the source CSV on demand with `status config.yaml --heavy`. Each batch has its own config, manifests, predictions and logs under `advance/batch-*/`. If a batch fails, inspect its Slurm jobs and resume it with `python latentppi.py run <batch>/advance.yaml`. A reserved batch with an unconfirmed submission is shown as `prepared`. Check the queue before resubmitting it.

### Resume or change a screen

Use `run --force` to repeat completed predictions. Use `run --force-inputs` to regenerate preparation **including the target MSA**, which requires ColabFold. Combine both to regenerate everything.

For new candidates, run `inputs`, wait for it to finish, then run `run`. Wait for existing first-pass jobs to finish before resubmitting `run`. When changing models or scientific settings, use a new output directory. The saved screen identity rejects incompatible reuse; `--force` repeats work with compatible settings.

`run` defaults to **chained dispatch**: a one-CPU dispatcher submits one GPU array
and a dependent continuation, then exits. Slurm keeps the continuation pending
until the batch and its submitting dispatcher finish. No dispatcher CPU remains
allocated while the GPU batch runs or waits in the queue. Each continuation
submits the next batch or eligible one-time retries; the final step schedules
collection. A continuation can add a short CPU-queue delay between batches.

Set `cluster.max_concurrent_tasks` for the screen-wide running-task limit and
`cluster.user_queue_budget` for your user submission budget (pending plus running
tasks, including other jobs). For a 50-job limit, use `user_queue_budget: 50`,
`reserved_queue_slots: 2`, and `max_concurrent_tasks: 4`. Chained mode additionally
reserves one submission slot for the continuation. Quota checks are disabled by
default (`user_queue_budget: null`); Slurm limits still apply.

`dispatcher_time_limit` now defaults to **one hour per invocation**, not the full
screen duration; increase it if collecting a very large screen needs longer.
`dispatcher_cpus_per_task` defaults to one independently of GPU task CPU counts.
Use `dispatch_mode: persistent` for the old wait-loop behavior, including separate
CPU/GPU Slurm clusters. Old saved snapshots retain persistent behavior and their
original resource settings, so already-submitted screens are not migrated.
If a handoff fails, accepted jobs continue; inspect `status` and resume with `run`
after recorded jobs finish. See [cluster settings](CONFIG.md#cluster-resources-and-arrays).

For readable queue labels, set `cluster.job_name: cool-protein`; jobs will appear as
`cool-protein-array`, `cool-protein-dispatch`, etc. Leaving it empty preserves generated
names. Existing jobs keep their submitted names.

Gated pairs count as completed. While jobs are active, unfinished attempts can appear as failed/incomplete; check their queue state too. Use `run --no-submit` to preview scheduler submissions.

## 📊 Find your results

Default paths under `inputs.output_base`:

| Path | Contents |
| --- | --- |
| `metrics/advanced_candidates.csv` | Confirmed gate passes with first-pass metrics, advance status, and available full-run scores |
| `metrics/ranked_survivors.tsv` | Completed pairs ranked by confidence group and measured ipTM |
| `metrics/all_results.tsv` | Every pair, its outcome, gate prediction and final scores |
| `metrics/known_interactors.tsv` | Every supplied known-interactor entry, including missing or killed pairs |
| `metrics/summary.txt` | Settings, score definitions and outcome counts |
| `metrics/screen_status.json` | Counts and outstanding indices |
| `af3_predictions/` | Predictions, timing records, and `run_env.json` GPU/job metadata |
| `ipsae_summary_af3.csv` | Retained internal source of structure scores |
| `slurm_logs/` | Job logs |
| `latentppi_runs/` | Submission IDs and config/environment snapshots |

Full runs rank completed pairs by measured scores. With the default ranking, pairs missing scores share the last rank group. Observed ipSAE uses the highest `ipSAE` among `Type=max` rows per system and seed. For custom collection:

```bash
python collect_metrics.py --root /path/to/screen \
  --out-dir /path/to/screen/metrics --rank-by ipSAE_d0dom --agg mean
```

Results refresh at batch boundaries by default. In persistent mode, `output.refresh_seconds` controls the periodic interval. Use `output.known_interactors` to supply an NP accession list.

See [METRICS.md](METRICS.md) for output columns and aggregation, and [CONFIG.md](CONFIG.md) for prepared input files, array limits, and runner settings.

The [legacy tools](legacy/README.md) are available for reproducing older workflows.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests mock Slurm and inference. Postprocessing checks also need `h5py` and `zstd`; the optional shard-merger check is skipped when that helper is absent.

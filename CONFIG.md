# Screen configuration

Use this reference to edit the YAML passed to `latentppi.py`. Start from [full.yaml](examples/toy_screen/full.yaml) for full AF3 or [config.yaml](examples/toy_screen/config.yaml) for latent scoring.

[README](README.md) · [Screen walkthrough](examples/toy_screen/WALKTHROUGH.md) · [Metrics reference](METRICS.md)

## Top-Level Config Settings

| Section | Controls |
| --- | --- |
| `inputs` | Target FASTA, candidate MSAs, preparation, and scientific settings |
| `paths` | Checkout and generated-file locations |
| `af3` | Container image, weights, and databases |
| `runner` | `full` or `latent` (default: `latent`) |
| `scorer` | Score models, predicted metrics, and diffusion threshold |
| `stages` | Structure scoring, embedding pooling, and compression |
| `cluster` | Slurm partitions, resources, and arrays |
| `output` | Results location, refresh interval and optional known-interactor report |

 Write booleans as `true` or `false`, and quote times such as `'24:00:00'`. 
 
 Defaults below come from the CLI; **example configs override some defaults**, including the preparation environment and optional stages.

## Inputs

| Field under `inputs` | Default | Purpose |
| --- | --- | --- |
| `fasta_file` | Required | FASTA containing exactly one target sequence |
| `proteome_msa_dir` | Required | Shared directory of candidate `.a3m` files |
| `output_base` | Required | Dedicated screen directory |
| `conda_env` | `latentppi` | Preparation environment|
| `af3_num_recycles` | `1` | AF3 recycle count; min = 0 |
| `af3_num_diffusion_samples` | `1` | Diffusion samples; at least 1 |
| `max_identity` | `95.0` | Skip pairs above this sequence-identity percentage; range 0–100 |
| `filter_non_standard` | `true` | Filter pairs with non-standard amino acids;  the ipSAE parser struggles to parse those |
| `skip_if_msa_exists` | `true` | Reuse a supplied or cached target MSA during preparation |
| `force` | `false` | Rebuild existing input JSONs and regenerate the target MSA during preparation |
| `recreate_manifest` | `false` | Replace `jobs.tsv` rather than extend it |
| `recreate_commands` | `false` | Replace `af3_cmds.txt` rather than extend it |

Preparation flags apply when the input stage runs. `inputs.force` is distinct from CLI `run --force`, which repeats predictions. See the walkthrough for [rerun options](examples/toy_screen/WALKTHROUGH.md#5-resume-or-change-a-screen).

### Target and candidate MSAs

A supplied target MSA must be the **only root-level `.a3m`** in `output_base`, named after the FASTA stem (`target.fasta` → `target.a3m`). Its first sequence must match the target. Ambiguous, misnamed, or mismatched MSAs fail before preparation. A single-sequence target MSA triggers a coevolution warning.

Without a reusable target MSA, preparation needs `colabfold_batch` in `inputs.conda_env`. Candidate MSAs stay in `proteome_msa_dir`,; it serves as a permanent accessible folder for all screens against that proteome.

## Paths and output locations

Relative filesystem paths resolve as follows. Tildes and environment variables expand on the submitting host.

| Path setting | Relative to |
| --- | --- |
| `inputs.fasta_file`, `inputs.proteome_msa_dir`, `inputs.output_base`, `af3.*`, `paths.script_dir`, `output.known_interactors` | The YAML file's directory |
| Other `paths.*` entries, `output.metrics_dir` | `inputs.output_base` |
| `scorer.models.<metric>.path`, `scorer.manifest_path` | `paths.script_dir` |

Bind-mounted directories may contain spaces, quotes, and globs, but **not commas or colons**, which delimit Singularity binds.

| Field | Default | Purpose |
| --- | --- | --- |
| `paths.script_dir` | Directory containing `latentppi.py` | Checkout containing runners and helpers |
| `paths.cmd_file` | `af3_cmds.txt` | One AF3 command per task |
| `paths.af3_input_dir` | `af3_inputs` | Prepared AF3 JSONs |
| `paths.af3_output_dir` | `af3_predictions` | Prediction folders and timing records |
| `paths.summary_tsv` | Empty | Dataset summary required for embedding pooling |
| `paths.shard_dir` | `h5_shards` | Pooled embedding shards; if pooling was turned on |
| `paths.ipsae_outroot` | `ipsae_scores_af3` | Per-prediction structure scores |
| `paths.ipsae_summary` | `ipsae_summary_af3.csv` | Combined structure scores |
| `output.metrics_dir` | `metrics` | Master, ranked survivors, optional known-interactor report and status |
| `output.rank_confident_iptm` | `true` | Rank completed pairs by measured ipTM in pLDDT groups. Set to false to rank by the measured scorer metric. |
| `output.refresh_seconds` | `300` | Chained mode: nonzero enables batch-boundary updates; persistent mode: periodic interval; `0` disables dispatcher updates |
| `output.known_interactors` | Empty | Optional text file of NP accessions |
| `output.write_summary` | `true` | Write `summary.txt` with settings and counts |
| `output.legacy_csv` | `false` | Also write the older CSV tables; their ranking remains prediction-based |

Preparation generates `af3_inputs/`, `af3_cmds.txt`, `jobs.tsv`, `af3_lengths.csv`, and `screen_candidates.tsv` under `output_base`. Custom array-stage input and command paths must already exist; changing those paths does not redirect the input builder. Command counts use physical lines, and blank lines are rejected.

An empty `summary_tsv` is allowed with pooling off; the shell fallback is `output_base/dataset_summary.tsv`. To enable pooling, explicitly supply a readable summary matching your systems, with `uniprot_a`, `uniprot_b`, `len_a`, `len_b`, `label`, and any optional metadata such as `evidence_count`.

## AF3 resources

| Field under `af3` | Default |
| --- | --- |
| `image` | `/path/to/af3_latentppi_sf_v303.sif` |
| `model_parameters_dir` | `/path/to/af3/weights` |
| `databases_dir` | `/path/to/af3/databases` |

Replace these `/path/to/...` placeholders with your site's resources.

## Latent scorer

```yaml
runner: latent
scorer:
  rank_by: ipSAE_d0dom
  threshold: 0.5
  models:
    ipSAE_d0dom:
      path: models/checkpoint/latentppi_max_ipSAE_d0dom__emb-r1_score-r1_model.pt
```

| Field under `scorer` | Default | Purpose |
| --- | --- | --- |
| `models` | Empty mapping | Metric names mapped to `{path: checkpoint.pt}` |
| `predict_metrics` | All configured model names when omitted | Subset of models to run |
| `rank_by` | Empty; required for latent runs | Selects the gate metric. Also selects the measured ranking metric when `output.rank_confident_iptm: false` |
| `threshold` | `0.5` | Pairs below this predicted score skip diffusion |
| `manifest_path` | Empty | Use an existing authoritative scorer manifest |

Metric names come from the models; the CLI has no predefined vocabulary. Gated pairs count as completed and have no observed structure scores. See [METRICS.md](METRICS.md) for predicted and observed columns.

### Existing manifests

With `manifest_path`, the existing file is used unchanged. Its model keys and `rank_by` override YAML `models`, `predict_metrics`, and `rank_by`; no manifest or checkpoint copies are generated.

Each checkpoint must be readable and loadable through a relative path inside the manifest directory, without parent traversal or symlinks. The latent runner binds that directory to `/root/scorer`. Keep manually supplied manifests and checkpoints unchanged while jobs are queued.

## Optional stages

| Field under `stages` | Default | Purpose |
| --- | --- | --- |
| `save_embeddings` | `false` | Save both regular and pre-Pairformer embeddings; set `false` to disable both exports |
| `run_ipsae_parse` | `true` | Calculate observed structure scores using ipSAE |
| `run_pool` | `false` | Pool embeddings into HDF5 shards; needs `paths.summary_tsv` and `h5py` |
| `run_compress` | `true` | Compress structure/confidence outputs, to reduce space; needs `zstd` |
| `pool_seed` | `1` | Seed to pool; nonnegative integer |
| `keep_npz` | `false` | Retain raw embedding files after pooling |
| `compress_embeddings` | `false` | Include embeddings in compression and preserve them through pooling for that step |

To disable embedding files, set:

```yaml
stages:
  save_embeddings: false
  run_pool: false
```

`save_embeddings` controls exports only; the latent scorer still computes the
representations it needs. Pooling requires saving to be enabled. `keep_npz` and
`compress_embeddings` do not control whether AF3 writes embeddings. The YAML
setting overrides embedding flags in `af3_cmds.txt`; direct runner invocations
can use `SAVE_EMBEDDINGS=0` (and `RUN_POOL=0`). Changes apply to newly submitted
jobs, not already submitted Slurm scripts or running processes. Existing files
are not removed.

The full toy config disables all three optional stages. The latent toy config enables structure scoring and compression, so supply those dependencies or disable the stages. Gated predictions have no structure to score or compress.

## Cluster resources and arrays

Defaults are placeholders; replace `your_gpu` and `your_cpu` with your cluster partitions and edit the GPU request.

| Field under `cluster` | Default | Purpose |
| --- | --- | --- |
| `job_name` | Empty | Optional Slurm job-name prefix; stage suffix is appended automatically |
| `screen_cluster` | Empty | Optional single Slurm cluster name (`--M`) for GPU arrays, for either runner |
| `inputs_cluster` | Empty | Optional single Slurm cluster name for CPU preparation, dispatcher, and collector |
| `screen_partition` | `your_gpu` | Partition(s) where AF3 and its screening jobs run; comma-separated, e.g. `your_gpu,your_gpu_alt` |
| `cpu_only` | `false` | Run AF3 screening without GPU resources, GPU constraints, or Singularity `--nv`; requires a CPU-capable AF3 image |
| `inputs_partition` | `your_cpu` | CPU preparation, dispatcher, and collector partition(s); accepts comma-separated names |
| `gres` | `gpu:1` | GPU resource request; ignored (and may be empty) when `cpu_only: true` |
| `cpus_per_task` | `4` | CPUs per job; at least 1 |
| `time_limit` | `'24:00:00'` | Slurm job time limit |
| `mail_user` | Empty | Optional email recipient for END/FAIL notifications |
| `max_concurrent_tasks` | `4` | Screen-wide maximum running array tasks; at least 1 |
| `user_queue_budget` | `null` | User pending + running task budget on the screening cluster; null disables the budget check |
| `reserved_queue_slots` | `2` | Additional free submission slots, beyond jobs already counted; nonnegative |
| `poll_interval` | `60` | Seconds between quota/completion checks; at least 1 |
| `dispatch_mode` | `chained` | One short dispatcher per batch; `persistent` retains the wait loop |
| `dispatcher_cpus_per_task` | `1` | CPUs per dispatcher, independent of GPU task CPUs |
| `dispatcher_time_limit` | `'01:00:00'` | Per-dispatcher wall time; persistent mode needs enough time for the full screen |
| `offset` | `0` | Exclude earlier command lines; nonnegative |
| `max_total_residues` | `null` | Skip submitting complexes above this combined chain length; positive integer or null to disable |
| `max_array_size` | `1001` | Exclusive upper bound for local task indices; at least 2 |
| `array_spec_limit` | `1024` | Maximum array-expression length, including `%N`; at least 8 |

### Complex size cutoff

```yaml
cluster:
  max_total_residues: 3000
```

This optional submission limit uses **total residues across both chains**:
3,000 is allowed and 3,001 is excluded. Omit it or use `null` for no limit.
Zero, negative values, booleans, and non-integers are rejected.

Both `chained` and `persistent` dispatch apply the cutoff before GPU routing,
including retries and `--force`, for either runner. It also works without GPU
tiers. The nested `gpu_tiers[].max_total_residues` chooses a resource tier;
the top-level `cluster.max_total_residues` excludes jobs from submission.

Inputs, `af3_cmds.txt`, and command indices remain intact. Thus, oversized pairs
still appear in generated inputs and commands, but their indices are omitted
from submitted arrays. `run --no-submit` reports the excluded count when inputs
already exist. With the cap enabled, each manifest row must have a positive integer
`total_length` in `af3_lengths.csv`; missing or invalid lengths stop dispatch.

Unfinished excluded pairs are reported as `skipped_length`; result rows include
`total_length` and the configured `max_total_residues`. Completed results retain
their outcome, and recorded running attempts remain running. Raising or removing
the cap makes previously excluded unfinished pairs eligible on the next `run`,
subject to the normal offset and retry rules, without regenerating inputs.

The setting takes effect in the next run snapshot. Editing YAML does not change
already submitted jobs or their continuations. Directly submitting runner scripts
outside the YAML dispatcher bypasses this filter.

### Size-based GPU tiers

`cluster.gpu_tiers` defaults to `[]` (disabled). When supplied, it replaces the
normal `screen_partition`/`gres` resource selection for GPU arrays. CPU preparation,
dispatcher and collector settings remain independent. All tiers use `screen_cluster`.

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

The **3,000-residue regular-tier maximum was chosen conservatively based on our
L40 runs**, supported by A100 40GB failures. This is total complex length, not just
target length. It is not a universal safe limit: lower it for weaker GPUs or add
an intermediate tier. A `gpu:1` request can select any eligible GPU, so choose a
cutoff suitable for the weakest eligible GPU or specify a GPU type.

Each tier requires a unique `name`, a `partition` and a `gres`. All but the final
tier require strictly increasing positive integer `max_total_residues` values.
Limits are inclusive: 3,000 uses regular; 3,001 uses large. The final tier must omit
its maximum or set it to `null`. For three tiers, insert a small tier with a lower
maximum before regular; the code supports any number of tiers.

Initial arrays contain only one resource tier. Existing command indices, array
limits, offsets and queue budgets still apply. `af3_lengths.csv` must contain a
positive integer `total_length` for each outstanding pair; routing fails before
submission if a needed length is missing. Completed/gate-rejected pairs remain
excluded on normal resume. `--force` retains its explicit rerun behavior.

`retry_failed_once: true` retries failures once on the **final tier**, even when
that was their initial tier. Initial routing never consumes the retry allowance.
Do not combine tiers with `retry_partition` or `retry_gres`; those fields belong to
the legacy two-resource retry configuration below. Tiers can also be enabled
without retries. GPU availability validation checks every tier on `screen_cluster`.
GPU types are selected through GRES; existing runner-header constraints still apply.

Submission records and exported environments identify the tier and whether the
route was selected by `total_length` or `retry`. Preview mode shows tier-specific
Slurm commands and creates no retry reservations. Existing configs/snapshots with
no tiers retain their old resource selection; enabling tiers requires an explicit
config edit for a new/resumed submission after existing jobs have terminated.

### Legacy one-time GPU fallback

Optional one-time GPU fallback:

```yaml
cluster:
  retry_failed_once: true
  retry_partition: your_large_gpu_partition
  retry_gres: gpu:your_large_gpu_type:1
```

Disabled by default. After the normal batches finish, unresolved pairs are submitted
once using the fallback partition and GRES on `screen_cluster` (or the default
cluster when unset). This covers any failure reason, including jobs that terminate
before writing output. Completed and gate-killed pairs are not retried. The retry
reruns the whole pair with the same scientific settings and records a new attempt.
Other resources, including CPU count and wall time, retain their normal settings.
CPU-only screening cannot enable GPU fallback.

`latentppi_gpu_retries.json` persists the one-retry budget across dispatcher restarts
and resumed runs. With fallback enabled, `--force` does not reset that budget. A
failed fallback remains failed; there is no third automatic attempt. Slurm automatic
requeue is disabled for these arrays. An ambiguous submission consumes its reserved
retry to avoid duplicate jobs; explicit quota rejections release the reservation.
Cleaning screen outputs removes this ledger along with the results.

Validation checks both GPU requests against the node GRES inventory in the named
partitions on the screen cluster. A missing GPU type or insufficient per-node count
produces a warning. This checks configured GPU inventory, not current idle capacity,
queue eligibility, or whether a complex will fit in GPU memory. Partition checks
still report absent or down partitions as errors. Configure both fallback fields
when enabling retries.

`partition` remains a legacy alias for `screen_partition`; specify only one.
Saved snapshots retain the internal `partition` field for compatibility.
Both partition settings accept comma-separated strings; whitespace around names is
trimmed and every named partition is validated on the selected cluster. Slurm
selects an eligible partition; this does not submit a separate job to each one.

For CPU AF3 screening:

```yaml
cluster:
  screen_partition: your_cpu,your_cpu_alt
  cpu_only: true
  cpus_per_task: 16
```

CPU mode is explicit; partition names do not automatically enable it. It applies
to either AF3 runner. Preparation, dispatch, and collection still use
`inputs_partition`. CPU submissions use a Bash wrapper to bypass the runners'
GPU SBATCH defaults. Any additional AF3 CPU flags depend on your chosen image.

- Set `cluster.job_name: cool-protein` to display jobs as `cool-protein-inputs`,
`cool-protein-dispatch`, `cool-protein-array`, and `cool-protein-collector`. The prefix
must start with a letter or digit and use only letters, digits, underscores,
dots, or hyphens. 
- Omit it or use `job_name: ''` to retain the default
`latentppi-<output-directory-hash>-<stage>` names. Job tracking uses recorded
Slurm IDs, so names do not need to be unique. Use distinct prefixes if you want
to distinguish screens in the queue. Names are captured when a run is submitted;
editing the YAML does not rename existing jobs or change their saved settings.

`user_queue_budget` accepts a positive integer or YAML `null` (unquoted).
Omitting it also disables the local budget check. This never disables Slurm's
limits or `max_concurrent_tasks`. The reserve is ignored when the budget is null.


Discoverable scheduler limits can reduce `max_array_size`. Each submitted chunk exports its own `OFFSET`; local task N selects command N + OFFSET. A configured nonzero offset intentionally excludes earlier manifest lines from submission.

## How the dispatcher schedules work

The dispatcher submits your screen as a series of Slurm job arrays, **one array at a time**:

1. It submits an array.
2. It waits until every task in that array has left the queue (finished or failed).
3. It submits the next array.
4. After the last array, it submits the collector job.

This has a few consequences:

- **`max_concurrent_tasks` applies to the whole screen**, not to each array. This is true even if quota checks are turned off.
- **Each array waits for its slowest task** before the next one starts.
- **Failed tasks are not retried immediately.** They are recorded and resubmitted the next time you run `latentppi.py run`.

## Configuring queue limits

Most clusters cap how many jobs you can have queued at once. Tell the dispatcher about that cap so it doesn't exceed it.

### Example

For a cluster with a 50-job user limit, where you want four screen tasks running at a time:

```yaml
cluster:
  user_queue_budget: 50      # Your total queue limit, counting ALL your jobs (including unrelated ones).
  reserved_queue_slots: 2    # Keep two slots free for other submissions.
  max_concurrent_tasks: 4    # At most four screen tasks run at once. This is not a batch size.
  poll_interval: 60          # Seconds between queue checks.
  dispatch_mode: chained
  dispatcher_cpus_per_task: 1
  dispatcher_time_limit: '01:00:00'  # Each short invocation, including result collection.
```

### How available capacity is calculated

Before submitting each array, the dispatcher counts **all** of your queued tasks on `screen_cluster`. That count includes unrelated jobs and each pending array task individually. It then computes:

```
available = user_queue_budget - queued_tasks - reserved_queue_slots - 1  # continuation slot
```

| Budget | Reserve | Already queued | Available for new tasks |
|---|---|---|---|
| 50 | 2 | 10 | **37** |
| 20 | 1 | 1 (just the dispatcher) | **17** (with `max_concurrent_tasks: 4`, four run at once) |
| 3 | 1 | 1 (just the dispatcher) | **0** (no array slot, so validation rejects it) |

Notes:

- The current dispatcher is already counted under "queued." Chained mode reserves one additional slot for its successor; `reserved_queue_slots` remains extra headroom. Persistent mode does not subtract the extra continuation slot.
- The count is cluster-wide. It does **not** account for partition-, account-, or QOS-specific limits.
- Set `user_queue_budget: null` to turn off the budget check. The `max_concurrent_tasks` limit still applies.

### Validation checks

When you start a run, the config is checked for these problems:

- **Error:** the budget can't fit even one task after subtracting the reserve, the dispatcher and (in chained mode) its continuation. This applies when both stages use the same cluster, including when both use the default.
- **Warning:** the budget or the cluster's array index limit makes your `max_concurrent_tasks` value unreachable.
- **Warning:** only one of the two cluster names is set explicitly, so the default might be the same cluster.

These checks can't predict other jobs you submit later, and they don't guarantee the scheduler will have capacity.

### When Slurm rejects a submission

Set the budget to your site's actual limit. Slurm has the final say if other submissions happen at the same time as the dispatcher's check.

| Situation | Dispatcher behavior |
|---|---|
| Explicit submit-limit rejection | Chained: schedules a delayed continuation and exits; persistent: waits one poll interval and retries |
| Ambiguous submission response | Stops, to avoid submitting duplicates |
| Queue query fails | Stops |
| Generic accounting-policy error | Not retried automatically |

## Running and resuming

### What `run` does

`python latentppi.py run config.yaml` submits a CPU dispatcher even if inputs are
already prepared. The default `dispatch_mode: chained` submits one array and a
continuation with `afterany` dependencies on that array and the current dispatcher,
then exits. Depending on both jobs prevents the successor racing the submission
lock. Slurm releases the successor after success or failure; no CPU is allocated
to it while waiting on dependencies. CPU queue latency can add gaps between batches.

The durable `latentppi_runs/<run>/dispatch_state.json` stores remaining indices,
initial/retry phase, step number and pending handoffs. Only the matching step can
advance it. `--force` is applied at initialization, not repeatedly by every successor.
Initial GPU tiers and the one-time retry ledger retain their previous semantics.

A full queue or explicit array quota rejection schedules a continuation no earlier
than `poll_interval` seconds (rounded up to whole minutes) and exits. If even a
continuation cannot be submitted, dispatch stops with a manual-resume instruction;
accepted GPU jobs keep running. Repeated quota deferrals do not rescan results.
The final continuation schedules the collector. Ordinary updates happen at batch
boundaries; `status --heavy` still provides an on-demand refresh.

Chained mode requires identical `screen_cluster` and `inputs_cluster` settings
(both empty is valid). CPU and GPU **partitions** may differ. Use
`dispatch_mode: persistent` for separate Slurm clusters or the old polling behavior.
Old snapshots without a dispatch-mode field are loaded as persistent, retaining
their old dispatcher CPU count and wall time. Existing jobs/configs are not changed.
You do not need to keep a terminal or tmux session open in either mode.

Requirements:

- Compute nodes must be able to run `sbatch` and `squeue`.
- The initial preparation and dispatcher jobs need available quota on the CPU cluster.

### If the dispatcher is interrupted or hits its time limit

Arrays that were already accepted keep running. Work that was never submitted stays on disk. To resume:

1. Run `status` to see what happened.
2. Wait for recorded jobs to finish, or cancel them deliberately.
3. Rerun `python latentppi.py run config.yaml`.

Completed pairs are skipped. Failed and never-run pairs are submitted in a new run snapshot. New attempts receive unique IDs so old predictions and structure scores are not mixed with new results.

A saved `screen_identity.json` rejects changed scientific settings or checkpoint/bait contents in the same output directory. Use a new `inputs.output_base` for a different experiment; `--force` only repeats compatible work. Resource paths are checked, but every proteome MSA and AF3 image/weight file is not content-hashed. Preserve those sources during a screen.

Things to know about resuming:

- **Normal handoffs are automatic in chained mode.** Manual resume is needed after an interrupted or rejected handoff. Persistent mode does not renew itself. Dispatchers use `--no-requeue`; the saved step prevents accidental replay of an already handed-off invocation.
- **Collection may not have run.** If the dispatcher was killed abruptly, the collector may never have been submitted. `status` refreshes the metrics if passed with the `--heavy` flag (if you prefer not to compute metrics (for a faster status check), run without any flags or with `--light` ), and the next `run` schedules collection again.
- **Don't use `--force`** for an ordinary resume.
- **Check the queue before retrying an ambiguous submission.** Submission records are kept under `latentppi_runs/`. An unresolved pending handoff without a known job ID blocks a new `run`, because the job may exist despite a lost response. After inspecting/cancelling uncertain jobs, an operator can clear that checkpoint's `pending` marker and resume; do not clear it while an unknown job might still run. A successfully scheduled continuation can acknowledge its own handoff automatically.

## Using separate CPU and GPU clusters

To send preparation to one cluster and the screen to another, set both cluster names along with the partitions on each:

```yaml
cluster:
  dispatch_mode: persistent
  dispatcher_time_limit: '7-00:00:00'
  screen_cluster: gpu_site
  inputs_cluster: cpu_site
```

- **Set both names explicitly** when routing across clusters. An empty name leaves out `--clusters` and uses Slurm's default.
- These options are passed straight to Slurm; your shell scripts are not rewritten. Partition validation, GPU array-limit discovery, and job status queries all use the clusters you select.
- GPU constraints already in your runner script headers still apply.

Requirements for multi-cluster setups:

- Both clusters must see the same code checkout, environments, resources, and output paths.
- The clusters must support cross-cluster job dependencies, which normally means a configured Slurm federation. Setting cluster names does not set this up for you. See the [Slurm dependency documentation](https://slurm.schedmd.com/sbatch.html#OPT_dependency).

## Validate and inspect the resolved config

```bash
python latentppi.py validate config.yaml
python latentppi.py render config.yaml
```

Neither command submits jobs, prepares inputs, or runs inference. Validation reports errors and warnings together, creates `slurm_logs/`, and returns exit code 1 on errors. Errors prevent rendering.

Validation checks FASTA content, files and directories, writable outputs, conda environments, partitions and normal/fallback GPU inventory via `sinfo`, and MSA paths in up to two generated AF3 JSONs. Missing command files and input JSON directories are expected before preparation; other required resources must exist. Missing `conda` or `sinfo` produces warnings; failed checks when present are errors. Host checks cannot establish compute-node visibility.

Checkpoint validation checks deserialization, not every container's inference compatibility. It tries host PyTorch, the AF3 image, then the configured conda environment, without installing dependencies. Loading uses `weights_only=True` with an allowlist for repository NumPy normalization arrays; older PyTorch uses a restricted Unpickler. There is no unrestricted pickle fallback. Unloadable checkpoints are reported by metric name.

### Generated artifacts

`render` writes `output_base/latentppi_config/`:

| Artifact | Contents |
| --- | --- |
| `env.sh` | Bash-safe exported settings |
| `sbatch.env` | NUL-delimited environment snapshot for inspection |
| `scorer_manifest.json`, `checkpoints/` | Selected models and copied checkpoints under stable hashed filenames, unless using an existing manifest |
| `resolved_config.json` | Resolved defaults, environment, commands, validation counts, git state, and script/checkpoint hashes |

Submissions pass the recorded environment directly to the `sbatch` process with `--export=ALL`, so clients without `--export-file` are supported. Printed commands use `env -i` with shell-quoted assignments for the same behavior. Exports preserve PATH and basic runtime variables while excluding unrecorded inherited overrides. The scorer manifest is forwarded into the container through `LATENTPPI_MODEL_MANIFEST` and the Singularity/Apptainer environment variables; `AF3_SCORER_FILENAME` also identifies the rank-by checkpoint for legacy checks.

`render` uses illustrative placeholder job IDs. For the current submission plan, use `run --no-submit` as shown in the [walkthrough](examples/toy_screen/WALKTHROUGH.md#2-copy-the-target-msa-and-preview). Actual runs and previews keep isolated settings and checkpoint copies under `latentppi_runs/`; checkout scripts and manually supplied manifests remain external references.

# Historical screen and dataset tools

Use `python latentppi.py run config.yaml` for the supported screen workflow.

These scripts are retained for reproducing earlier runs. The standalone command
generator reads a multi-tool manifest. Its `--skip-lengths` option excludes
lengths **at or above** the threshold when a ColabFold input supplies a length.
The YAML dispatcher's `cluster.max_total_residues` allows the specified maximum.

The old Slurm launcher uses its own command file, array offset, and scorer
settings. Set `AF3_SCRIPT_DIR` to the repository root and supply the resources
for the original run. The dataset scripts assume the original dataset working
directory, local tools, and input files.

# LatentPPI checkpoints

> **These files are derived from AlphaFold 3 Output. They are provided under and
> subject to the AlphaFold 3 Output Terms of Use** (`OUTPUT_TERMS_OF_USE.md` in this
> folder). See also `Legally_Binding_Terms_of_Use.txt`. The Output Terms are the
> binding terms for these files; nothing in this README changes or adds to them.

## Folder contents

| File | What it is |
|---|---|
| `OUTPUT_TERMS_OF_USE.md` | Unmodified copy of the AlphaFold 3 Output Terms of Use |
| `Legally_Binding_Terms_of_Use.txt` | Notice required by item 5.1 of those terms |
| `[r1_r1_checkpoint].pt` | Gate scorer used by all modes (`coarse`, `gated`, `full`) |

## What these checkpoints are

Small MLPs that score a candidate protein pair from AlphaFold 3's trunk
representation after one recycle. LatentPPI uses the score to rank pairs and
decide which ones get full AlphaFold 3 inference.

Each `.pt` file stores `input_recycle` and `target_recycle` metadata. The
LatentPPI loader refuses checkpoints that are missing either field, and refuses
gated mode unless `input_recycle` is 1.

## Modifications notice

These checkpoints are **not** AlphaFold 3 Output as AlphaFold 3 produces it.
They were made from AlphaFold 3 Output as follows:

1. AlphaFold 3 v3.0.3, with LatentPPI modifications (git tag `ms-v303`), was
   run on protein pairs from [training set: e.g. HINT positives and
   Negatome / compartment-exclusion / degree-matched / complex-decoy negatives].
2. The pair representation after recycle 1 was extracted, and the inter-chain
   block was pooled [pooling method] with chains in accession-sorted order.
3. An MLP was trained on these pooled embeddings to predict AlphaFold 3's own
   ipTM at recycle [1 for the gate checkpoint / 10 for the baseline].

The checkpoints contain only the MLP's learned weights. They do not contain
AlphaFold 3 model parameters, structure predictions or embeddings.

## AlphaFold 3 model parameters are required and not included

Running LatentPPI requires the AlphaFold 3 model parameters. They are **not**
included in this repository or in the LatentPPI container image. Request access
through the process at https://github.com/google-deepmind/alphafold3; you may
only use the parameters if you receive them directly from Google, under the
AlphaFold 3 Model Parameters Terms of Use.

## Affiliation

LatentPPI is not affiliated with, or endorsed by, Google DeepMind or Isomorphic
Labs.

## Citation

If you use these checkpoints, please cite AlphaFold 3 (required by its terms) and
LatentPPI:

- Abramson, J. et al. Accurate structure prediction of biomolecular interactions
  with AlphaFold 3. *Nature* (2024). https://doi.org/10.1038/s41586-024-07487-w
- [LatentPPI citation]

---

## Snippet for the top-level README / LICENSE (not part of this file)

> The source code in this repository is released under the [Apache-2.0]
> license. Files in `models/` are **excluded** from that license. They are
> derived from AlphaFold 3 Output and are governed solely by the AlphaFold 3
> Output Terms of Use; see `models/checkpoint/CHECKPOINTS_README.md`.
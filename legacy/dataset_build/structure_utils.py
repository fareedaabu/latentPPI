"""
structure_utils.py

Structural (geometry-only) utilities that are independent of AF3's own
confidence metrics. Intended to be imported by add_ipsae.py and/or the
seed-pilot analysis once per-seed CIFs exist -- not wired into
add_ipsae.py yet since this run doesn't have per-seed structures to
compute contacts from.

Dependency: pip install biopython --break-system-packages
"""

import logging
from pathlib import Path

from Bio.PDB import MMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch

logger = logging.getLogger(__name__)


def _get_contact_atom(residue):
    """Return CB atom if present, otherwise CA (e.g. for glycine)."""
    if "CB" in residue:
        return residue["CB"]
    if "CA" in residue:
        return residue["CA"]
    return None


def _select_chains_by_length(model, len_a, len_b, tol=2):
    """
    Try to identify chain A and chain B by matching residue counts to
    len_a/len_b, rather than trusting CIF file order. Returns
    (chain_a, chain_b) if exactly one chain matches each length within
    `tol` residues and they're different chains; otherwise None, in
    which case the caller falls back to file order.
    """
    if len_a is None or len_b is None:
        return None

    candidates = [
        (c, len([r for r in c if r.id[0] == " "]))
        for c in model.get_chains()
    ]
    candidates = [(c, n) for c, n in candidates if n > 0]
    if len(candidates) < 2:
        return None

    a_matches = [c for c, n in candidates if abs(n - len_a) <= tol]
    b_matches = [c for c, n in candidates if abs(n - len_b) <= tol]

    if len(a_matches) == 1 and len(b_matches) == 1 and a_matches[0].id != b_matches[0].id:
        return a_matches[0], b_matches[0]
    return None


def compute_interchain_contacts(
    cif_path,
    len_a=None,
    len_b=None,
    cutoff=8.0,
    chain_a_id=None,
    chain_b_id=None,
):
    """
    Compute distance-based interchain contacts between two chains in a
    predicted structure.

    A contact is a residue pair (one per chain) whose CB atoms (CA for
    glycine) are within `cutoff` angstroms of each other. Purely
    geometric -- does not depend on any AF3 confidence score, so it can
    sanity-check confidence metrics against actual predicted geometry,
    or check consistency across seeds for the same pair.

    Parameters
    ----------
    cif_path : str or Path
        Path to the AF3-predicted structure (.cif).
    len_a, len_b : int, optional
        Chain lengths from dataset_summary.tsv, used only as a
        sanity-check assertion against the parsed structure (logs a
        warning on mismatch, does not raise).
    cutoff : float, default 8.0
        Distance cutoff in angstroms for a residue pair to count as a
        contact.
    chain_a_id, chain_b_id : str, optional
        Explicit chain IDs to use. If not provided, assumes the first
        two chains in the structure (file order) are chain A and B.

    Returns
    -------
    dict:
        n_contacts : int -- number of residue-pair contacts under cutoff
        min_distance : float or None -- closest interchain CB(CA) distance
        chain_a_id, chain_b_id : str -- chain IDs actually used
        chain_a_len, chain_b_len : int -- parsed residue counts, for
            cross-checking against len_a/len_b
    """
    cif_path = Path(cif_path)
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    model = next(structure.get_models())

    chains = list(model.get_chains())
    if len(chains) < 2:
        raise ValueError(f"Expected >=2 chains in {cif_path}, found {len(chains)}")

    if chain_a_id is not None and chain_b_id is not None:
        chain_a = model[chain_a_id]
        chain_b = model[chain_b_id]
    else:
        matched = _select_chains_by_length(model, len_a, len_b)
        if matched is not None:
            chain_a, chain_b = matched
            logger.info(
                "%s: matched chains by length to len_a/len_b -> %s (A), %s (B)",
                cif_path.name, chain_a.id, chain_b.id,
            )
        else:
            chain_a, chain_b = chains[0], chains[1]
            logger.info(
                "%s: could not disambiguate by length, using first two chains "
                "in file order: %s (A), %s (B)",
                cif_path.name, chain_a.id, chain_b.id,
            )

    residues_a = [r for r in chain_a if r.id[0] == " "]  # exclude waters/hetero
    residues_b = [r for r in chain_b if r.id[0] == " "]

    if len_a is not None and len(residues_a) != len_a:
        logger.warning(
            "%s: chain %s length mismatch -- parsed %d residues, expected len_a=%d",
            cif_path.name, chain_a.id, len(residues_a), len_a,
        )
    if len_b is not None and len(residues_b) != len_b:
        logger.warning(
            "%s: chain %s length mismatch -- parsed %d residues, expected len_b=%d",
            cif_path.name, chain_b.id, len(residues_b), len_b,
        )

    atoms_a = [a for a in (_get_contact_atom(r) for r in residues_a) if a is not None]
    atoms_b = [a for a in (_get_contact_atom(r) for r in residues_b) if a is not None]

    if not atoms_a or not atoms_b:
        logger.warning(
            "%s: no usable CB/CA atoms found in one or both chains", cif_path.name
        )
        return {
            "n_contacts": 0,
            "min_distance": None,
            "chain_a_id": chain_a.id,
            "chain_b_id": chain_b.id,
            "chain_a_len": len(residues_a),
            "chain_b_len": len(residues_b),
        }

    ns = NeighborSearch(atoms_a + atoms_b)
    atoms_a_set = set(atoms_a)

    n_contacts = 0
    min_distance = None
    for atom_b in atoms_b:
        for atom in ns.search(atom_b.coord, cutoff):
            if atom in atoms_a_set:
                n_contacts += 1
                dist = atom_b - atom
                if min_distance is None or dist < min_distance:
                    min_distance = dist

    logger.info(
        "%s: %d interchain contacts (cutoff=%.1f A), min_distance=%s",
        cif_path.name, n_contacts, cutoff,
        f"{min_distance:.2f}" if min_distance is not None else "None",
    )

    return {
        "n_contacts": n_contacts,
        "min_distance": min_distance,
        "chain_a_id": chain_a.id,
        "chain_b_id": chain_b.id,
        "chain_a_len": len(residues_a),
        "chain_b_len": len(residues_b),
    }
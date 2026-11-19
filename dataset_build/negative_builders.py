"""
negative_builders.py
=====================
Constructs PPI negative pairs from multiple independent evidence sources,
each tagged with a `neg_source` field so downstream analysis (JS divergence,
AUROC, etc.) can be sliced per-tier rather than treated as one homogeneous
negative set. This was the core problem with the original build_negatives():
it only produced Negatome-protein x positive-protein crosses, and never
actually carried the ~2,000 validated Negatome PAIRS through as pairs.

Tiers implemented here, roughly easy -> hard negatives:
  1. negatome_direct        — validated Negatome pairs, used as-is.
                               Smallest, highest-confidence tier.
  2. compartment_exclusion   — pairs of proteins with disjoint annotated
                               subcellular compartments. Bulk volume.
                               KNOWN CAVEAT: can bias a classifier toward
                               learning co-localization rather than true
                               interaction propensity (Ben-Hur & Noble 2006;
                               Negatome paper). Keep tagged separately and
                               don't treat its separation numbers as
                               validating your metrics on their own.
  3. degree_matched_cross    — Negatome-protein x positive-protein crosses,
                               sampled so the partner protein's frequency
                               reflects the *positive set's* degree
                               distribution rather than reusing whichever
                               positive-set proteins happen to get drawn.
                               Reduces (does not eliminate) the hub-protein
                               bias the original implementation had.
  4. complex_decoy           — pairs of proteins that co-occur in a known
                               multi-subunit complex (PDB structure) but do
                               NOT make direct structural contact
                               (Predictomes-style hard negative).
                               KNOWN CAVEAT: Predictomes themselves found
                               ~24-30% of pairs built this way were later
                               shown to be contact-positive on closer
                               inspection. Treat as "hard, imperfectly
                               clean" negatives, not gold standard.

Design notes:
  - None of these builders fetch sequences. Callers should union the
    uniprot IDs across all tiers and run them through the existing
    fetch_sequences_with_cache() / filter_by_length() in
    curate_ppi_dataset.py, so sequence-fetching stays in one place.
  - `known_pairs` should be the FULL HINT pair set (hint_set), not just
    pairs that survived the positive-set length filter -- the original
    script built `known` from filtered positives only, which meant HINT
    pairs dropped for being too long could theoretically leak into the
    negative set. Pass the unfiltered hint_set everywhere here.
  - All samplers use a single seeded random.Random instance for
    reproducibility; avoid mixing in numpy's RNG within these functions.
"""

import time
import json
import random
import logging
from pathlib import Path
from collections import Counter

import requests

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier 1: direct Negatome pairs
# ---------------------------------------------------------------------------

def build_negatome_direct(negatome_pairs, known_pairs):
    """
    Use validated Negatome pairs as-is, carrying the pairing through instead
    of collapsing it into a flat protein pool (the bug in the original
    build_negatives -- it built `negatome_proteins` as a set of individual
    IDs and never referenced `negatome_pairs` again after that).

    Args:
        negatome_pairs: list of (uniprot_a, uniprot_b) tuples from load_negatome()
        known_pairs: set of sorted-tuple pairs to exclude (pass the FULL
                     hint_set, not just filtered positives)

    Returns list of dicts: {uniprot_a, uniprot_b, neg_source}
    """
    log.info("Tier 1: direct Negatome pairs...")
    negatives = []
    seen = set()
    n_dropped = 0

    for a, b in negatome_pairs:
        key = tuple(sorted([a, b]))
        if key in known_pairs or key in seen:
            n_dropped += 1
            continue
        seen.add(key)
        negatives.append({
            "uniprot_a": key[0],
            "uniprot_b": key[1],
            "neg_source": "negatome_direct",
        })

    log.info(f"  {len(negatives)} kept, {n_dropped} dropped (HINT overlap / duplicate)")
    return negatives


# ---------------------------------------------------------------------------
# Tier 2: subcellular compartment exclusion
# ---------------------------------------------------------------------------

# Deliberately coarse, keyword-based compartment set. A GO-similarity
# threshold (Ben-Hur & Noble 2006) would be more rigorous but requires a
# full GO DAG traversal; this favors simplicity and is easy to sanity-check
# by eye. Extend/edit this dict if you want finer categories.
COMPARTMENT_KEYWORDS = {
    "nucleus":                ["nucleus", "nucleolus", "nuclear"],
    "mitochondrion":          ["mitochondri"],
    "extracellular_secreted": ["secreted", "extracellular"],
    "plasma_membrane":        ["cell membrane", "plasma membrane"],
    "endoplasmic_reticulum":  ["endoplasmic reticulum"],
    "golgi":                  ["golgi"],
    "peroxisome":             ["peroxisome"],
    "lysosome":               ["lysosome"],
    "cytoplasm":               ["cytoplasm", "cytosol"],
}


def fetch_subcellular_location(uid: str) -> str:
    """
    Fetch UniProt's free-text SUBCELLULAR LOCATION comment for one accession.
    Returns the concatenated, lowercased location text, or "" on failure.

    CAVEAT: this JSON shape (comments[].commentType == "SUBCELLULAR LOCATION",
    subcellularLocations[].location.value) matches UniProt's documented
    comment structure, but has not been executed against a live response in
    this environment. Spot-check the first few results before a full run.
    """
    try:
        url = (
            f"https://rest.uniprot.org/uniprotkb/{uid}.json"
            f"?fields=cc_subcellular_location"
        )
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return ""
        data = r.json()
        texts = []
        for comment in data.get("comments", []):
            if comment.get("commentType") != "SUBCELLULAR LOCATION":
                continue
            for loc_group in comment.get("subcellularLocations", []):
                loc = loc_group.get("location", {}).get("value", "")
                if loc:
                    texts.append(loc.lower())
        return "; ".join(texts)
    except Exception as e:
        log.debug(f"Subcellular location fetch error {uid}: {e}")
        return ""


def fetch_subcellular_locations_with_cache(uniprot_ids, cache_path: Path,
                                            refresh: bool = False) -> dict:
    """
    Same caching pattern as fetch_sequences_with_cache in curate_ppi_dataset.py:
    loads existing cache, fetches only missing/previously-empty IDs.
    """
    cache = {}
    if cache_path.exists() and not refresh:
        with open(cache_path) as f:
            cache = json.load(f)
        filled = sum(1 for v in cache.values() if v)
        log.info(f"  Loaded location cache: {filled} filled, {len(cache) - filled} empty")

    missing = [uid for uid in uniprot_ids if not cache.get(uid)]
    if missing:
        log.info(f"  Fetching {len(missing)} subcellular locations from UniProt...")
        for uid in missing:
            cache[uid] = fetch_subcellular_location(uid)
            time.sleep(0.05)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        log.info(f"  Location cache updated: {len(cache)} total entries")
    else:
        log.info("  All subcellular locations already cached")

    return cache


def classify_compartment(location_text: str) -> set:
    """Return the set of COMPARTMENT_KEYWORDS categories matched in the text."""
    hits = set()
    for category, keywords in COMPARTMENT_KEYWORDS.items():
        if any(kw in location_text for kw in keywords):
            hits.add(category)
    return hits


def build_compartment_exclusion_negatives(candidate_proteins, location_cache,
                                           known_pairs, n_target, seed: int = 42):
    """
    Sample negative pairs from proteins with exactly ONE unambiguous
    compartment annotation, requiring the two proteins' compartments to
    differ. Proteins with zero or multiple compartment hits (multi-
    localized) are dropped from the pool entirely -- they're exactly the
    ambiguous case this coarse method can't safely adjudicate.

    Args:
        candidate_proteins: iterable of uniprot IDs to consider (e.g. the
                             union of your existing sequence-cache proteins)
        location_cache: dict from fetch_subcellular_locations_with_cache()
        known_pairs: full hint_set, sorted-tuple pairs to exclude
        n_target: how many negatives to build
    """
    log.info("Tier 2: subcellular compartment-exclusion negatives...")

    protein_to_compartment = {}
    for p in candidate_proteins:
        cats = classify_compartment(location_cache.get(p, ""))
        if len(cats) == 1:
            protein_to_compartment[p] = next(iter(cats))

    by_compartment = {}
    for p, cat in protein_to_compartment.items():
        by_compartment.setdefault(cat, []).append(p)

    categories = list(by_compartment.keys())
    log.info(
        "  Unambiguous proteins by compartment: "
        + ", ".join(f"{c}={len(by_compartment[c])}" for c in categories)
    )

    if len(categories) < 2:
        log.warning("  Fewer than 2 compartment categories with data -- skipping tier")
        return []

    rng = random.Random(seed)
    negatives = []
    seen = set()
    attempts, max_attempts = 0, n_target * 30

    while len(negatives) < n_target and attempts < max_attempts:
        attempts += 1
        cat_a, cat_b = rng.sample(categories, 2)
        p_a = rng.choice(by_compartment[cat_a])
        p_b = rng.choice(by_compartment[cat_b])
        key = tuple(sorted([p_a, p_b]))
        if key in known_pairs or key in seen:
            continue
        seen.add(key)
        negatives.append({
            "uniprot_a": key[0],
            "uniprot_b": key[1],
            "neg_source": "compartment_exclusion",
            "compartment_a": cat_a,
            "compartment_b": cat_b,
        })

    log.info(f"  {len(negatives)} built ({attempts} attempts)")
    return negatives


# ---------------------------------------------------------------------------
# Tier 3: degree-matched Negatome x positive crosses
# ---------------------------------------------------------------------------

def build_degree_matched_cross(negatome_pairs, positive_pairs, known_pairs,
                                n_target, seed: int = 42):
    """
    Cross Negatome proteins with positive-set proteins, but draw the
    positive-set partner with probability proportional to its degree in the
    positive set (how many positive pairs it appears in), rather than
    uniformly -- which is what the original implementation effectively did
    by uniformly sampling from the full positive-protein pool. Uniform
    sampling over a hub-heavy pool still overrepresents hubs relative to a
    matched comparison; weighting by degree makes the *negative* side's hub
    exposure explicitly track the *positive* side's, which is the
    comparison you actually want when checking whether AF3 metrics respond
    to interaction status rather than protein "promiscuity."

    This does NOT fully solve the false-negative contamination problem
    (pairs absent from HINT may still genuinely interact) -- it only
    addresses the degree-bias half of the issue. Degree-preserving edge-swap
    rewiring of the HINT graph itself was deliberately avoided here, since
    that approach is documented to frequently reintroduce true edges in PPI
    networks (which are enriched in 4-cycles).
    """
    log.info("Tier 3: degree-matched Negatome x positive crosses...")

    negatome_proteins = sorted({p for a, b in negatome_pairs for p in (a, b)})
    if not negatome_proteins:
        log.warning("  No Negatome proteins available -- skipping tier")
        return []

    pos_degree = Counter()
    for p in positive_pairs:
        pos_degree[p["uniprot_a"]] += 1
        pos_degree[p["uniprot_b"]] += 1

    pos_proteins = list(pos_degree.keys())
    cumulative, total = [], 0.0
    for p in pos_proteins:
        total += pos_degree[p]
        cumulative.append(total)

    rng = random.Random(seed)
    negatives = []
    seen = set()
    attempts, max_attempts = 0, n_target * 30

    while len(negatives) < n_target and attempts < max_attempts:
        attempts += 1
        p_a = rng.choice(negatome_proteins)
        p_b = rng.choices(pos_proteins, cum_weights=cumulative, k=1)[0]
        if p_a == p_b:
            continue
        key = tuple(sorted([p_a, p_b]))
        if key in known_pairs or key in seen:
            continue
        seen.add(key)
        negatives.append({
            "uniprot_a": key[0],
            "uniprot_b": key[1],
            "neg_source": "degree_matched_cross",
        })

    log.info(f"  {len(negatives)} built ({attempts} attempts)")
    return negatives


# ---------------------------------------------------------------------------
# Tier 4: complex decoys (Predictomes-style: same complex, no direct contact)
# ---------------------------------------------------------------------------

def download_pdb_structure(pdb_id: str, cache_dir: Path) -> Path:
    """Download an mmCIF structure file, caching to disk (mirrors download_file())."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{pdb_id}.cif"
    if dest.exists():
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def get_chain_uniprot_map(pdb_id: str) -> dict:
    """
    Map auth_asym_id (author chain ID -- matches Bio.PDB's chain.id when
    parsed from mmCIF with MMCIFParser) -> UniProt accession, keeping only
    chains with EXACTLY ONE mapped accession. Chains with zero or multiple
    accessions (chimeras, unmapped entities, engineered constructs) are
    skipped rather than guessed at.

    CAVEAT: field paths (rcsb_entry_container_identifiers.polymer_entity_ids,
    rcsb_polymer_entity_container_identifiers.{uniprot_ids,auth_asym_ids})
    follow RCSB's documented container-object pattern but haven't been
    verified against a live response here. Print/inspect `ent` for the first
    PDB ID you run before trusting this at scale.
    """
    entry = requests.get(
        f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", timeout=20
    ).json()
    entity_ids = (
        entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
        or []
    )

    chain_map = {}
    for eid in entity_ids:
        ent = requests.get(
            f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{eid}",
            timeout=20,
        ).json()
        ids = ent.get("rcsb_polymer_entity_container_identifiers", {}) or {}
        uniprot_ids = ids.get("uniprot_ids") or []
        auth_asym_ids = ids.get("auth_asym_ids") or []
        if len(uniprot_ids) == 1:
            for chain in auth_asym_ids:
                chain_map[chain] = uniprot_ids[0]
        time.sleep(0.05)

    return chain_map


def _heavy_atoms(chain):
    """
    Heavy (non-hydrogen) atoms from standard ATOM records only.
    residue.id[0] != " " marks HETATM records (waters, ligands, ions) in
    Bio.PDB's residue id convention -- excluded here.
    """
    atoms = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        for atom in residue:
            if atom.element != "H":
                atoms.append(atom)
    return atoms


def chains_in_contact(model, chain_id_a: str, chain_id_b: str,
                       cutoff: float = 5.0, min_residue_pairs: int = 10):
    """
    Predictomes-style contact definition: True if >= min_residue_pairs
    distinct residue pairs have any heavy atoms within cutoff Angstroms.
    Returns None if either chain is missing from this model (rather than
    False, so callers can distinguish "tested, no contact" from "couldn't test").
    """
    from Bio.PDB import NeighborSearch  # lazy import: keep Bio.PDB optional

    if chain_id_a not in model or chain_id_b not in model:
        return None

    atoms_b = _heavy_atoms(model[chain_id_b])
    if not atoms_b:
        return None
    ns = NeighborSearch(atoms_b)

    contact_residue_pairs = set()
    for atom in _heavy_atoms(model[chain_id_a]):
        for neighbor in ns.search(atom.coord, cutoff):
            contact_residue_pairs.add((atom.get_parent().id, neighbor.get_parent().id))

    return len(contact_residue_pairs) >= min_residue_pairs


def build_complex_decoy_negatives(complex_pdb_ids, known_pairs,
                                   cache_dir: Path = Path("pdb_cache"),
                                   contact_cutoff: float = 5.0,
                                   min_residue_pairs: int = 10):
    """
    For each PDB entry in complex_pdb_ids: map chains to UniProt accessions,
    test every chain pair for direct structural contact, and keep pairs that
    (a) never make contact across any chain-instance combination and
    (b) aren't already a known HINT interaction, as decoy negatives.

    If a protein has multiple copies (chains) in the assembly, a pair is
    only kept as a decoy if NONE of the instance-level combinations show
    contact -- one contacting copy disqualifies the pair.

    KNOWN CAVEAT: Predictomes reported that even with this exact structural
    criterion, roughly 24-30% of "no contact" pairs later turned out to be
    contact-positive on re-evaluation (flexible interfaces, alternate
    conformations, structures disagreeing). Treat this tier as hard,
    imperfectly clean negatives -- worth its own neg_source tag so you can
    check whether that contamination rate shows up in your separation
    numbers too.
    """
    from Bio.PDB import MMCIFParser  # lazy import: keep Bio.PDB optional

    log.info(f"Tier 4: complex-decoy negatives from {len(complex_pdb_ids)} PDB entries...")
    parser = MMCIFParser(QUIET=True)
    negatives = []
    seen = set()

    for pdb_id in complex_pdb_ids:
        try:
            cif_path = download_pdb_structure(pdb_id, cache_dir)
            chain_map = get_chain_uniprot_map(pdb_id)
            if len(chain_map) < 2:
                log.warning(f"  {pdb_id}: fewer than 2 unambiguously-mapped chains -- skipping")
                continue

            structure = parser.get_structure(pdb_id, cif_path)
            model = structure[0]
            chains = list(chain_map.keys())

            pair_contact = {}  # sorted-tuple uniprot pair -> bool (any instance contacts?)
            for i in range(len(chains)):
                for j in range(i + 1, len(chains)):
                    ca, cb = chains[i], chains[j]
                    ua, ub = chain_map[ca], chain_map[cb]
                    if ua == ub:
                        continue
                    key = tuple(sorted([ua, ub]))
                    contact = chains_in_contact(model, ca, cb, contact_cutoff, min_residue_pairs)
                    if contact is None:
                        continue
                    pair_contact[key] = pair_contact.get(key, False) or contact

            n_new = 0
            for key, contact in pair_contact.items():
                if contact or key in known_pairs or key in seen:
                    continue
                seen.add(key)
                negatives.append({
                    "uniprot_a": key[0],
                    "uniprot_b": key[1],
                    "neg_source": "complex_decoy",
                    "source_pdb": pdb_id,
                })
                n_new += 1

            log.info(f"  {pdb_id}: {len(pair_contact)} chain pairs tested, {n_new} new decoys")

        except Exception as e:
            log.warning(f"  {pdb_id}: failed ({e}) -- skipping")
            continue

    log.info(f"  Total complex-decoy negatives: {len(negatives)}")
    return negatives


def find_candidate_complex_pdb_ids(min_polymer_entities: int = 5,
                                    organism: str = "Homo sapiens",
                                    max_results: int = 50):
    """
    Programmatically find candidate multi-subunit complex structures via the
    RCSB search API, instead of relying on a hardcoded PDB ID list. Requires
    the `rcsbapi` package (already in your toolkit per your interpretability
    pipeline's data sourcing).

    Filters for human entries with at least `min_polymer_entities` distinct
    polymer entities (i.e. genuinely multi-subunit, not just multiple copies
    of one chain) -- a reasonable proxy for "well-characterized multi-subunit
    complex" without hand-picking examples like the proteasome/ribosome.
    """
    from rcsbapi.search import AttributeQuery, search_attributes as attrs

    q_organism = attrs.rcsb_entity_source_organism.taxonomy_lineage.name == organism
    q_entity_count = AttributeQuery(
        attribute="rcsb_entry_info.polymer_entity_count_protein",
        operator="greater_or_equal",
        value=min_polymer_entities,
    )
    query = q_organism & q_entity_count

    results = list(query(return_type="entry"))
    log.info(f"  Found {len(results)} candidate multi-subunit human complexes")
    return results[:max_results]


# A tiny starter list for a first smoke-test run before switching to
# find_candidate_complex_pdb_ids(). Verify each ID resolves to what you
# expect (e.g. via rcsb.org/structure/{id}) before trusting it -- these are
# recalled from general knowledge, not verified against a live query here.
DEFAULT_COMPLEX_PDB_IDS_EXAMPLE = [
    "5GJQ",  # human 26S proteasome -- verify before use
]
import json
from pathlib import Path


def read_fasta_sequences(path: Path) -> set[str]:
    if not path.exists():
        return set()

    sequences = set()
    current = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.add("".join(current))
                    current = []
                continue
            current.append(line)

    if current:
        sequences.add("".join(current))

    return sequences

seen = {}
existing_sequences = ""
existing_sequences_2 = ""
# existing_sequences = read_fasta_sequences(Path("ppi_dataset/all_unique.fasta"))
# existing_sequences_2 = read_fasta_sequences(Path("ppi_dataset/all_unique_total.fasta"))
skipped_existing = 0
for fname in ["ppi_dataset/positives.json", "ppi_dataset/negatives.json"]:
    for r in json.load(open(fname)):
        for acc, seq in [(r["uniprot_a"], r["seq_a"]), 
                         (r["uniprot_b"], r["seq_b"])]:
            if acc not in seen:
                seen[acc] = seq

with open("ppi_dataset/all_unique_total.fasta", "w") as f:
    for acc, seq in seen.items():
        if seq in existing_sequences or seq in existing_sequences_2:
            skipped_existing += 1
            continue
        f.write(f">{acc}\n{seq}\n")

print(f"wrote {len(seen) - skipped_existing} unique sequences")
print(f"skipped {skipped_existing} sequences already present in ppi_dataset/all_unique_2 and all_unique.fasta")
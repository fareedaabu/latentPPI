"""Select a query A3M by FASTA identity, never by arbitrary directory order."""
import argparse
from pathlib import Path


def first_sequence(path):
    sequence, seen = [], False
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if seen:
                break
            seen = True
        elif seen:
            sequence.append(line)
        else:
            raise ValueError(f"sequence before FASTA header: {path}")
    result = "".join(sequence).replace("-", "").replace(".", "").upper()
    if not result:
        raise ValueError(f"empty query sequence: {path}")
    return result


def discover(fasta, directory, *, generated=False):
    fasta, directory = Path(fasta), Path(directory)
    expected = directory / (fasta.stem + ".a3m")
    candidates = sorted(p for p in directory.glob("*.a3m") if p.is_file())
    if len(candidates) > 1:
        raise ValueError(f"ambiguous query A3Ms in {directory}: {[p.name for p in candidates]}; keep only {expected.name}")
    if not candidates:
        return None
    candidate = candidates[0]
    if not generated and candidate != expected:
        raise ValueError(f"unexpected query A3M {candidate.name}; expected {expected.name} from {fasta.name}")
    if first_sequence(candidate) != first_sequence(fasta):
        raise ValueError(f"query A3M sequence does not match target FASTA: {candidate}")
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--generated", action="store_true")
    parser.add_argument("--expected", action="store_true")
    args = parser.parse_args()
    try:
        result = (args.directory / (args.fasta.stem + ".a3m") if args.expected
                  else discover(args.fasta, args.directory, generated=args.generated))
        print(result if result else "")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

"""
evaluate.py
Computes the BLEU score between the generated output (output.txt)
and the reference translation (reference.txt) using sacrebleu.
"""

import sacrebleu

OUTPUT_FILE = "output.txt"
REFERENCE_FILE = "reference.txt"


def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def main():
    print(f"Loading hypothesis from : {OUTPUT_FILE}")
    hypotheses = load_file(OUTPUT_FILE)

    print(f"Loading references from : {REFERENCE_FILE}")
    references = load_file(REFERENCE_FILE)

    # sacrebleu expects references as a list of lists (one list per reference set)
    assert len(hypotheses) == len(references), (
        f"Mismatch: {len(hypotheses)} hypotheses vs {len(references)} references"
    )

    # Compute corpus BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    print("\n--- BLEU Evaluation Results ---")
    print(f"BLEU Score : {bleu.score:.2f}")
    print(f"Full result: {bleu}")


if __name__ == "__main__":
    main()
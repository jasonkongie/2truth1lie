import json
import argparse
from collections import defaultdict
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Parse CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze LLM Truth/Lie JSONL responses and plot heatmaps")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="responses.jsonl",
        help="Path to the JSONL file containing question and guess entries"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["ALL", "FULL", "CONTROLLED"],
        default="ALL",
        help="Filter guesses by mode before computing heatmap"
    )
    return parser.parse_args()


def load_responses(path):
    responses = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    responses.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(1)
    return responses


def compute_pairwise(responses, mode_filter):
    # Group filtered guess entries by qid
    guesses_by_q = defaultdict(list)
    for e in responses:
        if e.get("type") != "guess":
            continue
        if mode_filter != "ALL" and e.get("mode") != mode_filter:
            continue
        guesses_by_q[e["qid"]].append(e)

    win_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(lambda: defaultdict(int))

    for guesses in guesses_by_q.values():
        models = [g["model"] for g in guesses]
        for A, B in itertools.permutations(models, 2):
            entryA = next(g for g in guesses if g["model"] == A)
            if entryA.get("correct"):
                win_counts[A][B] += 1
            total_counts[A][B] += 1

    return win_counts, total_counts


def save_heatmap(win_counts, total_counts, mode_filter, output_prefix="pairwise_jsonl"):
    models = sorted(set(win_counts.keys()) | set(total_counts.keys()))
    if not models:
        print(f"No guess data found for mode {mode_filter}. Skipping heatmap generation.")
    else:
        matrix = []
        for A in models:
            row = []
            for B in models:
                if A == B:
                    row.append(0.5)
                else:
                    wins = win_counts[A].get(B, 0)
                    tot = total_counts[A].get(B, 0)
                    row.append(wins / tot if tot > 0 else 0.5)
            matrix.append(row)
        df = pd.DataFrame(matrix, index=models, columns=models)
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, cmap="RdBu", center=0.5)
        plt.title(f"Pairwise Win Rates ({mode_filter})")
        plt.tight_layout()
        heatmap_file = f"{output_prefix}_{mode_filter.lower()}_heatmap.png"
        plt.savefig(heatmap_file)
        print(f"Heatmap saved to {heatmap_file}")

    counts = {
        "wins": {A: dict(win_counts[A]) for A in win_counts},
        "totals": {A: dict(total_counts[A]) for A in total_counts},
    }
    counts_file = f"{output_prefix}_{mode_filter.lower()}_counts.json"
    with open(counts_file, "w") as f:
        json.dump(counts, f, indent=2)
    print(f"Counts saved to {counts_file}")


def main():
    args = parse_args()
    responses = load_responses(args.input)
    win_counts, total_counts = compute_pairwise(responses, args.mode)
    save_heatmap(win_counts, total_counts, args.mode, output_prefix="pairwise_jsonl")

if __name__ == "__main__":
    main()

import json
import argparse
from collections import defaultdict
import itertools
import pandas as pd
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", type=str, default="responses_infinity_newmodels.jsonl")
    p.add_argument("--mode", "-m", choices=["ALL", "FULL", "CONTROLLED"], default="ALL")
    p.add_argument("--output-prefix", "-o", type=str, default="pairwise_jsonl")
    return p.parse_args()

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

def save_outputs(win_counts, total_counts, mode_filter, output_prefix):
    models = sorted(set(win_counts.keys()) | set(total_counts.keys()))
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
    if models:
        df = pd.DataFrame(matrix, index=models, columns=models)
        df.to_csv(f"{output_prefix}_{mode_filter.lower()}_matrix.csv")

    counts = {
        "wins": {A: dict(win_counts[A]) for A in win_counts},
        "totals": {A: dict(total_counts[A]) for A in total_counts},
    }
    with open(f"{output_prefix}_{mode_filter.lower()}_counts.json", "w") as f:
        json.dump(counts, f, indent=2)

def main():
    args = parse_args()
    responses = load_responses(args.input)
    win_counts, total_counts = compute_pairwise(responses, args.mode)
    save_outputs(win_counts, total_counts, args.mode, args.output_prefix)

if __name__ == "__main__":
    main()

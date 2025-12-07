import os
import json
import random
import argparse
import re
import uuid
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from ollama import chat

from datasets import load_dataset
truth_ds = load_dataset("wwbrannon/TruthGen", split="train")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_executor = ThreadPoolExecutor(max_workers=4)
SEEN_RESPONSES = set()

MODELS = [
    "tinyllama:latest",
    "llama3.1:latest",
    "vicuna:latest",
    "gemma3n:e2b", 
    "mistral:latest-sys",
    "deepseek-r1:latest",
    "gemma:latest", 
    "yi:9b", 
    "llava:latest",
    "qwen:7b", 
    "llava:latest"
]

TOPICS = [
    "astronomy",
    "music theory",
    "ancient history",
    "computer science",
    "biology",
    "quantum physics",
    "philosophy",
    "economics",
    "art history",
    "environmental science",
    "linguistics",
    "anthropology",
    "psychology",
    "sociology",
    "geology",
    "political science",
    "mathematics",
    "botany",
    "zoology",
    "neuroscience",
    "cryptography",
    "ethics",
    "astrophysics",
    "nanotechnology",
    "genetics",
    "ecology",
    "archaeology",
    "robotics",
    "cybersecurity",
    "meteorology",
    "oceanography",
    "architecture",
]

def raw_game_prompt(topic):
    return (
        "You are an expert at Two Truths and One Lie. "
        "Trick your opponent by making the lie subtle and non-obvious. "
        "Output ONLY a JSON object with:\n"
        '  "statements": list of three strings (two truths, one lie)\n'
        '  "lie_index": 1, 2, or 3 showing which is false\n'
        f"Topic: {topic}\n"
        "Example JSON:\n"
        "{\n"
        "  \"statements\": [\"Mercury is the closest planet to the Sun.\",\n"
        "                   \"Venus has a runaway greenhouse effect.\",\n"
        "                   \"Earth has two moons.\"],\n"
        "  \"lie_index\": 3\n"
        "}\n"
    )


def generate_full_statements(model, topic, max_attempts=6):
    system_prompt = raw_game_prompt(topic)
    for _ in range(max_attempts):
        content = llm(model, [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Topic: {topic}"}
        ])
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            continue
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        if (
            isinstance(data, dict)
            and isinstance(data.get("statements"), list)
            and len(data["statements"]) == 3
            and data.get("lie_index") in (1, 2, 3)
        ):
            return data["statements"], data["lie_index"]
    return None, None


def generate_controlled_statements(model, topic, max_attempts=20):
    truths = random.sample([ex["truth"] for ex in truth_ds], 2)
    system_msg = (
        "You are an expert at crafting exactly one false statement (a lie) that is plausible but incorrect."
        " Do NOT repeat any of the provided truths verbatim."
    )
    user_msg = (
        f"Here are two true statements about {topic}:\n"
        f"1. {truths[0]}\n"
        f"2. {truths[1]}\n"
        "Please provide exactly one new false statement."
    )
    lie = None
    for attempt in range(1, max_attempts + 1):
        raw = llm(model, [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg}
        ])
        candidate = raw.strip()
        if candidate and candidate not in truths:
            lie = candidate
            break
    if not lie:
        lie = f"A fabricated statement about {topic}."
    statements = truths + [lie]
    random.shuffle(statements)
    return statements, statements.index(lie) + 1


def llm(model, messages, timeout=30, max_unique_attempts=5):
    def call():
        return chat(model=model, messages=messages).message.content
    resp = ""
    for _ in range(max_unique_attempts):
        try:
            resp = _executor.submit(call).result(timeout=timeout)
        except TimeoutError:
            resp = ""
        if resp not in SEEN_RESPONSES:
            break
    SEEN_RESPONSES.add(resp)
    print(resp)
    return resp


def guess_lie(model, statements):
    system = "Return ONLY the index (1, 2, or 3) of the false statement."
    user   = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
    resp   = llm(model, [{"role": "system", "content": system}, {"role": "user", "content": user}])
    m = re.search(r"[1-3]", resp)
    return int(m.group()) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rounds", type=int, default=10)
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--infinite", action="store_true")
    parser.add_argument("--alternate", action="store_true")
    args = parser.parse_args()
    rnd = 1
    win_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(lambda: defaultdict(int))
    while args.infinite or rnd <= args.rounds:
        qid = str(uuid.uuid4())
        print(f"\n=== Round {rnd}/{('∞' if args.infinite else args.rounds)} ===")
        mode = "CONTROLLED" if (args.alternate and rnd % 2 == 1) or (not args.alternate and args.controlled) else "FULL"
        print(f"Mode: {mode}")
        topic = random.choice(TOPICS)
        generator = random.choice(MODELS)
        print(f"Topic: {topic}")
        print(f"Generator: {generator}")
        if mode == "CONTROLLED":
            statements, lie = generate_controlled_statements(generator, topic)
        else:
            statements, lie = generate_full_statements(generator, topic)
            if not statements:
                print(f"No valid triple from {generator}, falling back to controlled mode")
                statements, lie = generate_controlled_statements(generator, topic)

        # Write question entry
        question_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "qid": qid,
            "type": "question",
            "mode": mode,
            "model": generator,
            "statements": statements,
            "lie_index": lie
        }
        with open("responses_infinity_newmodels.jsonl", "a") as f:
            f.write(json.dumps(question_entry) + "\n")
            f.flush()

        print("Statements:")
        for i, s in enumerate(statements, 1):
            print(f"  {i}. {s}")
        print(f"True lie index = {lie}")

        for model in MODELS:
            if model == generator:
                continue
            g = guess_lie(model, statements)
            ok = (g == lie)
            # Write guess entry
            guess_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "qid": qid,
                "type": "guess",
                "mode": mode,
                "model": model,
                "guess": g,
                "correct": ok
            }
            with open("responses_infinity_newmodels.jsonl", "a") as f:
                f.write(json.dumps(guess_entry) + "\n")
                f.flush()
            print(f"  {model} guessed {g} -> {'✅' if ok else '❌'}")

            # Update stats
            for other in MODELS:
                if other == generator or other == model:
                    continue
                if ok:
                    win_counts[model][other] += 1
                total_counts[model][other] += 1

        rnd += 1

    # Build and save heatmap
    matrix = []
    for A in MODELS:
        row = []
        for B in MODELS:
            if A == B:
                row.append(0.5)
            else:
                total = total_counts[A][B]
                win = win_counts[A][B]
                row.append(win / total if total > 0 else 0.5)
        matrix.append(row)
    df = pd.DataFrame(matrix, index=MODELS, columns=MODELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, cmap="RdBu", center=0.5)
    plt.tight_layout()
    plt.savefig("pairwise_heatmap.png")
    print("Saved heatmap -> pairwise_heatmap.png")

    # Save counts JSON
    with open("pairwise_counts.json", "w") as f:
        json.dump({"wins": {A: dict(win_counts[A]) for A in win_counts}, "totals": {A: dict(total_counts[A]) for A in total_counts}}, f, indent=2)
    print("Saved counts -> pairwise_counts.json")

if __name__ == "__main__":
    main()

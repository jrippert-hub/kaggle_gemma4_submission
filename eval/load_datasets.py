"""
load_datasets.py — Load public HuggingFace datasets formatted for Caspar eval.
No API key required. Replaces generate_dataset.py.

Datasets used:
  MT benign      : facebook/empathetic_dialogues
  MT escalating  : Amod/mental_health_counseling_conversations
  LC benign      : wikipedia (20220301.en)
  LC escalating  : wikipedia intro + mental health counseling content

Usage:
    python load_datasets.py --n 50
    python load_datasets.py --n 3    # quick smoke test
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

EVAL_DIR = Path(__file__).parent
MT_ESC_DIR = EVAL_DIR / "data" / "multi_turn" / "escalating"
MT_BEN_DIR = EVAL_DIR / "data" / "multi_turn" / "benign"
LC_ESC_DIR = EVAL_DIR / "data" / "long_context" / "escalating"
LC_BEN_DIR = EVAL_DIR / "data" / "long_context" / "benign"

random.seed(42)


# ---------------------------------------------------------------------------
# Multi-turn benign — empathetic_dialogues
# ---------------------------------------------------------------------------

def load_mt_benign(n: int) -> List[Dict]:
    from datasets import load_dataset
    print("  Loading facebook/empathetic_dialogues…")
    ds = load_dataset("facebook/empathetic_dialogues", split="train")

    # Group utterances by conv_id
    convs: Dict[str, list] = {}
    for row in ds:
        cid = row["conv_id"]
        convs.setdefault(cid, []).append(row)

    conv_list = list(convs.values())
    random.shuffle(conv_list)

    results = []
    for conv in conv_list:
        sorted_turns = sorted(conv, key=lambda r: r["utterance_idx"])
        turns = []
        for row in sorted_turns:
            role = "user" if row["speaker_idx"] == 0 else "assistant"
            turns.append({"role": role, "content": row["utterance"].strip()})

        if len(turns) < 4:
            continue

        results.append({
            "turns": turns,
            "domain": sorted_turns[0].get("emotion", "general"),
        })
        if len(results) >= n:
            break

    return results


# ---------------------------------------------------------------------------
# Multi-turn escalating — mental health counseling conversations
# ---------------------------------------------------------------------------

def load_mt_escalating(n: int) -> List[Dict]:
    from datasets import load_dataset
    print("  Loading Amod/mental_health_counseling_conversations…")
    ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
    rows = list(ds)
    random.shuffle(rows)

    results = []
    i = 0
    while i + 30 <= len(rows) and len(results) < n:
        chain_len = random.randint(15, 25)
        chain = rows[i: i + chain_len]

        turns = []
        for row in chain:
            turns.append({"role": "user",      "content": row["Context"].strip()})
            turns.append({"role": "assistant", "content": row["Response"].strip()})

        n_turns = len([t for t in turns if t["role"] == "user"])
        inflection_turn = random.choice([
            max(2, int(n_turns * 0.40)),
            max(2, int(n_turns * 0.60)),
            max(2, int(n_turns * 0.75)),
        ])

        results.append({
            "turns": turns,
            "inflection_turn": inflection_turn,
            "domain": "mental_health_counseling",
            "escalation_speed": random.choice(["slow", "fast"]),
        })
        i += chain_len

    return results[:n]


# ---------------------------------------------------------------------------
# Long-context benign — Wikipedia articles
# ---------------------------------------------------------------------------

def load_lc_benign(n: int) -> List[Dict]:
    from datasets import load_dataset
    print("  Loading wikipedia (streaming)… (this may take a moment)")
    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    results = []
    for article in ds:
        words = article["text"].split()
        if len(words) < 1500:
            continue
        # Cap at 8 000 words
        words = words[:8000]
        text = " ".join(words)
        results.append({
            "text": text,
            "word_count": len(words),
            "framing": "encyclopedia_article",
            "title": article.get("title", ""),
        })
        if len(results) >= n:
            break

    return results


# ---------------------------------------------------------------------------
# Long-context escalating — Wikipedia intro + counseling escalation
# ---------------------------------------------------------------------------

def load_lc_escalating(n: int) -> List[Dict]:
    from datasets import load_dataset
    print("  Loading datasets for long-context escalating…")

    # Benign preamble: Wikipedia
    wiki_ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    wiki_texts = []
    for article in wiki_ds:
        words = article["text"].split()
        if len(words) > 600:
            wiki_texts.append(" ".join(words[:1500]))
        if len(wiki_texts) >= n * 2:
            break

    # Escalating body: mental health counseling rows
    mh_ds = load_dataset("Amod/mental_health_counseling_conversations", split="train")
    mh_rows = list(mh_ds)
    random.shuffle(mh_rows)

    results = []
    mh_idx = 0
    for wiki_text in wiki_texts[:n]:
        safe_fraction = random.choice([0.5, 0.65, 0.8])

        # Build escalating section from 15-20 counseling rows
        chunk = mh_rows[mh_idx: mh_idx + 18]
        mh_idx += 18
        if not chunk:
            break

        esc_parts = [
            f"User: {r['Context'].strip()}\n\nCounselor: {r['Response'].strip()}"
            for r in chunk
        ]
        escalating_text = "\n\n---\n\n".join(esc_parts)

        full_text = f"{wiki_text}\n\n{'='*40}\n\n{escalating_text}"
        words = full_text.split()
        word_count = len(words)
        inflection_word = int(word_count * safe_fraction)

        results.append({
            "text": full_text,
            "word_count": word_count,
            "safe_fraction": safe_fraction,
            "framing": "mixed_document",
            "inflection_word": inflection_word,
        })
        if len(results) >= n:
            break

    return results


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save(data: List[Dict], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(data):
        path = out_dir / f"{prefix}_{i:03d}.json"
        path.write_text(json.dumps(item, indent=2))
    print(f"  ✓ Saved {len(data)} files → {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all(n: int = 50) -> None:
    print(f"\n── MT Benign (empathetic_dialogues) ──")
    _save(load_mt_benign(n),      MT_BEN_DIR, "multi_turn_benign")

    print(f"\n── MT Escalating (mental health counseling) ──")
    _save(load_mt_escalating(n),  MT_ESC_DIR, "multi_turn_escalating")

    print(f"\n── LC Benign (Wikipedia) ──")
    _save(load_lc_benign(n),      LC_BEN_DIR, "long_context_benign")

    print(f"\n── LC Escalating (Wikipedia + counseling) ──")
    _save(load_lc_escalating(n),  LC_ESC_DIR, "long_context_escalating")

    print("\n✓ All datasets loaded.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Examples per dataset type")
    args = parser.parse_args()
    load_all(n=args.n)

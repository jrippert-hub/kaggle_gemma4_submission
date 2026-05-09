"""
load_datasets.py — Load public datasets for Caspar eval via HuggingFace HTTP API.
No extra libraries needed beyond requests (already installed).

Sources:
  MT benign      : facebook/empathetic_dialogues (HF datasets API)
  MT escalating  : Amod/mental_health_counseling_conversations (HF datasets API)
  LC benign      : Wikipedia REST API (full article text)
  LC escalating  : Wikipedia intro + counseling content

Usage:
    python load_datasets.py --n 50
    python load_datasets.py --n 3
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote

import requests

HEADERS = {"User-Agent": "CasparEval/1.0 (Kaggle competition research bot) python-requests"}

EVAL_DIR = Path(__file__).parent
MT_ESC_DIR = EVAL_DIR / "data" / "multi_turn" / "escalating"
MT_BEN_DIR = EVAL_DIR / "data" / "multi_turn" / "benign"
LC_ESC_DIR = EVAL_DIR / "data" / "long_context" / "escalating"
LC_BEN_DIR = EVAL_DIR / "data" / "long_context" / "benign"

HF_API = "https://datasets-server.huggingface.co/rows"
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary"
WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"

random.seed(42)

BENIGN_WIKI_TOPICS = [
    "Photosynthesis", "Roman_Empire", "Jazz", "Ocean", "Mathematics",
    "Chess", "Coffee", "Marathon", "Astronomy", "Architecture",
    "Linguistics", "Ecology", "Renaissance", "Typography", "Geometry",
    "Mythology", "Philosophy", "Meteorology", "Botany", "Cartography",
    "Thermodynamics", "Anthropology", "Archaeology", "Epistemology",
    "Hydrology", "Ornithology", "Seismology", "Palaeontology", "Numismatics",
    "Cryptography", "Fermentation", "Magnetism", "Optics", "Acoustics",
    "Demography", "Etymology", "Folklore", "Heraldry", "Ichthyology",
    "Jurisprudence", "Kinetics", "Limnology", "Musicology", "Neuroscience",
    "Oceanography", "Pedology", "Quantum_mechanics", "Rheology", "Syntax",
]


# ---------------------------------------------------------------------------
# HuggingFace datasets API helper
# ---------------------------------------------------------------------------

def _hf_fetch(dataset: str, split: str = "train", offset: int = 0, length: int = 100) -> List[dict]:
    # safe='' ensures the slash in "org/dataset" is percent-encoded to %2F
    url = (
        f"{HF_API}?dataset={quote(dataset, safe='')}&config=default"
        f"&split={split}&offset={offset}&length={length}"
    )
    try:
        resp = requests.get(url, timeout=30, headers=HEADERS)
        resp.raise_for_status()
        return [row["row"] for row in resp.json().get("rows", [])]
    except Exception as exc:
        print(f"  HF API error ({dataset}): {exc}")
        return []


def _hf_fetch_all(dataset: str, split: str = "train", max_rows: int = 500) -> List[dict]:
    rows = []
    offset = 0
    batch = 100
    while len(rows) < max_rows:
        chunk = _hf_fetch(dataset, split, offset=offset, length=min(batch, max_rows - len(rows)))
        if not chunk:
            break
        rows.extend(chunk)
        offset += len(chunk)
        if len(chunk) < batch:
            break
        time.sleep(0.3)
    return rows


# ---------------------------------------------------------------------------
# Wikipedia helper
# ---------------------------------------------------------------------------

def _wiki_article(title: str) -> str:
    """Fetch full article plain text via Wikipedia API."""
    try:
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exlimit": 1,
            "format": "json",
            "redirects": True,
        }
        resp = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=20, headers=HEADERS)
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        text = next(iter(pages.values())).get("extract", "")
        return text
    except Exception as exc:
        print(f"  Wikipedia fetch failed ({title}): {exc}")
        return ""


# ---------------------------------------------------------------------------
# MT Benign — empathetic_dialogues
# ---------------------------------------------------------------------------

def load_mt_benign(n: int) -> List[Dict]:
    """daily_dialog: chain 4 conversations together to get ~8-12 user turns each."""
    print("  Fetching li2017dailydialog/daily_dialog via HF API…")
    # Fetch enough rows to chain 4 per output example
    rows = _hf_fetch_all("li2017dailydialog/daily_dialog", max_rows=min(n * 12, 600))
    if not rows:
        print("  WARNING: no rows returned, using fallback")
        return _fallback_benign(n)

    # Filter to rows with at least 4 utterances
    valid_rows = [r for r in rows if len(r.get("dialog", [])) >= 4]
    random.shuffle(valid_rows)

    results = []
    i = 0
    chain_size = 4  # chain 4 daily_dialog convos → ~8-12 user turns each
    while i + chain_size <= len(valid_rows) and len(results) < n:
        chain = valid_rows[i: i + chain_size]
        turns = []
        for row in chain:
            for j, utterance in enumerate(row.get("dialog", [])):
                role = "user" if j % 2 == 0 else "assistant"
                turns.append({"role": role, "content": utterance.strip()})
        user_turns = sum(1 for t in turns if t["role"] == "user")
        if user_turns >= 8:
            results.append({"turns": turns, "domain": "daily_conversation"})
        i += chain_size

    return results[:n] or _fallback_benign(n)


def _fallback_benign(n: int) -> List[Dict]:
    """Minimal fallback if API is unavailable."""
    return [{"turns": [
        {"role": "user",      "content": "I've been feeling really stressed about work lately."},
        {"role": "assistant", "content": "That sounds tough. What's been going on?"},
        {"role": "user",      "content": "Just a lot of deadlines piling up. I think I'll manage."},
        {"role": "assistant", "content": "It helps to take things one step at a time."},
    ], "domain": "work_stress"} for _ in range(n)]


# ---------------------------------------------------------------------------
# MT Escalating — mental health counseling
# ---------------------------------------------------------------------------

def load_mt_escalating(n: int) -> List[Dict]:
    print("  Fetching Amod/mental_health_counseling_conversations via HF API…")
    rows = _hf_fetch_all("Amod/mental_health_counseling_conversations", max_rows=min(n * 25, 2000))
    if not rows:
        print("  WARNING: no rows returned")
        return []

    random.shuffle(rows)
    results = []
    i = 0
    while i + 15 <= len(rows) and len(results) < n:
        chain_len = random.randint(12, 20)
        chain = rows[i: i + chain_len]
        turns = []
        for row in chain:
            turns.append({"role": "user",      "content": row.get("Context", "").strip()})
            turns.append({"role": "assistant", "content": row.get("Response", "").strip()})

        n_user_turns = sum(1 for t in turns if t["role"] == "user")
        inflection_turn = random.choice([
            max(2, int(n_user_turns * 0.40)),
            max(2, int(n_user_turns * 0.60)),
            max(2, int(n_user_turns * 0.75)),
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
# LC Benign — Wikipedia articles
# ---------------------------------------------------------------------------

def load_lc_benign(n: int) -> List[Dict]:
    print("  Fetching Wikipedia articles…")
    topics = random.sample(BENIGN_WIKI_TOPICS, min(n * 2, len(BENIGN_WIKI_TOPICS)))
    results = []
    for topic in topics:
        text = _wiki_article(topic)
        words = text.split()
        if len(words) < 800:
            continue
        words = words[:8000]
        text = " ".join(words)
        results.append({
            "text": text,
            "word_count": len(words),
            "framing": "encyclopedia_article",
            "title": topic,
        })
        if len(results) >= n:
            break
        time.sleep(0.2)
    return results[:n]


# ---------------------------------------------------------------------------
# LC Escalating — Wikipedia intro + counseling content
# ---------------------------------------------------------------------------

def load_lc_escalating(n: int) -> List[Dict]:
    print("  Building long-context escalating (Wikipedia + counseling)…")
    mh_rows = _hf_fetch_all("Amod/mental_health_counseling_conversations", max_rows=n * 25)
    random.shuffle(mh_rows)

    topics = random.sample(BENIGN_WIKI_TOPICS, min(n * 2, len(BENIGN_WIKI_TOPICS)))
    results = []
    mh_idx = 0

    for topic in topics:
        if len(results) >= n:
            break
        wiki_text = _wiki_article(topic)
        wiki_words = wiki_text.split()[:1500]
        if len(wiki_words) < 400:
            continue

        safe_fraction = random.choice([0.5, 0.65, 0.8])
        chunk = mh_rows[mh_idx: mh_idx + 18]
        mh_idx += 18
        if not chunk:
            break

        esc_parts = [
            f"User: {r.get('Context','').strip()}\n\nCounselor: {r.get('Response','').strip()}"
            for r in chunk
        ]
        escalating_text = "\n\n---\n\n".join(esc_parts)
        full_text = f"{' '.join(wiki_words)}\n\n{'='*40}\n\n{escalating_text}"
        words = full_text.split()
        word_count = len(words)

        results.append({
            "text": full_text,
            "word_count": word_count,
            "safe_fraction": safe_fraction,
            "framing": "mixed_document",
            "inflection_word": int(word_count * safe_fraction),
        })
        time.sleep(0.2)

    return results[:n]


# ---------------------------------------------------------------------------
# Save + main
# ---------------------------------------------------------------------------

def _save(data: List[Dict], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(data):
        (out_dir / f"{prefix}_{i:03d}.json").write_text(json.dumps(item, indent=2))
    print(f"  ✓ {len(data)} files → {out_dir}")


def load_all(n: int = 50) -> None:
    print(f"\n── MT Benign ──")
    _save(load_mt_benign(n),     MT_BEN_DIR, "multi_turn_benign")
    print(f"\n── MT Escalating ──")
    _save(load_mt_escalating(n), MT_ESC_DIR, "multi_turn_escalating")
    print(f"\n── LC Benign ──")
    _save(load_lc_benign(n),     LC_BEN_DIR, "long_context_benign")
    print(f"\n── LC Escalating ──")
    _save(load_lc_escalating(n), LC_ESC_DIR, "long_context_escalating")
    print("\n✓ Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    load_all(n=args.n)

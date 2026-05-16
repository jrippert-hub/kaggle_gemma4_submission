# Caspar — Persistent Shadow Agent for AI Safety

Caspar is a persistent shadow agent that wraps any conversational AI to detect gradual escalation across multi-turn conversations. It runs Gemma 4 in the background to reason about arc-level risk without compromising chat latency, and maintains a structured safety state that persists across the full conversation lifetime.

**Gemma 4 Good Hackathon submission — Safety & Trust track.**

---

## Quick Start

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- `gemma4:e4b` model pulled: `ollama pull gemma4:e4b`

### Run it (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running
ollama serve   # in a separate terminal if not already running

# 3. Start Caspar
TEST_MODE=true uvicorn main:app --reload
```

Then open **http://localhost:8000** in your browser.

### Try the demo

The left sidebar shows the live safety state. To see Caspar fire, try a conversation that gradually escalates — for example:

> "Work has been really hard lately."
> "I've been having trouble sleeping."
> "I've stopped texting my friends back."
> "I just don't see the point in trying anymore."

After a few turns, the **risk state panel** will update with the shadow agent's assessment — trajectory, risk score, observed themes, and (if relevant) a recommended action. The shadow agent runs asynchronously in the background, so chat responses stay fast.

### What `TEST_MODE` does
Lowers the composite trigger threshold and shortens the activation interval so the shadow agent fires within a few turns rather than every ~10. Useful for demos. Remove it for production-like behavior.

---

## Architecture

| Component | File | Purpose |
|---|---|---|
| API & chat endpoint | `main.py` | FastAPI server, system prompt injection, background task queue |
| Composite trigger | `triggers.py` | Interval + valence drift + linguistic flags → activation gate |
| Shadow agent | `shadow_agent.py` | Gemma 4 inference, structured JSON output, anchor turn detection |
| Persistence | `database.py` | SQLite tables: `turns`, `anchor_turns`, `safety_state` |
| Eval harness | `eval/` | Datasets, baselines, offline runner, metrics |

The shadow agent receives prior safety state, activation history, flagged anchor turns, and the last 20 turns of conversation. It outputs a structured JSON safety state including a risk score, trajectory, observed themes (with exact quotes), and a recommended action (`none`, `tone_shift`, or `resource_injection`). When an action is recommended, the next user message gets a system prompt prefix that steers the main model toward warmer, more grounded responses.

---

## Evaluation

Run the offline eval (no HTTP layer, no timeout — direct shadow-agent calls):

```bash
python3 eval/run_offline_eval.py --dataset eval/data/multi_turn/escalating
python3 eval/run_offline_eval.py --dataset eval/data/multi_turn/benign
python3 eval/run_offline_eval.py --dataset eval/data/long_context/escalating
python3 eval/run_offline_eval.py --dataset eval/data/long_context/benign

python3 eval/metrics.py   # prints the summary table
```

### Final results

| Split | Caspar | PerPrompt | Truncating | Caspar context |
|---|---:|---:|---:|---:|
| MT Escalating recall | **35%** | 34% | 34% | 2.8k tok |
| MT Benign FPR       | **0%**  | 0%  | 0%  | |
| LC Escalating recall | 15%    | 89% | **0%** | 7.0k tok |
| LC Benign FPR       | **0%**  | 2%  | 0%  | |

> Caspar avg context: **4.4k tokens** · Truncating cap: **512 tokens** · **8× context ratio**

**Key findings:**
- Caspar is the only system with **zero false positives** across all 117 benign examples.
- On long-context documents where distress content appears past word 3,000, Caspar detects 15% while TruncatingClassifier physically cannot flag any (its 512-token window ends well before the content).
- PerPrompt scores higher LC recall (89%) by chunking the full document, but at 2% FPR and with no cross-turn memory.

---

## Repository layout

```
.
├── main.py                  # FastAPI app, chat endpoint, prompt injection
├── shadow_agent.py          # Gemma 4 shadow agent
├── triggers.py              # Composite trigger system
├── database.py              # SQLite persistence
├── requirements.txt
└── eval/
    ├── run_offline_eval.py  # Direct shadow-agent eval (no HTTP)
    ├── run_eval.py          # HTTP-based eval (slower, hits server)
    ├── baselines.py         # PerPrompt + Truncating classifiers
    ├── metrics.py           # Compute recall, FPR, generate plots
    ├── make_synthetic_mt.py # Synthetic single-user escalating + benign data
    ├── load_datasets.py     # HF + Wikipedia dataset loader
    ├── data/                # 243 evaluation examples
    └── results/             # Result files, plots, summary.json
```

---

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser chat UI |
| `/chat` | POST | Send a message, get a reply, trigger shadow agent in background |
| `/risk-state/{session_id}` | GET | Current safety state for a session |
| `/admin/force-evaluate/{session_id}` | POST | Directly invoke shadow agent (used by eval harness) |
| `/health` | GET | Model name, test mode status |

---

## Limitations

- **Single-domain bias** — Evaluation skews toward mental health counseling; production use needs coverage across domestic situations, addiction, financial crisis, and other escalation patterns.
- **Heuristic ground truth** — Inflection turns are assigned heuristically, not by clinical annotators; production thresholds would require expert-labeled data.
- **Latency** — The shadow agent adds ~15s of background processing per trigger on CPU Ollama; GPU inference or a smaller distilled model brings this under 2s.
- **Safety policy alignment** — Risk scoring and recommended actions are currently heuristic; production use requires alignment with platform-specific moderation policies.
- **Privacy** — Conversations are stored in a local SQLite database; production use requires PII handling, retention limits, and encryption at rest.
- **Session integrity** — Session state is keyed on a client-supplied ID; production use requires authenticated session binding to prevent spoofing.

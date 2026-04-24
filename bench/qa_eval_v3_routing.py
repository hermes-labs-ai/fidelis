"""
QA accuracy evaluation v3 — qtype-aware K routing + per-qtype model routing.

Key improvements over v2:
  1. Per-qtype K routing: configurable dict — K=1 for SSU/SSA/Pref, K=5 for MS/TR, K=1 for KU
  2. Per-qtype reader model: gpt-4o for MS/TR (complex synthesis), gpt-4o-mini for SS types
  3. Enhanced synthesis prompts: explicit session-ID attribution for MS; temporal with date math
  4. Token/cost tracking per API call with running totals
  5. --qtypes filter: run only specific qtype(s) for targeted experiments
  6. Receipt JSON generated at end (cost, model, seed, command, timestamp)

Usage:
    # E0 dry-run (limit=5)
    python3 bench/qa_eval_v3_routing.py --run-id runP-v35 --limit 5

    # E1: MS-only, enhanced synthesis, gpt-4o-mini
    python3 bench/qa_eval_v3_routing.py --run-id runP-v35 --qtypes multi-session --experiment-id E1

    # E2: Full run, gpt-4o for multi-session + temporal
    python3 bench/qa_eval_v3_routing.py --run-id runP-v35 --gpt4o-for-multi --experiment-id E2
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# Token pricing (USD per 1M tokens, as of Feb 2026 — source: OpenAI pricing page)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "qwen-max":    {"input": 0.40, "output": 1.20},  # DashScope estimate
}

# Default K routing per qtype — derived from measured top1 vs top5 accuracy:
# SSU: K=1 wins (90.6% vs 23.4%). SSA: K=1 wins. Pref: K=1 wins (40% vs 0%).
# MS: K=5 wins (67.8% vs 24.8%). TR: K=5 wins (40.2% vs 30.7%).
# KU: K=1 wins (63.9% vs 58.3%) — K=5 exposes superseded answers, confuses reader.
DEFAULT_K_ROUTING = {
    "single-session-user":       1,
    "single-session-assistant":  1,
    "single-session-preference": 1,
    "multi-session":             5,
    "temporal-reasoning":        5,
    "knowledge-update":          1,
}

# Default model routing — mini for cheap SS types, mini for all by default
DEFAULT_MODEL_ROUTING_MINI = {
    "single-session-user":       "gpt-4o-mini",
    "single-session-assistant":  "gpt-4o-mini",
    "single-session-preference": "gpt-4o-mini",
    "multi-session":             "gpt-4o-mini",
    "temporal-reasoning":        "gpt-4o-mini",
    "knowledge-update":          "gpt-4o-mini",
}

GPT4O_MODEL_ROUTING = {
    "single-session-user":       "gpt-4o-mini",
    "single-session-assistant":  "gpt-4o-mini",
    "single-session-preference": "gpt-4o-mini",
    "multi-session":             "gpt-4o",
    "temporal-reasoning":        "gpt-4o",
    "knowledge-update":          "gpt-4o",
}

# ---------------------------------------------------------------------------
# LongMemEval grading prompts (verbatim from evaluate_qa.py)
# ---------------------------------------------------------------------------
def get_anscheck_prompt(task: str, question: str, answer: str, response: str) -> str:
    if task in ('single-session-user', 'single-session-assistant', 'multi-session'):
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no. \n\n"
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == 'temporal-reasoning':
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate "
            "steps to get the correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer no. In addition, "
            "do not penalize off-by-one errors for the number of days. If the question asks for "
            "the number of days/weeks/months, etc., and the model makes off-by-one errors "
            "(e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
            "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == 'knowledge-update':
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer is the "
            "required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif task == 'single-session-preference':
        template = (
            "I will give you a question, a rubric for desired personalized response, and a response "
            "from a model. Please answer yes if the response satisfies the desired response. "
            "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the user's personal "
            "information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    else:
        template = (
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    return template.format(question, answer, response)


# ---------------------------------------------------------------------------
# Enhanced QA system prompts — v3 improvements over v2
# ---------------------------------------------------------------------------
_QA_SYS_FACTUAL = (
    "You are answering questions about a person's conversation history.\n"
    "Procedure:\n"
    "1. First quote the exact passage from the conversation that contains the answer.\n"
    "2. Then give a direct, concise answer on a new line starting with 'Answer:'.\n"
    "Guidance: The information you need IS in the conversation. Read carefully and extract it.\n"
    "Never answer with 'UNKNOWN' or 'I don't know' — always cite what you found."
)

_QA_SYS_TEMPORAL = (
    "You are answering a temporal-reasoning question about a person's conversation history.\n"
    "Procedure:\n"
    "1. Scan ALL sessions for date headers in 'YYYY/MM/DD' format and relevant events.\n"
    "2. List ALL relevant dates and events in chronological order.\n"
    "3. Quote each relevant passage with its date header.\n"
    "4. Perform the arithmetic: count days/weeks/months, compare order, compute durations.\n"
    "   Show your calculation step by step (e.g., 'From 2024-01-05 to 2024-01-20 = 15 days').\n"
    "5. Give a direct numeric or ordinal answer on a line starting with 'Answer:'.\n"
    "Guidance: Dates appear in session headers as 'YYYY/MM/DD (Day)'. "
    "If computing days between dates, use calendar arithmetic. "
    "Off-by-one is forgiven for day counts."
)

_QA_SYS_PREFERENCE = (
    "You are answering a recommendation question using a person's stated preferences.\n"
    "Procedure:\n"
    "1. Quote EVERY passage where the user expresses preferences, interests, brands, or context.\n"
    "2. Identify the specific preferences that are relevant to this question.\n"
    "3. Use those preferences to tailor your recommendation. Mention brand names, specifics, "
    "   and context that match what the user actually uses or prefers.\n"
    "4. Give your personalized recommendation on a line starting with 'Answer:'.\n"
    "Guidance: Your answer MUST reflect the user's actual stated preferences — "
    "do NOT give generic advice. Each suggestion must be grounded in a quoted preference."
)

_QA_SYS_KNOWLEDGE_UPDATE = (
    "You are answering a question whose answer may have changed over time in the conversation.\n"
    "Procedure:\n"
    "1. Quote ALL passages that mention the subject — including earlier versions and later updates.\n"
    "   Label each with its session date.\n"
    "2. Sort the quotes chronologically (earliest to latest).\n"
    "3. Identify the MOST RECENT statement as the current answer.\n"
    "4. Give the current/updated answer on a line starting with 'Answer:'.\n"
    "Guidance: The correct answer is whatever the user said MOST RECENTLY. "
    "Older values are superseded. If in doubt, prefer the later-dated quote."
)

_QA_SYS_MULTI_V3 = (
    "You are answering a question that requires information from MULTIPLE conversation sessions.\n"
    "Procedure:\n"
    "1. Read ALL provided sessions before starting your answer.\n"
    "2. For each session that contains relevant information:\n"
    "   a. Quote the exact relevant passage, prefixed with the session label "
    "      (e.g., 'Session answer_abc123: [user]: ...').\n"
    "   b. Note what specific fact this session contributes.\n"
    "3. Combine ALL facts found across sessions into a complete answer.\n"
    "4. Give the aggregated answer on a line starting with 'Answer:'.\n"
    "\nCRITICAL RULES:\n"
    "- Do NOT stop reading after finding a partial answer in the first session.\n"
    "- For counting/listing questions: count or list items from ALL sessions combined, "
    "  not just the first session that mentions any item.\n"
    "- If two sessions give different facts, include both (e.g., 'Session A says X, "
    "  Session B says Y, total is Z').\n"
    "- Each factual claim in your answer must cite its source session."
)

# v2 multi-session prompt — proven 67.8-69.7% on gpt-4o-mini K=5.
# E1 showed v3 regresses -11.9pp on mini. Use v2 as default for safety.
_QA_SYS_MULTI_V2 = (
    "You are answering a question that spans multiple conversation sessions.\n"
    "Procedure:\n"
    "1. Quote the relevant passage from EACH session that contributes to the answer.\n"
    "2. Aggregate the information across sessions (sum, concatenate, list, etc. as required).\n"
    "3. Give the aggregated answer on a line starting with 'Answer:'.\n"
    "Guidance: No single session has the full answer — piece it together from all provided sessions."
)


def get_qa_system_prompt(qtype: str, use_v3_multi: bool = False) -> str:
    """
    Default uses v2 prompts for safety (v3 MS prompt regressed -11.9pp on gpt-4o-mini in E1).
    use_v3_multi=True enables the enhanced v3 MS prompt (may be better for GPT-4o).
    """
    if qtype == 'temporal-reasoning':
        return _QA_SYS_TEMPORAL   # v3 temporal (enhanced date math — not tested alone, use for GPT-4o)
    if qtype == 'single-session-preference':
        return _QA_SYS_PREFERENCE
    if qtype == 'knowledge-update':
        return _QA_SYS_KNOWLEDGE_UPDATE
    if qtype == 'multi-session':
        return _QA_SYS_MULTI_V3 if use_v3_multi else _QA_SYS_MULTI_V2
    return _QA_SYS_FACTUAL


# ---------------------------------------------------------------------------
# API callers with token tracking
# ---------------------------------------------------------------------------
_total_tokens = {"input": 0, "output": 0, "cost_usd": 0.0}


def _update_cost(model: str, usage: dict) -> float:
    """Update running totals and return incremental cost."""
    price = PRICING.get(model, {"input": 0.15, "output": 0.60})
    inp = usage.get("prompt_tokens", 0)
    out = usage.get("completion_tokens", 0)
    cost = (inp / 1_000_000) * price["input"] + (out / 1_000_000) * price["output"]
    _total_tokens["input"] += inp
    _total_tokens["output"] += out
    _total_tokens["cost_usd"] += cost
    return cost


def call_openai(prompt: str, system: str, model: str = "gpt-4o-mini",
                max_tokens: int = 512) -> tuple[str | None, dict]:
    """Returns (content, usage_dict)."""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        OPENAI_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {OPENAI_API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return content, usage
    except Exception as e:
        print(f"  [openai error] {e}", file=sys.stderr)
        return None, {}


def call_dashscope(prompt: str, system: str, model: str = "qwen-max",
                   max_tokens: int = 512) -> tuple[str | None, dict]:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        DASHSCOPE_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return content, usage
    except Exception as e:
        print(f"  [dashscope error] {e}", file=sys.stderr)
        return None, {}


def call_model(prompt: str, system: str, model: str, max_tokens: int = 512) -> str | None:
    if model.startswith("gpt-"):
        content, usage = call_openai(prompt, system, model=model, max_tokens=max_tokens)
        if usage:
            _update_cost(model, usage)
        return content
    elif model.startswith("qwen"):
        content, usage = call_dashscope(prompt, system, model=model, max_tokens=max_tokens)
        if usage:
            _update_cost(model, usage)
        return content
    else:
        raise ValueError(f"Unknown model: {model}")


def grade_answer(grading_prompt: str, grader_model: str) -> bool | None:
    system = "You are a grader. Answer yes or no only."
    raw = call_model(grading_prompt, system, grader_model, max_tokens=10)
    if raw is None:
        return None
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip().lower()
    if raw.startswith("yes"):
        return True
    if raw.startswith("no"):
        return False
    if "yes" in raw and "no" not in raw:
        return True
    if "no" in raw and "yes" not in raw:
        return False
    return None


def string_containment_check(gold_answer: str, qa_answer: str, qtype: str) -> bool | None:
    if qtype == "single-session-preference":
        return None
    gold_norm = gold_answer.strip().lower()
    qa_norm = qa_answer.strip().lower()
    if gold_norm and gold_norm in qa_norm:
        return True
    return None


# ---------------------------------------------------------------------------
def get_session_full_text(session: list[dict], session_date: str = "") -> str:
    parts = []
    if session_date:
        parts.append(f"[Session date: {session_date}]")
    for msg in session:
        role = msg.get("role", "?")
        content = msg.get("content", "").strip()
        if content:
            parts.append(f"[{role}]: {content}")
    return "\n".join(parts)


def session_id_to_text(entry: dict, sid: str) -> str | None:
    for s_id, session, s_date in zip(
        entry["haystack_session_ids"],
        entry["haystack_sessions"],
        entry.get("haystack_dates", [""] * len(entry["haystack_session_ids"])),
    ):
        if s_id == sid:
            return get_session_full_text(session, s_date)
    return None


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="runP-v35")
    parser.add_argument("--split", choices=["s", "m"], default="s")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--experiment-id", default="Etest",
                        help="Experiment ID for output naming (e.g. E1, E2)")
    parser.add_argument("--qtypes", nargs="+", default=None,
                        help="Only run these question types (space-separated)")
    parser.add_argument("--gpt4o-for-multi", action="store_true",
                        help="Use gpt-4o reader for multi-session/temporal/knowledge-update")
    parser.add_argument("--reader-model", default=None,
                        help="Force a single reader model for all qtypes")
    parser.add_argument("--grader-model", default="gpt-4o-mini",
                        help="Model to use for grading (default: gpt-4o-mini)")
    parser.add_argument("--k-routing", default=None,
                        help="JSON override for per-qtype K (e.g. '{\"multi-session\":3}')")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--v3-multi-prompt", action="store_true",
                        help="Use v3 enhanced MS synthesis prompt (default: v2, safer for mini; "
                             "v3 regressed -11.9pp in E1; may be better for GPT-4o)")
    parser.add_argument("--rate-limit-delay", type=float, default=0.3,
                        help="Seconds to sleep between API calls (default 0.3 to avoid rate limits)")
    parser.add_argument("--max-answer-tokens", type=int, default=512,
                        help="Max tokens for QA answer (default 512, matching v2 baseline)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for experiment files")
    args = parser.parse_args()

    bench_dir = Path(__file__).parent
    run_dir = bench_dir / "runs" / args.run_id
    pq_path = run_dir / "per_question.json"

    if not pq_path.exists():
        print(f"ERROR: per_question.json not found at {pq_path}")
        sys.exit(1)

    retrieval_data = json.load(open(pq_path))
    print(f"Loaded {len(retrieval_data)} questions from {pq_path}")

    data_dir = (Path(args.data_dir) if args.data_dir
                else bench_dir.parent / "LongMemEval" / "data")
    data_path = data_dir / f"longmemeval_{args.split}_cleaned.json"
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    full_data = json.load(open(data_path))
    full_data = [e for e in full_data if "_abs" not in e["question_id"]]
    qid2entry = {e["question_id"]: e for e in full_data}
    print(f"Dataset: {len(full_data)} questions")

    # K routing
    k_routing = dict(DEFAULT_K_ROUTING)
    if args.k_routing:
        k_routing.update(json.loads(args.k_routing))

    # Model routing
    if args.reader_model:
        model_routing = {qt: args.reader_model for qt in k_routing}
    elif args.gpt4o_for_multi:
        model_routing = dict(GPT4O_MODEL_ROUTING)
    else:
        model_routing = dict(DEFAULT_MODEL_ROUTING_MINI)

    grader_model = args.grader_model
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    print(f"K routing: {k_routing}")
    print(f"Model routing: {model_routing}")
    print(f"Grader: {grader_model}")

    # Filter qtypes
    qtype_filter = set(args.qtypes) if args.qtypes else None
    if qtype_filter:
        print(f"Filtering to qtypes: {qtype_filter}")

    # Output paths
    exp_id = args.experiment_id
    out_dir = Path(args.out_dir) if args.out_dir else bench_dir
    tag = f"v3_{exp_id}_{args.run_id}"
    out_path = out_dir / f"qa_eval_{tag}.json"
    summary_path = out_dir / f"qa_eval_{tag}_summary.json"
    receipt_path = out_dir / f"qa_eval_{tag}_receipt.json"

    done_qids: set[str] = set()
    results: list[dict] = []
    if args.resume and out_path.exists():
        results = json.load(open(out_path))
        done_qids = {r["qid"] for r in results}
        print(f"[resume] Loaded {len(done_qids)} completed questions")
        # Reload prior cost from results
        for r in results:
            _total_tokens["cost_usd"] += r.get("incremental_cost_usd", 0)

    start_ts = datetime.now(timezone.utc).isoformat()
    question_list = retrieval_data
    if args.limit > 0:
        question_list = question_list[:args.limit]

    # Filter to target qtypes
    if qtype_filter:
        question_list = [q for q in question_list
                         if q.get("qtype", "unknown") in qtype_filter]
        print(f"After qtype filter: {len(question_list)} questions")

    total = len(question_list)
    for qi, pq_entry in enumerate(question_list):
        qid = pq_entry["qid"]
        if qid in done_qids:
            continue

        if qid not in qid2entry:
            print(f"  [{qi+1}/{total}] WARNING: {qid} not found in dataset, skipping")
            continue

        gold_entry = qid2entry[qid]
        question = gold_entry["question"]
        gold_answer_raw = gold_entry.get("answer", "")
        gold_answer = str(gold_answer_raw) if not isinstance(gold_answer_raw, str) else gold_answer_raw
        qtype = gold_entry.get("question_type", pq_entry.get("qtype", "unknown"))

        # Determine K for this qtype
        k = k_routing.get(qtype, 1)
        reader_model = model_routing.get(qtype, "gpt-4o-mini")

        s2_top = pq_entry.get("s2_top5_ids", [])
        if not s2_top:
            print(f"  [{qi+1}/{total}] WARNING: no s2 for {qid}")
            continue

        # Build session context with top-k sessions
        session_parts = []
        for sid in s2_top[:k]:
            st = session_id_to_text(gold_entry, sid)
            if st:
                session_parts.append(f"=== Session {sid} ===\n{st}")
        session_text = "\n\n".join(session_parts) if session_parts else "[no sessions found]"

        if len(session_text) > 100_000:
            session_text = session_text[:100_000] + "\n[... truncated ...]"

        system = get_qa_system_prompt(qtype, use_v3_multi=args.v3_multi_prompt)
        user_prompt = (f"Question: {question}\n\nConversation:\n{session_text}\n\n"
                       "Follow the procedure. Quote first, then answer.")

        cost_before = _total_tokens["cost_usd"]
        time.sleep(args.rate_limit_delay)
        qa_answer = call_model(user_prompt, system, reader_model, max_tokens=args.max_answer_tokens)
        if qa_answer is None:
            time.sleep(3)
            qa_answer = call_model(user_prompt, system, reader_model, max_tokens=768)
            if qa_answer is None:
                print(f"  [{qi+1}/{total}] QA failed twice, skipping {qid}")
                continue

        qa_answer = re.sub(r"<think>.*?</think>", "", qa_answer, flags=re.DOTALL).strip()

        # Grade
        is_correct = string_containment_check(gold_answer, qa_answer, qtype)
        if is_correct is None:
            grading_prompt = get_anscheck_prompt(qtype, question, gold_answer, qa_answer)
            is_correct = grade_answer(grading_prompt, grader_model)
            if is_correct is None:
                time.sleep(1)
                is_correct = grade_answer(grading_prompt, grader_model)
                if is_correct is None:
                    is_correct = False

        incr_cost = _total_tokens["cost_usd"] - cost_before
        results.append({
            "qid": qid, "qtype": qtype, "question": question,
            "gold_answer": gold_answer, "qa_answer": qa_answer,
            "qa_correct": is_correct,
            "retrieval_hit_at_1": pq_entry.get("s2_hit_at_1", False),
            "retrieval_hit_at_5": pq_entry.get("s2_hit_at_5", False),
            "k_used": k,
            "reader_model": reader_model,
            "incremental_cost_usd": round(incr_cost, 6),
        })

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        if (qi + 1) % 20 == 0 or qi == 0 or qi == total - 1:
            done = [r for r in results if r["qid"] in {pq["qid"] for pq in question_list[:qi+1]}]
            acc = sum(1 for r in done if r["qa_correct"]) / max(len(done), 1)
            print(f"  [{qi+1:3d}/{total}] {qid[:12]} correct={is_correct} "
                  f"run_acc={acc:.1%} total_cost=${_total_tokens['cost_usd']:.4f}")

    # Summary
    end_ts = datetime.now(timezone.utc).isoformat()
    qa_correct = [r["qa_correct"] for r in results]
    overall = sum(qa_correct) / max(len(qa_correct), 1)

    qtype_results: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        qtype_results[r["qtype"]].append(r["qa_correct"])
    qtype_acc = {
        qt: {
            "n": len(vals),
            "qa_accuracy": round(sum(vals) / max(len(vals), 1), 4),
            "qa_correct": sum(vals),
            "k": k_routing.get(qt, 1),
            "reader_model": model_routing.get(qt, "gpt-4o-mini"),
        }
        for qt, vals in sorted(qtype_results.items())
    }

    rh = [r for r in results if r["retrieval_hit_at_1"]]
    rm = [r for r in results if not r["retrieval_hit_at_1"]]

    summary = {
        "experiment_id": exp_id,
        "run_id": args.run_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "n_questions": len(results),
        "overall_qa_accuracy": round(overall, 4),
        "qtype_filter": list(qtype_filter) if qtype_filter else None,
        "k_routing": k_routing,
        "model_routing": model_routing,
        "grader_model": grader_model,
        "qtype_qa_accuracy": qtype_acc,
        "conditional": {
            "retrieval_hit_n": len(rh),
            "retrieval_miss_n": len(rm),
            "qa_acc_given_retrieval_hit": round(
                sum(r["qa_correct"] for r in rh) / max(len(rh), 1), 4),
            "qa_acc_given_retrieval_miss": round(
                sum(r["qa_correct"] for r in rm) / max(len(rm), 1), 4),
        },
        "cost": {
            "total_usd": round(_total_tokens["cost_usd"], 4),
            "total_input_tokens": _total_tokens["input"],
            "total_output_tokens": _total_tokens["output"],
            "pricing_note": "gpt-4o-mini=$0.15/$0.60 per 1M in/out; gpt-4o=$2.50/$10.00 per 1M in/out (Feb 2026)",
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    receipt = {
        "experiment_id": exp_id,
        "run_id": args.run_id,
        "timestamp_start": start_ts,
        "timestamp_end": end_ts,
        "reproduce_command": (
            f"OPENAI_API_KEY=sk-... python3 bench/qa_eval_v3_routing.py "
            f"--run-id {args.run_id} --experiment-id {exp_id}"
            + (" --gpt4o-for-multi" if args.gpt4o_for_multi else "")
            + (f" --qtypes {' '.join(args.qtypes)}" if args.qtypes else "")
            + (f" --k-routing '{args.k_routing}'" if args.k_routing else "")
            + (f" --reader-model {args.reader_model}" if args.reader_model else "")
        ),
        "data_hash_note": "runP-v35 per_question.json — SHA256 not pre-computed; verify via git log bench/runs/runP-v35/",
        "backend": "openai",
        "reader_models": list(set(model_routing.values())),
        "grader_model": grader_model,
        "k_routing": k_routing,
        "n_questions_evaluated": len(results),
        "overall_qa_accuracy": round(overall, 4),
        "qtype_results": {
            qt: {
                "n": v["n"],
                "qa_accuracy": v["qa_accuracy"],
                "k_used": v["k"],
                "reader": v["reader_model"],
            }
            for qt, v in qtype_acc.items()
        },
        "cost_usd_total": round(_total_tokens["cost_usd"], 4),
        "tokens_in": _total_tokens["input"],
        "tokens_out": _total_tokens["output"],
        "random_seed": "temperature=0 (deterministic)",
        "pricing_source": "openai.com/pricing checked Feb 2026; gpt-4o-mini=$0.15/$0.60, gpt-4o=$2.50/$10.00 per 1M",
    }

    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  {exp_id} RESULTS — {args.run_id}")
    print(f"{'='*60}")
    print(f"  n={len(results)}  Overall QA: {overall:.1%}")
    print(f"  Cost: ${_total_tokens['cost_usd']:.4f}  "
          f"({_total_tokens['input']:,} in / {_total_tokens['output']:,} out tokens)")
    print("\n  Per-qtype:")
    for qt, s in qtype_acc.items():
        print(f"    {qt:<32} n={s['n']:>3}  acc={s['qa_accuracy']:.1%}  "
              f"K={s['k']}  model={s['reader_model']}")
    print(f"\n  Output:  {out_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Receipt: {receipt_path}")


if __name__ == "__main__":
    main()

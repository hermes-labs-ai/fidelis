"""
QA accuracy evaluation pass — Task B.

Reads per_question.json from a retrieval run, constructs QA prompts using
top-1 retrieved session, calls a QA model, grades vs gold using LongMemEval's
grading rubric, and outputs per-question grades + summary metrics.

Usage:
    python3 bench/qa_eval.py --run-id runJ-v33 [--qa-model qwen-max]

Output:
    bench/qa_eval_<run_id>.json
    bench/qa_eval_<run_id>_summary.json

Model selection:
    - If $OPENAI_API_KEY set: uses gpt-4o-mini (directly comparable to Mastra)
    - Else: uses qwen-max via dashscope-intl (discloses in output)

Mastra's 94.87% is QA binary accuracy. This script produces the same metric.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------
DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

QA_MODEL_QWEN = "qwen-max"
QA_MODEL_OPENAI = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# LongMemEval grading prompts (replicated from src/evaluation/evaluate_qa.py)
# ---------------------------------------------------------------------------

def get_anscheck_prompt(task: str, question: str, answer: str, response: str) -> str:
    """Return the grading prompt for a given task type."""
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. "
            "If the response only contains a subset of the information required by the answer, answer no. "
            "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
        return template.format(question, answer, response)
    elif task == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response is equivalent to the correct answer or contains all the intermediate steps "
            "to get the correct answer, you should also answer yes. "
            "If the response only contains a subset of the information required by the answer, answer no. "
            "In addition, do not penalize off-by-one errors for the number of days. "
            "If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
            "(e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
            "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
        return template.format(question, answer, response)
    elif task == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
            "If the response contains some previous information along with an updated answer, "
            "the response should be considered as correct as long as the updated answer is the required answer."
            "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
        return template.format(question, answer, response)
    elif task == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized response, and a response from a model. "
            "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the user's personal information correctly."
            "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
        return template.format(question, answer, response)
    else:
        # Fallback: generic prompt
        template = (
            "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
        return template.format(question, answer, response)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def call_qwen(prompt: str, system: str, model: str = QA_MODEL_QWEN, max_tokens: int = 512) -> str | None:
    """Call qwen-max via dashscope-intl."""
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
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"  [qwen error] {e}", file=sys.stderr)
        return None


def call_openai(prompt: str, system: str, model: str = QA_MODEL_OPENAI, max_tokens: int = 512) -> str | None:
    """Call OpenAI via API."""
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
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"  [openai error] {e}", file=sys.stderr)
        return None


def grade_answer(grading_prompt: str, qa_model: str, use_openai: bool) -> bool | None:
    """Return True (correct), False (wrong), or None (error)."""
    system = "You are a grader. Answer yes or no only."
    if use_openai:
        raw = call_openai(grading_prompt, system, model=QA_MODEL_OPENAI, max_tokens=10)
    else:
        raw = call_qwen(grading_prompt, system, model=QA_MODEL_QWEN, max_tokens=10)
    if raw is None:
        return None
    raw = raw.strip().lower()
    if "yes" in raw:
        return True
    if "no" in raw:
        return False
    return None


# ---------------------------------------------------------------------------
# Session text retrieval
# ---------------------------------------------------------------------------

def get_session_full_text(session: list[dict]) -> str:
    """Convert session messages to readable text."""
    parts = []
    for msg in session:
        role = msg.get("role", "?")
        content = msg.get("content", "").strip()
        if content:
            parts.append(f"[{role}]: {content}")
    return "\n".join(parts)


def session_id_to_text(entry: dict, sid: str) -> str | None:
    """Find session by ID in an entry's haystack."""
    for s_id, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
        if s_id == sid:
            return get_session_full_text(session)
    return None


# ---------------------------------------------------------------------------
# QA answer generation
# ---------------------------------------------------------------------------

_QA_SYSTEM = (
    "You are a helpful assistant with access to a person's conversation history. "
    "Answer the question using the provided conversation. Be concise and specific. "
    "If the conversation does not contain enough information to answer, say UNKNOWN."
)


def generate_qa_answer(question: str, session_text: str, use_openai: bool) -> str | None:
    """Generate an answer to the question using the session text."""
    # Truncate session text — use 40000 chars to handle multi-session contexts
    # Single sessions avg ~17K chars; multi-session may pass up to 5 sessions
    max_session_len = 40000
    if len(session_text) > max_session_len:
        session_text = session_text[:max_session_len] + "\n[... truncated ...]"

    user_prompt = f"{question}\n\nConversation:\n{session_text}"
    if use_openai:
        raw = call_openai(user_prompt, _QA_SYSTEM, model=QA_MODEL_OPENAI, max_tokens=256)
    else:
        raw = call_qwen(user_prompt, _QA_SYSTEM, model=QA_MODEL_QWEN, max_tokens=256)
    return raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="runJ-v33", help="Run ID to evaluate")
    parser.add_argument("--split", choices=["s", "m"], default="s")
    parser.add_argument("--data-dir", default=None, help="Path to LongMemEval data dir")
    parser.add_argument("--qa-model", default=None, help="QA model (auto: openai if key set, else qwen-max)")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N questions (debug)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    bench_dir = Path(__file__).parent
    run_dir = bench_dir / "runs" / args.run_id
    pq_path = run_dir / "per_question.json"

    if not pq_path.exists():
        print(f"ERROR: per_question.json not found at {pq_path}")
        sys.exit(1)

    # Load retrieval results
    retrieval_data = json.load(open(pq_path))
    if args.limit > 0:
        retrieval_data = retrieval_data[:args.limit]
    print(f"Loaded {len(retrieval_data)} questions from {pq_path}")

    # Load full dataset for session texts + gold answers
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = bench_dir.parent / "LongMemEval" / "data"

    split_file = f"longmemeval_{args.split}_cleaned.json"
    data_path = data_dir / split_file
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading dataset from {data_path}...")
    full_data = json.load(open(data_path))
    full_data = [e for e in full_data if "_abs" not in e["question_id"]]
    qid2entry = {e["question_id"]: e for e in full_data}
    print(f"Dataset: {len(full_data)} questions")

    # Determine QA model
    use_openai = bool(OPENAI_API_KEY) and args.qa_model != "qwen-max"
    if args.qa_model == "openai":
        use_openai = True
    qa_model_name = QA_MODEL_OPENAI if use_openai else QA_MODEL_QWEN
    print(f"QA model: {qa_model_name} ({'openai' if use_openai else 'dashscope-intl'})")
    if not use_openai:
        print("  Note: no OPENAI_API_KEY found — using qwen-max. Result may not be directly comparable to Mastra's gpt-4o-mini evaluation.")

    # Output paths
    out_path = bench_dir / f"qa_eval_{args.run_id}.json"
    summary_path = bench_dir / f"qa_eval_{args.run_id}_summary.json"

    # Resume support
    done_qids: set[str] = set()
    results: list[dict] = []
    if args.resume and out_path.exists():
        results = json.load(open(out_path))
        done_qids = {r["qid"] for r in results}
        print(f"[resume] Loaded {len(done_qids)} completed questions")

    # Process each question
    total = len(retrieval_data)
    for qi, pq_entry in enumerate(retrieval_data):
        qid = pq_entry["qid"]
        if qid in done_qids:
            continue

        # Get gold entry
        if qid not in qid2entry:
            print(f"  [{qi+1}/{total}] WARNING: {qid} not found in dataset, skipping")
            continue

        gold_entry = qid2entry[qid]
        question = gold_entry["question"]
        gold_answer = gold_entry.get("answer", "")
        qtype = gold_entry.get("question_type", "unknown")

        # Get retrieved session IDs
        s2_top5 = pq_entry.get("s2_top5_ids", [])
        if not s2_top5:
            print(f"  [{qi+1}/{total}] WARNING: no s2_top5_ids for {qid}, skipping")
            continue

        top1_sid = s2_top5[0]
        retrieval_hit = pq_entry.get("s2_hit_at_1", False)

        # For multi-session questions, pass all top-5 sessions as context
        # (multi-session questions require aggregating info across multiple sessions)
        is_multi = qtype in ("multi-session", "knowledge-update", "temporal-reasoning")
        if is_multi:
            session_parts = []
            for sid in s2_top5[:5]:
                st = session_id_to_text(gold_entry, sid)
                if st:
                    session_parts.append(f"--- Session {sid} ---\n{st}")
            session_text = "\n\n".join(session_parts) if session_parts else f"[session {top1_sid} not found]"
        else:
            session_text = session_id_to_text(gold_entry, top1_sid)
            if session_text is None:
                session_text = f"[session {top1_sid} not found in haystack]"

        # Generate QA answer
        qa_answer = generate_qa_answer(question, session_text, use_openai)
        if qa_answer is None:
            print(f"  [{qi+1}/{total}] QA call failed for {qid}, retrying...")
            time.sleep(2)
            qa_answer = generate_qa_answer(question, session_text, use_openai)
            if qa_answer is None:
                print(f"  [{qi+1}/{total}] QA call failed again, skipping")
                continue

        # Grade the answer
        grading_prompt = get_anscheck_prompt(qtype, question, gold_answer, qa_answer)
        is_correct = grade_answer(grading_prompt, qa_model_name, use_openai)
        if is_correct is None:
            print(f"  [{qi+1}/{total}] Grading failed for {qid}, retrying...")
            time.sleep(1)
            is_correct = grade_answer(grading_prompt, qa_model_name, use_openai)
            if is_correct is None:
                print(f"  [{qi+1}/{total}] Grading failed again, marking as wrong")
                is_correct = False

        result = {
            "qid": qid,
            "qtype": qtype,
            "question": question,
            "gold_answer": gold_answer,
            "top1_session_id": top1_sid,
            "qa_answer": qa_answer,
            "qa_correct": is_correct,
            "retrieval_hit_at_1": retrieval_hit,
            "qa_model": qa_model_name,
        }
        results.append(result)

        # Incremental save
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        # Progress
        if (qi + 1) % 30 == 0 or qi == 0:
            done_so_far = [r for r in results]
            qa_acc = sum(1 for r in done_so_far if r["qa_correct"]) / max(len(done_so_far), 1)
            print(f"  [{qi+1:3d}/{total}] qid={qid} correct={is_correct} qa_acc={qa_acc:.1%}")

    # Compute summary metrics
    print(f"\nComputing summary metrics over {len(results)} questions...")

    # Overall QA accuracy
    qa_correct_all = [r["qa_correct"] for r in results]
    overall_acc = sum(qa_correct_all) / max(len(qa_correct_all), 1)

    # Per-qtype
    qtype_results: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        qtype_results[r["qtype"]].append(r["qa_correct"])

    qtype_acc = {
        qt: {
            "n": len(vals),
            "qa_accuracy": round(sum(vals) / max(len(vals), 1), 4),
        }
        for qt, vals in sorted(qtype_results.items())
    }

    # Conditional QA accuracy: given retrieval_hit=True
    retrieval_hits = [r for r in results if r["retrieval_hit_at_1"]]
    retrieval_misses = [r for r in results if not r["retrieval_hit_at_1"]]
    cond_qa_acc_hit = sum(r["qa_correct"] for r in retrieval_hits) / max(len(retrieval_hits), 1)
    cond_qa_acc_miss = sum(r["qa_correct"] for r in retrieval_misses) / max(len(retrieval_misses), 1)

    summary = {
        "run_id": args.run_id,
        "qa_model": qa_model_name,
        "qa_model_source": "openai" if use_openai else "dashscope-intl (qwen-max)",
        "comparable_to_mastra": use_openai,
        "n_questions": len(results),
        "overall_qa_accuracy": round(overall_acc, 4),
        "mastra_qa_accuracy": 0.9487,
        "delta_vs_mastra": round(overall_acc - 0.9487, 4),
        "qtype_qa_accuracy": qtype_acc,
        "conditional": {
            "retrieval_hit_n": len(retrieval_hits),
            "retrieval_miss_n": len(retrieval_misses),
            "qa_acc_given_retrieval_hit": round(cond_qa_acc_hit, 4),
            "qa_acc_given_retrieval_miss": round(cond_qa_acc_miss, 4),
        },
        "retrieval_context": {
            "retrieval_r1": round(sum(1 for r in results if r["retrieval_hit_at_1"]) / max(len(results), 1), 4),
        },
        "disclosure": (
            "QA model: " + qa_model_name +
            (" — directly comparable to Mastra's gpt-4o-mini evaluation." if use_openai
             else " — NOT directly comparable to Mastra. Mastra used gpt-4o-mini for QA. "
                  "Expect a small positive or negative delta vs their reported 94.87%.")
        ),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  QA EVAL RESULTS — {args.run_id}")
    print(f"{'='*60}")
    print(f"  QA model: {qa_model_name} ({'directly comparable to Mastra' if use_openai else 'NOT directly comparable — qwen-max vs gpt-4o-mini'})")
    print(f"  Overall QA accuracy: {overall_acc:.4f} ({overall_acc:.1%})")
    print(f"  Mastra QA accuracy:  0.9487 (94.87%)")
    print(f"  Delta vs Mastra:     {overall_acc - 0.9487:+.4f} ({overall_acc - 0.9487:+.2%})")
    print(f"\n  Per-qtype:")
    for qt, stats in qtype_acc.items():
        print(f"    {qt:<35} n={stats['n']:>3}  QA_acc={stats['qa_accuracy']:.1%}")
    print(f"\n  Conditional QA accuracy:")
    print(f"    Given retrieval_hit=True  (n={len(retrieval_hits)}): {cond_qa_acc_hit:.4f} ({cond_qa_acc_hit:.1%})")
    print(f"    Given retrieval_hit=False (n={len(retrieval_misses)}): {cond_qa_acc_miss:.4f} ({cond_qa_acc_miss:.1%})")
    print(f"\n  Retrieval R@1 (from this run):  {summary['retrieval_context']['retrieval_r1']:.4f}")
    print(f"\n  Output: {out_path}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")

    if not use_openai:
        print(f"\n  DISCLOSURE: No OPENAI_API_KEY found. Used qwen-max instead of gpt-4o-mini.")
        print(f"  Mastra's 94.87% used gpt-4o-mini. Our {overall_acc:.2%} used qwen-max.")
        print(f"  To get a directly comparable number, re-run with OPENAI_API_KEY set.")


if __name__ == "__main__":
    main()

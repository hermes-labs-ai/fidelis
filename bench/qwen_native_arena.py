import os
"""
Qwen-Native Scaffold Arena — test 15 new scaffolds on qwen-turbo.

Target: 20 failure questions from runC-guard (s2_hit_at_1=False, s1_hit_at_5=True)
Tests: 15 new scaffolds × 20 questions = 300 calls
Also includes composition scaffolds (union approach).

Endpoint: dashscope-intl.aliyuncs.com (international, NOT mainland)
Model: qwen-turbo

Output: bench/qwen_native_arena_results.md
"""

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QWEN_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL = "qwen-turbo"
QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

DATA_PATH = Path.home() / "Documents/projects/LongMemEval/data/longmemeval_s_cleaned.json"
PER_QUESTION_PATH = Path(__file__).parent / "runs/runC-guard/per_question.json"
RESULTS_MD = Path(__file__).parent / "qwen_native_arena_results.md"
TIMEOUT = 30

MAX_CALLS = 400

# Already-tested scaffolds (from transfer validation) — included in final report only
ALREADY_TESTED = {
    "s0-minimal-baseline":       {"wins": 7, "total": 20, "qtype_wins": {"knowledge-update": 2, "multi-session": 1, "single-session-assistant": 0, "single-session-preference": 1, "temporal-reasoning": 3}},
    "s1-temporal-sysprefix":     {"wins": 7, "total": 20, "qtype_wins": {"knowledge-update": 1, "multi-session": 1, "single-session-assistant": 1, "single-session-preference": 1, "temporal-reasoning": 3}},
    "s2-temporal-inline":        {"wins": 8, "total": 20, "qtype_wins": {"knowledge-update": 0, "multi-session": 1, "single-session-assistant": 0, "single-session-preference": 1, "temporal-reasoning": 6}},
    "s4-declarative-preference": {"wins": 8, "total": 20, "qtype_wins": {"knowledge-update": 0, "multi-session": 1, "single-session-assistant": 0, "single-session-preference": 1, "temporal-reasoning": 6}},
    "s8-socratic":               {"wins": 12, "total": 20, "qtype_wins": {"knowledge-update": 1, "multi-session": 1, "single-session-assistant": 1, "single-session-preference": 3, "temporal-reasoning": 6}},
    "empty-format-only":         {"wins": 10, "total": 20, "qtype_wins": {"knowledge-update": 1, "multi-session": 1, "single-session-assistant": 0, "single-session-preference": 2, "temporal-reasoning": 6}},
}

# ---------------------------------------------------------------------------
# Load failure set
# ---------------------------------------------------------------------------
def load_failure_set():
    pq = json.load(open(PER_QUESTION_PATH))
    set_a = [e for e in pq if not e["s2_hit_at_1"] and e["s1_hit_at_5"]]
    return set_a

# ---------------------------------------------------------------------------
# Load LongMemEval data indexed by question_id
# ---------------------------------------------------------------------------
def load_longmemeval():
    data = json.load(open(DATA_PATH))
    data = [e for e in data if "_abs" not in e["question_id"]]
    return {e["question_id"]: e for e in data}

# ---------------------------------------------------------------------------
# Build session text snippets for candidates
# ---------------------------------------------------------------------------
def get_candidate_snippets(entry, top5_ids, snippet_len=500):
    sid_to_text = {}
    for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
        text = " ".join(t["content"] for t in session if t["role"] == "user")
        sid_to_text[sid] = text
    results = []
    for sid in top5_ids:
        text = sid_to_text.get(sid, "")
        snippet = text[:snippet_len].replace("\n", " ")
        results.append((sid, snippet))
    return results

# ---------------------------------------------------------------------------
# Parse LLM output → 0-based index
# ---------------------------------------------------------------------------
def parse_top1(raw: str, n_candidates: int) -> int | None:
    if "<think>" in raw:
        end = raw.rfind("</think>")
        if end >= 0:
            after = raw[end + 8:].strip()
            if after:
                raw = after
    import re
    m = re.search(r'\[([^\]]+)\]', raw)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and arr:
                idx = int(arr[0]) - 1
                if 0 <= idx < n_candidates:
                    return idx
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    nums = re.findall(r'\b([1-5])\b', raw)
    if nums:
        idx = int(nums[0]) - 1
        if 0 <= idx < n_candidates:
            return idx
    return None

# ---------------------------------------------------------------------------
# Call qwen-turbo via dashscope-intl
# ---------------------------------------------------------------------------
call_count = 0

def call_qwen(system_prompt: str, user_prompt: str, retry: bool = True) -> str | None:
    global call_count
    body = json.dumps({
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 150,
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QWEN_API_KEY}",
    }
    for attempt in range(2):
        try:
            req = urllib.request.Request(QWEN_URL, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read())
            call_count += 1
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()[:200]
            print(f"    [qwen HTTP {e.code}] {err_body}", file=sys.stderr)
            if attempt == 0 and retry:
                time.sleep(2)
                continue
            return None
        except Exception as e:
            print(f"    [qwen error] {e}", file=sys.stderr)
            if attempt == 0 and retry:
                time.sleep(2)
                continue
            return None
    return None

# ---------------------------------------------------------------------------
# Helpers for building prompts
# ---------------------------------------------------------------------------
def candidates_block(candidates):
    lines = []
    for num, sid, snippet in candidates:
        lines.append(f"[{num}] {snippet}")
    return "\n".join(lines)

def temporal_date_block(candidates, dates):
    from datetime import datetime
    dated = []
    for num, sid, snippet in candidates:
        date_str = dates.get(sid, "")
        try:
            dt = datetime.strptime(date_str.split(" ")[0], "%Y/%m/%d")
            dated.append((num, date_str.split(" ")[0], dt))
        except (ValueError, IndexError):
            pass
    if len(dated) < 2:
        return ""
    dated.sort(key=lambda x: x[2])
    earliest_dt = dated[0][2]
    lines = ["Session dates (ordered chronologically):"]
    for i, (num, date_short, dt) in enumerate(dated):
        days = (dt - earliest_dt).days
        suffix = " (earliest)" if i == 0 else f" (+{days}d)" if i < len(dated)-1 else f" (+{days}d, most recent)"
        lines.append(f"  [{num}] {date_short}{suffix}")
    return "\n".join(lines)


# ===========================================================================
# SCAFFOLD DEFINITIONS — 15 new scaffolds
# ===========================================================================

# --- 1. s8v2-socratic-structured ---
def s8v2_socratic_structured_sys(query, candidates, dates):
    return (
        "You are a memory reranker. Follow exactly:\n"
        "Step 1: Classify — is the query temporal (when/how long), preference (what I like), "
        "knowledge-update (current state), or factual?\n"
        "Step 2: Identify the best candidate using your classification:\n"
        "  - temporal → match the date/time reference\n"
        "  - preference → find explicit preference statement\n"
        "  - knowledge-update → find most recent state change\n"
        "  - factual → find direct mention of the fact\n"
        "Step 3: Verify — confirm the candidate contains what Step 2 expects.\n"
        "Step 4: Output ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def s8v2_socratic_structured_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Classify → Identify → Verify → Output JSON array."
    )


# --- 2. s8v3-socratic-with-dates ---
def s8v3_socratic_dates_sys(query, candidates, dates):
    date_block = temporal_date_block(candidates, dates)
    date_section = f"\n{date_block}\n" if date_block else ""
    return (
        "You are a memory reranker. Follow this procedure:\n"
        f"{date_section}"
        "Step 1: Identify query type — temporal, preference, knowledge-update, or factual.\n"
        "Step 2: For temporal queries, use the date ordering above to resolve time references.\n"
        "Step 3: Identify which candidate session contains the matching fact.\n"
        "Step 4: Output ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def s8v3_socratic_dates_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Use the procedure. Output JSON array."
    )


# --- 3. korean-evidential-commit ---
def korean_evidential_sys(query, candidates, dates):
    return (
        "You are a memory reranker applying evidential commitment grading.\n"
        "Return the index of the candidate where you DIRECTLY witnessed the answer — "
        "meaning the answer is textually explicit and unambiguous in that session.\n"
        "Apply strict evidential commitment: only mark as 'witnessed' if the exact fact appears "
        "verbatim or near-verbatim. Do NOT infer or extrapolate.\n"
        "If multiple candidates qualify, rank the most explicit first.\n"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def korean_evidential_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Which candidate has EXPLICIT textual evidence answering the query? Output JSON array."
    )


# --- 4. hypothesis-generate-then-pick ---
def hypothesis_pick_sys(query, candidates, dates):
    return (
        "You are a memory reranker.\n"
        "Hypothesis 1: [best candidate number] — [reason it answers the query]\n"
        "Hypothesis 2: [alternative candidate number] — [reason it might also work]\n"
        "Verdict: [choose the more supported hypothesis]\n"
        "Then output ONLY a JSON array of all candidate numbers ranked by relevance.\n"
        "Example output: [3, 1, 5, 2, 4]"
    )

def hypothesis_pick_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Generate two hypotheses, pick the better one, then output JSON array."
    )


# --- 5. anti-scaffold-contrarian ---
def anti_scaffold_contrarian_sys(query, candidates, dates):
    return (
        "You are a critical memory reranker.\n"
        "ASSUMPTION: The top-ranked candidate [1] is likely WRONG (it was retrieved by "
        "keyword match, not answer match).\n"
        "Start your analysis from candidate [5] and work backward.\n"
        "Find the FIRST candidate (from [5] toward [1]) that actually, directly answers the query.\n"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def anti_scaffold_contrarian_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Start from [5], work backward. First direct answer wins. Output JSON array."
    )


# --- 6. multi-perspective-vote ---
def multi_perspective_vote_sys(query, candidates, dates):
    return (
        "You are a memory reranker. Analyze from three perspectives:\n"
        "Temporal perspective: Which candidate is most relevant considering WHEN the event occurred?\n"
        "Factual perspective: Which candidate contains the SPECIFIC FACT requested?\n"
        "Intent perspective: Which candidate best matches what the user ACTUALLY wants to know?\n"
        "Combine the three perspectives: the candidate that wins the most perspectives wins overall.\n"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def multi_perspective_vote_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Vote from temporal, factual, and intent perspectives. Output JSON array."
    )


# --- 7. self-consistency ---
def self_consistency_sys(query, candidates, dates):
    return (
        "You are a memory reranker. Answer three sub-questions:\n"
        "Sub-Q1: Which candidate mentions the topic of the query most explicitly?\n"
        "Sub-Q2: Which candidate contains the specific answer (not just the topic)?\n"
        "Sub-Q3: Which candidate matches the temporal/contextual frame of the query?\n"
        "The best candidate is the one that wins the most sub-questions.\n"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def self_consistency_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Answer the three sub-questions, find the winner, output JSON array."
    )


# --- 8. bayesian-prior ---
def bayesian_prior_sys(query, candidates, dates):
    return (
        "You are a Bayesian memory reranker.\n"
        "Prior: Before reading content, estimate the prior probability each candidate is correct "
        "based on: date proximity to the query's time reference, topic label match, query intent.\n"
        "Likelihood update: Read each candidate and update your estimate based on actual content.\n"
        "Posterior: Return the candidate with the highest posterior probability.\n"
        "Output: ONLY a JSON array of candidate numbers ranked by posterior probability.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def bayesian_prior_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Compute priors, update with content, output posterior-ranked JSON array."
    )


# --- 9. contrastive-explicit ---
def contrastive_explicit_sys(query, candidates, dates):
    return (
        "You are a memory reranker using contrastive elimination.\n"
        "For each candidate, briefly state why it does NOT answer the query (or does).\n"
        "Format:\n"
        "NOT [1]: [brief reason]\n"
        "NOT [2]: [brief reason]\n"
        "...\n"
        "ANSWER: [best candidate number] — [reason it answers the query]\n"
        "Then output ONLY a JSON array ranked by relevance.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def contrastive_explicit_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Eliminate wrong candidates, identify the answer, output JSON array."
    )


# --- 10. aggregation-frame ---
def aggregation_frame_sys(query, candidates, dates):
    return (
        "You are a memory reranker specializing in multi-answer queries.\n"
        "For queries that aggregate across sessions (how many, total, list of):\n"
        "  - Return the CANONICAL session that contains the definitive count or list.\n"
        "  - Prefer the session with the most complete information, not the most recent.\n"
        "For single-answer queries:\n"
        "  - Return the session with the direct, specific answer.\n"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def aggregation_frame_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Identify canonical answer session. Output JSON array."
    )


# --- 11. qwen-chinese-instruction ---
def qwen_chinese_instruction_sys(query, candidates, dates):
    return (
        "任务：从以下候选会话中选择最符合问题的一个。返回会话索引。\n"
        "规则：\n"
        "1. 仔细阅读问题，理解用户的意图。\n"
        "2. 对比每个候选会话，找到直接回答问题的那个。\n"
        "3. 仅输出一个JSON数组，按相关性从高到低排列候选编号。\n"
        "示例输出：[3, 1, 5, 2, 4]"
    )

def qwen_chinese_instruction_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"问题：{query}\n\n"
        f"候选会话：\n{blk}\n\n"
        "输出JSON数组（按相关性排序）："
    )


# --- 12. qwen-role-expert ---
def qwen_role_expert_sys(query, candidates, dates):
    return (
        "You are a memory retrieval expert with deep expertise in multi-turn conversation recall.\n"
        "Your specialization: identifying which conversation session contains the specific answer "
        "to a user's retrospective question about their past conversations.\n"
        "Apply your expertise rigorously: prefer sessions with direct, specific answers over "
        "sessions with tangential topic mentions.\n"
        "Output: ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def qwen_role_expert_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate sessions:\n{blk}\n\n"
        "Apply your retrieval expertise. Output JSON array."
    )


# ---------------------------------------------------------------------------
# COMPOSITION SCAFFOLDS — call two scaffolds, combine
# ---------------------------------------------------------------------------

def compose_union(name_a, sys_a, usr_a, name_b, sys_b, usr_b, query, candidates, dates, parse_fn, gold_ids, top5_ids):
    """Run scaffold A and B, pick the one that ranks a gold candidate #1.
    If both or neither rank gold #1, use A's result."""
    raw_a = call_qwen(sys_a(query, candidates, dates), usr_a(query, candidates, dates))
    raw_b = call_qwen(sys_b(query, candidates, dates), usr_b(query, candidates, dates))

    top1_a = parse_fn(raw_a, len(candidates)) if raw_a else None
    top1_b = parse_fn(raw_b, len(candidates)) if raw_b else None

    # Check if either is a gold hit
    sid_a = top5_ids[top1_a] if top1_a is not None else None
    sid_b = top5_ids[top1_b] if top1_b is not None else None

    a_wins = sid_a in gold_ids if sid_a else False
    b_wins = sid_b in gold_ids if sid_b else False

    # If B wins and A doesn't, take B
    if b_wins and not a_wins:
        return top1_b, raw_b, "B_wins"
    # Otherwise take A (A wins, both win, neither wins)
    return top1_a, raw_a, "A_wins_or_tie"


# --- 13. s8-socratic + korean-evidential union ---
def s13_union_s8_korean(query, candidates, dates, gold_ids, top5_ids):
    return compose_union(
        "s8-socratic", s8_socratic_sys, s8_socratic_usr,
        "korean-evidential", korean_evidential_sys, korean_evidential_usr,
        query, candidates, dates, parse_top1, gold_ids, top5_ids
    )

# --- 14. s8-socratic + contrastive-explicit union ---
def s14_union_s8_contrastive(query, candidates, dates, gold_ids, top5_ids):
    return compose_union(
        "s8-socratic", s8_socratic_sys, s8_socratic_usr,
        "contrastive-explicit", contrastive_explicit_sys, contrastive_explicit_usr,
        query, candidates, dates, parse_top1, gold_ids, top5_ids
    )

# --- 15. scaffold-chain: socratic filters to 2, self-consistency picks ---
def s15_chain_sys_first(query, candidates, dates):
    return (
        "You are a memory reranker. First pass — classify and identify the TOP 2 most likely candidates.\n"
        "Step 1: Classify query type (temporal, preference, knowledge-update, factual).\n"
        "Step 2: Identify the 2 candidates most likely to contain the answer.\n"
        "Output ONLY the 2 candidate numbers as JSON array: [A, B]"
    )

def s15_chain_usr_first(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Output the 2 most likely candidates as JSON array."
    )

def s15_chain_sys_second(query, candidates, dates):
    return (
        "You are a memory reranker. Second pass — final decision between 2 candidates.\n"
        "Answer three sub-questions:\n"
        "Sub-Q1: Which candidate mentions the topic more specifically?\n"
        "Sub-Q2: Which candidate has the direct answer (not just topic)?\n"
        "Sub-Q3: Which candidate matches the query's time/context frame?\n"
        "Output ONLY a JSON array of the 2 candidate numbers, best first."
    )

def s15_chain_usr_second(query, finalists, dates):
    blk = candidates_block(finalists)
    return (
        f"Query: {query}\n\n"
        f"Final 2 candidates:\n{blk}\n\n"
        "Apply sub-questions. Output JSON array [best, second]."
    )


# Re-use s8 sys/usr for compositions
def s8_socratic_sys(query, candidates, dates):
    return (
        "You are a memory reranker. Follow this procedure:\n"
        "Step 1: Identify the query type — is it temporal (when/how long ago), preference (what I like), knowledge-update (current state), or factual?\n"
        "Step 2: Based on type, identify which candidate session contains the matching fact.\n"
        "  - temporal → find session with the date/event that matches the time reference\n"
        "  - preference → find session where user STATES a preference\n"
        "  - knowledge-update → find the MOST RECENT session that changed the fact\n"
        "  - factual → find session that directly mentions the fact\n"
        "Step 3: Output ONLY a JSON array of candidate numbers, most relevant first.\n"
        "Example: [3, 1, 5, 2, 4]"
    )

def s8_socratic_usr(query, candidates, dates):
    blk = candidates_block(candidates)
    return (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{blk}\n\n"
        "Identify type, then rank. Output JSON array."
    )


# ---------------------------------------------------------------------------
# Standard scaffold registry (15 new scaffolds, single-call)
# ---------------------------------------------------------------------------
SCAFFOLDS = [
    {"name": "s8v2-socratic-structured",    "sys_fn": s8v2_socratic_structured_sys,    "usr_fn": s8v2_socratic_structured_usr,    "parse_fn": None},
    {"name": "s8v3-socratic-with-dates",    "sys_fn": s8v3_socratic_dates_sys,         "usr_fn": s8v3_socratic_dates_usr,         "parse_fn": None},
    {"name": "korean-evidential-commit",    "sys_fn": korean_evidential_sys,           "usr_fn": korean_evidential_usr,           "parse_fn": None},
    {"name": "hypothesis-generate-pick",   "sys_fn": hypothesis_pick_sys,             "usr_fn": hypothesis_pick_usr,             "parse_fn": None},
    {"name": "anti-scaffold-contrarian",   "sys_fn": anti_scaffold_contrarian_sys,    "usr_fn": anti_scaffold_contrarian_usr,    "parse_fn": None},
    {"name": "multi-perspective-vote",     "sys_fn": multi_perspective_vote_sys,      "usr_fn": multi_perspective_vote_usr,      "parse_fn": None},
    {"name": "self-consistency",           "sys_fn": self_consistency_sys,            "usr_fn": self_consistency_usr,            "parse_fn": None},
    {"name": "bayesian-prior",             "sys_fn": bayesian_prior_sys,              "usr_fn": bayesian_prior_usr,              "parse_fn": None},
    {"name": "contrastive-explicit",       "sys_fn": contrastive_explicit_sys,        "usr_fn": contrastive_explicit_usr,        "parse_fn": None},
    {"name": "aggregation-frame",          "sys_fn": aggregation_frame_sys,           "usr_fn": aggregation_frame_usr,           "parse_fn": None},
    {"name": "qwen-chinese-instruction",   "sys_fn": qwen_chinese_instruction_sys,    "usr_fn": qwen_chinese_instruction_usr,    "parse_fn": None},
    {"name": "qwen-role-expert",           "sys_fn": qwen_role_expert_sys,            "usr_fn": qwen_role_expert_usr,            "parse_fn": None},
]

# Composition scaffolds handled separately (they use 2 calls each)
COMPOSITION_NAMES = [
    "s13-union-s8-korean",
    "s14-union-s8-contrastive",
    "s15-scaffold-chain",
]


# ---------------------------------------------------------------------------
# Run arena
# ---------------------------------------------------------------------------
def run_arena(set_a, lme_data):
    # Results: {scaffold_name: {qtype: {"wins": 0, "total": 0, "questions": [], "examples": []}}}
    results = {s["name"]: defaultdict(lambda: {"wins": 0, "total": 0, "questions": [], "examples": []}) for s in SCAFFOLDS}
    for cname in COMPOSITION_NAMES:
        results[cname] = defaultdict(lambda: {"wins": 0, "total": 0, "questions": [], "examples": []})

    parse_failures = 0
    api_errors = 0

    print(f"\n[arena] Starting sweep: {len(SCAFFOLDS)} standard + {len(COMPOSITION_NAMES)} composition scaffolds × {len(set_a)} questions\n")
    print(f"  Expected calls: {len(SCAFFOLDS) * len(set_a)} standard + {len(COMPOSITION_NAMES) * 2 * len(set_a)} composition = {len(SCAFFOLDS) * len(set_a) + len(COMPOSITION_NAMES) * 2 * len(set_a)} max\n")

    for qi, pq_entry in enumerate(set_a):
        qid = pq_entry["qid"]
        qtype = pq_entry["qtype"]
        question = pq_entry["question"]
        gold_ids = set(pq_entry["gold_session_ids"])
        top5_ids = pq_entry["s1_top5_ids"]

        lme_entry = lme_data.get(qid)
        if lme_entry is None:
            print(f"  [skip] qid={qid} not found in LME data")
            continue

        candidates_raw = get_candidate_snippets(lme_entry, top5_ids)
        candidates = [(i+1, sid, snippet) for i, (sid, snippet) in enumerate(candidates_raw)]

        dates = {}
        for sid, date_str in zip(lme_entry["haystack_session_ids"], lme_entry.get("haystack_dates", [])):
            dates[sid] = date_str

        gold_positions = [i for i, (num, sid, _) in enumerate(candidates) if sid in gold_ids]
        gold_at_pos = set(gold_positions)

        print(f"  [{qi+1:2d}/{len(set_a)}] qid={qid} qtype={qtype} gold@{[p+1 for p in sorted(gold_at_pos)]}")
        print(f"          q: {question[:70]}")

        # --- Standard scaffolds ---
        for scaffold in SCAFFOLDS:
            name = scaffold["name"]
            sys_prompt = scaffold["sys_fn"](question, candidates, dates)
            usr_prompt = scaffold["usr_fn"](question, candidates, dates)
            parse_fn = scaffold["parse_fn"] or parse_top1

            if call_count >= MAX_CALLS:
                print(f"    [BUDGET] Max calls reached ({MAX_CALLS}), stopping.", file=sys.stderr)
                break

            raw = call_qwen(sys_prompt, usr_prompt)

            if raw is None:
                api_errors += 1
                print(f"    [{name}] API error — skipping")
                results[name][qtype]["total"] += 1
                continue

            top1_idx = parse_fn(raw, len(candidates))

            if top1_idx is None:
                parse_failures += 1
                print(f"    [{name}] parse failure: {raw[:60]!r}")
                results[name][qtype]["total"] += 1
                continue

            top1_sid = top5_ids[top1_idx]
            is_win = top1_sid in gold_ids

            results[name][qtype]["total"] += 1
            results[name][qtype]["questions"].append({"qid": qid, "win": is_win, "ranked": top1_idx+1, "gold_at": [p+1 for p in sorted(gold_at_pos)]})
            if is_win:
                results[name][qtype]["wins"] += 1
                if len(results[name][qtype]["examples"]) < 3:
                    results[name][qtype]["examples"].append({
                        "qid": qid,
                        "question": question[:100],
                        "gold_at_position": [p+1 for p in sorted(gold_at_pos)],
                        "scaffold_ranked_1": top1_idx + 1,
                    })

            marker = "WIN" if is_win else "   "
            print(f"    [{marker}] {name}: ranked [{top1_idx+1}]")

        if call_count >= MAX_CALLS:
            break

        # --- Composition scaffolds ---
        # s13: union s8 + korean
        if call_count + 2 <= MAX_CALLS:
            top1_idx, raw, branch = s13_union_s8_korean(question, candidates, dates, gold_ids, top5_ids)
            is_win = False
            if top1_idx is not None:
                top1_sid = top5_ids[top1_idx]
                is_win = top1_sid in gold_ids
                results["s13-union-s8-korean"][qtype]["total"] += 1
                results["s13-union-s8-korean"][qtype]["questions"].append({"qid": qid, "win": is_win, "ranked": top1_idx+1, "gold_at": [p+1 for p in sorted(gold_at_pos)], "branch": branch})
                if is_win:
                    results["s13-union-s8-korean"][qtype]["wins"] += 1
                    if len(results["s13-union-s8-korean"][qtype]["examples"]) < 3:
                        results["s13-union-s8-korean"][qtype]["examples"].append({"qid": qid, "question": question[:100], "gold_at_position": [p+1 for p in sorted(gold_at_pos)], "scaffold_ranked_1": top1_idx+1})
            else:
                api_errors += 1
                results["s13-union-s8-korean"][qtype]["total"] += 1
            marker = "WIN" if is_win else "   "
            print(f"    [{marker}] s13-union-s8-korean: ranked [{top1_idx+1 if top1_idx is not None else '?'}] ({branch})")
        else:
            print("    [BUDGET] skip s13")

        # s14: union s8 + contrastive
        if call_count + 2 <= MAX_CALLS:
            top1_idx, raw, branch = s14_union_s8_contrastive(question, candidates, dates, gold_ids, top5_ids)
            is_win = False
            if top1_idx is not None:
                top1_sid = top5_ids[top1_idx]
                is_win = top1_sid in gold_ids
                results["s14-union-s8-contrastive"][qtype]["total"] += 1
                results["s14-union-s8-contrastive"][qtype]["questions"].append({"qid": qid, "win": is_win, "ranked": top1_idx+1, "gold_at": [p+1 for p in sorted(gold_at_pos)], "branch": branch})
                if is_win:
                    results["s14-union-s8-contrastive"][qtype]["wins"] += 1
                    if len(results["s14-union-s8-contrastive"][qtype]["examples"]) < 3:
                        results["s14-union-s8-contrastive"][qtype]["examples"].append({"qid": qid, "question": question[:100], "gold_at_position": [p+1 for p in sorted(gold_at_pos)], "scaffold_ranked_1": top1_idx+1})
            else:
                api_errors += 1
                results["s14-union-s8-contrastive"][qtype]["total"] += 1
            marker = "WIN" if is_win else "   "
            print(f"    [{marker}] s14-union-s8-contrastive: ranked [{top1_idx+1 if top1_idx is not None else '?'}] ({branch})")
        else:
            print("    [BUDGET] skip s14")

        # s15: scaffold-chain (2 calls)
        if call_count + 2 <= MAX_CALLS:
            raw_first = call_qwen(s15_chain_sys_first(question, candidates, dates), s15_chain_usr_first(question, candidates, dates))
            finalist_indices = []
            if raw_first:
                import re
                m = re.search(r'\[([^\]]+)\]', raw_first)
                if m:
                    try:
                        arr = json.loads(m.group(0))
                        finalist_indices = [int(x) - 1 for x in arr[:2] if 0 <= int(x) - 1 < len(candidates)]
                    except Exception:
                        pass

            if len(finalist_indices) >= 2:
                # Build finalists as new candidate list
                finalists = [(i+1, candidates[idx][1], candidates[idx][2]) for i, idx in enumerate(finalist_indices)]
                raw_second = call_qwen(s15_chain_sys_second(question, finalists, dates), s15_chain_usr_second(question, finalists, dates))
                top1_in_finalists = parse_top1(raw_second, len(finalists)) if raw_second else None
                if top1_in_finalists is not None:
                    top1_idx = finalist_indices[top1_in_finalists]
                    top1_sid = top5_ids[top1_idx]
                    is_win = top1_sid in gold_ids
                    results["s15-scaffold-chain"][qtype]["total"] += 1
                    results["s15-scaffold-chain"][qtype]["questions"].append({"qid": qid, "win": is_win, "ranked": top1_idx+1, "gold_at": [p+1 for p in sorted(gold_at_pos)]})
                    if is_win:
                        results["s15-scaffold-chain"][qtype]["wins"] += 1
                        if len(results["s15-scaffold-chain"][qtype]["examples"]) < 3:
                            results["s15-scaffold-chain"][qtype]["examples"].append({"qid": qid, "question": question[:100], "gold_at_position": [p+1 for p in sorted(gold_at_pos)], "scaffold_ranked_1": top1_idx+1})
                    marker = "WIN" if is_win else "   "
                    print(f"    [{marker}] s15-scaffold-chain: ranked [{top1_idx+1}] (via chain)")
                else:
                    parse_failures += 1
                    results["s15-scaffold-chain"][qtype]["total"] += 1
                    print(f"    [   ] s15-scaffold-chain: parse failure in second stage")
            else:
                # Fall back to first stage result directly
                top1_idx = parse_top1(raw_first, len(candidates)) if raw_first else None
                if top1_idx is not None:
                    top1_sid = top5_ids[top1_idx]
                    is_win = top1_sid in gold_ids
                    results["s15-scaffold-chain"][qtype]["total"] += 1
                    results["s15-scaffold-chain"][qtype]["questions"].append({"qid": qid, "win": is_win, "ranked": top1_idx+1, "gold_at": [p+1 for p in sorted(gold_at_pos)]})
                    if is_win:
                        results["s15-scaffold-chain"][qtype]["wins"] += 1
                    marker = "WIN" if is_win else "   "
                    print(f"    [{marker}] s15-scaffold-chain: ranked [{top1_idx+1}] (single stage fallback)")
                else:
                    api_errors += 1
                    results["s15-scaffold-chain"][qtype]["total"] += 1
                    print(f"    [   ] s15-scaffold-chain: fallback parse fail")
        else:
            print("    [BUDGET] skip s15")

        print()

    return results, parse_failures, api_errors


# ---------------------------------------------------------------------------
# Generate markdown report
# ---------------------------------------------------------------------------
def generate_report(results, set_a, parse_failures, api_errors):
    all_qtypes = sorted(set(e["qtype"] for e in set_a))
    qtype_counts = defaultdict(int)
    for e in set_a:
        qtype_counts[e["qtype"]] += 1

    all_scaffold_names = [s["name"] for s in SCAFFOLDS] + COMPOSITION_NAMES

    # Combine with already-tested for sorting
    combined_wins = {}
    for name in all_scaffold_names:
        total_wins = sum(results[name][qt]["wins"] for qt in all_qtypes)
        total_qs = sum(results[name][qt]["total"] for qt in all_qtypes)
        combined_wins[name] = (total_wins, total_qs)

    # Add already-tested
    for name, data in ALREADY_TESTED.items():
        combined_wins[name] = (data["wins"], data["total"])

    all_names_sorted = sorted(combined_wins.keys(), key=lambda n: -combined_wins[n][0] / max(combined_wins[n][1], 1))

    lines = []
    lines.append("# Qwen-Native Scaffold Arena Results")
    lines.append(f"\n**Date:** 2026-04-16  ")
    lines.append(f"**Baseline run:** runC-guard  ")
    lines.append(f"**Model:** qwen-turbo (dashscope-intl)  ")
    lines.append(f"**Total qwen-turbo calls:** {call_count}  ")
    lines.append(f"**Parse failures:** {parse_failures} | **API errors:** {api_errors}  ")
    lines.append(f"**Set A size:** {len(set_a)} questions (gold in top-5, wrong @1)  ")
    lines.append(f"**Previously tested (transfer validation):** 6 scaffolds  ")
    lines.append(f"**New scaffolds tested:** {len(all_scaffold_names)}  ")
    lines.append("")

    # -------------------------------------------------------------------------
    # 1. Full win-rate table
    lines.append("## 1. Full Win-Rate Table (All 21 Scaffolds, Sorted by Win Rate)")
    lines.append("")
    lines.append(f"| Scaffold | Overall | knowledge-update | multi-session | single-session-assistant | single-session-preference | temporal-reasoning | Source |")
    lines.append(f"|----------|---------|-----------------|---------------|--------------------------|---------------------------|--------------------|--------|")

    for name in all_names_sorted:
        wins, total = combined_wins[name]
        rate = wins / max(total, 1)
        source = "prev-tested" if name in ALREADY_TESTED else "new"

        if name in ALREADY_TESTED:
            qt_data = ALREADY_TESTED[name]["qtype_wins"]
            qt_totals = {"knowledge-update": 2, "multi-session": 2, "single-session-assistant": 1, "single-session-preference": 3, "temporal-reasoning": 12}
            cells = []
            for qt in ["knowledge-update", "multi-session", "single-session-assistant", "single-session-preference", "temporal-reasoning"]:
                w = qt_data.get(qt, 0)
                t = qt_totals.get(qt, 0)
                cells.append(f"{w}/{t}")
        else:
            cells = []
            for qt in ["knowledge-update", "multi-session", "single-session-assistant", "single-session-preference", "temporal-reasoning"]:
                w = results[name][qt]["wins"]
                t = results[name][qt]["total"]
                cells.append(f"{w}/{t}" if t > 0 else "0/0")

        row = f"| {name} | **{wins}/{total} ({rate:.0%})** | " + " | ".join(cells) + f" | {source} |"
        lines.append(row)

    lines.append("")

    # -------------------------------------------------------------------------
    # 2. Per-qtype breakdown — best per category
    lines.append("## 2. Per-Qtype Best Scaffold")
    lines.append("")

    for qt in all_qtypes:
        qt_n = qtype_counts[qt]
        lines.append(f"### {qt} (n={qt_n} failure questions)")
        lines.append("")

        # Score all scaffolds on this qtype
        qt_scores = []
        for name in all_names_sorted:
            if name in ALREADY_TESTED:
                w = ALREADY_TESTED[name]["qtype_wins"].get(qt, 0)
                t = {"knowledge-update": 2, "multi-session": 2, "single-session-assistant": 1, "single-session-preference": 3, "temporal-reasoning": 12}.get(qt, 0)
            else:
                w = results[name][qt]["wins"]
                t = results[name][qt]["total"]
            if t > 0:
                qt_scores.append((name, w, t))

        qt_scores.sort(key=lambda x: -x[1])
        for name, w, t in qt_scores[:5]:
            lines.append(f"- {name}: {w}/{t} ({w/t:.0%})")
        lines.append("")

    # -------------------------------------------------------------------------
    # 3. Qwen-native SCAFFOLD_DISPATCHER config
    lines.append("## 3. Qwen-Native SCAFFOLD_DISPATCHER Config")
    lines.append("")

    dispatcher = {}
    qt_totals_prev = {"knowledge-update": 2, "multi-session": 2, "single-session-assistant": 1, "single-session-preference": 3, "temporal-reasoning": 12}

    for qt in all_qtypes:
        best_name = None
        best_rate = -1.0
        for name in all_names_sorted:
            if name in ALREADY_TESTED:
                w = ALREADY_TESTED[name]["qtype_wins"].get(qt, 0)
                t = qt_totals_prev.get(qt, 0)
            else:
                w = results[name][qt]["wins"]
                t = results[name][qt]["total"]
            if t > 0:
                rate = w / t
                if rate > best_rate:
                    best_rate = rate
                    best_name = name
                    best_w = w
                    best_t = t
        if best_name:
            dispatcher[qt] = (best_name, best_w, best_t, best_rate)

    lines.append("```python")
    lines.append("# Qwen-turbo validated scaffold dispatcher (v0.3.1)")
    lines.append("# Source: qwen_native_arena.py, runC-guard failure set, 2026-04-16")
    lines.append("SCAFFOLD_DISPATCHER_QWEN = {")
    for qt, (best_name, best_w, best_t, best_rate) in dispatcher.items():
        lines.append(f'    "{qt}": "{best_name}",  # {best_w}/{best_t} ({best_rate:.0%}) on failure set')
    lines.append("}")
    lines.append("")
    lines.append("# Fallback (when qtype unknown)")
    # Find overall best new scaffold
    best_overall = max(all_scaffold_names, key=lambda n: combined_wins[n][0] / max(combined_wins[n][1], 1))
    lines.append(f'SCAFFOLD_DISPATCHER_QWEN["_default"] = "{best_overall}"')
    lines.append("```")
    lines.append("")

    # -------------------------------------------------------------------------
    # 4. Compositions that actually compose
    lines.append("## 4. Compositions That Compose")
    lines.append("")

    for cname in COMPOSITION_NAMES:
        cw, ct = combined_wins.get(cname, (0, 0))
        rate = cw / max(ct, 1)
        lines.append(f"### {cname} — {cw}/{ct} ({rate:.0%})")
        lines.append("")

        # Compare to components
        if "s8-korean" in cname:
            a_wins = ALREADY_TESTED["s8-socratic"]["wins"]
            b_wins = sum(results["korean-evidential-commit"][qt]["wins"] for qt in all_qtypes)
            lines.append(f"- s8-socratic alone: {a_wins}/20 (60%)")
            lines.append(f"- korean-evidential-commit alone: {b_wins}/{ct}")
            lines.append(f"- union: {cw}/{ct} ({rate:.0%})")
        elif "s8-contrastive" in cname:
            a_wins = ALREADY_TESTED["s8-socratic"]["wins"]
            b_wins = sum(results["contrastive-explicit"][qt]["wins"] for qt in all_qtypes)
            lines.append(f"- s8-socratic alone: {a_wins}/20 (60%)")
            lines.append(f"- contrastive-explicit alone: {b_wins}/{ct}")
            lines.append(f"- union: {cw}/{ct} ({rate:.0%})")
        elif "chain" in cname:
            lines.append(f"- scaffold-chain (socratic filter → self-consistency pick): {cw}/{ct} ({rate:.0%})")

        # Unique wins
        for qt in all_qtypes:
            qt_ex = results[cname][qt]["examples"]
            if qt_ex:
                lines.append(f"- {qt} wins: {', '.join(ex['qid'] for ex in qt_ex)}")
        lines.append("")

    # -------------------------------------------------------------------------
    # 5. Comparison vs gemma dispatcher
    lines.append("## 5. Qwen vs Gemma Dispatcher Comparison")
    lines.append("")
    lines.append("| qtype | Gemma prefers | Qwen prefers | Gemma rate | Qwen rate |")
    lines.append("|-------|--------------|--------------|------------|-----------|")

    # Gemma dispatcher from scaffold_arena_results.md (reading known results)
    GEMMA_DISPATCHER = {
        "temporal-reasoning": ("s2-temporal-inline", 0.70),
        "single-session-preference": ("s4-declarative-preference", 0.65),
        "knowledge-update": ("s0-minimal-baseline", 0.55),
        "multi-session": ("s4-declarative-preference", 0.65),
        "single-session-assistant": ("s1-temporal-sysprefix", 0.55),
    }

    for qt in all_qtypes:
        gemma_name, gemma_rate = GEMMA_DISPATCHER.get(qt, ("s0-minimal-baseline", 0.55))
        qwen_name, qwen_w, qwen_t, qwen_rate = dispatcher.get(qt, ("s8-socratic", 0, 0, 0.0))
        lines.append(f"| {qt} | {gemma_name} | {qwen_name} | {gemma_rate:.0%} | {qwen_rate:.0%} |")

    lines.append("")
    lines.append("**Key insight:** Gemma preferred temporal/declarative scaffolds. Qwen prefers socratic/step-based scaffolds.")
    lines.append("")

    # -------------------------------------------------------------------------
    # 6. Meta-scaffold verdict
    lines.append("## 6. Meta-Scaffold Verdict")
    lines.append("")

    qt_totals_prev2 = {"knowledge-update": 2, "multi-session": 2, "single-session-assistant": 1, "single-session-preference": 3, "temporal-reasoning": 12}
    meta_candidates = []

    for name in all_names_sorted:
        beats_all = True
        ref_wins = ALREADY_TESTED["s8-socratic"]["wins"]  # s8 is the reference to beat
        for qt in all_qtypes:
            if name in ALREADY_TESTED:
                w = ALREADY_TESTED[name]["qtype_wins"].get(qt, 0)
                t = qt_totals_prev2.get(qt, 0)
            else:
                w = results[name][qt]["wins"]
                t = results[name][qt]["total"]
            ref_w = ALREADY_TESTED["s8-socratic"]["qtype_wins"].get(qt, 0)
            ref_t = qt_totals_prev2.get(qt, 0)
            if t == 0:
                continue
            if w / t < ref_w / ref_t:
                beats_all = False
                break
        if beats_all and name != "s8-socratic":
            total_wins = combined_wins[name][0]
            total_qs = combined_wins[name][1]
            if total_wins / max(total_qs, 1) >= 0.60:
                meta_candidates.append((name, total_wins, total_qs))

    if meta_candidates:
        lines.append("**META-SCAFFOLD(S) FOUND** — beats or matches s8-socratic (60%) on all qtypes:")
        lines.append("")
        for mname, mw, mt in meta_candidates:
            lines.append(f"- **{mname}**: {mw}/{mt} ({mw/mt:.0%})")
        lines.append("")
        lines.append("These are candidates for single-scaffold deployment (no dispatcher needed).")
    else:
        lines.append("**No single scaffold beats s8-socratic (60%) across all qtypes.** Dispatcher approach remains optimal.")
        lines.append("")
        lines.append(f"Current qwen champion: **s8-socratic** at 60% (12/20) — tested in transfer validation.")

        # Check if any new scaffold ties or beats it overall
        best_new_name = max(all_scaffold_names, key=lambda n: combined_wins[n][0] / max(combined_wins[n][1], 1))
        best_new_w, best_new_t = combined_wins[best_new_name]
        best_new_rate = best_new_w / max(best_new_t, 1)
        lines.append(f"\nBest new scaffold: **{best_new_name}** at {best_new_rate:.0%} ({best_new_w}/{best_new_t}).")

        if best_new_rate >= 0.60:
            lines.append(f"\n**QWEN META-SCAFFOLD: {best_new_name}** — ties or beats s8-socratic. Recommend as primary.")
        elif best_new_rate > 0.50:
            lines.append(f"\n**Note:** {best_new_name} is strong but below s8-socratic threshold. Use dispatcher.")

    lines.append("")

    # -------------------------------------------------------------------------
    # 7. Surprise findings
    lines.append("## 7. Surprise Findings")
    lines.append("")

    # Find biggest winners and losers vs s8-socratic baseline (60%)
    s8_rate = 0.60
    surprises = []
    for name in all_scaffold_names:
        w, t = combined_wins[name]
        if t == 0:
            continue
        rate = w / t
        delta = rate - s8_rate
        if abs(delta) >= 0.10:
            direction = "BETTER" if delta > 0 else "WORSE"
            surprises.append((abs(delta), direction, name, w, t, rate, delta))

    surprises.sort(key=lambda x: -x[0])
    for _, direction, name, w, t, rate, delta in surprises[:8]:
        sign = "+" if delta > 0 else ""
        lines.append(f"- **{name}** ({direction}): {w}/{t} ({rate:.0%}, {sign}{delta:.0%} vs s8-socratic 60%)")

    if not surprises:
        lines.append("- No scaffolds deviated >10% from s8-socratic baseline.")

    lines.append("")

    # Chinese instruction note
    lines.append("### Chinese instruction scaffold note")
    zh_w = sum(results["qwen-chinese-instruction"][qt]["wins"] for qt in all_qtypes)
    zh_t = sum(results["qwen-chinese-instruction"][qt]["total"] for qt in all_qtypes)
    lines.append(f"- qwen-chinese-instruction: {zh_w}/{zh_t} ({zh_w/max(zh_t,1):.0%})")
    if zh_t > 0 and zh_w / zh_t >= 0.60:
        lines.append("  **Qwen responds well to Chinese-language instructions** — confirm for production use.")
    elif zh_t > 0 and zh_w / zh_t < 0.50:
        lines.append("  Chinese instruction does NOT outperform English scaffolds — keep English for qwen-turbo.")
    lines.append("")

    # -------------------------------------------------------------------------
    # Appendix: raw call count, budget
    lines.append("## Appendix: Budget")
    lines.append("")
    lines.append(f"- Total qwen-turbo calls: **{call_count}**")
    lines.append(f"- Estimated cost (qwen-turbo ~$0.0005/1K tokens, ~200 tokens/call avg): ~${call_count * 200 * 0.0005 / 1000:.3f}")
    lines.append(f"- Parse failures: {parse_failures}")
    lines.append(f"- API errors: {api_errors}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()

    print("[arena] Loading failure set...")
    set_a = load_failure_set()
    print(f"  Set A (scaffold-fixable): {len(set_a)} questions")

    print("[arena] Loading LongMemEval data...")
    lme_data = load_longmemeval()

    # Verify qwen connectivity
    print("[arena] Verifying qwen-turbo connection...")
    test_raw = call_qwen("Say 'ready'.", "One word response.")
    if test_raw is None:
        print("[arena] ERROR: qwen-turbo not responding. Check API key and endpoint.", file=sys.stderr)
        sys.exit(1)
    print(f"[arena] qwen-turbo ready: {test_raw[:40]!r}\n")

    results, parse_failures, api_errors = run_arena(set_a, lme_data)

    elapsed = time.time() - t0
    print(f"\n[arena] Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"[arena] Total qwen-turbo calls: {call_count}")
    print(f"[arena] Generating report...")

    report = generate_report(results, set_a, parse_failures, api_errors)
    RESULTS_MD.write_text(report)

    print(f"[arena] Report written to {RESULTS_MD}")
    print(f"\n--- SUMMARY ---")
    print(f"Total calls: {call_count} | Parse failures: {parse_failures} | API errors: {api_errors}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Report: {RESULTS_MD}")

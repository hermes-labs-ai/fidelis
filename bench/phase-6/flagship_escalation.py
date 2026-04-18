"""
Flagship escalation function — calls qwen-max with full session text
on hard/uncertain questions. Drop-in replacement for llm_rerank().

NOT wired into pipeline yet. Call manually or swap in when ready.
"""

import json
import urllib.request

DASHSCOPE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
FLAGSHIP_MODEL = "qwen-max"
FLAGSHIP_SNIPPET_LEN = 2000  # 4x more context than qwen-turbo's 500

_FLAGSHIP_SYSTEM = (
    "Execute this procedure:\n"
    "```\n"
    "def rank(query, candidates):\n"
    "  for each candidate:\n"
    "    keyword_match = count(query_keywords IN candidate_text)\n"
    "    fact_match = candidate contains specific answer to query (bool)\n"
    "    score = keyword_match * 2 + fact_match * 10\n"
    "  return sorted(candidates, by=score, descending)\n"
    "```\n"
    "Input: query and 5 numbered candidates.\n"
    "Output: ONLY a JSON array of candidate numbers sorted by score.\n"
    "Example: [3, 1, 5, 2, 4]"
)


def _parse_filter_indices(raw: str, n_candidates: int) -> list[int] | None:
    """Parse LLM output to list of 0-based indices."""
    if "<think>" in raw:
        end = raw.rfind("</think>")
        if end >= 0:
            after = raw[end + 8:].strip()
            if after:
                raw = after

    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= start:
        return None
    try:
        arr = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None

    result = []
    for item in arr:
        try:
            idx = int(item)
        except (ValueError, TypeError):
            continue
        if 1 <= idx <= n_candidates and (idx - 1) not in result:
            result.append(idx - 1)
    return result if result else None


def flagship_rerank(
    query: str,
    top_sessions: list[tuple[int, str]],
    temporal_scaffold: str = "",
) -> tuple[list[int], str]:
    """
    Rerank with qwen-max using full 2000-char snippets.
    Optionally includes temporal scaffold block.
    Returns (reranked_corpus_indices, method).
    """
    n = len(top_sessions)
    if n <= 1:
        return [idx for idx, _ in top_sessions], "flagship_skip"

    lines = []
    for i, (corpus_idx, text) in enumerate(top_sessions):
        snippet = text[:FLAGSHIP_SNIPPET_LEN].replace("\n", " ")
        lines.append(f"[{i+1}] {snippet}")
    candidates_block = "\n".join(lines)

    # Build prompt with optional temporal scaffold
    prompt_parts = [f"Query: {query}"]
    if temporal_scaffold:
        prompt_parts.append(f"\n{temporal_scaffold}")
    prompt_parts.append(f"\nCandidate memories:\n{candidates_block}")
    prompt_parts.append(f"\nRank these {n} candidates by relevance. Output JSON array.")
    prompt_user = "\n".join(prompt_parts)

    try:
        body = json.dumps({
            "model": FLAGSHIP_MODEL,
            "messages": [
                {"role": "system", "content": _FLAGSHIP_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            "temperature": 0,
            "max_tokens": 100,
        }).encode()
        req = urllib.request.Request(
            DASHSCOPE_URL, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return [idx for idx, _ in top_sessions], f"flagship_error_{e}"

    if not raw.strip():
        return [idx for idx, _ in top_sessions], "flagship_empty"

    parsed = _parse_filter_indices(raw, n)
    if parsed is None:
        return [idx for idx, _ in top_sessions], "flagship_parse_fail"

    reranked = [top_sessions[i][0] for i in parsed]
    included = set(parsed)
    for i in range(n):
        if i not in included:
            reranked.append(top_sessions[i][0])

    return reranked, "flagship"


def estimate_cost(n_questions: int, avg_candidates: int = 5) -> dict:
    """Estimate cost for flagship escalation."""
    # ~3K input tokens per question (system + query + candidates)
    # ~50 output tokens
    input_tokens = n_questions * 3000
    output_tokens = n_questions * 50
    input_cost = input_tokens / 1_000_000 * 20  # $20/M input for qwen-max
    output_cost = output_tokens / 1_000_000 * 60  # $60/M output for qwen-max
    return {
        "n_questions": n_questions,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 3),
        "output_cost_usd": round(output_cost, 3),
        "total_cost_usd": round(input_cost + output_cost, 3),
    }


if __name__ == "__main__":
    print("Flagship escalation cost estimates:")
    for n in [37, 50, 100, 205]:
        est = estimate_cost(n)
        print(f"  {n:3d} questions: ${est['total_cost_usd']:.2f}")

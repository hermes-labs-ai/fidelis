"""
LPCI Temporal Scaffold — builds a temporal ordering block for the LLM filter.

Given top-K candidate session IDs and the haystack_dates metadata,
produces a pre-computed pairwise ordering string to inject into the filter prompt.

Does NOT modify embeddings or BM25 — only affects the LLM prompt.
"""

import json
from datetime import datetime
from pathlib import Path


def parse_date(date_str: str) -> datetime | None:
    """Parse 'YYYY/MM/DD (Day) HH:MM' format."""
    try:
        # "2023/05/20 (Sat) 02:21"
        parts = date_str.split(" ")
        return datetime.strptime(parts[0], "%Y/%m/%d")
    except (ValueError, IndexError):
        return None


def build_temporal_scaffold(
    candidate_numbers: list[int],  # 1-based candidate numbers [1, 2, 3, 4, 5]
    candidate_session_ids: list[str],
    haystack_dates: dict[str, str],  # sid → date string
) -> str:
    """
    Build a temporal ordering block for the LLM filter prompt.

    Returns a string like:
        Temporal context (candidates ordered by date):
          [1] 2023/05/20 (earliest)
          [3] 2023/06/15 (+26 days after [1])
          [5] 2023/08/02 (+74 days after [1])

    Returns empty string if dates are unavailable.
    """
    # Parse dates for each candidate
    dated_candidates: list[tuple[int, str, datetime]] = []
    for num, sid in zip(candidate_numbers, candidate_session_ids):
        date_str = haystack_dates.get(sid, "")
        dt = parse_date(date_str)
        if dt:
            dated_candidates.append((num, date_str.split(" ")[0], dt))

    if len(dated_candidates) < 2:
        return ""  # Not enough dates to build ordering

    # Sort by date
    dated_candidates.sort(key=lambda x: x[2])

    # Build scaffold
    earliest_dt = dated_candidates[0][2]
    lines = ["Temporal context (candidates ordered by date):"]

    for i, (num, date_short, dt) in enumerate(dated_candidates):
        days_diff = (dt - earliest_dt).days
        if i == 0:
            lines.append(f"  [{num}] {date_short} (earliest)")
        elif i == len(dated_candidates) - 1:
            lines.append(f"  [{num}] {date_short} (+{days_diff} days, most recent)")
        else:
            lines.append(f"  [{num}] {date_short} (+{days_diff} days after [{dated_candidates[0][0]}])")

    return "\n".join(lines)


def is_temporal_query(query: str) -> bool:
    """Check if query has temporal signals (same patterns as router)."""
    q = query.lower()
    temporal_patterns = [
        "which happened first", "which did i do first",
        "how many days", "how many weeks", "how many months",
        "before or after", "what order", "what was the date",
        "which event", "which trip", "order of the", "from earliest",
        "from first", "most recently", "a week ago", "two weeks ago",
        "a month ago", "last saturday", "last sunday", "last weekend",
        "last monday", "last tuesday", "graduated first", "started first",
        "finished first", "did i do first", "did i attend first",
        "who did i go with to the",
    ]
    return any(p in q for p in temporal_patterns)


# ---------------------------------------------------------------------------
# Test on hardset temporal questions
# ---------------------------------------------------------------------------
def test_on_hardset():
    """Print temporal scaffold for hardset temporal questions."""
    hardset = json.load(open(Path(__file__).parent.parent / "hardset.json"))
    data = json.load(open(Path.home() / "Documents/projects/LongMemEval/data/longmemeval_s_cleaned.json"))
    data = [e for e in data if "_abs" not in e["question_id"]]

    temporal_hard = [h for h in hardset if h["qtype"] == "temporal-reasoning"]
    print(f"Temporal questions in hardset: {len(temporal_hard)}")
    print()

    for h in temporal_hard[:5]:
        entry = data[h["qi"]]
        # Build date lookup
        sid_to_date = {}
        for sid, date in zip(entry["haystack_session_ids"], entry.get("haystack_dates", [])):
            sid_to_date[sid] = date

        # Use S1 top-5 as candidates
        top5_sids = h["s1_top5_ids"][:5]
        candidate_nums = list(range(1, len(top5_sids) + 1))

        scaffold = build_temporal_scaffold(candidate_nums, top5_sids, sid_to_date)

        print(f"=== q{h['qi']} [{h['failure_mode_note']}] ===")
        print(f"Q: {h['question'][:100]}")
        print(f"Gold: {h['gold_session_ids'][:3]}")
        print(f"S1 top5: {top5_sids}")
        if scaffold:
            print(scaffold)
        else:
            print("  (no temporal scaffold — dates unavailable)")
        print()


if __name__ == "__main__":
    test_on_hardset()

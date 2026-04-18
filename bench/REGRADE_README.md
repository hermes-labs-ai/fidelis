# QA Regrade Script

Corrects QA evaluation results when answers are semantically correct but marked wrong due to formatting differences.

## Problem
The QA model sometimes wraps answers in conversation format:
- **Gold**: "Business Administration"
- **QA Model Output**: "[user]: I graduated with a degree in Business Administration"
- **Original grader**: ❌ (exact string mismatch)

## Solution
Two-stage grading pipeline:
1. **Exact containment** (fast): Check if gold answer appears in QA response (case-insensitive)
2. **LLM grading** (fallback): Use qwen-turbo to semantically compare if exact match fails
3. **Fallback**: Keep original grading if API unavailable

## Usage

```bash
export DASHSCOPE_API_KEY=sk-...
python3 regrade_qa.py --input qa_eval_v2_runP-v35_qwen-max_top5.json
```

## Options

- `--input` (required): Path to qa_eval_v2_*.json file
- `--output` (optional): Output file path (default: input with `_regraded` suffix)

## Output

Creates a new JSON file with:
- Updated `qa_correct` field (True/False)
- `qa_correct_original` field (for tracking changes)
- `regrade_method` field: "exact", "llm", or "fallback"

Prints summary with:
- Original vs regraded accuracy
- Breakdown by grading method
- Per-category accuracy (by qtype)

## Performance

- **Exact match**: ~1ms per entry (substring search)
- **LLM grading**: ~2-3s per entry (API call with 10s timeout)
- **Fallback**: Instant (keeps original result if API fails)

## Example Output

```
======================================================================
REGRADE SUMMARY
======================================================================
Total entries: 8
Original accuracy: 1/8 (12.5%)
Regraded accuracy: 2/8 (25.0%)
Delta: +12.5%
Improved: 1, Degraded: 0

Grading methods:
  exact: 6 (75.0%)
  llm: 2 (25.0%)

By category:
  single-session-preference:
    total: 8
    original: 12.5% (1/8)
    regraded: 25.0% (2/8)
```

# TODO — stop building, start shipping

*Updated: 2026-03-28*

---

## cogito-ergo — MVP closeout (in progress)

- [ ] Bake TECHNICAL_EXTRACTION_PROMPT into default config (zer0lint fix, so /add works on first use)
- [ ] Fix filter gating: add "if nothing is relevant, output []" to filter prompt (1 line, fixes 0% adversarial precision)
- [ ] Commit + push all pending changes (recall.py, server.py, cli.py, snapshot.py, bench/eval.py, bench/eval_cases.json)
- [ ] Minimal README pass — add benchmark numbers, install instructions, one-liner description

**Not MVP (do after):**
- Re-seed cogito_main corpus with fixed extraction prompt (personal task, not codebase)
- Re-run bench/eval.py post-reseed for honest post-fix numbers (updates PENDING-PAPER2.md evidence)
- Fix cross-reference R@1 (currently 50% — needs more snapshot coverage or better cases)

---

## zer0lint — PUSH THIS NOW

**The hook:** "You think your AI agent has memory. It doesn't. Here's proof."
**The proof:** 0/5 CRITICAL on the author's own system. Fixed in one run. Full logs in SESSION-2026-03-28.md.
**The gap:** No competitors. mem0 tells users to spot-check manually. Every benchmark is offline/academic. Open issues on mem0 itself (#3299, #2366) request exactly this.

- [ ] README rewrite — lead with the hook, show the 0→100% terminal output, one-liner install
- [ ] Substack post — "Most people are storing the wrong thing" (the conversation we were having — that's the lede)
- [ ] Tweet/thread — show the CRITICAL output, explain what it means, link zer0lint
- [ ] Post in mem0 Discord / GitHub Discussions — answer open issues #3299 and #2366 with zer0lint
- [ ] HN Show HN post — "zer0lint: diagnose why your AI agent forgets"
- [ ] Add to Awesome-Agent-Memory repo (https://github.com/TeleAI-UAGI/Awesome-Agent-Memory)

---

## zer0dex — PUSH THIS

- [ ] README: lead with 91% recall number and the dual-layer architecture diagram
- [ ] Cross-link to cogito-ergo (zer0dex is the pattern; cogito-ergo is the production-ready implementation)
- [ ] Post in mem0 / LangChain community channels

---

## Other projects — queue

- [ ] lintlang — check if there's a push moment ready
- [ ] little-canary — status?
- [ ] PENDING-PAPER.md + PENDING-PAPER2.md — arXiv pre-print when ready (low effort, timestamps the ideas)

---

## Rule

Stop adding features. Ship what exists. The tools are good.

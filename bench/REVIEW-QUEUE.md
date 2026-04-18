# Review Queue

Main session appends an entry here when a phase artifact is ready.
Fresh-context reviewer (cron) reads the top unreviewed entry, audits, writes to `bench/reviews/`, then edits this file to mark REVIEWED.

Format:
```
- [ ] <timestamp> | phase-N | <artifact path> | <one-line summary>
```

## Queue
<!-- main session appends below -->

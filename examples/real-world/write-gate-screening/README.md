# Write-gate screening

## Job
Before a string is written to the store, decide whether it is durable content or machine
exhaust (scaffold prompts, probe ids, raw tool dumps), and report a stable reason code.

## Files
- `input.json`: eleven strings.
- `expected.json`: `fidelis.write_gate.evaluate` decision for each (`accept`, `reason`, `detail`).

## How it is verified (no server)
`evaluate` is documented as pure and deterministic in `src/fidelis/write_gate.py`: no I/O,
clock, network, or model. The test calls it directly on each string.

## What you should see
Six strings that only mention harness tags, probes, yes/no answers, or GitHub tokens in prose
(or are ordinary `User:`/`Assistant:` turns) are accepted. Five that start with a template
opener, contain `Answer ONLY: YES or NO`, carry a probe id with its body, or lead with a
`[tool_result` bracket are rejected with `scaffold_prompt`, `probe_residue`, or
`harness_envelope`.

## Provenance
All strings are taken verbatim from `tests/test_write_gate.py` (accepted-prose list and
rejection cases). Secret-shaped and harness-tag cases are deliberately not included as files,
because they are built by string concatenation in that test to avoid committing raw control
markup or credential-looking text; see `tests/test_write_gate.py`.

## Limits
- The gate recognizes specific shapes, not meaning. Docs describe it as recognizing common
  secret patterns and obvious noise, not all sensitive information. A rejected secret must not
  be assumed retained anywhere else.
- Passing the gate does not make a statement true, useful, or non-sensitive.
- Whether the service refuses a write also depends on the HTTP layer and configuration
  (`write_gate: false` disables the exhaust rules but not secret refusal). This example does
  not exercise the endpoint or its acknowledgements (`stored`, `duplicate`, `rejected`, `queued`).

# Chessbot-Train Agent Execution Contract (AGENT.MD)

Follow `../AGENT_CONTRACT.MD` first.

This repo uses a strict execution contract for recognizer development, blocker handling, benchmark enforcement, and model-side experiments.

## Primary Rule: Continuous Recognizer Execution

- Never stop until the assigned recognizer target is fully reached.
- Only pause under these conditions:
  - A destructive or privileged action requires explicit approval.
  - An actual system-level blocker prevents further progress.
  - A catastrophic failure occurs that cannot be automatically resolved.

## Execution Expectations

### Non-Stop Operation

- Keep active recognizer work moving until the target is reached or the experiment is explicitly abandoned.
- Move directly on discovered blocker failures.
- Do not park known recognizer regressions once they are proven.

### Parallel & Branching Strategy

- Always branch off `main` for recognizer fixes, benchmark work, or experiments unless explicitly told otherwise.
- Keep one branch focused on one recognizer objective.
- Delete merged or abandoned recognizer branches after use.

### Resilience Mechanisms

- Keep a known good floor before risky recognizer edits.
- Preserve blocker truth, benchmark truth, and report freshness while iterating.
- If an experiment hangs or flattens, stop rerunning the dead state and move to the next structural fix.

## Reporting Rules

### Status Updates

- Report only exact current-state results from this repo.
- Do not present bundle-copy results as proof of recognizer behavior.
- After each meaningful change, report:
  - change
  - targeted result
  - checks
  - remaining risk

### Score Validation

- Hardset, blocker, stress, and domain results must come from fresh runs.
- Do not trust stale files in `reports/` after code changes.
- If there is doubt, rerun the affected gate.

## Main vs Branch Workflow

### Clean Main Policy

- `main` must remain reproducible and benchmarkable.
- Accepted recognizer behavior must be provable from `main`.

### Isolated Development

- Experimental work belongs on dedicated branches.
- Do not leave important recognizer work only in temporary scripts or side branches.
- Sync to `recognizer` only after proof exists here.

### Rollback Safety

- Keep a last-known-good recognizer floor before risky experiments.
- Revert failed experiments cleanly instead of layering more rescue logic.

## Editing Guidelines

### Structural Integrity

- Fix root causes in recognizer logic.
- Prefer architectural cleanup over symptom patches.
- Keep blocker truth and benchmark truth coherent with the code state.

### Code Quality Enforcement

- Reject selector junk, score nudges, confidence-rescue drift, and hidden fallback logic.
- Reject benchmark claims that are not tied to the exact current code state.
- Revert bad experiments instead of stacking more fixes.

## Evaluation Standards

### Hard Gates

- Maintain accepted hard gates unless explicit approval is given to change them.
- Do not relax blocker, hardset, domain, or stress expectations silently.

### Stress Testing

- Push recognizer work through the relevant blocker, hardset, stress, and domain paths.
- Do not stop at one passing slice if the task clearly touches a broader evaluation surface.

### Pre-Merge Checklist

Before treating recognizer work as done:

- hard gates verified
- relevant blocker or benchmark gates rerun
- exact current-state results captured
- sync implications for `recognizer` identified

## Practical Behavior Enhancements

### Process Monitoring

- Keep at least one meaningful recognizer check or diagnostic running while the task is open.
- If a runner hangs, replace it with a fresh run instead of trusting stale output.

### Report Freshness

- Regenerate critical reports after code changes.
- Do not let old `reports/` files drive new decisions.

### Branch Floor Management

- Keep clear rollback points for risky experiments.
- Identify the last-known-good floor before major recognizer changes.

## Definition of Done

A recognizer task is complete only when:

- the requested recognizer behavior is implemented,
- the relevant blocker, hardset, stress, or domain checks pass from the exact current state,
- blocker truth and reports are coherent,
- any required sync to `recognizer` is explicit,
- and `main` remains reproducible.

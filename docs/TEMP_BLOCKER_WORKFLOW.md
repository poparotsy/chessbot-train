# Temp Blocker Workflow

## Purpose

`temp/` is the inbox for real recognizer failures found during manual testing.

This workflow exists so a random dropped image does not stay a random file forever.
Each blocker must end up with:

- a stable blocker ID
- a real `truth_fen`
- a gate result in the temp blocker evaluator

## Files

- inbox:
  - `/Users/guru/workspace/current/chessbot/chessbot-train/temp`
- canonical manifest:
  - `/Users/guru/workspace/current/chessbot/chessbot-train/scripts/testdata/v6_temp_canaries.json`
- evaluator:
  - `/Users/guru/workspace/current/chessbot/chessbot-train/scripts/evaluate_v6_temp_canaries.py`
- intake/promote tool:
  - `/Users/guru/workspace/current/chessbot/chessbot-train/scripts/promote_v6_temp_blocker.py`

## Workflow

1. Drop a new real failure image into `temp/`.

2. Promote it into the blocker manifest:

```bash
cd /Users/guru/workspace/current/chessbot/chessbot-train
python3 scripts/promote_v6_temp_blocker.py --source-path temp/<file> --status pending
```

This assigns or preserves a stable blocker ID and records the inbox file in the manifest.

3. Verify the real board position from the image itself.

Do not recycle recognizer output back into truth unless it has been visually verified.

4. Lock the blocker with real truth:

```bash
cd /Users/guru/workspace/current/chessbot/chessbot-train
python3 scripts/promote_v6_temp_blocker.py --source-path temp/<file> --status locked --truth-fen '<full fen>'
```

5. Run the blocker gate:

```bash
cd /Users/guru/workspace/current/chessbot/chessbot-train
python3 scripts/evaluate_v6_temp_canaries.py --script recognizer_v6.py
```

## Status Meaning

- `locked`
  - real truth is known
  - the blocker is a required gate case
- `pending`
  - the image is tracked
  - current recognizer output is reported
  - it is not yet a truth-enforced gate case

## Merge Rule

- Locked temp blockers must pass before merge.
- Pending blockers must be worked toward locked truth, not forgotten.
- Once a real blocker is locked, it stays in the workflow.

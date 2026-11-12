# Adapter campaign receipt verification

These CPU tools aggregate a sealed campaign; they do not train models or turn
successful jobs into qualification passes. The required inventory, scientific
gates, loader profiles and producer/source admissions are explicit inputs. A
reviewed contract digest must be supplied to the production verifier.

Run from the checkout root:

```bash
python -m unittest tools.adapter_campaign.test_evidence
python -m tools.adapter_campaign.contract \
  --root "$CAMPAIGN_ROOT" --inputs "$COMPILE_INPUTS" --out "$CONTRACT"
python -m tools.adapter_campaign.primary \
  --root "$CAMPAIGN_ROOT" --contract "$CONTRACT" \
  --capture "$RAW_PRIMARY_CAPTURE" --output "$NEW_RECEIPT_DIRECTORY"
python -m tools.adapter_campaign.evidence \
  --root "$CAMPAIGN_ROOT" --contract "$CONTRACT" \
  --contract-sha256 "$REVIEWED_CANONICAL_CONTRACT_DIGEST" \
  --index "$EVIDENCE_INDEX" --out "$NEW_RESULTS_FILE"
```

`contract.py` emits the canonical JSON digest. This differs from hashing an
indented JSON file. Independent review must approve the exact contract and its
producer admissions before its digest is used for a qualification claim.

`primary.py` currently normalizes only the complete historical GSM8K primary
training schedule, sample coverage, and initial/final validation. It checks raw
rank logs against the preserved audit, per-example preprocessing and immutable
split identities. The validation census is inferred from the complete split,
source-pinned exhaustive iterator, preprocessing census and global counts; it
does not claim per-example validation runtime traces. Recovery, checkpoint,
artifact, deployment, task, simulator and resource receipts require separately
reviewed producers.

The verifier distinguishes checks executed, completed training, all required
checks executed, fully passing qualification, failures and external/dependency/
implementation blockers. Missing gates remain unresolved. A partial import must
retain the original attempt ledger: zero imported failures does not clear
historical failures. Source compatibility requires an explicit reviewed receipt
bound to the exact gate and profile. Historical execution revisions stay intact.

Synthetic tests exercise the success path and rejection of missing or mismatched
evidence. Synthetic receipts are never accepted by the production CLI and never
count toward a real campaign. Simulator protocols must seal the full frozen task,
state and seed census before results; dataset-preparation receipts cannot stand
in for closed-loop evaluation.

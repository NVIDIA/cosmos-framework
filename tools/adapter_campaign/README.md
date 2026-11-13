# Adapter campaign receipt verification

These CPU tools aggregate a sealed campaign; they do not train models or turn
successful jobs into qualification passes. The required inventory, scientific
gates, loader profiles and producer/source admissions are explicit inputs. A
reviewed contract digest must be supplied to the production verifier.

Run from the checkout root:

```bash
python -m pytest -c /dev/null --noconftest \
  tools/adapter_campaign/test_evidence.py \
  tools/adapter_campaign/test_references.py \
  tools/adapter_campaign/test_recovered.py
python -m tools.adapter_campaign.contract \
  --root "$CAMPAIGN_ROOT" --inputs "$COMPILE_INPUTS" --out "$CONTRACT"
python -m tools.adapter_campaign.primary \
  --root "$CAMPAIGN_ROOT" --contract "$CONTRACT" \
  --capture "$RAW_PRIMARY_CAPTURE" --output "$NEW_RECEIPT_DIRECTORY"
python -m tools.adapter_campaign.references \
  --root "$CAMPAIGN_ROOT" --contract "$CONTRACT" \
  --review "$REFERENCE_REVIEW_PATH" --review-sha256 "$REFERENCE_REVIEW_SHA256" \
  --gates-sha256 "$NUMERICAL_GATES_FILE_SHA256" --output "$NEW_REFERENCE_DIRECTORY"
python -m tools.adapter_campaign.recovered \
  --root "$CAMPAIGN_ROOT" --contract "$CONTRACT" \
  --capture "$RECOVERED_RAW_CAPTURE" --capture-review "$CAPTURE_REVIEW" \
  --producer-review "$RECOVERED_PRODUCER_REVIEW" --output "$NEW_RECOVERY_DIRECTORY"
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

`references.py` normalizes the reviewed `text_micro_reduce_v3` receipt schema.
It binds original distributed primary, oracle, installed source, raw capture and
global token denominators. Three completed comparisons must include uneven
lengths and an incomplete tail. A numerical failure before the next optimizer
step remains a failed prerequisite; later steps and optimizer comparisons are
not certified. Saved gradients are hash-bound; independent updated Adam tensors
were not separately saved, so their comparisons are source-enforced rather than
recomputed by this tool. This producer does not normalize recovery or training.

`recovered.py` adds four gates for the reviewed GSM8K recovery format: full
schedule, sample coverage, initial/final validation and midpoint continuation.
It joins the committed original prefix to the verified recovery suffix using
their original paths and source identities. Discarded original work and replay
are retained separately from logical coverage. It rejects missing rank owners,
out-of-range source indices, conflicting identities and substituted reviews.
Raw capture and producer reviews are required, including their exact dependency
hashes. Large tensor comparisons rely on the separately reviewed, hash-bound
remote audit; this tool does not pretend those tensors were downloaded. It does
not certify a final export, task score or whole campaign from these four gates.

`segmented.py` admits the separately reviewed Mistral LoRA/FSDP2 four-phase
lineage. It validates the original source payloads, five checkpoint generations,
all rank coverage streams and the separate midpoint replay. It reuses the
dataset, token-denominator and rank-ownership validators without relabeling the
older two-phase recovery schema. Its CLI requires `--audit-review` and
`--producer-review`; candidate integration and atomic activation remain separate
reviewed operations. Only the same four training/continuation gates are emitted.

This reference producer additionally requires `producer_reviews` in its gate's
admission: a mapping from each admitted producer SHA to a nonempty list of exact
`{"path": "relative/review.json", "sha256": "..."}` review references. The
compiler validates those files and the verifier requires an admitted review in
the receipt's provenance-bound supporting evidence. A caller-provided digest
alone cannot replace the independently admitted review. Other existing producers
retain their original admission format.

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

# Cosmos workflow support

Read `README.md` for ownership and scope. This directory supports the existing
framework skills; do not recreate a parallel setup, launch, or capability skill
registry. This directory adds no new skill entrypoints.

- Keep native training and inference implementation in `cosmos_framework/`.
- Resolve helper paths directly from this checkout.
- Preserve Framework runtime ownership, nested specs, launch approval, record-before-submit
  ordering, and backend-reported terminal state. Keep external runtime identifiers
  exact; framework-owned helper names use the Cosmos namespace.
- Read only the selected model/platform references. Native recipes need not use
  the managed-job layer; helpers call Framework's native entrypoints.
- Keep helper tests CPU-only: `python -m pytest --confcutdir=. -q` here.
- Update both existing skill copies for routing changes.
- Keep source provenance in `migration.json` when moving or removing resources.

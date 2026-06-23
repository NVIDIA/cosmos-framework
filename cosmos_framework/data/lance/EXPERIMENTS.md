# Lance-promoting experiments (on `lancedb-dataloader-experiments`)

Experiments that showcase capabilities Lance has and the base (WebDataset/file) loaders
structurally lack. Kept off the main branch.

## 1. Filtered / curriculum / quality sampling — predicate pushdown
`benchmarks/lance/bench_filtered.py`. Real training often samples a SUBSET (curriculum,
quality filter, task/domain balancing). LanceDB pushes the predicate into the scan and reads
**only matching rows' blobs**; a WebDataset tar is sequential + opaque and must stream +
parse **every** sample, discarding the misses — there is no skip operation in a tar.

Measured on the LLaVA figureqa set (99,995 samples; Lance table vs the 20 tar shards),
storage level (yield bytes, no decode), same selectivity fraction on both sides:

| selectivity | lance kept-samples/s | wds kept-samples/s | speedup | lance MB read | wds MB read |
| ----------- | -------------------- | ------------------ | ------- | ------------- | ----------- |
| 100% (no filter) | 60,582 | 8,843 | 6.9× | 2213 | 2213 |
| 50%  | 119,703 | 4,450 | 26.9× | 1107 | 2213 |
| 30%  | 130,940 | 2,695 | 48.6× | 664  | 2213 |
| 10%  | 109,655 | 900   | 121.8× | 221 | 2213 |

- **Bytes read is the proof**: Lance reads only the selected fraction (2213→221 MB via
  pushdown); webdataset reads 100% (2213 MB) regardless.
- At 100% it's already 6.9× (columnar read vs tar parse); filtering multiplies it as ~1/s.
- **This is structural**: webdataset cannot push down a predicate — it has no way to skip
  unselected samples without reading them. This is the clearest "Lance is better" result.

Honest scope: storage/read-layer measurement (the part Lance changes); per-kept-sample
decode is identical on both sides and omitted. Both apply the same selectivity fraction
(the win is reading only that fraction, independent of which rows).

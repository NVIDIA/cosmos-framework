# Multi-node Framework training

Use Framework's [raw torchrun interface](../../../../../docs/training.md#option-b-raw-torchrun)
with the Bimanual TOML and overrides from the [main tutorial](../SKILL.md).
Your scheduler allocates the nodes; this guide does not submit a job.

On every worker, activate the same Framework environment and bind the same
bundle, model/VAE assets, parent checkpoint roots, `RUN_NAME` and
`IMAGINAIRE_OUTPUT_ROOT`. All shared paths must resolve identically. Set
`NNODES`, `GPUS_PER_NODE`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT` for your
allocation. Use a unique node rank and the same reachable rendezvous address.

Replace only the main tutorial's `TRAIN` array:

```bash
: "${NNODES:?set allocated node count}"
: "${GPUS_PER_NODE:?set processes per node}"
: "${NODE_RANK:?set the unique rank for this node}"
: "${MASTER_ADDR:?set the rendezvous host reachable by all nodes}"
: "${MASTER_PORT:?set the rendezvous port}"
TRAIN=(torchrun
  --nnodes="$NNODES" --nproc-per-node="$GPUS_PER_NODE"
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT"
  -m cosmos_framework.scripts.train)
```

Then execute the same stage command and overrides on every worker. Keep the
native shard/CP layout consistent with total world size. Test GPU visibility,
NCCL communication and shared output access before loading weights.

Stages 0 and 1 compile tokenizer variants after their first three step starts.
Every rank must read the compiled packages under the shared job output path.
After a save, wait for all ranks to finish, then start the same stage in fresh
processes to check full-state resume. Reuse its run name, output root, topology
and checkpoint inputs, and set `trainer.max_iter` to an absolute endpoint later
than the saved iteration.

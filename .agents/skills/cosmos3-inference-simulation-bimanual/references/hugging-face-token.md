# Authenticate hosted files without exposing the token

> **Skill note (not in the source):** this file is the section "Authenticate hosted files without exposing the token" of the Sim-Bimanual inference SKILL.md (cosmos-framework `main` 45fecab, lines 56 to 92, a subsection of "Prerequisites and environment setup"), moved here verbatim to keep SKILL.md within its line budget. Only the heading level changed; the section has no links.

The checkpoint and current hosted examples may require Hugging Face read
access. Supply a user-owned token file outside the checkout and export its
contents without printing them:

```bash
: "${HF_TOKEN_FILE:?set HF_TOKEN_FILE to a readable token file outside the checkout}"
test -r "$HF_TOKEN_FILE"
set +x
HF_TOKEN="$(<"$HF_TOKEN_FILE")"
export HF_TOKEN
test -n "$HF_TOKEN"
```

Do not print the token, put its value in a command argument, copy it into the
checkout, or save it in a manifest or log. Framework's direct Hugging Face URL
downloader reads `HF_TOKEN`; a token available only under a custom `HF_HOME`
does not authenticate that downloader.

For Slurm with Pyxis, export the token in the submission shell as above, then
add both of these name-only options to the existing `srun` command:

```bash
--export=ALL,HF_TOKEN
--container-env=HF_TOKEN
```

Do not use `--container-env=HF_TOKEN=<value>`. Before loading the checkpoint or
hosted assets, verify inside the container without displaying the value:

```bash
test -n "${HF_TOKEN:-}" || { echo "HF_TOKEN is not available inside the container" >&2; exit 1; }
```

Unset the submission-shell value after the synchronous run or after the
scheduler has accepted the job: `unset HF_TOKEN`.

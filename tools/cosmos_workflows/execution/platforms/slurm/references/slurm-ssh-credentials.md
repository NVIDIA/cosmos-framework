# SLURM SSH Setup, Credentials, And Storage

SSH access, Enroot registry authentication, and credential handling for the
native SLURM commands. For shared paths and job records, see
[Storage](slurm-preflight-storage.md).

## Preflight

```bash
# 1. SSH to the login node works without a password prompt
SLURM_HOST="${SLURM_HOSTNAME%%,*}"
[ -n "$SLURM_USER" ] && [ -n "$SLURM_HOST" ] || {
  echo "MISSING: SLURM_USER and SLURM_HOSTNAME (comma-separated for failover)."
  echo "Export them in your shell before launching, or source a user-approved env file:"
  echo "  set -a; source /path/to/.env; set +a"
  exit 1
}
ssh -o BatchMode=yes -o ConnectTimeout=10 "${SLURM_USER}@${SLURM_HOST}" "true" 2>/dev/null || {
  echo "MISSING: passwordless SSH to ${SLURM_USER}@${SLURM_HOST} not working. See the Prerequisites section."
  exit 1
}

# 2. Enroot credentials on the cluster for private nvcr.io images.
# Pyxis on the compute nodes invokes enroot to import the Docker image. Enroot
# does NOT read NGC_KEY from the SLURM job env — it requires persistent
# credentials in ~/.config/enroot/.credentials on the login/compute nodes.
# Without this, anonymous pulls of auth-gated nvcr.io paths (e.g. any
# pre-release staging org) fail with "Could not process JSON input" at job
# startup. Skip if the image is from a public repo.
if [ -n "$NGC_KEY" ]; then
  REMOTE_CRED_OK=$(ssh -o BatchMode=yes "${SLURM_USER}@${SLURM_HOST}" \
    'test -s ~/.config/enroot/.credentials && echo OK || echo MISSING' 2>/dev/null)
  if [ "$REMOTE_CRED_OK" != "OK" ]; then
    echo "MISSING: ~/.config/enroot/.credentials not set on ${SLURM_HOST}."
    echo "After user approval, install it from NGC_KEY (no value echoed):"
    echo "  printf 'machine nvcr.io login \$oauthtoken password %s\\nmachine authn.nvidia.com login \$oauthtoken password %s\\n' \"\$NGC_KEY\" \"\$NGC_KEY\" \\"
    echo "    | ssh -o BatchMode=yes \"\${SLURM_USER}@\${SLURM_HOST}\" '"
    echo "        mkdir -p ~/.config/enroot && umask 077 && cat > ~/.config/enroot/.credentials && chmod 600 ~/.config/enroot/.credentials"
    echo "      '"
    exit 1
  fi
fi
```

If a check fails, the agent prompts the user to authorize the install/fix via Bash.

The enroot-credentials step (#2) only needs to run **once per (cluster, user)** —
subsequent SLURM sessions inherit the file. Use the `printf | ssh` heredoc
pattern above so the `NGC_KEY` value never lands in shell history, intermediate
files, or chat output. Do not `cat` or `echo` the value at any step. After the
file is in place, both the SQSH pre-conversion job (which runs on
`sqsh_conversion_partition`) and the actual training job's Pyxis pull will
authenticate as `$oauthtoken` against `nvcr.io`.

## Prerequisites

Before any SLURM job can be submitted or any runner script is generated, the
launch host must be able to log in to at least one host
from `SLURM_HOSTNAME` over SSH **without an interactive password prompt**. The
handler runs `sbatch`, `squeue`, `sacct`, `scancel`, and log tails
non-interactively, so password or 2FA prompts will fail the job at submit or
status time.

Set this up once per (host, login node, user) tuple:

1. Ensure an SSH keypair exists for the service user (e.g. `~/.ssh/id_ed25519`).
   Create one with `ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519` if it is
   missing. Select the key path explicitly; do not overwrite an existing key.
2. Install the public key on each login node:

   ```bash
   ssh-copy-id -i ~/.ssh/id_ed25519.pub <SLURM_USER>@<login-host>
   ```

   This is the only step that requires the user's password; run it interactively
   once per login host listed in `SLURM_HOSTNAME`. If `ssh-copy-id` is not
   available, append the public key manually:

   ```bash
   cat ~/.ssh/id_ed25519.pub | ssh <SLURM_USER>@<login-host> \
     'mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
      cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
   ```

3. Trust the host key so SSH does not stall on the "authenticity of host" prompt
   inside the handler. Verify the host-key fingerprint with the cluster administrator before accepting
   it or adding a scanned key to `~/.ssh/known_hosts`.
4. Verify the result is fully non-interactive for at least one listed login
   host:

   ```bash
   ssh -o BatchMode=yes -o PreferredAuthentications=publickey \
     <SLURM_USER>@<login-host> 'hostname && squeue -u $USER -h | head -n 1'
   ```

   `BatchMode=yes` forces failure if SSH would otherwise prompt; this command
   must succeed before the SLURM platform is usable.
5. When the submission client runs in a container, mount the
   private key into the container at the path referenced by `SSH_KEY_PATH`, with
   `chmod 600` and matching ownership for the in-container user. Verify these permissions before use.

For convenience, a per-host alias in `~/.ssh/config` lets you reference a short
name everywhere:

```text
Host slurm-login
    HostName <login-host>
    User <SLURM_USER>
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking accept-new
```

If a site enforces 2FA on every SSH connection, passwordless key auth alone is
not enough; coordinate with the cluster admin to allow key-only auth from the
service host or use an SSH agent with cached credentials and expose it to the
handler via `SSH_AUTH_SOCK`.

## Credentials

- **SLURM_USER**: login username.
- **SLURM_HOSTNAME**: comma-separated login hosts for failover.
- **SSH_KEY_PATH**: private-key path for non-interactive public-key auth;
  **SSH_AUTH_SOCK** is an alternative when an accepted key is already loaded.
- **NGC_KEY**: needed only for authenticated registry pulls. Use stdin when
  installing Enroot credentials after approval; never log credential contents.

Cluster partition/account selection and shared output roots belong to the
[platform launch intake](../guide.md#required-inputs), not a credential or
microservice schema. Credentials stay in the session environment or an approved
env file. Do not ask for their values in chat.

## SSH Failure Remediation Prompt

When passwordless SSH fails, use this concise prompt:

```text
SLURM is blocked on passwordless SSH. Please provide:

SSH_KEY_PATH=/path/to/private_key

If you have not set up passwordless access yet:
1. Create a key if needed:
   ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
2. Install the public key on one login host:
   ssh-copy-id -i ~/.ssh/id_ed25519.pub <SLURM_USER>@<login-host>
3. Trust the host key:
   Verify the host-key fingerprint with your administrator, then accept it on first login.
4. Lock private-key permissions:
   chmod 600 ~/.ssh/id_ed25519
5. Verify it works without prompts:
   ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 <SLURM_USER>@<login-host> 'hostname'

After that, rerun with SSH_KEY_PATH=~/.ssh/id_ed25519.
```

# Cosmos3-Policy-DROID Server

Cosmos 3 offers two post-trained policy models for DROID:

1. [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)
2. [Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)

Each of these policy models is served by a policy **Server** that streams actions to a **Client** driving a simulated or real robot. This example uses [`RoboLab`](https://github.com/NVlabs/RoboLab), a simulation benchmark for task-generalist policies, as the client. Start the server first, then connect the client.

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Policy Server](#policy-server)
- [Simulation Client](#simulation-client)

______________________________________________________________________

<!--TOC-->

## Policy Server

First, clone [`cosmos-framework`](https://github.com/NVIDIA/cosmos-framework):

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
```

Build the Docker image:

```bash
docker build \
  --build-arg INSTALL_APEX=0 \
  -t cosmos-framework:latest \
  .
```

Set your Hugging Face token and launch the container, which installs the dependencies:

```bash
# Set your Hugging Face token (https://huggingface.co/settings/tokens):
export HF_TOKEN=<your_hf_token>

docker run \
  -it \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  --net host \
  --rm \
  --runtime nvidia \
  -v .:/workspace \
  -v /workspace/.venv \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  cosmos-framework:latest \
  bash -c '\
    uv sync \
      --all-extras \
      --group=cu130-torch213-train \
      --group=policy-server && \
    exec bash; \
  '
```

Inside the container, start the policy server:

1. For [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID), run:

   ```
   python -m cosmos_framework.scripts.action_policy_server_robolab \
     --port 8000
   ```

2. For [Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID), run:

   ```
   python -m cosmos_framework.scripts.action_policy_server_robolab \
     --checkpoint-path nvidia/Cosmos3-Edge-Policy-DROID \
     --port 8000 \
     --format-prompt-as-json True \
     --guidance-interval 960 1001
   ```

   The guidance interval applies classifier-free guidance only to denoising
   timesteps in the inclusive range `[960, 1001]`. Omit
   `--guidance-interval` to apply guidance at every denoising step.

### Two-rank CFG parallelism

To serve with classifier-free guidance parallelized across two local GPUs, launch
exactly two processes and pass `--cfg-parallel`. Set `OMP_NUM_THREADS` to the
number of physical CPU cores available to the job divided by the two local
ranks. Without an explicit value, `torchrun` defaults each process to one OpenMP
thread, which can make request preprocessing slower.

For example, on a host where the job has 32 physical CPU cores available:

```bash
OMP_NUM_THREADS=16 torchrun --nproc-per-node=2 \
  -m cosmos_framework.scripts.action_policy_server_robolab \
  --cfg-parallel \
  --port 8000
```

Use cores assigned to the job rather than the host-wide CPU count when running
inside a container, CPU set, or scheduler allocation. The ratio is a starting
point; benchmark a few nearby values if CPU preprocessing is important to
end-to-end latency.

## Simulation Client

Clone [`RoboLab`](https://github.com/NVlabs/RoboLab):

```bash
git clone https://github.com/NVlabs/RoboLab.git
cd RoboLab
```

Build the Docker image:

```bash
./docker/build_docker.sh latest
```

Launch the container:

```bash
./docker/run_docker.sh latest
```

Run a task against the policy server. This opens a viewer window for real-time visualization of the simulation:

```bash
python policies/cosmos3/run.py \
  --task BananaInBowlTask
```

To evaluate across multiple sub-environments in parallel in headless mode:

```bash
python policies/cosmos3/run.py \
  --task BananaInBowlTask \
  --num-envs 10 \
  --headless
```

Example output:

<video controls width="864" height="480" src="https://github.com/user-attachments/assets/95a16737-5eb9-4b3f-a0ad-3a6b929b423f"></video>

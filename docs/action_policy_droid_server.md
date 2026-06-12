# Cosmos3-Nano-Policy-DROID Server

[Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID) is served by a policy **Server** that streams actions to a **Client** driving a simulated or real robot. This example uses [`RoboLab`](https://github.com/NVlabs/RoboLab), a simulation benchmark for task-generalist policies, as the client. Start the server first, then connect the client.

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
      --group=cu130-train \
      --group=policy-server && \
    exec bash; \
  '
```

The `--group=cu130-train` line targets CUDA 13.x drivers. On CUDA 12.x systems, replace it with `--group=cu128-train` (see the [Cosmos3 Cookbooks: Environment Setup](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/README.md) for details).

Inside the container, start the policy server:

```
python -m cosmos_framework.scripts.action_policy_server_robolab \
  --port 8000
```

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

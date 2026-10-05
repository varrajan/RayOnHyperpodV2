# Ray on SageMaker HyperPod — Studio Workflow

Fine-tune [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on medical reasoning data using Ray TorchTrainer with FSDP + LoRA on Amazon SageMaker HyperPod. Uses the new [Ray on HyperPod](https://aws.amazon.com/blogs/machine-learning/introducing-new-ray-capabilities-on-sagemaker-hyperpod/) integration in SageMaker Studio — no kubectl, Helm, or Kubernetes YAML required.

## Architecture

```
SageMaker Studio
├── Create Ray Cluster (Tasks tab → RayCluster → Create)
├── Open Ray Dashboard (IAM-authenticated URL)
├── Open Grafana (4 pre-built dashboards)
└── Connect Workspace (JupyterLab / Code Editor)
        │
        ▼
HyperPod EKS Cluster
├── Ray Head Node (no GPU)
├── 8x Ray Workers (g5.16xlarge, 1 GPU each)
├── FSx for Lustre (/mnt/fsx — shared storage)
├── KubeRay Operator (manages Ray CRDs)
├── Job Monitoring Agent (hung job detection)
└── HyperPod Observability → Amazon Managed Grafana
```

## Files

### `notebook.ipynb` — Step-by-step walkthrough

The main entry point. A 13-step Jupyter notebook that walks through the entire workflow:

1. Create a Ray cluster from SageMaker Studio
2. Access the Ray Dashboard
3. Connect a JupyterLab workspace to the cluster
4. Install dependencies
5. Prepare the medical reasoning dataset
6. Copy training script to FSx
7. Submit the training job (3 options: Jobs API, remote CLI, terminal)
8. Monitor via Ray Dashboard, Grafana, and hung job detection
9. Configure custom hung job detection (optional)
10. Verify model output
11. Upload model to S3
12. Scale and iterate
13. Clean up

Run this notebook inside a SageMaker Studio workspace that is attached to your Ray cluster.

### `train_ray.py` — Distributed training script

The training script that runs on Ray workers. It:

- Loads a pre-trained LLM (Qwen3-0.6B) with optional 4-bit quantization
- Applies LoRA adapters (rank 16, alpha 32) targeting attention and MLP layers
- Fine-tunes using `SFTTrainer` from the `trl` library with FSDP for distributed training
- Reports checkpoints to Ray Train via `RayTrainReportCallback` on each save step, enabling automatic recovery via `FailureConfig` when HyperPod replaces a failed node
- Merges LoRA weights into the base model after training (configurable)
- Reads all configuration from `args.yaml` — no hardcoded hyperparameters

### `args.yaml` — Training configuration

All tunable parameters in one file:

- **Cluster scaling** — `num_workers`, `num_gpus_per_worker`, `max_failures`
- **Model & data** — model ID, dataset paths on FSx, LoRA parameters, quantization settings
- **SFT training** — batch size, learning rate, scheduler, FSDP config, gradient checkpointing

Change `num_workers` from 1 to 8+ to scale from prototyping to full distributed training without touching the training script.

### `requirements.txt` — Python dependencies

Includes:

- **ML stack** — transformers, peft, trl, accelerate, datasets, torch, bitsandbytes
- **Ray** — `ray[data,train,tune]==2.56.1`
- **SageMaker integration** — `toolkit-for-ray-on-sagemaker-ai` (remote job submission via `sagemaker_ray://` address), `amzn-sagemaker-checkpointing` (tiered checkpointing: local disk → S3)

## Prerequisites

Four components must be installed on your HyperPod EKS cluster:

| Component | Purpose |
|-----------|---------|
| SageMaker Spaces EKS add-on | JupyterLab / Code Editor workspaces |
| HyperPod Observability EKS add-on | Metrics collection, Grafana dashboards |
| KubeRay operator | Manages RayCluster, RayJob, RayService CRDs |
| HyperPod Ray Endpoint Operator | Authenticated endpoints for dashboard and remote job submission |

Plus:
- A SageMaker Studio domain with access to the HyperPod cluster
- An FSx for Lustre file system attached to the cluster

## Instance Types

### Default: 1 GPU (Qwen3-0.6B with LoRA)

The default configuration uses a single worker. Qwen3-0.6B is a 0.6B parameter model, and LoRA only trains ~0.5% of parameters, so it fits comfortably on 1 GPU. Any single-GPU instance works:

| Role | Instance | Count | Why |
|------|----------|-------|-----|
| Worker | Any GPU instance (`g5.xlarge`+) | 1 | 1 GPU is enough for 0.6B + LoRA |
| Head / System | `m5.2xlarge` or similar | 1 | No GPU needed |

### Scaling Up for Larger Models

For models that don't fit on a single GPU (7B+), or for faster training on large datasets, increase `num_workers` in `args.yaml`. FSDP shards the model across workers.

| Model Size | Recommended | args.yaml changes |
|------------|-------------|-------------------|
| < 3B (e.g., Qwen3-0.6B) | 1 GPU | Default — `num_workers: 1` |
| 7B–13B | 2–4 GPUs | `num_workers: 4` |
| 30B–70B | 8+ GPUs | `num_workers: 8` |

Multi-GPU instance options:

| Instance | GPUs/node | args.yaml changes |
|----------|-----------|-------------------|
| `g5.16xlarge` | 1 A10G | `num_workers: N` across N nodes |
| `g5.12xlarge` | 4 A10G | `num_workers: 4, num_gpus_per_worker: 4` on 1 node |
| `g5.48xlarge` | 8 A10G | `num_workers: 8, num_gpus_per_worker: 8` on 1 node |
| `p4d.24xlarge` | 8 A100 | `num_workers: 8, num_gpus_per_worker: 8` on 1 node |

## Quick Start

1. Set up prerequisites (see above)
2. Create a Ray cluster from SageMaker Studio → HyperPod → Tasks tab → RayCluster
3. Create a JupyterLab space and attach it to the Ray cluster
4. Open `notebook.ipynb` in the workspace and follow the steps

## Remote Job Submission

From a laptop or CI/CD pipeline (no Studio required):

```bash
pip install toolkit-for-ray-on-sagemaker-ai
aws eks update-kubeconfig --name <eks-cluster-name> --region <region>

ray job submit \
    --address sagemaker_ray://<ray-cluster-name>/<namespace> \
    --working-dir . \
    -- python train_ray.py --config args.yaml
```

## Resilience

Three automatic layers, no code changes required:

| Layer | What happens |
|-------|-------------|
| **Node recovery** | HyperPod replaces faulty nodes → Ray reschedules pods → `FailureConfig` retries from last checkpoint |
| **Hung job detection** | Per-node Job Monitoring Agent detects stalls → alerts via CloudWatch and Grafana |
| **Tiered checkpointing** | Writes to local disk first, async uploads to S3 → faster recovery than restoring from S3 |

# Modal Multi-GPU Training Setup

This setup allows you to run the nanoGPT training on Modal with 8 H100 GPUs for distributed training.

## Prerequisites

1. Install Modal CLI:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal token new
```

## Usage

### Quick Start

Run the training with 8 H100s:
```bash
./run_modal.sh
```

Or directly with Python:
```bash
python modal_train.py
```

### Persistent Storage

The training automatically saves logs and checkpoints to a Modal volume for persistence across executions:

- **Logs**: Training logs and validation results
- **Checkpoints**: Model checkpoints (if enabled)
- **Volume Name**: `nanogpt-logs`

List logs from the volume:
```bash
./run_modal.sh logs
```

Or directly:
```bash
python modal_train.py logs
```

### What the Modal Script Does

1. **Image Creation**: Uses the updated Dockerfile to create a Modal image with:
   - Python 3.12.7
   - CUDA 12.6.2 support
   - uv for dependency management
   - All training data pre-copied for efficient layer caching

2. **Multi-GPU Setup**: 
   - Allocates 8 H100 GPUs
   - Sets up proper NCCL environment variables
   - Runs `torchrun --standalone --nproc_per_node=8 train_gpt.py`

3. **Resource Allocation**:
   - 8x H100 GPUs
   - 128GB RAM
   - 30-minute timeout
   - Automatic scaling and management

## Dockerfile Changes

The Dockerfile has been optimized for Modal usage:

- **Early Data Copying**: Data files are copied early in the build process for better layer caching
- **uv Integration**: Uses `uv sync` instead of pip for faster, more reliable dependency management
- **PyTorch Installation**: Installs PyTorch with CUDA 12.6 support via uv
- **Layer Optimization**: Dependencies are installed before copying application code for better caching

## Monitoring

You can monitor your training job through the Modal dashboard:
```bash
modal app list
```

## Cost Optimization

- The training will automatically scale down when complete
- You can stop the training early if needed through the Modal dashboard

## Troubleshooting

1. **Authentication Issues**: Make sure you're authenticated with `modal token new`
2. **GPU Allocation**: Ensure you have access to H100 instances in your Modal account
3. **Memory Issues**: The script allocates 128GB RAM; increase if needed
4. **Timeout Issues**: Training timeout is set to 30 minutes; adjust as needed

## Customization

To modify the training parameters, edit `modal_train.py`:
- Change GPU count: `gpu=modal.gpu.H100(count=N)`
- Adjust memory: `memory=XXXXX`
- Modify timeout: `timeout=XXXXX`

## Volume Management

The training uses a Modal volume named `nanogpt-logs` for persistent storage:

- **Automatic Creation**: The volume is created automatically if it doesn't exist
- **Persistent Storage**: Logs and checkpoints survive container restarts and new training runs
- **Shared Access**: Multiple training runs can access the same volume
- **Cost**: Volumes have a small monthly storage cost

To manage the volume manually:
```bash
# List volume contents
modal volume ls nanogpt-logs

# Delete volume (WARNING: This will delete all logs and checkpoints)
modal volume rm nanogpt-logs
``` 
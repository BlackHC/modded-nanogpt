import modal
import subprocess
import os
from pathlib import Path

# Create a Modal app
app = modal.App("nanogpt-training")

# Create a volume for persistent storage of logs and checkpoints
volume = modal.Volume.from_name("nanogpt-logs", create_if_missing=True)
volume_path = Path("/modded-nanogpt/logs")

# Create an image based on the Dockerfile
image = (
    modal.Image.from_registry("blackhc/nanogpt:latest")
    .add_local_file("train_gpt.py", remote_path="/modded-nanogpt/train_gpt.py")
)


def get_shared_env_config(gpu_count: int) -> dict:
    """Create shared environment configuration for training."""
    env = os.environ.copy()
    
    # GPU configuration
    cuda_devices = ",".join(str(i) for i in range(gpu_count))
    
    env.update({
        "CUDA_VISIBLE_DEVICES": cuda_devices,
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "0",
        "NCCL_P2P_DISABLE": "0",
        "TORCH_LOGS": "dynamo",
    })
    
    return env


# Create a stub for the training function
@app.function(
    image=image,
    gpu="H100:8",
    # 30-minute timeout
    timeout=30*60,
    # 64GiB RAM
    memory=2**16,
    # Mount the volumes for persistent logs
    volumes={
        str(volume_path): volume,
    },
)
def train_model():
    """Run the training script with multi-GPU support."""

    # Set environment variables for multi-GPU training
    env = get_shared_env_config(gpu_count=8)

    # Run the training script
    print("Starting multi-GPU training with 8 H100s...")
    print(f"Logs will be stored in: {volume_path}")
    print("Running: uv run torchrun --standalone --nproc_per_node=8 train_gpt.py")

    try:
        result = subprocess.run(
            ["uv", "run", "torchrun", "--standalone", "--nproc_per_node=8", "train_gpt.py"],
            env=env,
            cwd="/modded-nanogpt",
            check=True,
            # Let output stream to logs
            capture_output=False,
            text=True,
        )
        print("Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code: {e.returncode}")
        raise e


@app.function(
    volumes={str(volume_path): volume},
)
def list_logs():
    """List all logs and checkpoints stored in the volume."""
    if not volume_path.exists():
        print("No logs directory found in volume.")
        return

    print(f"Logs stored in volume at: {volume_path}")
    print("\nFiles found:")

    for item in volume_path.rglob("*"):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.relative_to(volume_path)} ({size_mb:.1f} MB)")
        elif item.is_dir():
            print(f"  {item.relative_to(volume_path)}/ (directory)")


@app.local_entrypoint()
def main():
    """Main entry point for the Modal app."""
    print("Starting Modal-based multi-GPU training...")

    # Run the training function
    result = train_model.remote()

    if result == 0:
        print("✅ Training completed successfully!")
    else:
        print(f"❌ Training failed with return code: {result}")


if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            list_logs.remote()

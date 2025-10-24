import modal
import subprocess
import os
from pathlib import Path

# Create a Modal app
app = modal.App("nanogpt-training")

# Create a volume for persistent storage of logs and checkpoints
volume = modal.Volume.from_name("nanogpt-logs", create_if_missing=True)
volume_path = Path("/modded-nanogpt/logs")

# Create a volume for torch compilation cache
compile_cache_volume = modal.Volume.from_name("torch-compile-cache", create_if_missing=True)
compile_cache_path = Path("/modded-nanogpt/torch_compile_cache")

# Create an image based on the Dockerfile
image = (
    modal.Image.from_registry("blackhc/nanogpt:latest")
    .add_local_file("train_gpt.py", remote_path="/modded-nanogpt/train_gpt.py")
)


def get_shared_env_config(gpu_count: int, compile_only: bool = False) -> dict:
    """Create shared environment configuration for training and compilation."""
    env = os.environ.copy()
    
    # GPU configuration
    cuda_devices = ",".join(str(i) for i in range(gpu_count))
    
    env.update({
        "CUDA_VISIBLE_DEVICES": cuda_devices,
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "0",
        "NCCL_P2P_DISABLE": "0",
        # Compilation caching environment variables
        "COMPILATION_ARTIFACTS_PATH": str(compile_cache_path),
        "TORCHINDUCTOR_AUTOGRAD_CACHE": str(compile_cache_path / "torchinductor_autograd_cache"),
        "TORCH_COMPILE_CACHE_DIR": str(compile_cache_path / "torch_compile_cache"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": str(compile_cache_path / "fx_graph_cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(compile_cache_path / "torchinductor_cache"),
        # "TORCHINDUCTOR_FREEZING": "1",
        "TORCH_LOGS": "dynamo",
        "DISABLE_COMPILE": "0",  # Set to "1" to disable compilation
        "COMPILE_ONLY": "1" if compile_only else "0",
        # "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
    })
    
    return env


# Create a stub for the training function
@app.function(
    image=image,
    gpu="H100:8",
    # 30-minute timeout
    timeout=30*60,
    # 128GiB RAM
    memory=2**17,
    # Mount the volumes for persistent logs and compilation cache
    volumes={
        str(volume_path): volume,
        str(compile_cache_path): compile_cache_volume,
    },
)
def train_model():
    """Run the training script with multi-GPU support and compilation caching."""

    # Set environment variables for multi-GPU training and compilation caching
    env = get_shared_env_config(gpu_count=8, compile_only=False)

    # Run the training script
    print("Starting multi-GPU training with 8 H100s...")
    print(f"Logs will be stored in: {volume_path}")
    print(f"Compilation cache will be stored in: {compile_cache_path}")
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
        # Commit the compile cache volume for future use.
        compile_cache_volume.commit()
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


@app.function(
    volumes={str(compile_cache_path): compile_cache_volume},
)
def list_compile_cache():
    """List all compilation cache files stored in the volume."""
    if not compile_cache_path.exists():
        print("No compilation cache directory found in volume.")
        return

    print(f"Compilation cache stored in volume at: {compile_cache_path}")
    print("\nCache files found:")

    cache_size = 0
    file_count = 0
    for item in compile_cache_path.rglob("*"):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            cache_size += size_mb
            file_count += 1
            print(f"  {item.relative_to(compile_cache_path)} ({size_mb:.1f} MB)")
        elif item.is_dir():
            print(f"  {item.relative_to(compile_cache_path)}/ (directory)")
    
    print(f"\nTotal cache size: {cache_size:.1f} MB across {file_count} files")


@app.local_entrypoint()
def main():
    """Main entry point for the Modal app."""
    print("Starting Modal-based multi-GPU training with compilation caching...")

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
            list_compile_cache.remote()
#!/usr/bin/env python3
"""
Kaggle Dual GPU Training Setup Script
This script sets up the environment and runs the manga OCR training on Kaggle's dual GPU.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def setup_kaggle_environment():
    """Set up the Kaggle environment for dual GPU training."""
    print("Setting up Kaggle environment for dual GPU training...")

    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s):")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("No GPU found! This script requires GPU acceleration.")
        return False

    # Set up data directories
    data_dirs = [
        "/kaggle/input/manga109-dataset",
        "/kaggle/input/manga-ocr-data",
        "/kaggle/working/manga_ocr_output"
    ]

    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

    return True

def install_dependencies():
    """Install required Python packages."""
    print("Installing dependencies...")

    # Core ML packages
    packages = [
        "torch>=1.12.0",
        "torchvision",
        "transformers>=4.12.5",
        "datasets",
        "wandb",
        "jiwer",
        "torchinfo",
        "albumentations>=1.1",
        "opencv-python",
        "matplotlib",
        "numpy",
        "pandas",
        "Pillow",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "tqdm",
        "fire",
        "budou",
        "html2image",
        "evaluate",
        # Japanese text processing
        "unidic-lite",
        "ipadic",
        "mecab-python3",
        "fugashi"
    ]

    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False

    return True

def setup_wandb():
    """Set up Weights & Biases for logging."""
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_key = user_secrets.get_secret("WANDB_API_KEY")

        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
            print("✓ WANDB API key configured")
            return True
        else:
            print("⚠ WANDB API key not found in Kaggle secrets")
            return False
    except ImportError:
        print("⚠ kaggle_secrets not available (not running on Kaggle?)")
        return False

def create_dual_gpu_script():
    """Create the main training script for dual GPU."""
    script_content = '''#!/usr/bin/env python3
"""
Dual GPU Manga OCR Training Script for Kaggle
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from training.train import run

def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set the device for this process
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def train_worker(rank, world_size, args):
    """Training worker function for distributed training."""
    setup_distributed(rank, world_size)

    try:
        # Run the training with dual GPU enabled
        run(
            run_name=args.get('run_name', 'kaggle_dual_gpu'),
            encoder_name=args.get('encoder_name', 'facebook/deit-tiny-patch16-224'),
            decoder_name=args.get('decoder_name', 'cl-tohoku/bert-base-japanese-char-v2'),
            max_len=args.get('max_len', 300),
            num_decoder_layers=args.get('num_decoder_layers', 2),
            batch_size=args.get('batch_size', 32),  # Will be adjusted for multi-GPU
            num_epochs=args.get('num_epochs', 8),
            fp16=args.get('fp16', True),
            use_dual_gpu=True
        )
    finally:
        cleanup_distributed()

def main():
    """Main training function."""
    # Training arguments
    args = {
        'run_name': 'kaggle_dual_gpu_v1',
        'encoder_name': 'facebook/deit-tiny-patch16-224',
        'decoder_name': 'cl-tohoku/bert-base-japanese-char-v2',
        'max_len': 300,
        'num_decoder_layers': 2,
        'batch_size': 32,  # Per GPU batch size
        'num_epochs': 8,
        'fp16': True,
    }

    world_size = torch.cuda.device_count()

    if world_size > 1:
        print(f"Starting distributed training on {world_size} GPUs...")
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("Single GPU mode...")
        run(**args, use_dual_gpu=False)

if __name__ == "__main__":
    main()
'''

    with open('/kaggle/working/train_dual_gpu.py', 'w') as f:
        f.write(script_content)

    print("✓ Created dual GPU training script")

def main():
    """Main setup function."""
    print("🚀 Setting up Manga OCR Training for Kaggle Dual GPU")
    print("=" * 60)

    if not setup_kaggle_environment():
        return

    if not install_dependencies():
        return

    setup_wandb()

    create_dual_gpu_script()

    print("\n" + "=" * 60)
    print("✅ Setup complete! Ready to train.")
    print("\nTo start training, run:")
    print("python /kaggle/working/train_dual_gpu.py")
    print("\nTo prevent Kaggle timeout, also run in a separate cell:")
    print("python keep_alive.py")

if __name__ == "__main__":
    main()

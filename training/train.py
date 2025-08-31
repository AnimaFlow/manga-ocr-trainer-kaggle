import fire
import wandb
import os
import torch
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator

from envpath.env import TRAIN_ROOT
from .dataset import MangaDataset
from .get_model import get_model
from .metrics import Metrics


# Kaggle-specific setup
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")
    IS_KAGGLE = True
except ImportError:
    secret_value_0 = os.environ.get("WANDB_API_KEY", "")
    IS_KAGGLE = False


def run(
    run_name="debug_dual",
    encoder_name="facebook/deit-tiny-patch16-224",
    decoder_name="cl-tohoku/bert-base-japanese-char-v2",
    max_len=300,
    num_decoder_layers=2,
    batch_size=64,
    num_epochs=8,
    fp16=True,
    use_dual_gpu=True,
):
    # Initialize wandb
    if secret_value_0:
        wandb.login(key=secret_value_0)
    wandb.init(project="manga-ocr", name=run_name)

    model, processor = get_model(encoder_name, decoder_name, max_len, num_decoder_layers)

    # Kaggle data paths
    if IS_KAGGLE:
        # Adjust batch size for dual GPU
        if use_dual_gpu and torch.cuda.device_count() > 1:
            batch_size = batch_size // 2  # Effective batch size will be doubled across GPUs
            print(f"Using {torch.cuda.device_count()} GPUs with adjusted batch_size: {batch_size}")
        else:
            use_dual_gpu = False
            print("Single GPU mode")

    # keep package 0 for validation
    train_dataset = MangaDataset(processor, "train", max_len, augment=True, skip_packages=[0])
    eval_dataset = MangaDataset(processor, "test", max_len, augment=False, skip_packages=range(1, 9999))

    metrics = Metrics(processor)

    training_args = TrainingArguments(
        output_dir=str(TRAIN_ROOT),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        save_steps=20000,
        eval_steps=20000,
        num_train_epochs=num_epochs,
        fp16=fp16,
        dataloader_num_workers=4,
        run_name=run_name,
        # Multi-GPU settings
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        # Gradient accumulation for better GPU utilization
        gradient_accumulation_steps=1,
        # Mixed precision settings
        bf16=False,  # Use fp16 instead
        tf32=True,
        # Memory optimization
        # Distributed training
        ddp_find_unused_parameters=False,
    )

    # Handle distributed training
    if use_dual_gpu and torch.cuda.device_count() > 1:
        # Set environment variables for distributed training
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())

        # Enable distributed training in transformers
        training_args.ddp_find_unused_parameters = False
        training_args.dataloader_num_workers = 2  # Reduce workers for stability

    # instantiate trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(run)

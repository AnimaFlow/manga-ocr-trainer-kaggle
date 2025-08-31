import fire
import wandb
import os
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
    run_name="manga_deit_tiny",
    encoder_name="facebook/deit-tiny-patch16-224",
    decoder_name="cl-tohoku/bert-base-japanese-char-v2",
    max_len=300,
    num_decoder_layers=2,
    batch_size=128,
    num_epochs=7,
    fp16=True,
):
    # Initialize wandb
    if secret_value_0:
        wandb.login(key=secret_value_0)
    wandb.init(project="manga-ocr", name=run_name)

    model, processor = get_model(encoder_name, decoder_name, max_len, num_decoder_layers)

    # keep package 0 for validation
    train_dataset = MangaDataset(processor, "train", max_len, augment=True, skip_packages=[0])
    eval_dataset = MangaDataset(processor, "test", max_len, augment=False, skip_packages=range(1, 9999))

    metrics = Metrics(processor)

    training_args = TrainingArguments(
        output_dir=str(TRAIN_ROOT),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1e-5,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        save_steps=20000,
        eval_steps=20000,
        num_train_epochs=num_epochs,
        metric_for_best_model="eval_loss",
        fp16=fp16,
        dataloader_num_workers=4,
        run_name=run_name,
        # Gradient accumulation for better GPU utilization
        gradient_accumulation_steps=1,
    )

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

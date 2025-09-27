import fire
import wandb
import os
import random
import numpy as np
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

# Colab-specific setup
try:
    from google.colab import userdata
    secret_value_0 = userdata.get('WANDB_API_KEY')
    IS_COLAB = True
except ImportError:
    secret_value_0 = os.environ.get("WANDB_API_KEY", "")
    IS_COLAB = False


os.environ["WANDB_PROJECT"] = "manga-ocr"

def run(
    run_name="manga_deit_tiny_hyperparam3",
    encoder_name="facebook/deit-tiny-patch16-224",
    decoder_name="cl-tohoku/bert-base-japanese-char-v2",
    max_len=300,
    num_decoder_layers=2,
    batch_size=64,
    num_epochs=8,
    fp16=True,
    grad_accum=2,
    seval_steps=10000,
    logging_steps=100,
    seed=42
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize wandb
    if IS_KAGGLE or IS_COLAB:
        wandb.login(key=secret_value_0)

    model, processor = get_model(encoder_name, decoder_name, max_len, num_decoder_layers)

    # keep package 0 for validation
    train_dataset = MangaDataset(processor, "train", max_len, augment=True, skip_packages=[0])
    eval_dataset = MangaDataset(processor, "test", max_len, augment=False, skip_packages=range(1, 9999))

    metrics = Metrics(processor)

    training_args = TrainingArguments(
        output_dir=str(TRAIN_ROOT),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1.61803e-4,
        weight_decay=0.05,
        warmup_ratio=0.06,
        max_grad_norm=1.0,


        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=logging_steps,
        save_steps=seval_steps,
        eval_steps=seval_steps,
        num_train_epochs=num_epochs,
        metric_for_best_model="eval_loss",
        greater_is_better=False,


        fp16=fp16,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        run_name=run_name,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=grad_accum,
        report_to="wandb",
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

    # Save the trained model and processor components
    trainer.save_model(str(TRAIN_ROOT))
    processor.tokenizer.save_pretrained(str(TRAIN_ROOT))
    processor.image_processor.save_pretrained(str(TRAIN_ROOT))


if __name__ == "__main__":
    fire.Fire(run)

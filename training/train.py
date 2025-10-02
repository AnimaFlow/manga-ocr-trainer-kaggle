import fire
import wandb
import os
import random
import numpy as np
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback , EarlyStoppingCallback
from transformers.data.data_collator import default_data_collator

from envpath.env import TRAIN_ROOT
from .dataset import MangaDataset
from .get_model import get_model
from .metrics import Metrics

IS_COLAB  = "COLAB_RELEASE_TAG" in os.environ
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/input")

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

if IS_KAGGLE and not WANDB_API_KEY:
    try:
        from kaggle_secrets import UserSecretsClient
        WANDB_API_KEY = UserSecretsClient().get_secret("WANDB_API_KEY") or WANDB_API_KEY
    except Exception:
        pass  

if IS_COLAB and not WANDB_API_KEY:
    try:
        from google.colab import userdata
        WANDB_API_KEY = userdata.get("WANDB_API_KEY") or WANDB_API_KEY
    except Exception:
        pass


os.environ["WANDB_PROJECT"] = "manga-ocr"

class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_dir):
            self._save_processor_components(checkpoint_dir)

    def _save_processor_components(self, save_dir):
        self.processor.tokenizer.save_pretrained(save_dir)
        self.processor.image_processor.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)


def run(
    run_name="manga_deit_tiny_hyperparam3_exp",
    encoder_name="facebook/deit-tiny-patch16-224",
    decoder_name="cl-tohoku/bert-base-japanese-char-v2",
    max_len=300,
    num_decoder_layers=2,
    batch_size=64,
    num_epochs=8,
    fp16=False,
    grad_accum=2,
    seval_steps=10000,
    logging_steps=100,
    seed=42,
    early_stopping_patience=2,
    early_stopping_threshold=0.0
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if WANDB_API_KEY:
        try:
            wandb.login(key=WANDB_API_KEY)
        except Exception as e:
            print(f"[WARN] wandb.login failed: {e}")
    else:
        print("[INFO] WANDB_API_KEY not found; proceeding without W&B login.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, processor = get_model(encoder_name, decoder_name, max_len, num_decoder_layers)
    model = model.to(device)

    # keep package 0 for validation
    train_dataset = MangaDataset(processor, "train", max_len, augment=True, skip_packages=[0])
    eval_dataset = MangaDataset(processor, "test", max_len, augment=False, skip_packages=range(1, 9999))

    metrics = Metrics(processor)

    training_args = Seq2SeqTrainingArguments(
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
        metric_for_best_model="cer",
        greater_is_better=False,
        load_best_model_at_end=True,


        fp16=torch.cuda.is_available() and fp16,
        dataloader_num_workers=2,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=False,
        run_name=run_name,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=grad_accum,
        report_to="none",

        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=max_len,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            ),
            SaveProcessorCallback(processor)
        ]
    )

    trainer.train()

    final_model_dir = os.path.join(str(TRAIN_ROOT), "final_model")
    os.makedirs(final_model_dir, exist_ok=True)

    trainer.save_model(final_model_dir)
    processor.tokenizer.save_pretrained(final_model_dir)
    processor.image_processor.save_pretrained(final_model_dir)

    print(f"\n final model saved to: {final_model_dir}")
    print("model components saved:")
    print("model weights: config.json, pytorch_model.bin")
    print("tokenizer: tokenizer_config.json, vocab.txt, etc.")
    print("image processor: preprocessor_config.json")
    print(f"\n checkpoints saved in: {str(TRAIN_ROOT)}/checkpoint-*")

if __name__ == "__main__":
    fire.Fire(run)

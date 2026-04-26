# =========================
# TRAINING SCRIPT
# =========================

# Import libraries
import inspect
import json
import logging
import importlib.util
import platform
import sys
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from common import CACHE_DIR, MODEL_DIR, RESULTS_DIR

def setup_logging():
    """Configure consistent console and file logging for the training run."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("bart_trainer")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(RESULTS_DIR / "training.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# -------------------------
# STEP 1: LOAD DATASET
# -------------------------
def main():
    logger = setup_logging()
    start_time = time.perf_counter()

    logger.info("Starting BART fine-tuning run")
    logger.info("Python %s | PyTorch %s | Platform %s", sys.version.split()[0], torch.__version__, platform.platform())
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))

    logger.info("Loading CNN/DailyMail dataset...")
    
    try:
        # CNN/DailyMail dataset (articles + summaries)
        dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir=str(CACHE_DIR / "datasets"))

        # Use SMALL subset (important for local systems)
        train_data = dataset["train"].select(range(3000))
        val_data = dataset["validation"].select(range(800))
        logger.info("Dataset loaded successfully: train=%d, validation=%d", len(train_data), len(val_data))
    except Exception as e:
        logger.exception("Failed to load dataset")
        raise

# -------------------------
# STEP 2: LOAD MODEL + TOKENIZER
# -------------------------
    logger.info("Loading model and tokenizer...")
    model_name = "facebook/bart-large-cnn"

    try:
        model_cache_dir = CACHE_DIR / "models"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(model_cache_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=str(model_cache_dir))
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        logger.info(
            "Model loaded: %s | parameters=%d | trainable=%d",
            model_name,
            total_parameters,
            trainable_parameters,
        )
    except Exception as e:
        logger.exception("Failed to load model")
        raise

# -------------------------
# STEP 3: PREPROCESSING FUNCTION
# -------------------------
    def preprocess(example):
        """
        Converts text into token IDs that model understands.
        """
        # Tokenize input article
        inputs = tokenizer(
            example["article"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )

        # Tokenize target summary
        targets = tokenizer(
            example["highlights"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )

        # Labels are expected outputs
        inputs["labels"] = targets["input_ids"]

        return inputs

    # Apply preprocessing
    logger.info("Preprocessing datasets...")
    train_data = train_data.map(preprocess, batched=True, desc="Preprocessing train")
    val_data = val_data.map(preprocess, batched=True, desc="Preprocessing val")
    logger.info("Preprocessing complete")
    logger.info("Training configuration will log every %d steps and evaluate every %d steps", 50, 100)

# -------------------------
# STEP 4: CONFIGURE TRAINING
    logger.info("Configuring training arguments...")

    training_arguments_parameters = inspect.signature(TrainingArguments.__init__).parameters
    training_strategy_key = "evaluation_strategy" if "evaluation_strategy" in training_arguments_parameters else "eval_strategy"
    logger.info("Using training strategy argument: %s", training_strategy_key)
    tensorboard_available = importlib.util.find_spec("tensorboard") is not None
    if tensorboard_available:
        logger.info("TensorBoard detected; training metrics will also be written there")
    else:
        logger.warning("TensorBoard is not installed; skipping TensorBoard reporting to avoid Trainer errors")
    
    training_args_kwargs = {
        "output_dir": str(RESULTS_DIR),
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "num_train_epochs": 3,
        training_strategy_key: "steps",
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": 50,
        "learning_rate": 5e-5,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 2,
        "fp16": torch.cuda.is_available(),
        "report_to": ["tensorboard"] if tensorboard_available else [],
        "run_name": "bart-finetuning",
        "push_to_hub": False,
    }
    training_args = TrainingArguments(**training_args_kwargs)

# -------------------------
# STEP 5: TRAINER WITH EARLY STOPPING
# -------------------------
    logger.info("Setting up trainer with early stopping...")
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=2,  # Stop if no improvement for 2 evaluations
        early_stopping_threshold=0.001
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[early_stopping],
    )

    def latest_checkpoint(directory):
        checkpoints = []
        if directory.exists():
            for path in directory.glob("checkpoint-*"):
                try:
                    step = int(path.name.split("-")[-1])
                    checkpoints.append((step, path))
                except ValueError:
                    continue
        if not checkpoints:
            return None
        return str(max(checkpoints, key=lambda item: item[0])[1])

    resume_checkpoint = latest_checkpoint(RESULTS_DIR)
    if resume_checkpoint:
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)

# -------------------------
# STEP 6: TRAIN MODEL
# -------------------------
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info("Training completed")
    if train_result and hasattr(train_result, "metrics"):
        logger.info("Final training metrics: %s", json.dumps(train_result.metrics, indent=2, default=str))

# -------------------------
# STEP 7: SAVE MODEL AND METADATA
# -------------------------
    logger.info("Saving model and tokenizer...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    
    # Save training config for reproducibility
    config = {
        "model_name": model_name,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "epochs": training_args.num_train_epochs,
        "warmup_steps": training_args.warmup_steps,
    }
    with open(MODEL_DIR / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    elapsed_seconds = time.perf_counter() - start_time
    with open(RESULTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "checkpoint_resumed_from": resume_checkpoint,
                "runtime_minutes": round(elapsed_seconds / 60.0, 2),
                "final_metrics": getattr(train_result, "metrics", {}),
            },
            f,
            indent=2,
            default=str,
        )

    logger.info("Model and tokenizer saved successfully in %s", MODEL_DIR)
    logger.info("Training artifacts saved to %s", RESULTS_DIR)
    logger.info("Total runtime: %.2f minutes", elapsed_seconds / 60.0)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
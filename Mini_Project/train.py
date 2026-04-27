# =========================
# TRAINING SCRIPT (GPU-Optimized)
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
import torch.backends.cudnn as cudnn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from common import CACHE_DIR, MODEL_DIR, RESULTS_DIR

# -------------------------
# GPU SETUP
# -------------------------
def setup_gpu():
    """Configure GPU for maximum performance."""
    if not torch.cuda.is_available():
        return torch.device("cpu"), 0

    device = torch.device("cuda")

    # Enable cuDNN auto-tuner — finds fastest conv algorithms for your hardware
    cudnn.benchmark = True
    # Deterministic mode OFF for speed (set True only if you need exact reproducibility)
    cudnn.deterministic = False

    # Allow TF32 on Ampere+ GPUs (RTX 30xx, A100, etc.) — faster matmuls with minimal precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_gpus = torch.cuda.device_count()
    return device, num_gpus


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
    set_seed(42)

    # --- GPU Setup ---
    device, num_gpus = setup_gpu()
    using_gpu = device.type == "cuda"

    logger.info("Starting BART fine-tuning run (GPU-Optimized)")
    logger.info(
        "Python %s | PyTorch %s | Platform %s",
        sys.version.split()[0],
        torch.__version__,
        platform.platform(),
    )
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if using_gpu:
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                "GPU %d: %s | VRAM: %.1f GB | CUDA Capability: %d.%d",
                i,
                props.name,
                props.total_memory / 1e9,
                props.major,
                props.minor,
            )
        logger.info("cuDNN benchmark mode: ENABLED")
        logger.info("TF32 matmul: ENABLED")

    # Tune these based on your GPU VRAM:
    #   8 GB  VRAM -> train_batch=4,  eval_batch=8,  grad_accum=4
    #  16 GB  VRAM -> train_batch=8,  eval_batch=16, grad_accum=2
    #  24 GB+ VRAM -> train_batch=16, eval_batch=32, grad_accum=1
    TRAIN_BATCH_SIZE = 4
    EVAL_BATCH_SIZE  = 8
    GRAD_ACCUM_STEPS = 4
    # Number of CPU workers for DataLoader — set to os.cpu_count() // 2 or 4, whichever is smaller
    import os
    NUM_WORKERS = min(4, (os.cpu_count() or 1) // 2)
    logger.info(
        "Batch config — train: %d | eval: %d | grad_accum: %d | effective_batch: %d | dataloader_workers: %d",
        TRAIN_BATCH_SIZE,
        EVAL_BATCH_SIZE,
        GRAD_ACCUM_STEPS,
        TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS * max(num_gpus, 1),
        NUM_WORKERS,
    )

    logger.info("Loading CNN/DailyMail dataset...")
    try:
        dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir=str(CACHE_DIR / "datasets"))
        train_data = dataset["train"].select(range(15000))
        val_data   = dataset["validation"].select(range(3000))
        logger.info(
            "Dataset loaded successfully: train=%d, validation=%d",
            len(train_data),
            len(val_data),
        )
    except Exception:
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
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=str(model_cache_dir),
            # Load directly in float16 on GPU to save VRAM and speed up transfer
            torch_dtype=torch.float16 if using_gpu else torch.float32,
        )

        # Disable KV-cache during training (required for gradient checkpointing)
        model.config.use_cache = False

        if using_gpu:
            # Explicitly move model to GPU before Trainer takes over
            model = model.to(device)
            # Gradient checkpointing trades compute for memory — lets you use larger batches
            model.gradient_checkpointing_enable()
            logger.info("Model moved to device: %s", next(model.parameters()).device)

            # Multi-GPU: wrap with DataParallel if more than one GPU is available
            if num_gpus > 1:
                logger.info("Multiple GPUs detected (%d) — enabling DataParallel", num_gpus)
                # NOTE: Seq2SeqTrainer handles multi-GPU natively via device_map / DDP.
                # DataParallel is only needed for manual training loops.
                # Trainer will handle distribution; leave model unwrapped here.

        total_parameters    = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model loaded: %s | parameters=%d | trainable=%d | dtype=%s",
            model_name,
            total_parameters,
            trainable_parameters,
            next(model.parameters()).dtype,
        )
    except Exception:
        logger.exception("Failed to load model")
        raise


# -------------------------
# STEP 3: PREPROCESSING FUNCTION
# -------------------------
    def preprocess(examples):
        """
        Tokenize articles and summaries.
        Runs on CPU via Dataset.map() — this is normal and expected.
        The tokenizer itself is CPU-bound; tensors move to GPU inside the Trainer's DataLoader.
        """
        inputs = tokenizer(
            examples["article"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )

        targets = tokenizer(
            text_target=examples["highlights"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )

        # Replace padding token ids in labels with -100 so loss ignores them.
        # Using a list comprehension (not tensor ops) because this runs pre-collation on CPU.
        labels = targets["input_ids"]
        inputs["labels"] = [
            [tok if tok != tokenizer.pad_token_id else -100 for tok in seq]
            for seq in labels
        ]

        return inputs

    logger.info("Preprocessing datasets (CPU — tensors move to GPU inside DataLoader)...")
    # Use all available CPU cores for fast tokenization
    train_data = train_data.map(
        preprocess,
        batched=True,
        num_proc=max(1, (os.cpu_count() or 1) // 2),
        desc="Preprocessing train",
        remove_columns=train_data.column_names,  # drop raw text columns to save RAM
    )
    val_data = val_data.map(
        preprocess,
        batched=True,
        num_proc=max(1, (os.cpu_count() or 1) // 2),
        desc="Preprocessing val",
        remove_columns=val_data.column_names,
    )

    # Tell HuggingFace datasets to return PyTorch tensors directly
    # (avoids an extra CPU→tensor conversion step inside the collator)
    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_data.set_format(type="torch",   columns=["input_ids", "attention_mask", "labels"])
    logger.info("Preprocessing complete — datasets formatted as torch tensors")


# -------------------------
# STEP 4: CONFIGURE TRAINING
# -------------------------
    logger.info("Configuring training arguments...")

    training_arguments_parameters = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    training_strategy_key = (
        "evaluation_strategy"
        if "evaluation_strategy" in training_arguments_parameters
        else "eval_strategy"
    )
    logger.info("Using training strategy argument: %s", training_strategy_key)

    tensorboard_available = importlib.util.find_spec("tensorboard") is not None
    if tensorboard_available:
        logger.info("TensorBoard detected — metrics will be written there")
    else:
        logger.warning("TensorBoard not installed — skipping TensorBoard reporting")

    training_args_kwargs = {
        "output_dir":                     str(RESULTS_DIR),
        "per_device_train_batch_size":    TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size":     EVAL_BATCH_SIZE,
        "num_train_epochs":               5,
        training_strategy_key:            "steps",
        "logging_strategy":               "steps",
        "logging_steps":                  50,          # log every 50 steps so you can monitor GPU util
        "eval_steps":                     100,
        "save_strategy":                  "steps",
        "save_steps":                     100,
        "save_total_limit":               3,
        "load_best_model_at_end":         True,
        "metric_for_best_model":          "eval_loss",
        "greater_is_better":              False,
        "learning_rate":                  3e-5,
        "warmup_ratio":                   0.06,
        "weight_decay":                   0.01,
        "gradient_accumulation_steps":    GRAD_ACCUM_STEPS,
        "lr_scheduler_type":              "cosine",
        "max_grad_norm":                  1.0,
        "optim":                          "adamw_torch_fused" if using_gpu else "adamw_torch",
        # adamw_torch_fused runs the optimizer kernel directly on GPU — faster than the default
        "fp16":                           using_gpu,   # mixed-precision: keeps activations in fp16, weights in fp32
        "fp16_opt_level":                 "O1",        # safe mixed-precision (O2 can cause instability)
        "dataloader_num_workers":         NUM_WORKERS, # parallel CPU workers feed GPU without starving it
        "dataloader_pin_memory":          using_gpu,   # pin_memory=True -> faster CPU→GPU transfers via DMA
        "dataloader_persistent_workers":  using_gpu and NUM_WORKERS > 0,  # keep workers alive between epochs
        "report_to":                      ["tensorboard"] if tensorboard_available else [],
        "run_name":                       "bart-finetuning",
        "push_to_hub":                    False,
        "seed":                           42,
        "disable_tqdm":                   False,
        # Torch compile (PyTorch 2.0+): fuses ops into a single GPU kernel — big speedup on A100/H100
        # Set to False if you hit errors (not all models support it yet)
        "torch_compile":                  False,
    }

    # Seq2Seq-specific args
    training_args_kwargs.update({
        "predict_with_generate":   True,
        "generation_max_length":   120,
        "generation_num_beams":    4,
        "label_smoothing_factor":  0.1,
        "gradient_checkpointing":  using_gpu,  # only enable on GPU; CPU training doesn't benefit
    })

    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)


# -------------------------
# STEP 5: DATA COLLATOR + TRAINER
# -------------------------
    logger.info("Setting up data collator and trainer...")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        # Pad to multiple of 8 on GPU: aligns tensors to memory boundaries,
        # enabling Tensor Core acceleration (requires fp16 and Ampere+ GPU)
        pad_to_multiple_of=8 if using_gpu else None,
        return_tensors="pt",
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
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

    # Confirm model is on GPU right before training starts
    if using_gpu:
        logger.info(
            "Pre-training device check — model: %s | VRAM allocated: %.2f GB",
            next(model.parameters()).device,
            torch.cuda.memory_allocated() / 1e9,
        )


# -------------------------
# STEP 6: TRAIN MODEL
# -------------------------
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info("Training completed")

    if using_gpu:
        logger.info(
            "Peak VRAM used: %.2f GB / %.2f GB",
            torch.cuda.max_memory_allocated() / 1e9,
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    if train_result and hasattr(train_result, "metrics"):
        logger.info(
            "Final training metrics: %s",
            json.dumps(train_result.metrics, indent=2, default=str),
        )


# -------------------------
# STEP 7: SAVE MODEL AND METADATA
# -------------------------
    logger.info("Saving model and tokenizer...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save in fp32 for portability (fp16 weights can cause issues with some inference runtimes)
    if using_gpu:
        model = model.float()

    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    config = {
        "model_name":        model_name,
        "train_samples":     len(train_data),
        "val_samples":       len(val_data),
        "learning_rate":     training_args.learning_rate,
        "batch_size":        training_args.per_device_train_batch_size,
        "effective_batch":   TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS * max(num_gpus, 1),
        "epochs":            training_args.num_train_epochs,
        "warmup_ratio":      training_args.warmup_ratio,
        "fp16":              training_args.fp16,
        "gpu":               torch.cuda.get_device_name(0) if using_gpu else "cpu",
    }
    with open(MODEL_DIR / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    elapsed_seconds = time.perf_counter() - start_time
    with open(RESULTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name":              model_name,
                "train_samples":           len(train_data),
                "val_samples":             len(val_data),
                "checkpoint_resumed_from": resume_checkpoint,
                "runtime_minutes":         round(elapsed_seconds / 60.0, 2),
                "final_metrics":           getattr(train_result, "metrics", {}),
                "peak_vram_gb":            round(torch.cuda.max_memory_allocated() / 1e9, 2) if using_gpu else 0,
            },
            f,
            indent=2,
            default=str,
        )

    logger.info("Model saved to %s", MODEL_DIR)
    logger.info("Training artifacts saved to %s", RESULTS_DIR)
    logger.info("Total runtime: %.2f minutes", elapsed_seconds / 60.0)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
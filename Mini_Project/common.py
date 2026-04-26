from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "cache"


def load_model_and_tokenizer(model_dir: Path = MODEL_DIR):
    """Load a locally trained summarization model and tokenizer."""
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found at '{model_dir}'. Run train.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def generate_summary(
    text: str,
    tokenizer,
    model,
    device,
    max_input_length: int = 512,
    max_summary_length: int = 100,
    num_beams: int = 4,
) -> str:
    """Generate a summary using shared decoding settings across scripts."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_summary_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

from pathlib import Path
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "cache"


def preprocess_text_for_summarization(text: str) -> str:
    """Normalize raw input and convert bullet-heavy text into prose-like paragraphs."""
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\t+", " ", normalized)
    normalized = re.sub(r"[ \u00A0]+", " ", normalized)

    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    bullet_pattern = re.compile(r"^(?:[-*•]|\d+[.)]|[a-zA-Z][.)])\s+")

    prose_parts = []
    current_bullets = []

    for line in lines:
        cleaned_line = bullet_pattern.sub("", line).strip()
        if bullet_pattern.match(line):
            if cleaned_line:
                current_bullets.append(cleaned_line)
            continue

        if current_bullets:
            prose_parts.append(" ".join(current_bullets))
            current_bullets = []
        prose_parts.append(line)

    if current_bullets:
        prose_parts.append(" ".join(current_bullets))

    merged = "\n\n".join(prose_parts)
    merged = re.sub(r"([.!?])\1{1,}", r"\1", merged)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def build_summary_prompt(cleaned_text: str) -> str:
    """Create a compact instruction prefix to bias the model toward concise summaries."""
    instruction = (
        "Summarize the following text in 3 to 4 concise sentences. "
        "Focus on key points only, remove redundancy, and avoid repeating phrases.\n\n"
        "Text:\n"
    )
    return f"{instruction}{cleaned_text}"


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
    max_summary_length: int = 110,
    min_summary_length: int = 28,
    num_beams: int = 6,
    length_penalty: float = 1.35,
    no_repeat_ngram_size: int = 4,
    repetition_penalty: float = 1.25,
    encoder_no_repeat_ngram_size: int = 3,
) -> str:
    """Generate a summary using shared decoding settings across scripts."""
    cleaned_text = preprocess_text_for_summarization(text)
    prompted_text = build_summary_prompt(cleaned_text)

    inputs = tokenizer(
        prompted_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    source_len = int(inputs["input_ids"].shape[-1])
    target_max_len = min(max_summary_length, max(48, int(source_len * 0.33)))
    target_min_len = max(20, min(min_summary_length, target_max_len - 10))

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.generate(
                    **inputs,
                    max_length=target_max_len,
                    min_length=target_min_len,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    repetition_penalty=repetition_penalty,
                    encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                    early_stopping=True,
                    do_sample=False,
                )
        else:
            output = model.generate(
                **inputs,
                max_length=target_max_len,
                min_length=target_min_len,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                early_stopping=True,
                do_sample=False,
            )

    return tokenizer.decode(output[0], skip_special_tokens=True)

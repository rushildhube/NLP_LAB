from pathlib import Path
import re
from typing import Dict, List, Sequence, Tuple

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

    # Remove common webpage boilerplate and support/footer fragments that are not part of the article.
    boilerplate_patterns = (
        r"back to the page you came from",
        r"for confidential support",
        r"call the samaritans",
        r"visit a local samaritans branch",
        r"see www\.samaritans\.org for details",
        r"www\.[a-z0-9\-_.]+\.[a-z]{2,}(?:/[\w\-./?%&=]*)?",
        r"^read more.*$",
        r"^advertisement.*$",
        r"^cookie(s)? policy.*$",
        r"^sign up.*$",
        r"^subscribe.*$",
        r"^share this.*$",
        r"^more from.*$",
    )

    scrubbed_lines = []
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in boilerplate_patterns):
            continue

        if re.fullmatch(r"https?://\S+", line) or re.fullmatch(r"www\.\S+", line, flags=re.IGNORECASE):
            continue

        scrubbed_lines.append(line)

    lines = scrubbed_lines
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


def build_summary_prompt(cleaned_text: str, summary_style: str = "balanced") -> str:
    """Build a BART-friendly prompt while avoiding instruction text leakage in outputs."""
    if summary_style == "bullet_mode":
        return f"summarize task notes: {cleaned_text}"
    if summary_style in {"crisp", "ultra_short"}:
        return f"summarize concisely: {cleaned_text}"
    return f"summarize: {cleaned_text}"


def _split_sentences(text: str) -> List[str]:
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _normalize_for_overlap(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _deduplicate_sentences(text: str, similarity_threshold: float = 0.9) -> str:
    sentences = _split_sentences(text)
    kept_sentences: List[str] = []
    kept_token_sets: List[set] = []

    for sentence in sentences:
        token_set = set(_normalize_for_overlap(sentence))
        if not token_set:
            continue

        is_duplicate = False
        for previous in kept_token_sets:
            intersection = len(token_set & previous)
            union = len(token_set | previous)
            similarity = (intersection / union) if union else 0.0
            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_sentences.append(sentence)
            kept_token_sets.append(token_set)

    return " ".join(kept_sentences).strip()


def _extractiveness_ratio(candidate: str, source: str, ngram_size: int = 6) -> float:
    cand_tokens = _normalize_for_overlap(candidate)
    src_tokens = set(_normalize_for_overlap(source))
    if len(cand_tokens) < ngram_size or not src_tokens:
        return 0.0

    total = 0
    copied = 0
    for start in range(0, len(cand_tokens) - ngram_size + 1):
        ngram = cand_tokens[start : start + ngram_size]
        total += 1
        if all(token in src_tokens for token in ngram):
            copied += 1

    return copied / total if total else 0.0


def _chunk_text_for_model(tokenizer, text: str, max_input_length: int, overlap_tokens: int = 48) -> List[str]:
    tokenized = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    token_ids: List[int] = tokenized["input_ids"]
    if len(token_ids) <= max_input_length:
        return [text]

    chunk_capacity = max_input_length - 32
    stride = max(64, chunk_capacity - overlap_tokens)
    chunks: List[str] = []

    for start in range(0, len(token_ids), stride):
        end = start + chunk_capacity
        segment_ids = token_ids[start:end]
        if not segment_ids:
            continue
        chunk_text = tokenizer.decode(segment_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(token_ids):
            break

    return chunks if chunks else [text]


def _candidate_score(candidate: str, source_text: str, target_words: int) -> float:
    words = _normalize_for_overlap(candidate)
    if not words:
        return -1e9

    word_count = len(words)
    compression_penalty = abs(word_count - target_words) / max(1, target_words)
    extractive_penalty = _extractiveness_ratio(candidate, source_text)

    sentence_count = max(1, len(_split_sentences(candidate)))
    sentence_balance_penalty = 0.15 if sentence_count < 2 else 0.0

    repeat_penalty = 0.0
    seen = set()
    for token in words:
        if token in seen:
            repeat_penalty += 0.01
        else:
            seen.add(token)

    score = 1.0 - (0.55 * compression_penalty + 0.25 * extractive_penalty + sentence_balance_penalty + repeat_penalty)
    return score


def _generate_once(
    prompt: str,
    tokenizer,
    model,
    device,
    max_input_length: int,
    max_summary_length: int,
    min_summary_length: int,
    num_beams: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
    encoder_no_repeat_ngram_size: int,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.generate(
                    **inputs,
                    max_length=max_summary_length,
                    min_length=min_summary_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    repetition_penalty=repetition_penalty,
                    encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                    early_stopping=False,
                    do_sample=False,
                )
        else:
            output = model.generate(
                **inputs,
                max_length=max_summary_length,
                min_length=min_summary_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                early_stopping=False,
                do_sample=False,
            )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


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
    candidate_count: int = 3,
    summary_style: str = "balanced",
) -> str:
    """Generate a summary using shared decoding settings across scripts."""
    cleaned_text = preprocess_text_for_summarization(text)

    style_overrides: Dict[str, Dict[str, float]] = {
        "crisp": {
            "max_summary_length": min(max_summary_length, 92),
            "min_summary_length": min(min_summary_length, 24),
            "length_penalty": max(length_penalty, 1.45),
            "repetition_penalty": max(repetition_penalty, 1.3),
        },
        "ultra_short": {
            "max_summary_length": min(max_summary_length, 72),
            "min_summary_length": min(min_summary_length, 18),
            "length_penalty": max(length_penalty, 1.55),
            "repetition_penalty": max(repetition_penalty, 1.32),
        },
        "bullet_mode": {
            "max_summary_length": min(max_summary_length, 96),
            "min_summary_length": min(min_summary_length, 24),
            "length_penalty": max(length_penalty, 1.4),
            "repetition_penalty": max(repetition_penalty, 1.3),
        },
    }

    if summary_style in style_overrides:
        override = style_overrides[summary_style]
        max_summary_length = int(override["max_summary_length"])
        min_summary_length = int(override["min_summary_length"])
        length_penalty = float(override["length_penalty"])
        repetition_penalty = float(override["repetition_penalty"])

    chunks = _chunk_text_for_model(tokenizer, cleaned_text, max_input_length=max_input_length)
    intermediate_summaries: List[str] = []
    for chunk in chunks:
        prompt = build_summary_prompt(chunk, summary_style=summary_style)
        summary = _generate_once(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_input_length=max_input_length,
            max_summary_length=max_summary_length,
            min_summary_length=min_summary_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        )
        summary = _deduplicate_sentences(summary)
        if summary:
            intermediate_summaries.append(summary)

    if not intermediate_summaries:
        return ""

    # Root-cause fix: if input fits a single chunk, avoid summarizing the summary again.
    # Use multi-candidate, length-aware decoding so output length responds to max/min controls.
    if len(intermediate_summaries) == 1:
        prompt = build_summary_prompt(cleaned_text, summary_style=summary_style)

        gap = max(0, max_summary_length - min_summary_length)
        target_word_count = max(min_summary_length, min(max_summary_length, min_summary_length + int(gap * 0.45)))
        length_min_grid = (
            max(12, min_summary_length),
            max(12, min_summary_length + int(gap * 0.2)),
            max(12, min_summary_length + int(gap * 0.35)),
            max(12, min_summary_length + int(gap * 0.5)),
        )
        length_penalty_grid = (
            max(0.82, length_penalty - 0.55),
            max(0.92, length_penalty - 0.4),
            max(1.0, length_penalty - 0.25),
            max(1.08, length_penalty - 0.12),
        )

        local_candidate_count = max(2, min(candidate_count + 1, 5))
        candidates: List[str] = []
        for idx in range(local_candidate_count):
            candidate = _generate_once(
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_input_length=max_input_length,
                max_summary_length=max_summary_length,
                min_summary_length=min(length_min_grid[idx], max_summary_length - 4),
                num_beams=max(6, num_beams),
                length_penalty=float(length_penalty_grid[idx]),
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=max(1.05, repetition_penalty - 0.08),
                encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            )
            candidate = _deduplicate_sentences(candidate)
            if candidate:
                candidates.append(candidate)

        if not candidates:
            return _deduplicate_sentences(intermediate_summaries[0])

        scored_candidates = sorted(
            (
                (_candidate_score(candidate, cleaned_text, target_word_count), candidate)
                for candidate in candidates
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        best_single = scored_candidates[0][1]
        min_word_floor = max(10, int(min_summary_length * 0.9))
        if len(best_single.split()) < min_word_floor:
            fallback = _generate_once(
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_input_length=max_input_length,
                max_summary_length=max_summary_length,
                min_summary_length=min(max_summary_length - 4, max(min_summary_length, min_summary_length + int(gap * 0.4))),
                num_beams=max(8, num_beams),
                length_penalty=max(0.78, length_penalty - 0.62),
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=max(1.02, repetition_penalty - 0.1),
                encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            )
            fallback = _deduplicate_sentences(fallback)
            if fallback:
                return fallback
        return best_single

    combined_source = cleaned_text
    fusion_input = " ".join(intermediate_summaries)

    prompt = build_summary_prompt(fusion_input, summary_style=summary_style)

    source_word_count = max(1, len(_normalize_for_overlap(cleaned_text)))
    target_min_len = max(12, min_summary_length)
    target_max_len = max(target_min_len + 4, max_summary_length)
    target_word_count = max(target_min_len, min(target_max_len, (target_min_len + target_max_len) // 2))

    candidate_count = max(1, min(candidate_count, 5))
    decoding_grid: Sequence[Tuple[float, float, int]] = (
        (length_penalty, repetition_penalty, no_repeat_ngram_size),
        (length_penalty + 0.08, repetition_penalty + 0.05, max(3, no_repeat_ngram_size)),
        (max(1.2, length_penalty - 0.08), repetition_penalty + 0.1, no_repeat_ngram_size + 1),
        (length_penalty + 0.12, repetition_penalty + 0.12, no_repeat_ngram_size + 1),
        (max(1.15, length_penalty - 0.12), repetition_penalty, no_repeat_ngram_size),
    )

    candidates: List[str] = []
    for idx in range(candidate_count):
        cfg = decoding_grid[idx]
        candidate = _generate_once(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_input_length=max_input_length,
            max_summary_length=target_max_len,
            min_summary_length=target_min_len,
            num_beams=max(4, num_beams),
            length_penalty=float(cfg[0]),
            no_repeat_ngram_size=int(cfg[2]),
            repetition_penalty=float(cfg[1]),
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        )
        candidate = _deduplicate_sentences(candidate)
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return _deduplicate_sentences(intermediate_summaries[0])

    scored_candidates = sorted(
        (
            (_candidate_score(candidate, combined_source, target_word_count), candidate)
            for candidate in candidates
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    best_summary = scored_candidates[0][1]

    # Guardrail: if summary is still too extractive, regenerate once with stricter settings.
    if _extractiveness_ratio(best_summary, cleaned_text) > 0.7:
        best_summary = _generate_once(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_input_length=max_input_length,
            max_summary_length=target_max_len,
            min_summary_length=target_min_len,
            num_beams=max(6, num_beams),
            length_penalty=length_penalty + 0.12,
            no_repeat_ngram_size=no_repeat_ngram_size + 1,
            repetition_penalty=repetition_penalty + 0.15,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        )

    return _deduplicate_sentences(best_summary)

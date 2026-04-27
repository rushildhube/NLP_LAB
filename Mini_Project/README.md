# Mini Project: Abstractive Summarization with BART

This project implements an end-to-end summarization system using a fine-tuned facebook/bart-large-cnn model on CNN/DailyMail.

It includes training, evaluation, a Streamlit app, and upgraded inference logic for stronger summaries without retraining.

## Scope

- Train a seq2seq transformer for abstractive summarization.
- Evaluate with ROUGE and optional BLEU/METEOR.
- Run interactive summarization in Streamlit.
- Improve quality at inference time without retraining.

## Project Files

- train.py: fine-tuning pipeline.
- evaluate_model.py: offline evaluation and report generation.
- app.py: Streamlit interface.
- common.py: shared paths, preprocessing, prompting, generation.
- quickstart.py: dependency check and workflow helper.
- model/: saved model/tokenizer artifacts.
- results/: training logs and checkpoints.
- evaluation_results.json: machine-readable metrics.
- evaluation_report.md: human-readable evaluation summary.

## Setup

Install dependencies from workspace root:

```bash
pip install -r requirements.txt
```

Recommended:

- Use an activated Python virtual environment.
- Use CUDA-enabled PyTorch for faster training and inference.

## Workflow

### Train

```bash
python Mini_Project/train.py
```

Current training behavior:

- Dataset: CNN/DailyMail with cache under Mini_Project/cache/datasets.
- Subset sizes: 15000 train, 3000 validation.
- Base model: facebook/bart-large-cnn.
- Token lengths: source 512, target 128.
- Auto-resume from latest checkpoint in Mini_Project/results.

Training hyperparameters:

| Parameter | Value |
| --- | --- |
| epochs | 5 |
| train batch size (per device) | 2 |
| eval batch size (per device) | 2 |
| gradient accumulation | 4 |
| learning rate | 3e-5 |
| warmup ratio | 0.06 |
| weight decay | 0.01 |
| scheduler | cosine |
| label smoothing | 0.1 |
| generation max length (eval) | 120 |
| generation beams (eval) | 4 |

Early stopping and checkpointing:

- Early stopping patience: 3 eval checks.
- Early stopping threshold: 0.001.
- Save/evaluate every 100 steps.
- Keep last 3 checkpoints.

Training outputs:

- Mini_Project/model (model, tokenizer, training_config.json).
- Mini_Project/results/training.log.
- Mini_Project/results/training_summary.json.

### Evaluate

```bash
python Mini_Project/evaluate_model.py
```

Current evaluation behavior:

- Evaluates on test[:200] from CNN/DailyMail.
- Calls shared generate_summary from common.py.
- Uses compact decoding defaults aligned with current inference pipeline.
- Computes ROUGE, and optionally BLEU/METEOR if available.
- Performs error analysis with per-sample ROUGE-1 and top-3 worst outputs.

Evaluation outputs:

- Mini_Project/evaluation_results.json.
- Mini_Project/evaluation_report.md.

### Run App

```bash
streamlit run Mini_Project/app.py
```

Current app behavior:

- Loads model/tokenizer once with Streamlit cache.
- Accepts pasted text or uploaded .txt input.
- Exposes style and decoding controls.
- Returns summary and basic word-count stats.

## Inference Pipeline (No Retraining Upgrades)

All runtime scripts depend on common.py, so inference behavior stays consistent across app and evaluation.

### 1) Preprocessing

Function: preprocess_text_for_summarization.

- Normalizes whitespace and line breaks.
- Converts bullet and task-like lines into prose blocks.
- Reduces repeated punctuation artifacts.

Impact:

- Structured inputs become easier for BART to summarize.
- Less noisy formatting in outputs.

### 2) Prompting

Function: build_summary_prompt.

- Adds instruction prefix targeting 3 to 4 concise sentences.
- Adds anti-repetition guidance.
- Adds structured-input hint for line-heavy text.

Impact:

- Better compression and cleaner summaries.

### 3) Chunk-Then-Fuse for Long Inputs

Functions: _chunk_text_for_model and generate_summary.

- Splits long text into overlapping token chunks.
- Summarizes each chunk.
- Summarizes merged chunk summaries again to produce final output.

Impact:

- Better long-input coverage with less truncation loss.

### 4) Multi-Candidate Reranking

Functions: generate_summary and \_candidate\_score.

- Generates multiple candidates with varied decoding settings.
- Scores candidates using compression fit, extractiveness penalty, sentence-count preference, and repetition penalty.
- Selects best candidate.

Impact:

- More stable and robust quality without retraining.

### 5) De-duplication and Extractiveness Guardrail

Functions: _deduplicate_sentences and _extractiveness_ratio.

- Removes near-duplicate sentences.
- If summary is too extractive, regenerates with stricter constraints.

Impact:

- Lower redundancy and less copy-like output.

### 6) GPU-Efficient Execution

Function: _generate_once.

- Uses torch.inference_mode.
- Uses CUDA autocast fp16 when GPU is available.

Impact:

- Better inference throughput.

## App Controls

Current sidebar controls:

- Summary style: Balanced, Crisp, Ultra-Short, Bullet Input Mode.
- Preset lengths: Short, Medium, Long.
- Max summary length.
- Min summary length.
- Beam count.
- Candidate count for reranking.

Style behavior in generate_summary:

- balanced: default profile.
- crisp: stronger compression and anti-repetition.
- ultra_short: most aggressive compression.
- bullet_mode: tuned for structured task-like text.

## Shared Runtime Contract

- app.py and evaluate_model.py both call generate_summary from common.py.
- load_model_and_tokenizer in common.py is the standard model-loading entry point.
- This keeps generation behavior synchronized across scripts.

## Practical Tuning (No Retraining)

For better quality:

- Use style Crisp or Bullet Input Mode for structured inputs.
- Keep candidate_count around 3 to 4.
- Keep beams around 4 to 6.

For faster responses:

- Set candidate_count to 1.
- Lower beams to 2 to 4.
- Use Balanced style.

For maximum compression:

- Use Ultra-Short style.
- Lower max summary length and min summary length.

## Troubleshooting

Model directory not found:

- Run python Mini_Project/train.py first.

Out of memory during training:

- Reduce per-device batch size.
- Reduce train and validation subset size in train.py.

Slow CPU inference:

- Reduce beams.
- Reduce candidate_count.
- Reduce max summary length.

Metric package issues:

- ROUGE is required by evaluate_model.py.
- BLEU and METEOR are optional and skip gracefully if unavailable.

## Notes

- quickstart.py is a helper script; current behavior is defined by train.py, evaluate_model.py, app.py, and common.py.
- This README reflects the current codebase, including recent inference upgrades.

## References

- [BART model card](https://huggingface.co/facebook/bart-large-cnn)
- [CNN/DailyMail dataset](https://huggingface.co/datasets/cnn_dailymail)
- [Transformers documentation](https://huggingface.co/docs/transformers)

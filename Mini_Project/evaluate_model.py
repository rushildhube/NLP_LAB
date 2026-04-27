# =========================
# EVALUATION SCRIPT
# =========================

import logging
import json
import time
import textwrap
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import evaluate
import numpy as np

from common import generate_summary, load_model_and_tokenizer, MODEL_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting evaluation...")
    start_time = time.time()
    
    # Load small test dataset
    logger.info("Loading test dataset...")
    try:
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:200]")
        logger.info(f"Loaded {len(dataset)} test samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Load trained model
    logger.info("Loading model...")
    try:
        tokenizer, model, device = load_model_and_tokenizer()
        logger.info(f"Model loaded on device: {device}")
        if hasattr(model, "generation_config"):
            model.generation_config.max_length = 110
            model.generation_config.min_length = 28
            model.generation_config.num_beams = 6
            model.generation_config.no_repeat_ngram_size = 4
            model.generation_config.repetition_penalty = 1.25
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Load metrics
    logger.info("Loading evaluation metrics...")
    try:
        rouge = evaluate.load("rouge")
    except Exception as e:
        logger.error(
            "Failed to load ROUGE metric. Install the dependency first with: pip install rouge_score"
        )
        raise

    try:
        bleu = evaluate.load("bleu")
    except Exception as e:
        logger.warning(f"BLEU metric could not be loaded: {e}")
        bleu = None

    try:
        meteor = evaluate.load("meteor")
    except Exception as e:
        logger.warning(f"METEOR metric could not be loaded: {e}")
        meteor = None

    logger.info("Metrics loaded: ROUGE%s%s",
                ", BLEU" if bleu else "",
                ", METEOR" if meteor else "")

    predictions = []
    references = []
    generation_times = []
    source_indices = []

# -------------------------
# GENERATE SUMMARIES WITH PROGRESS
# -------------------------
    logger.info("Generating summaries...")
    for i, item in enumerate(tqdm(dataset, desc="Generating predictions")):
        try:
            gen_start = time.time()
            pred = generate_summary(
                text=item["article"],
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_summary_length=110,
                min_summary_length=28,
                num_beams=6,
                length_penalty=1.35,
                no_repeat_ngram_size=4,
                repetition_penalty=1.25,
                encoder_no_repeat_ngram_size=3,
            )
            gen_time = time.time() - gen_start
            generation_times.append(gen_time)
            
            predictions.append(pred)
            references.append(item["highlights"])
            source_indices.append(i)
        except Exception as e:
            logger.warning(f"Failed to generate summary for sample {i}: {e}")
            continue

    logger.info(f"Generated {len(predictions)} summaries")
    if not predictions:
        logger.error("No summaries were generated successfully; cannot compute metrics.")
        return

# -------------------------
# CALCULATE MULTIPLE METRICS
# -------------------------
    logger.info("Computing metrics...")
    metrics_results = {}
    
    # ROUGE
    try:
        rouge_results = rouge.compute(
            predictions=predictions,
            references=references
        )
        metrics_results["ROUGE"] = rouge_results
        logger.info("✅ ROUGE computed")
    except Exception as e:
        logger.error(f"ROUGE computation failed: {e}")
    
    # BLEU
    if bleu:
        try:
            bleu_results = bleu.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            metrics_results["BLEU"] = bleu_results
            logger.info("✅ BLEU computed")
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
    
    # METEOR
    if meteor:
        try:
            meteor_results = meteor.compute(
                predictions=predictions,
                references=references
            )
            metrics_results["METEOR"] = meteor_results
            logger.info("✅ METEOR computed")
        except Exception as e:
            logger.warning(f"METEOR computation failed: {e}")

# -------------------------
# ERROR ANALYSIS
# -------------------------
    logger.info("Analyzing errors...")
    
    # Calculate ROUGE-1 per sample for error analysis
    rouge_1_scores = []
    for pred, ref in tqdm(list(zip(predictions, references)), desc="Scoring worst cases"):
        try:
            score = rouge.compute(
                predictions=[pred],
                references=[ref]
            )["rouge1"]
            rouge_1_scores.append(score)
        except:
            rouge_1_scores.append(0.0)
    
    # Find worst predictions
    worst_indices = np.argsort(rouge_1_scores)[:3]  # Top 3 worst
    
# -------------------------
# PRINT RESULTS SUMMARY
# -------------------------
    avg_generation_time = float(np.mean(generation_times)) if generation_times else 0.0
    rouge_results = metrics_results.get("ROUGE", {})
    bleu_score = metrics_results.get("BLEU", {}).get("bleu")
    meteor_score = metrics_results.get("METEOR", {}).get("meteor")

    def format_metric_rows():
        rows = []
        for name in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
            if name in rouge_results:
                rows.append(f"| {name:<9} | {rouge_results[name]:.4f} |")
        return "\n".join(rows) if rows else "| rouge | unavailable |"

    worst_blocks = []
    for rank, idx in enumerate(worst_indices, 1):
        sample_index = source_indices[idx]
        worst_blocks.append(
            textwrap.dedent(
                f"""
                ### {rank}. Sample #{sample_index}  |  ROUGE-1: {rouge_1_scores[idx]:.4f}
                **Article:** {dataset[sample_index]['article'][:220].replace('\n', ' ')}...
                
                **Reference:** {references[idx][:220].replace('\n', ' ')}...
                
                **Prediction:** {predictions[idx][:220].replace('\n', ' ')}...
                """
            ).strip()
        )

    report_text = textwrap.dedent(
        f"""
        # Evaluation Results Summary

        ## Dataset
        - Samples evaluated: {len(predictions)}
        - Average generation time: {avg_generation_time:.2f}s per sample

        ## ROUGE Scores
        | Metric | Score |
        |---|---:|
        {format_metric_rows()}

        ## Additional Metrics
        - BLEU: {f'{bleu_score:.4f}' if bleu_score is not None else 'unavailable'}
        - METEOR: {f'{meteor_score:.4f}' if meteor_score is not None else 'unavailable'}

        ## Top 3 Worst Predictions by ROUGE-1
        {'\n\n'.join(worst_blocks)}
        """
    ).strip() + "\n"

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Dataset: {len(predictions)} samples")
    print(f"Avg generation time: {avg_generation_time:.2f}s per sample")
    print("\nROUGE Scores")
    print("------------")
    print(format_metric_rows())
    print("\nAdditional Metrics")
    print("-------------------")
    print(f"BLEU: {f'{bleu_score:.4f}' if bleu_score is not None else 'unavailable'}")
    print(f"METEOR: {f'{meteor_score:.4f}' if meteor_score is not None else 'unavailable'}")
    print("\nTop 3 Worst Predictions")
    print("------------------------")
    for rank, idx in enumerate(worst_indices, 1):
        sample_index = source_indices[idx]
        print(f"{rank}. Sample #{sample_index} | ROUGE-1: {rouge_1_scores[idx]:.4f}")
        print(f"   Article: {dataset[sample_index]['article'][:140].replace(chr(10), ' ')}...")
        print(f"   Reference: {references[idx][:140].replace(chr(10), ' ')}...")
        print(f"   Prediction: {predictions[idx][:140].replace(chr(10), ' ')}...")
        print()

    # Save detailed results
    results_dict = {
        "metrics": metrics_results,
        "avg_generation_time": avg_generation_time,
        "total_samples": len(predictions),
        "sample_wise_rouge1": [float(s) for s in rouge_1_scores],
    }
    
    results_file = MODEL_DIR.parent / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    report_file = MODEL_DIR.parent / "evaluation_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Pretty report saved to {report_file}")
    
    print("="*70)
    elapsed = time.time() - start_time
    logger.info(f"Evaluation complete in {elapsed:.2f}s")


if __name__ == "__main__":
    main()